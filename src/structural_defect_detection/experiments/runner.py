"""Run lifecycle, identities and failure provenance."""

from __future__ import annotations

import json
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from filelock import FileLock

from ..config import ExperimentConfig
from ..data import DataModule
from ..engine import CheckpointManager, Trainer
from ..evaluation import Evaluator
from ..models import build_model
from ..utils import file_hash, resolve_device, runtime_metadata, seed_everything, write_json


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(
        self,
        run_id: str | None = None,
        resume: Path | None = None,
        retry: bool = False,
        plan_id: str | None = None,
    ) -> dict:
        if self.config.runtime.device.startswith("cuda"):
            # Cross-process protection for runs launched from the same checkout,
            # even when their output roots differ. OS locks release on process exit.
            locks = Path.cwd() / ".cache" / "device-locks"
            locks.mkdir(parents=True, exist_ok=True)
            device_index = self.config.runtime.device.partition(":")[2] or "0"
            with FileLock(str(locks / f"cuda-{device_index}.lock"), timeout=0):
                return self._run(run_id, resume, retry, plan_id)
        return self._run(run_id, resume, retry, plan_id)

    def _run(
        self,
        run_id: str | None = None,
        resume: Path | None = None,
        retry: bool = False,
        plan_id: str | None = None,
    ) -> dict:
        cfg = self.config
        if resume:
            resume = resume.resolve()
            output = resume.parent.parent
            run_id = run_id or output.name
            if output.name != run_id:
                raise ValueError("Resume run ID mismatch")
        else:
            run_id = (
                run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}-{cfg.experiment_hash[:10]}"
            )
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id):
                raise ValueError("Invalid run ID")
            output = Path(cfg.output.root) / run_id
            if output.exists():
                if not retry or not (output / "state.json").exists():
                    raise FileExistsError(f"Run exists; use explicit resume or retry: {output}")
                previous = json.loads((output / "state.json").read_text())
                if previous["status"] != "failed":
                    raise ValueError("Only failed runs can be retried")
                checkpoint = output / "checkpoints" / "last.pt"
                if checkpoint.exists():
                    resume = checkpoint.resolve()
        output.mkdir(parents=True, exist_ok=True)
        if (output / "state.json").exists():
            previous = json.loads((output / "state.json").read_text())
            if (
                previous["experiment_hash"] != cfg.experiment_hash
                or previous.get("plan_id") != plan_id
            ):
                raise ValueError("Existing run identity/config does not match")
        lock = output / ".lock"
        try:
            with lock.open("x") as handle:
                handle.write(str(time.time()))
        except FileExistsError as exc:
            raise RuntimeError(
                f"Run lock exists: {lock}; inspect the owning process before removing a stale lock"
            ) from exc
        state = {
            "run_id": run_id,
            "plan_id": plan_id,
            "status": "running",
            "experiment_hash": cfg.experiment_hash,
            "config_hash": cfg.config_hash,
        }
        started = time.perf_counter()
        try:
            write_json(output / "state.json", state)
            (output / "resolved.yaml").write_text(
                yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8"
            )
            seed_everything(cfg.training.seed, cfg.runtime.threads)
            device = resolve_device(cfg.runtime.device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            data = DataModule(cfg)
            write_json(output / "data_report.json", data.report)
            identity = {
                "experiment_hash": cfg.experiment_hash,
                "manifest_hash": data.report["manifest_hash"],
                "split_hash": data.report["split_hash"],
                "initial_weights_hash": file_hash(Path(cfg.model.init_weights))
                if cfg.model.init_weights
                else None,
            }
            metadata = runtime_metadata(device)
            metadata.update(
                identity=identity,
                micro_batch=cfg.runtime.micro_batch,
                accumulation=cfg.training.effective_batch // cfg.runtime.micro_batch,
                effective_batch=cfg.training.effective_batch,
                amp=cfg.runtime.amp,
                amp_dtype=cfg.runtime.amp_dtype,
                input_size=cfg.data.size,
            )
            metadata_path = output / "environment.json"
            if metadata_path.exists():
                write_json(output / f"environment_resume_{time.time_ns()}.json", metadata)
            else:
                write_json(metadata_path, metadata)
            model = build_model(cfg.model, initialize=resume is None).to(device)
            if resume is None:
                write_json(output / "initialization.json", model.initialization_report)
            trainer = Trainer(model, cfg, data, device, output, identity)
            if resume:
                trainer.restore(resume)
            result = trainer.fit()
            result.update(state)
            result.update(
                status="completed",
                identity=identity,
                seed=cfg.training.seed,
                wall_seconds=time.perf_counter() - started,
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
            )
            write_json(output / "result.json", result)
            write_json(output / "state.json", state | {"status": "completed"})
            return result
        except BaseException as exc:
            write_json(
                output / "state.json",
                state
                | {
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        finally:
            lock.unlink(missing_ok=True)

    def load_model(self, checkpoint: Path, allow_external_data: bool = False):
        cfg = self.config
        seed_everything(cfg.training.seed, cfg.runtime.threads)
        device = resolve_device(cfg.runtime.device)
        state = CheckpointManager.load(checkpoint)
        comparable = cfg
        if allow_external_data:
            comparison = cfg.to_dict()
            comparison["data"]["sources"] = state["config"]["data"]["sources"]
            from ..config import ConfigLoader

            comparable = ConfigLoader.from_dict(comparison)
        if state["identity"]["experiment_hash"] != comparable.experiment_hash:
            raise ValueError(
                "Checkpoint/config mismatch; use the run's resolved.yaml with machine overrides"
            )
        model = build_model(cfg.model, initialize=False).to(device)
        model.load_state_dict(state["model"])
        return model, device, state

    def evaluate(self, checkpoint: Path, split: str, output: Path) -> dict:
        model, device, state = self.load_model(
            checkpoint, allow_external_data=split == "external_test"
        )
        data = DataModule(self.config)
        if (
            data.report["manifest_hash"] != state["identity"]["manifest_hash"]
            and split != "external_test"
        ):
            raise ValueError(
                "Evaluation manifest differs from training; use a declared external_test protocol"
            )
        if split == "external_test":
            if "image_hashes_by_split" not in state:
                raise ValueError(
                    "Checkpoint lacks training image fingerprints for external leakage audit"
                )
            training_images = set(state["image_hashes_by_split"]["train"]) | set(
                state["image_hashes_by_split"]["validation"]
            )
            if training_images & set(data.report["image_hashes_by_split"]["external_test"]):
                raise ValueError("External-test images overlap training/validation content")
        result = Evaluator(model, self.config, device).evaluate(
            data.select(split), split, output, save_predictions=True
        )
        result.update(
            checkpoint_hash=file_hash(checkpoint),
            evaluation_data=data.report,
            training_identity=state["identity"],
        )
        write_json(output / f"{split}_metrics.json", result)
        return result
