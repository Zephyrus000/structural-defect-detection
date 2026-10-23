"""Real-data forward/backward checks without optimizer updates or checkpoints."""

from __future__ import annotations

import json
import re
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from filelock import FileLock
from PIL import Image

from .config import ExperimentConfig
from .data.dataset import collate
from .data.probe import RealDataSampler
from .inference import PredictionWriter
from .losses import CompositeLoss
from .models import build_model
from .utils import resolve_device, runtime_metadata, seed_everything, write_json


def patch_image(tensor, config):
    """Recover the actual transformed RGB patch for visual inspection."""
    rgb = tensor.detach().cpu() * torch.tensor(config.data.std)[:, None, None]
    rgb += torch.tensor(config.data.mean)[:, None, None]
    return rgb.mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()


class TrainingProbe:
    def __init__(self, model, config, device):
        self.model, self.config, self.device = model, config, device
        self.criterion = CompositeLoss(config.loss)

    def check(self, batch):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        tensors = {k: batch[k].to(self.device) for k in ("image", "target", "valid")}
        if not bool(tensors["valid"].flatten(1).any(1).all()):
            raise ValueError("A selected training crop has no supervised pixels")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.config.runtime.amp,
            dtype=getattr(torch, self.config.runtime.amp_dtype),
        ):
            logits = self.model(tensors["image"])
        if logits.shape != tensors["target"].shape or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Invalid logits shape or non-finite logits")
        loss = self.criterion(logits, tensors["target"], tensors["valid"], step=0, epoch=0)
        if not bool(torch.isfinite(loss.total)):
            raise FloatingPointError("Non-finite real-data probe loss")
        ph_gradient = None
        if self.config.loss.ph.weight > 0:
            if loss.statistics.get("ph_terms", 0) == 0:
                raise ValueError("No fully valid PH windows in the selected training crops")
            gradient = torch.autograd.grad(loss.components["ph"], logits, retain_graph=True)[0]
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("Non-finite PH gradient")
            ph_gradient = float(gradient.detach().float().norm())
        # Match the initial training FP16 scale without constructing/updating an optimizer.
        scale = (
            128.0 if self.config.runtime.amp and self.config.runtime.amp_dtype == "float16" else 1.0
        )
        (loss.total * scale).backward()
        squared_norm = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(scale)
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError("Non-finite model gradient in real-data probe")
                squared_norm += float(parameter.grad.detach().double().square().sum())
        if squared_norm == 0:
            raise ValueError("Real-data probe did not produce a model gradient")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        report = {
            "batch_size": len(tensors["image"]),
            "logits_shape": list(logits.shape),
            "loss": {
                "total": float(loss.total.detach()),
                **{k: float(v.detach()) for k, v in loss.components.items()},
            },
            "ph": loss.statistics,
            "ph_gradient_norm": ph_gradient,
            "model_gradient_norm": squared_norm**0.5,
            "forward_backward_ms": (time.perf_counter() - started) * 1000,
            "optimizer_steps": 0,
        }
        probabilities = logits.detach().float().sigmoid().cpu().numpy()[:, 0]
        self.model.zero_grad(set_to_none=True)
        return report, probabilities


class DryRunRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self, run_id: str | None = None, layout: Path | None = None, max_samples: int = 2):
        if type(max_samples) is not int or max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        cfg = self.config
        run_id = run_id or f"dryrun-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id):
            raise ValueError("Invalid dry-run ID")
        output = Path(cfg.output.root) / "dry-runs" / run_id
        lock = nullcontext()
        if cfg.runtime.device.startswith("cuda"):
            locks = Path.cwd() / ".cache" / "device-locks"
            locks.mkdir(parents=True, exist_ok=True)
            index = cfg.runtime.device.partition(":")[2] or "0"
            lock = FileLock(str(locks / f"cuda-{index}.lock"), timeout=0)
        with lock:
            output.mkdir(parents=True, exist_ok=False)
            report = {
                "kind": "real_data_dry_run",
                "run_id": run_id,
                "status": "running",
                "output": str(output.resolve()),
                "config_hash": cfg.config_hash,
                "max_samples": max_samples,
                "samples": [],
                "batches": [],
                "optimizer_steps": 0,
                "selection": "first sorted training samples, in configured source order",
                "scope": "selected training crops only; not a full dataset/leakage audit or accuracy result",
            }
            write_json(output / "report.json", report)
            try:
                (output / "resolved.yaml").write_text(
                    yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8"
                )
                seed_everything(cfg.training.seed, cfg.runtime.threads)
                device = resolve_device(cfg.runtime.device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                report["environment"] = runtime_metadata(device)
                model = build_model(cfg.model).to(device)
                report["initialization"] = model.initialization_report
                probe = TrainingProbe(model, cfg, device)
                writer = PredictionWriter(output / "previews", cfg.evaluation.threshold)
                pending = []

                def check_batch():
                    batch_report, probabilities = probe.check(collate(pending))
                    report["batches"].append(batch_report)
                    for sample, probability in zip(pending, probabilities, strict=True):
                        metadata = sample["metadata"] | {
                            "coordinates_scope": "transformed_training_patch"
                        }
                        rgb = patch_image(sample["image"], cfg)
                        path = writer.write(rgb, probability, metadata)
                        preview = json.loads(path.read_text(encoding="utf-8"))
                        preview["coordinates"] = "transformed_training_patch_pixels"
                        write_json(path, preview)
                        key = path.stem
                        Image.fromarray(rgb).save(path.parent / f"{key}_input.png")
                        for name in ("target", "valid"):
                            mask = sample[name][0].numpy().astype(np.uint8) * 255
                            Image.fromarray(mask).save(path.parent / f"{key}_{name}.png")
                    pending.clear()
                    write_json(output / "report.json", report)

                for raw in RealDataSampler(cfg, layout).samples(max_samples):
                    transformed = raw.transformed(cfg)
                    report["samples"].append(
                        {
                            "sample": transformed["metadata"],
                            "evidence": raw.evidence,
                            "foreground_pixels": int(transformed["target"].sum()),
                            "valid_pixels": int(transformed["valid"].sum()),
                        }
                    )
                    pending.append(transformed)
                    if len(pending) == cfg.runtime.micro_batch:
                        check_batch()
                if pending:
                    check_batch()
                report.update(
                    status="completed",
                    samples_checked=len(report["samples"]),
                    samples_with_foreground=sum(
                        s["foreground_pixels"] > 0 for s in report["samples"]
                    ),
                    peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                )
                write_json(output / "report.json", report)
                return report
            except BaseException as exc:
                report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
                write_json(output / "report.json", report)
                raise
