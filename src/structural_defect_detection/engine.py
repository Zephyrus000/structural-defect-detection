"""Training and epoch-boundary checkpointing; no YAML or CLI dependencies."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .evaluation import Evaluator
from .losses import CompositeLoss
from .utils import write_json


class CheckpointManager:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, state: dict) -> Path:
        path = self.root / name
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)
        return path

    @staticmethod
    def load(path: str | Path) -> dict:
        # Training checkpoints contain RNG/optimizer objects: only load trusted local artifacts.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint version")
        return state

    @staticmethod
    def rng_state() -> dict:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        }

    @staticmethod
    def restore_rng(state: dict) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state["cuda"]:
            if len(state["cuda"]) != torch.cuda.device_count():
                raise ValueError(
                    "CUDA RNG device count changed; cannot restore this checkpoint faithfully"
                )
            torch.cuda.set_rng_state_all(state["cuda"])


class Trainer:
    def __init__(self, model, config: ExperimentConfig, data, device, output: Path, identity: dict):
        self.model, self.config, self.data, self.device = model, config, data, device
        self.output, self.identity = output, identity
        self.loss = CompositeLoss(config.loss)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.weight_decay,
        )
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.training.epochs)
            if config.training.scheduler == "cosine"
            else torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1)
        )
        self.amp_dtype = getattr(torch, config.runtime.amp_dtype)
        if (
            config.runtime.amp
            and self.amp_dtype == torch.bfloat16
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError(
                "bfloat16 AMP is not supported on this device; select float16 explicitly"
            )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.runtime.amp and self.amp_dtype == torch.float16, init_scale=128
        )
        self.checkpoints = CheckpointManager(output / "checkpoints")
        self.step, self.start_epoch, self.best = 0, 0, -1.0
        self.history = []

    def restore(self, path: Path):
        state = self.checkpoints.load(path)
        if state["identity"] != self.identity or state["amp"] != self.config.runtime.amp:
            raise ValueError("Checkpoint experiment/data/precision is incompatible with this run")
        if (
            self.config.runtime.amp
            and state.get("amp_dtype", "float16") != self.config.runtime.amp_dtype
        ):
            raise ValueError("Checkpoint AMP dtype is incompatible with this run")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.start_epoch, self.step, self.best = state["next_epoch"], state["step"], state["best"]
        self.history = state["history"]
        self.checkpoints.restore_rng(state["rng"])

    def _state(self, next_epoch):
        return {
            "format_version": 1,
            "image_hashes_by_split": self.data.report["image_hashes_by_split"],
            "identity": self.identity,
            "config": self.config.to_dict(),
            "amp": self.config.runtime.amp,
            "amp_dtype": self.config.runtime.amp_dtype,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": self.checkpoints.rng_state(),
            "next_epoch": next_epoch,
            "step": self.step,
            "best": self.best,
            "history": self.history,
            "sampler": {
                "epoch": next_epoch,
                "seed": self.config.training.seed,
                "policy": "deterministic_epoch_generator_and_sample_augmentation",
            },
        }

    def _update(self, count: int):
        self.scaler.unscale_(self.optimizer)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(count)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.training.grad_clip, error_if_nonfinite=True
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loader = self.data.train_loader(epoch)
        accumulation = self.config.training.effective_batch // self.config.runtime.micro_batch
        counts, pending, skipped, totals = (
            0,
            0,
            0,
            {"total": 0.0, "dice": 0.0, "focal": 0.0, "ph": 0.0},
        )
        ph_seconds, ph_terms, ph_skipped, ph_truncated = 0.0, 0, 0, 0
        for index, batch in enumerate(loader):
            eligible = batch["valid"].flatten(1).any(1)
            count = int(eligible.sum())
            skipped += len(eligible) - count
            if count:
                image, target, valid = (
                    batch[key][eligible].to(self.device) for key in ("image", "target", "valid")
                )
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.config.runtime.amp,
                    dtype=self.amp_dtype,
                ):
                    logits = self.model(image)
                # Persistence uses FP32; its discrete CPU pairing never runs in autocast.
                loss = self.loss(logits, target, valid, self.step, epoch)
                if not bool(torch.isfinite(loss.total)):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={self.step}")
                self.scaler.scale(loss.total * count).backward()
                pending += count
                counts += count
                totals["total"] += float(loss.total.detach()) * count
                for key, value in loss.components.items():
                    totals[key] += float(value.detach()) * count
                ph_seconds += loss.statistics.get("ph_seconds", 0)
                ph_terms += loss.statistics.get("ph_terms", 0)
                ph_skipped += loss.statistics.get("ph_skipped_windows", 0)
                ph_truncated += loss.statistics.get("ph_truncated_pairs", 0)
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                if pending:
                    self._update(pending)
                    pending = 0
        if not counts:
            raise ValueError("No valid training supervision in this epoch")
        if (
            self.config.loss.ph.weight > 0
            and not ph_terms
            and self.config.loss.ph.every_n_steps == 1
        ):
            raise ValueError(
                "PH enabled but every topology window was ignored; fix coverage/window settings explicitly"
            )
        return {
            **{k: v / counts for k, v in totals.items()},
            "samples": counts,
            "ignored_samples": skipped,
            "ph_seconds": ph_seconds,
            "ph_terms": ph_terms,
            "ph_skipped_windows": ph_skipped,
            "ph_truncated_pairs": ph_truncated,
        }

    def fit(self) -> dict:
        evaluator = Evaluator(self.model, self.config, self.device)
        for epoch in range(self.start_epoch, self.config.training.epochs):
            started = time.perf_counter()
            metrics = self._train_epoch(epoch)
            validation = evaluator.evaluate(
                self.data.select("validation"), "validation", self.output
            )
            score = validation["selection"]["crack_dice"]
            if score is None:
                raise ValueError("Validation has no valid supervised pixels")
            improved = score > self.best
            if improved:
                self.best = score
                write_json(self.output / "best_validation_metrics.json", validation)
            self.scheduler.step()
            entry = {
                "epoch": epoch,
                "optimizer_steps": self.step,
                "training": metrics,
                "validation_dice": score,
                "seconds": time.perf_counter() - started,
            }
            self.history.append(entry)
            state = self._state(epoch + 1)
            if improved:
                self.checkpoints.save("best.pt", state)
            self.checkpoints.save("last.pt", state)
            write_json(self.output / "history.json", self.history)
            print(
                f"epoch={epoch + 1}/{self.config.training.epochs} loss={metrics['total']:.5f} "
                f"val_dice={score:.5f} ph={metrics['ph']:.5f}",
                flush=True,
            )
        if not (self.output / "checkpoints" / "best.pt").exists():
            raise ValueError("Missing best checkpoint for completed training")
        self.model.load_state_dict(
            self.checkpoints.load(self.output / "checkpoints" / "best.pt")["model"]
        )
        best_metrics = evaluator.evaluate(
            self.data.select("validation"),
            "validation",
            self.output / "best",
            save_predictions=True,
        )
        return {
            "best_validation": best_metrics,
            "history": self.history,
            "optimizer_steps": self.step,
            "best_checkpoint": "checkpoints/best.pt",
            "last_checkpoint": "checkpoints/last.pt",
        }
