from __future__ import annotations

import time

import numpy as np
import torch

from .config import ExperimentConfig
from .data.dataset import collate
from .inference import Predictor
from .utils import runtime_metadata
from .verification import TrainingProbe, patch_image


class Profiler:
    def __init__(self, model, config: ExperimentConfig, device):
        self.model, self.config, self.device = model, config, device

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def measure(self, sample, iterations: int = 10, warmup: int = 2) -> dict:
        if iterations < 1 or warmup < 0:
            raise ValueError("Invalid profiling iteration count")
        self.model.eval()
        h, w = self.config.data.size
        transformed = sample.transformed(self.config)
        batch = collate([transformed])
        tensor = batch["image"].to(self.device)
        image = patch_image(transformed["image"], self.config)
        predictor = Predictor(self.model, self.config, self.device)
        timings = {}
        for name, action in (
            ("model", lambda: self.model(tensor)),
            ("pipeline", lambda: predictor.predict(image)),
        ):
            samples = []
            with torch.inference_mode():
                for i in range(iterations + warmup):
                    self._sync()
                    started = time.perf_counter()
                    action()
                    self._sync()
                    if i >= warmup:
                        samples.append((time.perf_counter() - started) * 1000)
            timings[name] = {
                "mean_ms": float(np.mean(samples)),
                "p95_ms": float(np.percentile(samples, 95)),
                "fps": 1000 / float(np.mean(samples)),
            }
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        probe, _ = TrainingProbe(self.model, self.config, self.device).check(batch)
        return {
            "environment": runtime_metadata(self.device),
            "size": [h, w],
            "batch": 1,
            "inference_precision": "fp32",
            "training_amp": self.config.runtime.amp,
            "training_amp_dtype": self.config.runtime.amp_dtype,
            "iterations": iterations,
            "warmup": warmup,
            "timings": timings,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "training_step_probe_ms": probe["forward_backward_ms"],
            "training_probe": probe,
            "ph": probe["ph"],
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else None,
            "pipeline_scope": "normalize + resize/tile + model + restore; excludes file IO and overlay writing",
            "energy_joules": None,
            "input": "real transformed training patch",
            "sample": transformed["metadata"],
            "evidence": sample.evidence,
        }
