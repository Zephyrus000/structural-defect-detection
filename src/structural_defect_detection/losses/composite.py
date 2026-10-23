from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..config import LossConfig
from .topology import PersistentHomologyLoss


@dataclass
class LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    statistics: dict


class CompositeLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        self.ph = PersistentHomologyLoss(config.ph) if config.ph.weight > 0 else None

    def forward(self, logits, target, valid, step: int = 0, epoch: int = 0) -> LossOutput:
        logits, target, weights = logits.float(), target.float(), valid.float()
        if logits.shape != target.shape or logits.shape != valid.shape:
            raise ValueError("Logits, target and valid mask must have the same shape")
        probability = logits.sigmoid()
        axes = (1, 2, 3)
        counts = weights.sum(axes)
        eligible = counts > 0
        if not bool(eligible.any()):
            zero = logits.sum() * 0
            return LossOutput(zero, {"dice": zero, "focal": zero, "ph": zero}, {"valid_samples": 0})
        dice = 1 - (2 * (probability * target * weights).sum(axes) + 1) / (
            ((probability + target) * weights).sum(axes) + 1
        )
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = probability * target + (1 - probability) * (1 - target)
        alpha = self.config.focal_alpha * target + (1 - self.config.focal_alpha) * (1 - target)
        focal = (alpha * (1 - pt).pow(self.config.focal_gamma) * ce * weights).sum(
            axes
        ) / counts.clamp_min(1)
        dice, focal = dice[eligible].mean(), focal[eligible].mean()
        ph_loss = logits.sum() * 0
        statistics = {"valid_samples": int(eligible.sum()), "ph_seconds": 0.0, "ph_weight": 0.0}
        if self.ph is not None and step % self.config.ph.every_n_steps == 0:
            started = time.perf_counter()
            ph_loss = self.ph(probability, target, valid)
            statistics.update(self.ph.last_stats)
            statistics["ph_seconds"] = time.perf_counter() - started
            warmup = self.config.ph.warmup_epochs
            statistics["ph_weight"] = self.config.ph.weight * (
                min(1.0, (epoch + 1) / warmup) if warmup else 1
            )
        total = (
            self.config.dice_weight * dice
            + self.config.focal_weight * focal
            + statistics["ph_weight"] * ph_loss
        )
        return LossOutput(total, {"dice": dice, "focal": focal, "ph": ph_loss}, statistics)
