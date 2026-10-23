"""Cubical persistence diagram loss with differentiable critical-cell values.

Pairing and optimal assignment are discrete, computed on detached CPU data.
Loss values are gathered from the original tensor, retaining piecewise gradients.
H0 essential intervals are capped at filtration value 1, including empty targets.
Domains are closed, non-overlapping windows; windows touching ignore are skipped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import gudhi
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from ..config import PHConfig


class TopologyBackend(ABC):
    @abstractmethod
    def diagram(self, probability: torch.Tensor, dimension: int) -> torch.Tensor:
        """Return birth/death pairs with gradients to probability."""


class GudhiTopologyBackend(TopologyBackend):
    def __init__(self, min_persistence: float = 0.01, max_pairs: int = 64):
        self.min_persistence, self.max_pairs = min_persistence, max_pairs
        self.truncated_pairs = 0

    def diagram(self, probability: torch.Tensor, dimension: int = 0):
        filtration = 1.0 - probability.float()
        complex_ = gudhi.CubicalComplex(top_dimensional_cells=filtration.detach().cpu().numpy())
        complex_.compute_persistence(homology_coeff_field=2, min_persistence=0)
        finite, essential = complex_.cofaces_of_persistence_pairs()
        # GUDHI coface indices use Fortran order, even for a 2-D input array.
        flat = filtration.T.contiguous().view(-1)
        parts = []
        if dimension < len(finite) and len(finite[dimension]):
            indices = torch.as_tensor(finite[dimension], device=flat.device, dtype=torch.long)
            parts.append(flat[indices])
        if dimension == 0 and essential and len(essential[0]):
            indices = torch.as_tensor(essential[0], device=flat.device, dtype=torch.long)
            birth = flat[indices]
            parts.append(torch.stack([birth, torch.ones_like(birth)], dim=1))
        if not parts:
            return flat[:0].reshape(0, 2)
        diagram = torch.cat(parts)
        persistence = diagram[:, 1] - diagram[:, 0]
        diagram = diagram[persistence.detach() > self.min_persistence]
        self.truncated_pairs += max(0, len(diagram) - self.max_pairs)
        if len(diagram) > self.max_pairs:
            order = torch.argsort(
                (diagram[:, 1] - diagram[:, 0]).detach(), descending=True, stable=True
            )
            diagram = diagram[order[: self.max_pairs]]
        return diagram


def diagram_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared L2 optimal matching including diagonal assignments (unnormalized)."""
    n, m = len(prediction), len(target)
    if n == 0 and m == 0:
        return prediction.sum() * 0
    pred_diagonal = (prediction[:, 1] - prediction[:, 0]).square() / 2
    target_diagonal = (target[:, 1] - target[:, 0]).square() / 2
    if m == 0:
        return pred_diagonal.sum()
    if n == 0:
        return prediction.sum() * 0 + target_diagonal.sum()
    distances = (prediction[:, None, :] - target[None, :, :]).square().sum(-1)
    costs = np.full((n + m, n + m), np.inf, dtype=np.float64)
    costs[:n, :m] = distances.detach().cpu().numpy()
    costs[np.arange(n), m + np.arange(n)] = pred_diagonal.detach().cpu().numpy()
    costs[n + np.arange(m), np.arange(m)] = target_diagonal.detach().cpu().numpy()
    costs[n:, m:] = 0
    rows, cols = linear_sum_assignment(costs)
    loss = prediction.sum() * 0
    for row, col in zip(rows, cols, strict=True):
        if row < n and col < m:
            loss = loss + distances[row, col]
        elif row < n:
            loss = loss + pred_diagonal[row]
        elif col < m:
            loss = loss + target_diagonal[col]
    return loss


class PersistentHomologyLoss(nn.Module):
    def __init__(self, config: PHConfig, backend: TopologyBackend | None = None):
        super().__init__()
        self.config = config
        self.backend = backend or GudhiTopologyBackend(config.min_persistence, config.max_pairs)
        self.last_stats = {}

    def forward(self, probability, target, valid):
        samples, skipped, term_count = [], 0, 0
        if hasattr(self.backend, "truncated_pairs"):
            self.backend.truncated_pairs = 0
        size = self.config.window_size
        for batch in range(probability.shape[0]):
            if not bool(valid[batch].any()):
                continue
            terms = []
            for top in range(0, probability.shape[-2], size):
                for left in range(0, probability.shape[-1], size):
                    mask = valid[batch, 0, top : top + size, left : left + size]
                    if min(mask.shape) < 2 or not bool(mask.all()):
                        skipped += 1
                        continue
                    pred = probability[batch, 0, top : top + size, left : left + size]
                    truth = target[batch, 0, top : top + size, left : left + size]
                    for dimension in self.config.dimensions:
                        predicted_diagram = self.backend.diagram(pred, dimension)
                        with torch.no_grad():
                            target_diagram = self.backend.diagram(truth, dimension)
                        terms.append(diagram_distance(predicted_diagram, target_diagram))
            term_count += len(terms)
            samples.append(torch.stack(terms).mean() if terms else probability[batch].sum() * 0)
        self.last_stats = {
            "ph_terms": term_count,
            "ph_skipped_windows": skipped,
            "ph_truncated_pairs": getattr(self.backend, "truncated_pairs", 0),
        }
        return torch.stack(samples).mean() if samples else probability.sum() * 0
