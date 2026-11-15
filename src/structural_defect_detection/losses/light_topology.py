"""PyTorch topology losses for binary 2-D segmentation."""

import torch
from torch.nn import functional as F


def _soft_erode(image):
    vertical = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_skeleton(image):
    eroded = _soft_erode(image)
    skeleton = (image - F.max_pool2d(eroded, 3, stride=1, padding=1)).relu()
    for _ in range(5):
        image = eroded
        eroded = _soft_erode(image)
        detail = (image - F.max_pool2d(eroded, 3, stride=1, padding=1)).relu()
        skeleton = skeleton + (detail * (1 - skeleton)).relu()
    return skeleton


def soft_cldice_loss(probability, target, valid):
    probability = probability * valid
    target = target * valid
    pred_skeleton = _soft_skeleton(probability) * valid
    truth_skeleton = _soft_skeleton(target) * valid
    axes = (1, 2, 3)
    pred_mass = pred_skeleton.sum(axes)
    truth_mass = truth_skeleton.sum(axes)
    eps = 1e-6
    precision = ((pred_skeleton * target).sum(axes) + eps) / (pred_mass + eps)
    recall = ((truth_skeleton * probability).sum(axes) + eps) / (truth_mass + eps)
    loss = 1 - 2 * precision * recall / (precision + recall)
    positive = truth_mass > 0
    supervised = valid.flatten(1).any(1)
    return (loss * positive).sum() / supervised.sum().clamp_min(1)


def euler_curve(image, thresholds, *, smooth):
    """Euler characteristic of unions of closed pixel squares at superlevel thresholds."""
    thresholds = thresholds[:, None, None]
    if smooth:
        lower = (-20 * thresholds).sigmoid()
        upper = (20 * (1 - thresholds)).sigmoid()
        occupancy = ((20 * (image.unsqueeze(-3) - thresholds)).sigmoid() - lower) / (upper - lower)
    else:
        occupancy = (image.unsqueeze(-3) >= thresholds).to(image.dtype)
    padded = F.pad(occupancy, (1, 1, 1, 1))
    a, b = padded[..., :-1, :-1], padded[..., :-1, 1:]
    c, d = padded[..., 1:, :-1], padded[..., 1:, 1:]
    # At each vertex, count that vertex, its right and lower edges, and the lower-right face.
    right = torch.maximum(b, d)
    down = torch.maximum(c, d)
    vertex = torch.maximum(torch.maximum(a, c), right)
    return (vertex - right - down + d).sum((-2, -1))


def euler_curve_loss(probability, target, valid):
    size = min(32, *probability.shape[-2:])
    # Edge windows smaller than the chosen square are omitted.
    pred = probability.unfold(-2, size, size).unfold(-2, size, size)
    truth = target.unfold(-2, size, size).unfold(-2, size, size)
    usable = valid.unfold(-2, size, size).unfold(-2, size, size).all(dim=(-2, -1))
    thresholds = probability.new_tensor([0.2, 0.35, 0.5, 0.65, 0.8])
    pred_curve = euler_curve(pred, thresholds, smooth=True)
    expected = euler_curve(truth, thresholds, smooth=False)
    error = F.smooth_l1_loss(pred_curve, expected, reduction="none").mean(dim=-1)
    per_image = (error * usable).flatten(1).sum(1)
    per_image = per_image / usable.flatten(1).sum(1).clamp_min(1)
    supervised = valid.flatten(1).any(1)
    return per_image.sum() / supervised.sum().clamp_min(1)
