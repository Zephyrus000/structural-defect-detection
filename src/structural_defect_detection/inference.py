"""Original-coordinate prediction using resize or overlapping tiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.nn import functional as F

from .config import ExperimentConfig, stable_hash
from .data.dataset import image_tensor
from .utils import write_json


def tile_starts(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    return sorted(set([*range(0, length - size + 1, size - overlap), length - size]))


class Predictor:
    def __init__(self, model, config: ExperimentConfig, device: torch.device):
        self.model, self.config, self.device = model, config, device

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> np.ndarray:
        self.model.eval()
        data, evaluation = self.config.data, self.config.evaluation
        value = image_tensor(image, data.mean, data.std)
        height, width = image.shape[:2]
        h, w = data.size
        if evaluation.mode == "resize":
            tensor = F.interpolate(value[None], (h, w), mode="bilinear", align_corners=False).to(
                self.device
            )
            result = self.model(tensor).sigmoid().float().cpu()
            return F.interpolate(result, (height, width), mode="bilinear", align_corners=False)[
                0, 0
            ].numpy()
        total = torch.zeros((height, width))
        counts = torch.zeros_like(total)
        for top in tile_starts(height, h, evaluation.overlap):
            for left in tile_starts(width, w, evaluation.overlap):
                tile = value[:, top : top + h, left : left + w]
                th, tw = tile.shape[-2:]
                tile = F.pad(tile, (0, w - tw, 0, h - th))
                probability = (
                    self.model(tile[None].to(self.device)).sigmoid().float().cpu()[0, 0, :th, :tw]
                )
                total[top : top + th, left : left + tw] += probability
                counts[top : top + th, left : left + tw] += 1
        return (total / counts).numpy()


class PredictionWriter:
    def __init__(self, root: Path, threshold: float):
        self.root, self.threshold = root, threshold
        root.mkdir(parents=True, exist_ok=True)

    def write(self, image: np.ndarray, probability: np.ndarray, metadata: dict) -> Path:
        key = stable_hash([metadata.get("dataset_id"), metadata["sample_id"]])[:20]
        mask = probability >= self.threshold
        Image.fromarray(mask.astype(np.uint8) * 255).save(self.root / f"{key}_mask.png")
        overlay = image.astype(np.float32).copy()
        overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
        Image.fromarray(overlay.astype(np.uint8)).save(self.root / f"{key}_overlay.png")
        labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
        regions = []
        for i, slices in enumerate(ndimage.find_objects(labels), start=1):
            if slices is not None:
                y, x = slices
                regions.append(
                    {
                        "component": i,
                        "xywh_pixels": [x.start, y.start, x.stop - x.start, y.stop - y.start],
                        "area_pixels": int((labels[slices] == i).sum()),
                    }
                )
        path = self.root / f"{key}.json"
        write_json(
            path,
            {
                "sample": metadata,
                "threshold": self.threshold,
                "shape": list(mask.shape),
                "coordinates": "original_image_pixels",
                "components": count,
                "regions": regions,
            },
        )
        return path
