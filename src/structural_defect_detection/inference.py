"""Original-coordinate prediction using resize or overlapping tiles."""

from itertools import batched, product
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
        height, width = image.shape[:2]
        h, w = data.size
        if evaluation.mode == "resize":
            value = image_tensor(image, data.mean, data.std)
            tensor = F.interpolate(value[None], (h, w), mode="bilinear", align_corners=False).to(
                self.device
            )
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.config.runtime.amp,
                dtype=getattr(torch, self.config.runtime.amp_dtype),
            ):
                logits = self.model(tensor)
            result = logits.float().sigmoid().cpu()
            return F.interpolate(result, (height, width), mode="bilinear", align_corners=False)[
                0, 0
            ].numpy()
        gpu_tiles = self.config.runtime.gpu_tiles
        if gpu_tiles:
            value = torch.from_numpy(np.ascontiguousarray(image)).to(self.device).permute(2, 0, 1)
            mean = value.new_tensor(data.mean, dtype=torch.float32)[None, :, None, None]
            std = value.new_tensor(data.std, dtype=torch.float32)[None, :, None, None]
        else:
            value = image_tensor(image, data.mean, data.std)
        total = torch.zeros((height, width), device=value.device)
        counts = torch.zeros_like(total)
        positions = product(
            tile_starts(height, h, evaluation.overlap), tile_starts(width, w, evaluation.overlap)
        )
        for batch in batched(positions, self.config.runtime.micro_batch):
            tiles = []
            for top, left in batch:
                tile = value[:, top : top + h, left : left + w]
                tiles.append(tile)
            inputs = torch.stack(tiles)
            th, tw = inputs.shape[-2:]
            if gpu_tiles:
                inputs = (inputs.float().div_(255) - mean) / std
            if (th, tw) != (h, w):
                inputs = F.pad(inputs, (0, w - tw, 0, h - th))
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.config.runtime.amp,
                dtype=getattr(torch, self.config.runtime.amp_dtype),
            ):
                logits = self.model(inputs.to(self.device))
            probabilities = logits.float().sigmoid().to(total.device)[:, 0]
            for (top, left), probability in zip(batch, probabilities, strict=True):
                th, tw = min(h, height - top), min(w, width - left)
                total[top : top + th, left : left + tw] += probability[:th, :tw]
                counts[top : top + th, left : left + tw] += 1
        return total.div_(counts).cpu().numpy()


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
