"""Stateless per-epoch augmentation makes resume independent of worker scheduling."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

from ..config import ExperimentConfig, stable_hash
from .manifest import ManifestBuilder, load_arrays


def image_tensor(image: np.ndarray, mean, std) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255
    return (value - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


class SegmentationDataset(Dataset):
    def __init__(self, records, config: ExperimentConfig, epoch: int = 0):
        self.records, self.config, self.epoch = records, config, epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        source, row = self.records[index]
        image, target, valid = load_arrays(source, row)
        return self.transform(image, target, valid, source, row)

    def transform(self, image, target, valid, source, row):
        """Apply the training transforms to an annotated image, including raw-data probes."""
        cfg = self.config.data
        original = image.shape[:2]
        seed = int(
            stable_hash([self.config.training.seed, self.epoch, source.name, row["sample_id"]])[
                :16
            ],
            16,
        )
        rng = np.random.default_rng(seed)
        h, w = cfg.size
        record = {"original_size": list(original), "mode": cfg.train_mode}
        if cfg.train_mode == "resize":
            image = image_tensor(image, cfg.mean, cfg.std)
            target = torch.from_numpy(target.copy()).float()[None]
            valid = torch.from_numpy(valid.copy())[None]
            image = F.interpolate(image[None], (h, w), mode="bilinear", align_corners=False)[0]
            target = F.interpolate(target[None], (h, w), mode="nearest")[0]
            valid = F.interpolate(valid[None].float(), (h, w), mode="nearest")[0].bool()
        else:
            # Crop uint8 arrays before normalization: a 4K source should not create
            # multiple full-size float32 copies for a small training patch.
            pad = (0, max(0, w - original[1]), 0, max(0, h - original[0]))
            top = int(rng.integers(0, max(original[0], h) - h + 1))
            left = int(rng.integers(0, max(original[1], w) - w + 1))
            image = image_tensor(image[top : top + h, left : left + w], cfg.mean, cfg.std)
            target = torch.from_numpy(target[top : top + h, left : left + w].copy()).float()[None]
            valid = torch.from_numpy(valid[top : top + h, left : left + w].copy())[None]
            image, target, valid = (
                F.pad(image, pad),
                F.pad(target, pad),
                F.pad(valid, pad, value=False),
            )
            record.update({"crop": [top, left, h, w], "pad": list(pad)})
        flips = []
        if cfg.augment:
            for axis in (-1, -2):
                if rng.random() < 0.5:
                    image, target, valid = (x.flip(axis) for x in (image, target, valid))
                    flips.append(axis)
        record["flips"] = flips
        return {
            "image": image,
            "target": target,
            "valid": valid,
            "metadata": {**row, "transform": record},
        }


def collate(samples):
    return {key: torch.stack([s[key] for s in samples]) for key in ("image", "target", "valid")} | {
        "metadata": [s["metadata"] for s in samples]
    }


class DataModule:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.records, self.report = ManifestBuilder(config.data).validate()

    def select(self, split: str):
        records = [(s, r) for s, r in self.records if r["split"] == split]
        if not records:
            raise ValueError(f"No samples for split {split}")
        return records

    def train_loader(self, epoch: int):
        records = self.select("train")
        dataset = SegmentationDataset(records, self.config, epoch)
        generator = torch.Generator().manual_seed(self.config.training.seed + epoch)
        counts = Counter(s.name for s, _ in records)
        if len(counts) == 1:
            sampler = RandomSampler(dataset, generator=generator)
        else:
            weights = [s.weight / counts[s.name] for s, _ in records]
            sampler = WeightedRandomSampler(
                weights, len(records), replacement=True, generator=generator
            )
        return DataLoader(
            dataset,
            batch_size=self.config.runtime.micro_batch,
            sampler=sampler,
            num_workers=self.config.runtime.workers,
            collate_fn=collate,
            generator=generator,
            persistent_workers=False,
            pin_memory=self.config.runtime.device.startswith("cuda"),
        )
