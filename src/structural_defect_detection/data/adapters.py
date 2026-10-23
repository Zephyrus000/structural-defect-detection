"""Adapters do not infer annotation completeness or licenses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

from .manifest import safe_path, write_manifest


class DatasetAdapter(ABC):
    @abstractmethod
    def rows(self) -> list[dict]:
        """Return portable records with explicit annotation coverage."""

    def write(self, destination: Path) -> None:
        write_manifest(destination, self.rows())


class PairedFolderAdapter(DatasetAdapter):
    def __init__(
        self,
        root: Path,
        dataset_id: str,
        image_dir: str,
        mask_dir: str,
        release: str,
        coverage: str,
        split: str = "unassigned",
        license_name: str = "unknown",
    ):
        if coverage != "full_crack":
            raise ValueError(
                "Folder adapter requires confirmed full_crack coverage; use an explicit masked manifest otherwise"
            )
        self.root, self.dataset_id = root, dataset_id
        self.image_dir, self.mask_dir = image_dir, mask_dir
        self.release, self.coverage, self.split, self.license = (
            release,
            coverage,
            split,
            license_name,
        )

    def rows(self) -> list[dict]:
        image_root = safe_path(self.root, self.image_dir)
        mask_root = safe_path(self.root, self.mask_dir)
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        def index(root):
            result = {}
            for path in sorted(root.rglob("*")):
                if path.suffix.lower() in extensions:
                    key = path.relative_to(root).with_suffix("").as_posix()
                    if key in result:
                        raise ValueError(f"Ambiguous duplicate stem: {key}")
                    result[key] = path
            return result

        images, masks = index(image_root), index(mask_root)
        if not images or images.keys() != masks.keys():
            raise ValueError("Image and mask relative stems must match exactly and be nonempty")
        result = []
        for stem, path in images.items():
            with Image.open(path) as image:
                w, h = image.size
            result.append(
                {
                    "sample_id": stem,
                    "dataset_id": self.dataset_id,
                    "image": path.relative_to(self.root).as_posix(),
                    "mask": masks[stem].relative_to(self.root).as_posix(),
                    "valid_mask": "",
                    "release": self.release,
                    "split": self.split,
                    "width": w,
                    "height": h,
                    "annotation_coverage": self.coverage,
                    "source_image_id": stem,
                    "structure_id": "unknown",
                    "capture_session_id": "unknown",
                    "platform": "unknown",
                    "material": "unknown",
                    "domain": "unknown",
                    "license": self.license,
                }
            )
        return result


class DeepCrackAdapter(DatasetAdapter):
    """Liu DeepCrack layout: train_img/train_lab and test_img/test_lab.

    A deterministic subset of training images becomes internal validation; official
    test remains test. Source-image identity is the strongest available grouping.
    """

    def __init__(self, root: Path, release: str, validation_fraction: float = 0.2, seed: int = 42):
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        self.root, self.release, self.fraction, self.seed = root, release, validation_fraction, seed

    def rows(self) -> list[dict]:
        import random

        train = PairedFolderAdapter(
            self.root, "deepcrack", "train_img", "train_lab", self.release, "full_crack", "train"
        ).rows()
        test = PairedFolderAdapter(
            self.root, "deepcrack", "test_img", "test_lab", self.release, "full_crack", "test"
        ).rows()
        if len(train) < 2:
            raise ValueError("DeepCrack needs at least two training images for internal validation")
        order = list(range(len(train)))
        random.Random(self.seed).shuffle(order)
        for i in order[: max(1, min(len(train) - 1, round(len(train) * self.fraction)))]:
            train[i]["split"] = "validation"
        for prefix, rows in (("train", train), ("test", test)):
            for row in rows:
                row["sample_id"] = prefix + "/" + row["sample_id"]
                row["source_image_id"] = row["sample_id"]
                row["domain"] = "deepcrack"
        return train + test


def merge_instance_masks(
    classes: dict[str, list[np.ndarray]], shape: tuple[int, int], target_class: str = "crack"
) -> np.ndarray:
    """Merge instances of a class; reject cross-class overlaps rather than overwrite."""
    occupied = np.zeros(shape, dtype=bool)
    target = np.zeros(shape, dtype=bool)
    for label, instances in classes.items():
        merged = np.zeros(shape, dtype=bool)
        for mask in instances:
            if mask.shape != shape:
                raise ValueError("Instance mask shape mismatch")
            merged |= mask.astype(bool)
        if (occupied & merged).any():
            raise ValueError("Cross-class overlap requires an explicit annotation policy")
        occupied |= merged
        if label == target_class:
            target |= merged
    return target
