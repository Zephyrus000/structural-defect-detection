"""Bounded real training samples; never generates data or changes official splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import ExperimentConfig, SourceConfig
from ..utils import file_hash
from .cubit_yolo import CubitYoloAdapter
from .dataset import SegmentationDataset
from .manifest import load_arrays, pixel_hash, read_manifest, safe_path


@dataclass
class ProbeSample:
    source: SourceConfig
    row: dict
    image: np.ndarray
    target: np.ndarray
    valid: np.ndarray
    evidence: dict

    def transformed(self, config: ExperimentConfig):
        return SegmentationDataset([], config).transform(
            self.image, self.target, self.valid, self.source, self.row
        )


class RealDataSampler:
    """Read the first N sorted training records, with no fallback or invalid-label skipping."""

    def __init__(self, config: ExperimentConfig, layout: Path | None = None):
        self.config, self.layout = config, layout

    def samples(self, limit: int = 2):
        if type(limit) is not int or limit < 1:
            raise ValueError("max_samples must be a positive integer")
        if self.layout:
            yield from self._raw(limit)
            return
        count = 0
        for source in self.config.data.sources:
            root = Path(source.root)
            manifest = safe_path(root, source.manifest)
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"Missing prepared manifest: {manifest}. Prepare the dataset first, "
                    "or pass --layout for an in-memory CUBIT YOLO training-sample probe."
                )
            rows = sorted(
                (row for row in read_manifest(manifest) if row["split"] == "train"),
                key=lambda row: row["sample_id"],
            )
            for row in rows:
                if row["dataset_id"] != source.name:
                    raise ValueError("Probe dataset_id does not match configured source")
                image, target, valid = load_arrays(source, row)
                if image.shape[:2] != (int(row["height"]), int(row["width"])):
                    raise ValueError(f"Manifest dimensions mismatch: {row['sample_id']}")
                evidence = {
                    "mode": "prepared_manifest",
                    "manifest": str(manifest),
                    "manifest_sha256": file_hash(manifest),
                    "image_sha256": file_hash(safe_path(root, row["image"])),
                    "mask_sha256": file_hash(safe_path(root, row["mask"])),
                    "valid_mask_sha256": file_hash(safe_path(root, row["valid_mask"]))
                    if row.get("valid_mask")
                    else None,
                }
                yield self._sample(source, row, image, target, valid, evidence)
                count += 1
                if count == limit:
                    return
        if count == 0:
            raise ValueError("No training samples for the real-data probe")

    def _raw(self, limit):
        if len(self.config.data.sources) != 1:
            raise ValueError("Raw YOLO --layout requires exactly one configured source")
        source = self.config.data.sources[0]
        adapter = CubitYoloAdapter(Path(source.root), source.name, self.layout)
        paths = adapter.layout["splits"].get("train")
        if not isinstance(paths, dict) or set(paths) != {"images", "labels"}:
            raise ValueError("Raw probe requires train images and labels directories")
        images, labels = (safe_path(adapter.root, paths[k]) for k in ("images", "labels"))
        if not images.is_dir() or not labels.is_dir():
            raise FileNotFoundError("Extract real training images and labels before --dry-run")
        candidates = sorted(
            p
            for p in images.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        )
        if not candidates:
            raise ValueError("No real training images in the configured directory")
        selected = candidates[:limit]
        if len({p.relative_to(images).with_suffix(".txt") for p in selected}) != len(selected):
            raise ValueError("Ambiguous image/label stems in selected training samples")
        for image_path in selected:
            relative = image_path.relative_to(images)
            image_path = safe_path(images, relative.as_posix())
            label_path = safe_path(labels, relative.with_suffix(".txt").as_posix())
            with Image.open(image_path) as handle:
                image = np.array(handle.convert("RGB"))
            height, width = image.shape[:2]
            target, valid, counts = adapter.rasterize(label_path, width, height)
            row = {
                "sample_id": f"train/{relative.with_suffix('').as_posix()}",
                "dataset_id": source.name,
                "split": "train",
                "image": image_path.relative_to(adapter.root).as_posix(),
                "release": adapter.layout["release"],
                "width": width,
                "height": height,
                "annotation_coverage": "masked",
                "domain": "building_facade",
            }
            evidence = {
                "mode": "raw_yolo_in_memory",
                "label": str(label_path),
                "image_sha256": file_hash(image_path),
                "label_sha256": file_hash(label_path),
                "layout_sha256": file_hash(self.layout),
                "class_names": adapter.names,
                "class_mapping_source": adapter.layout["class_mapping_source"],
                "coverage_assumption": adapter.layout["coverage"],
                "instances_by_class": counts,
            }
            yield self._sample(source, row, image, target, valid, evidence)

    @staticmethod
    def _sample(source, row, image, target, valid, evidence):
        if not valid.any():
            raise ValueError(f"No supervised pixels: {row['sample_id']}")
        evidence.update(
            target_pixel_hash=pixel_hash(target),
            valid_pixel_hash=pixel_hash(valid),
            source_root=str(Path(source.root).resolve()),
        )
        return ProbeSample(source, row, image, target, valid, evidence)
