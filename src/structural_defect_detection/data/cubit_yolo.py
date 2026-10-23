"""Streaming ingestion of CUBIT YOLO polygon labels with explicit class semantics."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from ..utils import file_hash, write_json
from .cubit import decode_segmentation
from .manifest import safe_path, write_manifest


def read_yolo_polygons(path: Path, width: int, height: int, class_names: dict):
    """Yield category and pixel-space polygon; missing labels never mean background."""
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"{path}:{number}: expected polygon, not bounding box")
        try:
            category = int(fields[0])
            coordinates = np.asarray(fields[1:], dtype=np.float64).reshape(-1, 2)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: invalid polygon numbers") from exc
        if category not in class_names:
            raise ValueError(f"{path}:{number}: unknown class ID {category}")
        if not np.isfinite(coordinates).all() or (coordinates < 0).any() or (coordinates > 1).any():
            raise ValueError(f"{path}:{number}: coordinates must be finite and normalized [0,1]")
        x, y = coordinates.T
        if abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) < 1e-12:
            raise ValueError(f"{path}:{number}: degenerate polygon")
        yield category, (coordinates * [width, height]).reshape(-1).tolist()


class CubitYoloAdapter:
    def __init__(self, root: Path, dataset_id: str, layout_path: Path):
        self.root, self.dataset_id = root.resolve(), dataset_id
        self.layout = yaml.safe_load(layout_path.read_text(encoding="utf-8"))
        allowed = {
            "format",
            "release",
            "coverage",
            "class_names",
            "class_mapping_source",
            "target_categories",
            "other_defects",
            "output_dir",
            "splits",
            "groups_csv",
        }
        if not isinstance(self.layout, dict) or self.layout.keys() - allowed:
            raise ValueError("Unknown YOLO ingestion layout fields")
        if self.layout.get("format") != "yolo_polygon":
            raise ValueError("Expected format: yolo_polygon")
        if (
            self.layout.get("coverage") != "full_crack"
            or self.layout.get("other_defects", "ignore") != "ignore"
        ):
            raise ValueError("Confirm full_crack coverage; other defects must be ignored")
        self.names = self.layout.get("class_names")
        if (
            not isinstance(self.names, dict)
            or not self.names
            or any(
                type(k) is not int or k < 0 or not isinstance(v, str) or not v.strip()
                for k, v in self.names.items()
            )
        ):
            raise ValueError("class_names requires a verified integer-ID to name mapping")
        if not self.layout.get("class_mapping_source") or not self.layout.get("release"):
            raise ValueError("Provide release and class_mapping_source evidence")
        targets = self.layout.get("target_categories")
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(n, str) for n in targets)
        ):
            raise ValueError("target_categories must contain explicit category names")
        targets = {n.casefold() for n in targets}
        self.target_ids = {k for k, v in self.names.items() if v.casefold() in targets}
        if {self.names[k].casefold() for k in self.target_ids} != targets:
            raise ValueError("Target categories not found in class_names")
        if not isinstance(self.layout.get("splits"), dict) or not self.layout["splits"]:
            raise ValueError("Provide at least one split with images and labels directories")

    def _inputs(self):
        inputs = []
        for split, paths in self.layout["splits"].items():
            if (
                split not in {"train", "validation", "test", "external_test"}
                or not isinstance(paths, dict)
                or set(paths) != {"images", "labels"}
            ):
                raise ValueError("Each split requires images and labels directories")
            images, labels = (
                safe_path(self.root, paths["images"]),
                safe_path(self.root, paths["labels"]),
            )
            if not images.is_dir() or not labels.is_dir():
                raise FileNotFoundError(
                    f"Extract each archive into its own split first: {images}, {labels}"
                )
            image_paths = sorted(
                p
                for p in images.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
            )
            expected = {p.relative_to(images).with_suffix(".txt").as_posix() for p in image_paths}
            actual = {p.relative_to(labels).as_posix() for p in labels.rglob("*.txt")}
            if not image_paths or len(expected) != len(image_paths) or expected != actual:
                raise ValueError(
                    f"Image/label pairing mismatch in {split}: missing={len(expected - actual)}, extra={len(actual - expected)}; do not treat absent labels as negatives"
                )
            inputs.append((split, images, labels, image_paths))
        return inputs

    def rasterize(self, label_path: Path, width: int, height: int):
        """Use the same union/ignore policy for preparation and in-memory dry runs."""
        target, ignored = np.zeros((height, width), bool), np.zeros((height, width), bool)
        counts = {k: 0 for k in self.names}
        for category, polygon in read_yolo_polygons(label_path, width, height, self.names):
            mask = decode_segmentation([polygon], height, width)
            counts[category] += 1
            if category in self.target_ids:
                target |= mask
            else:
                ignored |= mask
        return target & ~ignored, ~ignored, counts

    def prepare(self, destination: Path) -> dict:
        if destination.exists():
            raise FileExistsError(f"Manifest exists: {destination}")
        output = safe_path(self.root, self.layout.get("output_dir", "prepared/crack_semantic"))
        if output.exists():
            raise FileExistsError(f"Prepared masks already exist: {output}; use a fresh output_dir")
        inputs = self._inputs()
        groups = {}
        if self.layout.get("groups_csv"):
            with safe_path(self.root, self.layout["groups_csv"]).open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                for row in csv.DictReader(handle):
                    if row["image"] in groups:
                        raise ValueError("Duplicate image in group metadata")
                    groups[row["image"]] = row
        output.mkdir(parents=True)
        rows, reports = [], []
        for split, images, labels, image_paths in inputs:
            split_output = output / split
            split_output.mkdir()
            annotation_hashes, counts, ignored_pixels = {}, {k: 0 for k in self.names}, 0
            for image_path in image_paths:
                relative = image_path.relative_to(images)
                label_path = safe_path(labels, relative.with_suffix(".txt").as_posix())
                image_path = safe_path(images, relative.as_posix())
                with Image.open(image_path) as image:
                    width, height = image.size
                target, valid, sample_counts = self.rasterize(label_path, width, height)
                for category, count in sample_counts.items():
                    counts[category] += count
                ignored_pixels += int((~valid).sum())
                annotation_hashes[label_path.relative_to(self.root).as_posix()] = file_hash(
                    label_path
                )
                sample = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:24]
                mask_path, valid_path = (
                    split_output / f"{sample}.png",
                    split_output / f"{sample}_valid.png",
                )
                Image.fromarray(target.astype(np.uint8) * 255).save(mask_path)
                Image.fromarray(valid.astype(np.uint8) * 255).save(valid_path)
                relative_image = image_path.relative_to(self.root).as_posix()
                group = groups.get(relative_image, {})
                rows.append(
                    {
                        "sample_id": f"{split}/{relative.with_suffix('').as_posix()}",
                        "dataset_id": self.dataset_id,
                        "image": relative_image,
                        "mask": mask_path.relative_to(self.root).as_posix(),
                        "valid_mask": valid_path.relative_to(self.root).as_posix(),
                        "release": self.layout["release"],
                        "split": split,
                        "width": width,
                        "height": height,
                        "annotation_coverage": "masked",
                        "platform": "UAV",
                        "material": "unknown",
                        "domain": "building_facade",
                        "license": "CC-BY-4.0",
                        **{
                            key: group.get(key) or "unknown"
                            for key in ("source_image_id", "structure_id", "capture_session_id")
                        },
                    }
                )
            reports.append(
                {
                    "split": split,
                    "samples": len(image_paths),
                    "instances_by_class": counts,
                    "annotation_hashes": annotation_hashes,
                    "ignored_pixels": ignored_pixels,
                }
            )
        write_manifest(destination, rows)
        report = {
            "dataset": "CUBIT-InSeg",
            "source": "https://github.com/BenyunZhao/CUBIT-InSeg",
            "layout": self.layout,
            "splits": reports,
            "manifest": str(destination),
            "policy": "union target polygons; other defects/overlaps ignored; official splits retained",
            "rasterization": "normalized coordinates times original width/height; COCO polygon rasterizer",
            "group_coverage": len(groups),
            "requires_audit": "Coverage, class mapping and near-duplicates",
        }
        write_json(output / "preparation_report.json", report)
        return report
