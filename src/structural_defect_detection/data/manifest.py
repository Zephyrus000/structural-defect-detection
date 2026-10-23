"""Portable CSV manifests with content verification and split leakage checks."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import DataConfig, SourceConfig, stable_hash

FIELDS = (
    "sample_id",
    "dataset_id",
    "image",
    "mask",
    "valid_mask",
    "release",
    "split",
    "width",
    "height",
    "annotation_coverage",
    "source_image_id",
    "structure_id",
    "capture_session_id",
    "platform",
    "material",
    "domain",
    "license",
)
SPLITS = {"train", "validation", "test", "external_test", "unassigned"}


def safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Manifest paths must be relative and contained: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return resolved


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "dataset_id",
            "image",
            "mask",
            "release",
            "split",
            "width",
            "height",
            "annotation_coverage",
        }
        if not required <= set(reader.fieldnames or []):
            raise ValueError(
                f"Missing manifest columns: {sorted(required - set(reader.fieldnames or []))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_arrays(source: SourceConfig, row: dict):
    root = Path(source.root)
    with Image.open(safe_path(root, row["image"])) as handle:
        image = np.array(handle.convert("RGB"))
    with Image.open(safe_path(root, row["mask"])) as handle:
        raw = np.array(handle)
    if raw.ndim != 2 or raw.shape != image.shape[:2]:
        raise ValueError(f"Mask must be single-channel and image-aligned: {row['sample_id']}")
    allowed = set(source.foreground_values + source.background_values + source.ignore_values)
    unknown = set(np.unique(raw).tolist()) - allowed
    if unknown:
        raise ValueError(f"Unmapped mask values {unknown}: {row['sample_id']}")
    target = np.isin(raw, source.foreground_values)
    valid = ~np.isin(raw, source.ignore_values)
    coverage = row["annotation_coverage"]
    if coverage not in ("full_crack", "masked"):
        raise ValueError(
            "annotation_coverage must be full_crack or masked; unknown negatives are unsafe"
        )
    if coverage == "masked" and not row.get("valid_mask"):
        raise ValueError("Partial annotations require an explicit valid_mask")
    if row.get("valid_mask"):
        with Image.open(safe_path(root, row["valid_mask"])) as handle:
            coverage_mask = np.array(handle)
        if coverage_mask.ndim != 2 or coverage_mask.shape != target.shape:
            raise ValueError("valid_mask shape mismatch")
        valid &= coverage_mask > 0
    return image, target, valid


def pixel_hash(array: np.ndarray) -> str:
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


class ManifestBuilder:
    """Validate both labels and split identities before a training run."""

    def __init__(self, config: DataConfig):
        self.config = config

    def validate(self) -> tuple[list[tuple[SourceConfig, dict]], dict]:
        records, contents, split_records = [], [], []
        seen_ids, groups, images = set(), {}, {}
        duplicate_images, missing_groups = [], 0
        for source in self.config.sources:
            rows = read_manifest(safe_path(Path(source.root), source.manifest))
            for row in rows:
                key = (source.name, row["sample_id"])
                if key in seen_ids or row["dataset_id"] != source.name:
                    raise ValueError(f"Duplicate identity or dataset_id mismatch: {key}")
                seen_ids.add(key)
                if row["split"] not in SPLITS:
                    raise ValueError(f"Unknown split: {row['split']}")
                image, target, valid = load_arrays(source, row)
                h, w = image.shape[:2]
                if (int(row["height"]), int(row["width"])) != (h, w):
                    raise ValueError(f"Manifest dimensions mismatch: {key}")
                image_digest = pixel_hash(image)
                if image_digest in images:
                    previous = images[image_digest]
                    if previous[0] != row["split"]:
                        raise ValueError(f"Image content leakage between {previous} and {key}")
                    duplicate_images.append([list(previous[1]), list(key)])
                images[image_digest] = (row["split"], key)
                has_group = False
                for field in ("structure_id", "capture_session_id", "source_image_id"):
                    value = row.get(field, "")
                    if value and value != "unknown":
                        has_group = True
                        group_key = (source.name, field, value)
                        if group_key in groups and groups[group_key] != row["split"]:
                            raise ValueError(f"Group leakage: {group_key}")
                        groups[group_key] = row["split"]
                missing_groups += int(not has_group)
                clean = {k: row.get(k, "") for k in FIELDS}
                contents.append(
                    {
                        "row": clean,
                        "image_hash": image_digest,
                        "target_hash": pixel_hash(target),
                        "valid_hash": pixel_hash(valid),
                    }
                )
                split_records.append([*key, row["split"]])
                records.append((source, clean))
        report = {
            "samples": len(records),
            "image_hashes_by_split": {
                split: sorted(
                    {entry["image_hash"] for entry in contents if entry["row"]["split"] == split}
                )
                for split in sorted(SPLITS)
            },
            "splits": dict(Counter(r["split"] for _, r in records)),
            "manifest_hash": stable_hash(
                sorted(contents, key=lambda v: (v["row"]["dataset_id"], v["row"]["sample_id"]))
            ),
            "split_hash": stable_hash(sorted(split_records)),
            "same_split_duplicates": duplicate_images,
            "samples_without_group_metadata": missing_groups,
            "leakage_limit": "Exact decoded-image duplicates and available group IDs only; near-duplicates need source audit.",
        }
        return records, report


class SplitBuilder:
    """Group split with transitive metadata/content linkage, preserving assigned splits."""

    def build(
        self,
        source: SourceConfig,
        seed: int = 42,
        ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> list[dict]:
        if any(r < 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
            raise ValueError("Split ratios must be nonnegative and sum to 1")
        rows = read_manifest(safe_path(Path(source.root), source.manifest))
        parent = list(range(len(rows)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        seen = {}
        for i, row in enumerate(rows):
            image, _, _ = load_arrays(source, row)
            keys = [("content", pixel_hash(image))]
            keys += [
                (field, row[field])
                for field in ("structure_id", "capture_session_id", "source_image_id")
                if row.get(field) not in (None, "", "unknown")
            ]
            for key in keys:
                if key in seen:
                    parent[find(i)] = find(seen[key])
                else:
                    seen[key] = i
        grouped = defaultdict(list)
        for i, row in enumerate(rows):
            grouped[find(i)].append(row.copy())
        randomizer = random.Random(seed)
        result = []
        for group in grouped.values():
            fixed = {r["split"] for r in group if r["split"] != "unassigned"}
            if len(fixed) > 1:
                raise ValueError(
                    "Existing official splits have linked samples; retain and audit separately"
                )
            chosen = (
                next(iter(fixed))
                if fixed
                else randomizer.choices(["train", "validation", "test"], weights=ratios, k=1)[0]
            )
            for row in group:
                row["split"] = chosen
            result.extend(group)
        return result
