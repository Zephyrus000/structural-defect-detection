"""Explicit-layout COCO instance adapter for CUBIT-InSeg crack experiments.

The public repository does not expose the downloaded archive layout. File paths
and category names therefore come from an ingestion YAML, never guessed class IDs.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as coco_mask

from ..utils import file_hash, write_json
from .manifest import safe_path, write_manifest


def decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        if not segmentation or any(len(p) < 6 or len(p) % 2 for p in segmentation):
            raise ValueError("Invalid/empty COCO polygons; bounding boxes are not masks")
        rle = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict):
        if segmentation.get("size") != [height, width]:
            raise ValueError("COCO RLE size does not match image")
        if isinstance(segmentation.get("counts"), list):
            rle = coco_mask.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation.copy()
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode("ascii")
    else:
        raise ValueError("Expected COCO polygon or RLE segmentation")
    mask = coco_mask.decode(rle)
    if mask.shape != (height, width):
        raise ValueError("Decoded COCO mask shape mismatch")
    return mask.astype(bool)


class CubitInSegAdapter:
    def __init__(self, root: Path, dataset_id: str, layout_path: Path):
        self.root, self.dataset_id = root.resolve(), dataset_id
        self.layout = yaml.safe_load(layout_path.read_text(encoding="utf-8"))
        allowed = {
            "format",
            "release",
            "coverage",
            "target_categories",
            "other_defects",
            "output_dir",
            "splits",
            "groups_csv",
        }
        if not isinstance(self.layout, dict) or self.layout.keys() - allowed:
            raise ValueError("Unknown CUBIT ingestion layout fields")
        if self.layout.get("format") != "coco":
            raise ValueError(
                "This adapter requires COCO instance JSON; inspect the downloaded archive before selecting it"
            )
        if (
            self.layout.get("coverage") != "full_crack"
            or self.layout.get("other_defects", "ignore") != "ignore"
        ):
            raise ValueError(
                "Confirm full crack annotation coverage; v1 ignores other defects/crowd/overlaps"
            )
        if not self.layout.get("release") or not self.layout.get("target_categories"):
            raise ValueError("Explicit release and target category names are required")

    def prepare(self, destination: Path) -> dict:
        if destination.exists():
            raise FileExistsError(f"Manifest exists: {destination}")
        output = safe_path(self.root, self.layout.get("output_dir", "prepared/crack_semantic"))
        if output.exists():
            raise FileExistsError(f"Prepared masks already exist: {output}; use a fresh output_dir")
        groups = {}
        if self.layout.get("groups_csv"):
            with safe_path(self.root, self.layout["groups_csv"]).open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                for row in csv.DictReader(handle):
                    if row["image"] in groups:
                        raise ValueError("Duplicate image in group metadata")
                    groups[row["image"]] = row
        inputs = []
        for split, paths in self.layout["splits"].items():
            if split not in ("train", "validation", "test", "external_test") or set(paths) != {
                "images",
                "annotations",
            }:
                raise ValueError("Each declared split requires images and annotations paths")
            annotation_path = safe_path(self.root, paths["annotations"])
            content = json.loads(annotation_path.read_text(encoding="utf-8"))
            categories = {c["id"]: c["name"] for c in content["categories"]}
            names = {str(n).casefold() for n in self.layout["target_categories"]}
            target_ids = {i for i, name in categories.items() if name.casefold() in names}
            if not target_ids or {categories[i].casefold() for i in target_ids} != names:
                raise ValueError(
                    f"Target categories {names} not found exactly in {categories}; do not guess IDs"
                )
            image_ids = [image["id"] for image in content["images"]]
            known_images = set(image_ids)
            if len(known_images) != len(image_ids):
                raise ValueError("Duplicate COCO image IDs")
            instances = defaultdict(list)
            seen_annotations = set()
            for annotation in content["annotations"]:
                if (
                    annotation["image_id"] not in known_images
                    or annotation["category_id"] not in categories
                ):
                    raise ValueError("Unknown image/category referenced by annotation")
                if annotation["id"] in seen_annotations:
                    raise ValueError("Duplicate COCO annotation IDs")
                seen_annotations.add(annotation["id"])
                instances[annotation["image_id"]].append(annotation)
            inputs.append(
                (
                    split,
                    paths,
                    content,
                    categories,
                    target_ids,
                    instances,
                    file_hash(annotation_path),
                )
            )
        output.mkdir(parents=True)
        rows, reports = [], []
        for split, paths, content, categories, target_ids, instances, digest in inputs:
            split_output = output / split
            split_output.mkdir()
            ignored_pixels = 0
            for image_info in content["images"]:
                image_path = safe_path(
                    safe_path(self.root, paths["images"]), image_info["file_name"]
                )
                relative_image = image_path.relative_to(self.root).as_posix()
                with Image.open(image_path) as image:
                    width, height = image.size
                if (height, width) != (image_info["height"], image_info["width"]):
                    raise ValueError(f"COCO/image dimension mismatch: {relative_image}")
                target, ignored = np.zeros((height, width), bool), np.zeros((height, width), bool)
                for annotation in instances[image_info["id"]]:
                    mask = decode_segmentation(annotation.get("segmentation"), height, width)
                    if (
                        annotation["category_id"] not in target_ids
                        or annotation.get("iscrowd", 0)
                        or annotation.get("ignore", 0)
                    ):
                        ignored |= mask
                    else:
                        target |= mask
                target &= ~ignored
                ignored_pixels += int(ignored.sum())
                sample = str(image_info["id"])
                if not sample.isdigit():
                    raise ValueError("COCO image IDs must be nonnegative integers")
                mask_path, valid_path = (
                    split_output / f"{sample}.png",
                    split_output / f"{sample}_valid.png",
                )
                Image.fromarray(target.astype(np.uint8) * 255).save(mask_path)
                Image.fromarray((~ignored).astype(np.uint8) * 255).save(valid_path)
                group = groups.get(relative_image, {})
                rows.append(
                    {
                        "sample_id": f"{split}/{sample}",
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
                    "samples": len(content["images"]),
                    "categories": categories,
                    "annotation_hash": digest,
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
            "policy": "union crack instances; other defects, crowd and cross-class overlaps ignored; official splits retained",
            "group_coverage": len(groups),
            "requires_audit": "Near-duplicates and missing source/structure metadata",
        }
        write_json(output / "preparation_report.json", report)
        return report
