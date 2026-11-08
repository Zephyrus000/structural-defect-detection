"""Build a small, real CiF crack-segmentation dataset for the Studio 7 workflow."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from structural_defect_detection.config import DataConfig, SourceConfig
from structural_defect_detection.data.cubit import decode_segmentation
from structural_defect_detection.data.manifest import ManifestBuilder, write_manifest
from structural_defect_detection.utils import write_json

DATASET = "ibm-research/cif-dataset"
DATASET_ID = "cif-dataset"
VIEWER_HOST = "datasets-server.huggingface.co"
LICENSE = "CDLA-Permissive-2.0"
PAGE_SIZE = 20
MAX_ROWS = 160
CROP_SIZE = 96
PH_WINDOW = 16
CRACK_CATEGORY = 2


def fetch(url: str) -> bytes:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != VIEWER_HOST:
        raise ValueError("Expected an HTTPS URL from the Hugging Face Dataset Viewer")
    request = Request(url, headers={"User-Agent": "structural-defect-detection-studio7"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                return response.read()
        except (URLError, TimeoutError) as exc:
            if isinstance(exc, HTTPError) and exc.code < 500 and exc.code != 429:
                raise RuntimeError(f"CiF Dataset Viewer returned HTTP {exc.code}") from exc
            if attempt == 2:
                raise RuntimeError(f"CiF Dataset Viewer could not be reached: {exc}") from exc
            time.sleep(2**attempt)


def viewer_rows(split: str):
    for offset in range(0, MAX_ROWS, PAGE_SIZE):
        params = urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        result = json.loads(fetch(f"https://{VIEWER_HOST}/rows?{params}"))
        if not isinstance(result, dict) or not isinstance(result.get("rows"), list):
            raise ValueError(f"Invalid CiF Dataset Viewer response for {split} at row {offset}")
        yield from result["rows"]
        if len(result["rows"]) < PAGE_SIZE:
            break


def crop_mask(row: dict) -> tuple[int, int, np.ndarray] | None:
    width, height = row["width"], row["height"]
    if (width, height) != (1024, 1024):
        raise ValueError("Expected a 1024×1024 CiF tile")
    objects = row["objects"]
    categories, crowds = objects["category_id"], objects["iscrowd"]
    if CRACK_CATEGORY not in categories:
        return None

    target = np.zeros((height, width), dtype=bool)
    ignore = np.zeros_like(target)
    for category, crowd, polygons in zip(categories, crowds, objects["segmentation"], strict=True):
        if not isinstance(polygons, list) or any(
            not isinstance(polygon, list) or not all(math.isfinite(x) for x in polygon)
            for polygon in polygons
        ):
            raise ValueError("Expected finite COCO polygon coordinates")
        instance = decode_segmentation(polygons, height, width)
        if category == CRACK_CATEGORY and not crowd:
            target |= instance
        else:
            ignore |= instance

    points = np.argwhere(target & ~ignore)
    midpoint = len(points) // 2
    indices = np.linspace(0, len(points) - 1, min(len(points), 9), dtype=int)
    for index in sorted(indices, key=lambda i: abs(i - midpoint)):
        y, x = (int(value) for value in points[index])
        top = min(max(y - CROP_SIZE // 2, 0), height - CROP_SIZE)
        left = min(max(x - CROP_SIZE // 2, 0), width - CROP_SIZE)
        region = np.s_[top : top + CROP_SIZE, left : left + CROP_SIZE]
        # Ignored defects/crowds take precedence over overlapping cracks.
        mask = np.where(ignore[region], 128, np.where(target[region], 255, 0)).astype(np.uint8)
        windows = (mask != 128).reshape(
            CROP_SIZE // PH_WINDOW, PH_WINDOW, CROP_SIZE // PH_WINDOW, PH_WINDOW
        )
        if np.any(mask == 0) and windows.all(axis=(1, 3)).any():
            return top, left, mask
    return None


def choose_rows(split: str, count: int, excluded_parents: set[str]):
    selected, fallback = [], []
    parents = set(excluded_parents)
    invalid = 0
    for entry in viewer_rows(split):
        try:
            if set(entry.get("truncated_cells", [])) & {"objects", "image", "file_name_original"}:
                raise ValueError("Truncated annotation or image metadata")
            row = entry["row"]
            parent = row["file_name_original"]
            if not isinstance(parent, str) or not parent or type(row["image_id"]) is not int:
                raise ValueError("Missing original-image or tile identity")
            if parent in excluded_parents or (parent in parents and len(fallback) >= count):
                continue
            crop = crop_mask(row)
        except (KeyError, TypeError, ValueError, IndexError):
            invalid += 1
            continue
        if crop is None:
            continue
        candidate = (entry, crop)
        if parent in parents:
            fallback.append(candidate)
        else:
            selected.append(candidate)
            parents.add(parent)
            if len(selected) == count:
                break
    chosen = (selected + fallback)[:count]
    if len(chosen) != count:
        raise ValueError(
            f"Found {len(chosen)}/{count} usable crops in the first {MAX_ROWS} CiF {split} rows "
            f"({invalid} malformed or truncated rows)"
        )
    return chosen


def save_sample(root: Path, viewer_split: str, split: str, entry: dict, crop: tuple):
    row = entry["row"]
    top, left, mask = crop
    tile_id = row["image_id"]
    stem = f"{split}-{tile_id}"
    image_file = f"{stem}.png"
    mask_file = f"{stem}-mask.png"
    valid_file = f"{stem}-valid.png"
    image_bytes = fetch(row["image"]["src"])
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.size != (row["width"], row["height"]):
            raise ValueError(f"Downloaded CiF tile {tile_id} does not match its annotations")
        region = (left, top, left + CROP_SIZE, top + CROP_SIZE)
        image.crop(region).convert("RGB").save(root / image_file)
    Image.fromarray(mask).save(root / mask_file)
    Image.fromarray((mask != 128).astype(np.uint8) * 255).save(root / valid_file)

    parent = row["file_name_original"]
    manifest_row = {
        "sample_id": f"{split}/{tile_id}",
        "dataset_id": DATASET_ID,
        "split": split,
        "image": image_file,
        "mask": mask_file,
        "valid_mask": valid_file,
        "width": CROP_SIZE,
        "height": CROP_SIZE,
        "annotation_coverage": "masked",
        "source_image_id": parent,
        "release": DATASET,
        "license": LICENSE,
    }
    annotations = json.dumps(row["objects"], sort_keys=True, separators=(",", ":")).encode()
    evidence = {
        "sample_id": manifest_row["sample_id"],
        "split": viewer_split,
        "row_offset": entry["row_idx"],
        "tile_id": tile_id,
        "file_name_original": parent,
        "parent_image_id": row["parent_image_id"],
        "source_parent_image_id": row.get("source_parent_image_id"),
        "tile_file_name": row["file_name"],
        "crop_top_left_height_width": [top, left, CROP_SIZE, CROP_SIZE],
        "original_tile_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "tile_annotations_sha256": hashlib.sha256(annotations).hexdigest(),
    }
    return manifest_row, evidence


def prepare(output: Path) -> None:
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {output}")
    train = choose_rows("train_tiled", 2, set())
    train_parents = {entry["row"]["file_name_original"] for entry, _ in train}
    validation = choose_rows("val_tiled", 1, train_parents)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".cif-") as temp:
        staging = Path(temp) / "prepared"
        staging.mkdir()
        rows, samples = [], []
        for viewer_split, split, selected in (
            ("train_tiled", "train", train),
            ("val_tiled", "validation", validation),
        ):
            for entry, crop in selected:
                row, evidence = save_sample(staging, viewer_split, split, entry, crop)
                rows.append(row)
                samples.append(evidence)
        write_manifest(staging / "manifest.csv", rows)
        ManifestBuilder(
            DataConfig(sources=(SourceConfig(name=DATASET_ID, root=str(staging)),))
        ).validate()
        write_json(
            staging / "provenance.json",
            {
                "dataset": DATASET,
                "dataset_viewer_config": "default",
                "license": LICENSE,
                "target_category_id": CRACK_CATEGORY,
                "coverage_policy": "Unmarked pixels are treated as background for this CI check only.",
                "samples": samples,
            },
        )
        staging.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/cif-studio7"))
    args = parser.parse_args()
    try:
        prepare(args.output)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Could not prepare CiF dataset: {exc}\n")
    print(f"Prepared 2 train crops and 1 validation crop in {args.output}")


if __name__ == "__main__":
    main()
