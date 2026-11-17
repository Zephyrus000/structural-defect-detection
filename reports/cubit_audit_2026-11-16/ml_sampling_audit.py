"""Reproduce the 2026-11-16 TRAIN-only CUBIT crop-exposure audit.

Run from the repository root:
  .venv/Scripts/python reports/cubit_audit_2026-11-16/ml_sampling_audit.py --output reports/cubit_sampling_rerun
The output directory must not exist. Inputs and production code are never modified.
"""

import argparse
import csv
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from structural_defect_detection.config import ConfigLoader, stable_hash
from structural_defect_detection.data.dataset import SegmentationDataset
from structural_defect_detection.data.manifest import load_arrays, read_manifest, safe_path
from structural_defect_detection.inference import tile_starts
from structural_defect_detection.utils import file_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
output = args.output
if output.exists():
    parser.error(f"Output directory already exists: {output}")
output.mkdir(parents=True, exist_ok=False)

torch.set_num_threads(2)
config = ConfigLoader.load("configs/experiments/hybrid_ecc_no_ph.yaml")
source = config.data.sources[0]
root = Path(source.root)
rows = read_manifest(root / source.manifest)
train = sorted([r for r in rows if r["split"] == "train"], key=lambda r: r["sample_id"])
assert len(train) == 2523
assert config.data.train_mode == "crop" and config.evaluation.mode == "tile"
assert config.data.size == (256, 256) and config.training.epochs == 20
with (root / "prepared/crack_v1/audit/inventory.csv").open(
    encoding="utf-8", newline=""
) as handle:
    inventory = {r["sample_id"]: r for r in csv.DictReader(handle)}
ordered = sorted(
    train,
    key=lambda r: (
        int(inventory[r["sample_id"]]["crack_pixels"])
        / (int(r["width"]) * int(r["height"])),
        r["sample_id"],
    ),
)
strata = [list(group) for group in np.array_split(np.asarray(ordered, dtype=object), 4)]
selection_rng = random.Random(42)
selected = []
for index, group in enumerate(strata):
    sample = selection_rng.sample(group, 32)
    selected.extend((index, row) for row in sorted(sample, key=lambda r: r["sample_id"]))
assert len(selected) == 128 and len({r["sample_id"] for _, r in selected}) == 128
assert all(r["split"] == "train" for _, r in selected)
h, w = config.data.size


def coordinates(row, epoch):
    seed = int(
        stable_hash([config.training.seed, epoch, source.name, row["sample_id"]])[:16], 16
    )
    rng = np.random.default_rng(seed)
    top = int(rng.integers(0, max(int(row["height"]), h) - h + 1))
    left = int(rng.integers(0, max(int(row["width"]), w) - w + 1))
    return top, left


def measure(item):
    stratum, row = item
    with Image.open(safe_path(root, row["mask"])) as handle:
        raw = np.asarray(handle)
    assert raw.shape == (int(row["height"]), int(row["width"]))
    assert set(np.unique(raw).tolist()) <= set(
        source.foreground_values + source.background_values + source.ignore_values
    )
    target = np.isin(raw, source.foreground_values)
    valid = ~np.isin(raw, source.ignore_values)
    positives = int(target.sum())
    assert positives == int(inventory[row["sample_id"]]["crack_pixels"]) and positives > 0
    coverage = np.zeros_like(target, dtype=bool)
    crops = []
    for epoch in range(config.training.epochs):
        top, left = coordinates(row, epoch)
        patch = target[top : top + h, left : left + w]
        patch_valid = valid[top : top + h, left : left + w]
        assert patch.shape == (h, w) and patch_valid.all()
        count = int(patch.sum())
        crops.append(
            {
                "sample_id": row["sample_id"],
                "split": "train",
                "stratum": stratum,
                "epoch": epoch,
                "top": top,
                "left": left,
                "height": h,
                "width": w,
                "foreground_pixels": count,
                "foreground_fraction": count / (h * w),
                "valid_pixels": int(patch_valid.sum()),
            }
        )
        coverage[top : top + h, left : left + w] = True
    return {
        "sample_id": row["sample_id"],
        "stratum": stratum,
        "foreground_pixels": positives,
        "image_pixels": int(target.size),
        "foreground_fraction": positives / target.size,
        "blank_crop_count": sum(c["foreground_pixels"] == 0 for c in crops),
        "positive_crop_count": sum(c["foreground_pixels"] > 0 for c in crops),
        "unique_image_pixels_seen": int(coverage.sum()),
        "unique_image_fraction_seen": float(coverage.mean()),
        "unique_foreground_pixels_seen": int(np.count_nonzero(coverage & target)),
        "unique_foreground_fraction_seen": int(np.count_nonzero(coverage & target)) / positives,
    }, crops


started = time.perf_counter()
per_image, crops = [], []
with ThreadPoolExecutor(max_workers=4) as pool:
    for count, (image_result, crop_result) in enumerate(
        pool.map(measure, selected, buffersize=4), 1
    ):
        per_image.append(image_result)
        crops.extend(crop_result)
        if count % 32 == 0:
            print(f"ML crop audit: {count}/128 training masks", flush=True)

# Check the measured sampler against production transforms on real image/mask pairs.
checks = 0
for index in [0, 16, 32, 48, 64, 80, 96, 112]:
    _, row = selected[index]
    image, target, valid = load_arrays(source, row)
    for epoch in [0, 19]:
        actual = SegmentationDataset([], config, epoch).transform(
            image, target, valid, source, row
        )
        top, left = coordinates(row, epoch)
        assert actual["metadata"]["transform"]["crop"] == [top, left, h, w]
        expected = target[top : top + h, left : left + w]
        for axis in actual["metadata"]["transform"]["flips"]:
            expected = np.flip(expected, axis)
        assert np.array_equal(actual["target"][0].numpy().astype(bool), expected)
        observed = next(
            c for c in crops if c["sample_id"] == row["sample_id"] and c["epoch"] == epoch
        )
        assert int(actual["target"].sum()) == observed["foreground_pixels"]
        assert int(actual["valid"].sum()) == observed["valid_pixels"]
        checks += 1
assert checks == 16 and len(crops) == 2560


def stats(values):
    array = np.asarray(list(values), dtype=float)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


stratum_results = []
weights = np.asarray([len(group) / len(train) for group in strata])
image_blank_rates = []
for stratum, group in enumerate(strata):
    samples = [r for r in per_image if r["stratum"] == stratum]
    rates = np.asarray([r["blank_crop_count"] / config.training.epochs for r in samples])
    image_blank_rates.append(rates)
    stratum_results.append(
        {
            "stratum": stratum,
            "population_images": len(group),
            "sample_images": len(samples),
            "blank_crop_rate": float(rates.mean()),
            "population_foreground_fraction_range": [
                int(inventory[r["sample_id"]]["crack_pixels"])
                / (int(r["height"]) * int(r["width"]))
                for r in [group[0], group[-1]]
            ],
        }
    )
bootstrap_rng = np.random.default_rng(43)
bootstrap = np.zeros(2000)
for weight, rates in zip(weights, image_blank_rates, strict=True):
    bootstrap += weight * bootstrap_rng.choice(
        rates, size=(2000, len(rates)), replace=True
    ).mean(axis=1)
weighted_blank = float(
    sum(weight * rates.mean() for weight, rates in zip(weights, image_blank_rates, strict=True))
)
validation = [r for r in rows if r["split"] == "validation"]
validation_tiles = sum(
    len(tile_starts(int(r["height"]), h, config.evaluation.overlap))
    * len(tile_starts(int(r["width"]), w, config.evaluation.overlap))
    for r in validation
)
report = {
    "scope": "Training-only crop exposure under the current hybrid_ecc_no_ph configuration; no model training or test-label model probe.",
    "config": "configs/experiments/hybrid_ecc_no_ph.yaml",
    "config_hash": config.config_hash,
    "manifest_file_sha256": file_hash(root / source.manifest),
    "population": {
        "train_images": len(train),
        "validation_images": len(validation),
        "train_positive_fraction": sum(
            int(inventory[r["sample_id"]]["crack_pixels"]) for r in train
        ) / sum(int(r["width"]) * int(r["height"]) for r in train),
    },
    "sampling": {
        "selection": "32 images without replacement from each foreground-density rank quartile, random.Random(42); all20 configured epoch crops per image",
        "selection_seed": 42,
        "transform_seed": config.training.seed,
        "images": len(selected),
        "crops": len(crops),
        "epochs": config.training.epochs,
        "actual_transform_checks": checks,
        "self_check": "exact crop coordinates, post-flip target arrays and foreground/valid counts matched actual SegmentationDataset.transform on16 cases",
    },
    "blank_crop_rate_unweighted": sum(c["foreground_pixels"] == 0 for c in crops) / len(crops),
    "blank_crop_rate_population_weighted": weighted_blank,
    "blank_crop_rate_95pct_stratified_image_bootstrap_interval": np.percentile(
        bootstrap, [2.5, 97.5]
    ).tolist(),
    "bootstrap_note": "2,000 replicates resampling images within strata; holds the20 configured epoch crops fixed; does not represent annotation uncertainty or new-site generalization.",
    "per_stratum": stratum_results,
    "per_epoch": [
        {
            "epoch": epoch,
            "blank_crops": sum(
                c["foreground_pixels"] == 0 for c in crops if c["epoch"] == epoch
            ),
            "crops": 128,
        }
        for epoch in range(config.training.epochs)
    ],
    "foreground_pixels_per_crop": stats(c["foreground_pixels"] for c in crops),
    "foreground_fraction_per_crop": stats(c["foreground_fraction"] for c in crops),
    "positive_crop_foreground_pixels": stats(
        c["foreground_pixels"] for c in crops if c["foreground_pixels"] > 0
    ),
    "images_with_no_positive_crop_in20_epochs": sum(
        r["positive_crop_count"] == 0 for r in per_image
    ),
    "sample_image_foreground_fraction": stats(r["foreground_fraction"] for r in per_image),
    "unique_image_fraction_seen_over20_epochs": stats(
        r["unique_image_fraction_seen"] for r in per_image
    ),
    "unique_foreground_fraction_seen_over20_epochs": stats(
        r["unique_foreground_fraction_seen"] for r in per_image
    ),
    "current_transforms": {
        "train_mode": config.data.train_mode,
        "size": list(config.data.size),
        "augment": config.data.augment,
        "augmentation": "horizontal and vertical flips only",
        "normalization_mean": list(config.data.mean),
        "normalization_std": list(config.data.std),
        "evaluation_mode": config.evaluation.mode,
        "evaluation_overlap": config.evaluation.overlap,
        "validation_tiles_per_epoch": validation_tiles,
        "resolution_note": "Current crop+tile pipeline uses native-resolution pixels; no downsize interpolation in active modes. The resize branch and model-internal thin-feature retention were not evaluated.",
    },
    "seconds": time.perf_counter() - started,
    "limitations": [
        "The128image sample is stratified by annotation occupancy, not structure or capture session; those metadata are missing.",
        "A blank crop means no annotated crack pixels, not a verified defect-free negative.",
        "Measurements describe this fixed seed and20 configured epochs; they are not model accuracy or a benchmark result.",
    ],
}

# Preserve the explicit denominators/IDs added to the original measurement report.
assert all(int(inventory[r["sample_id"]]["crack_pixels"]) > 0 for r in rows)
report["blank_crops"] = sum(c["foreground_pixels"] == 0 for c in crops)
report["positive_crops"] = sum(c["foreground_pixels"] > 0 for c in crops)
report["images_with_no_positive_crop_in20_epochs_ids"] = [
    r["sample_id"] for r in per_image if r["positive_crop_count"] == 0
]
report["prepared_inventory_ground_truth_positive_images_by_split"] = dict(
    Counter(r["split"] for r in rows)
)
report["prepared_inventory_empty_ground_truth_image_count"] = 0
assert report["blank_crops"] == 1879 and report["positive_crops"] == 681

for name, items in [
    ("ml_sampling_crops.csv", crops),
    ("ml_sampling_images.csv", per_image),
]:
    with (output / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)
(output / "ml_sampling.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            key: report[key]
            for key in [
                "blank_crop_rate_population_weighted",
                "blank_crop_rate_95pct_stratified_image_bootstrap_interval",
                "foreground_pixels_per_crop",
                "images_with_no_positive_crop_in20_epochs",
                "unique_foreground_fraction_seen_over20_epochs",
                "current_transforms",
                "seconds",
            ]
        },
        indent=2,
    ),
    flush=True,
)
