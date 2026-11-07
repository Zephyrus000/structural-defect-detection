"""Read-only, reproducible EDA of the local CUBIT-InSeg YOLO polygon release.

Full scan: .venv/Scripts/python scripts/eda_cubit_inseg.py --root data/CUBIT-InSeg
Figures: python scripts/eda_cubit_inseg.py --render-only --output reports/cubit_eda

The full scan uses numpy, Pillow, PyYAML and pycocotools. Figure-only mode additionally
uses matplotlib and can run in a separate environment. Source images/labels are never
modified. This is a dataset audit, not model evaluation or annotation correction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not fields and rows:
        fields = list(rows[0])
    if not fields:
        fields = ["no_records"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_annotations(path: Path, classes: dict[int, str]):
    """Return validated normalized polygons plus explicit line-level defects."""
    polygons, issues, counts = [], [], Counter()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        reason = None
        category = None
        coordinates = None
        try:
            category = int(fields[0])
            counts[category] += 1
            if category not in classes:
                reason = "unknown_class"
            elif len(fields) < 7 or (len(fields) - 1) % 2:
                reason = "not_a_polygon_coordinate_sequence"
            else:
                coordinates = np.asarray(fields[1:], dtype=np.float64).reshape(-1, 2)
                if not np.isfinite(coordinates).all():
                    reason = "non_finite_coordinate"
                elif (coordinates < 0).any() or (coordinates > 1).any():
                    reason = "coordinate_outside_normalized_range"
                else:
                    x, y = coordinates.T
                    twice_area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    if twice_area < 1e-12:
                        # A zero signed/shoelace area can also result from cancelling
                        # lobes of a self-intersecting polygon; it does not prove that
                        # every rasterizer would produce a zero-pixel region.
                        reason = "zero_signed_shoelace_area"
        except (ValueError, OverflowError):
            reason = "invalid_numeric_value"
        if reason:
            issues.append({"line": number, "class_id": category, "reason": reason})
        else:
            polygons.append((category, coordinates))
    return polygons, issues, counts


def raster_masks(polygons, width: int, height: int, classes: dict[int, str]):
    """COCO union of valid polygons, at original resolution; overlaps remain explicit."""
    from pycocotools import mask as coco_mask

    result = {}
    for category, name in classes.items():
        shapes = [
            (coordinates * [width, height]).reshape(-1).tolist()
            for label, coordinates in polygons
            if label == category
        ]
        result[name] = (
            coco_mask.decode(coco_mask.merge(coco_mask.frPyObjects(shapes, height, width)))
            .astype(bool)
            if shapes
            else np.zeros((height, width), dtype=bool)
        )
    return result


def summarize_numbers(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {"n": 0}
    return {
        "n": len(array),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def build_cubit_eda(args) -> dict:
    import yaml

    root, output = args.root.resolve(), args.output.resolve()
    if root == output or root in output.parents:
        raise ValueError("Outputs must be outside the dataset tree")
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8-sig"))
    names = config["names"]
    classes = {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else dict(
        enumerate(names)
    )
    rows, issues, pairing, split_summaries = [], [], [], []
    hashes = defaultdict(list)
    for split in SPLITS:
        images_root, labels_root = root / split / "images", root / split / "labels"
        images = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in EXTENSIONS)
        labels = {p.relative_to(labels_root).as_posix() for p in labels_root.rglob("*.txt")}
        expected = {p.relative_to(images_root).with_suffix(".txt").as_posix() for p in images}
        for kind, paths in (("missing_label", expected - labels), ("orphan_label", labels - expected)):
            pairing.extend({"split": split, "kind": kind, "relative_label": p} for p in sorted(paths))
        start_index = len(rows)
        for index, image_path in enumerate(images, 1):
            relative = image_path.relative_to(images_root)
            label_path = labels_root / relative.with_suffix(".txt")
            row = {
                "split": split,
                "image": image_path.relative_to(root).as_posix(),
                "label": label_path.relative_to(root).as_posix(),
                "image_bytes": image_path.stat().st_size,
                "width": "",
                "height": "",
                "image_mode": "",
                "header_readable": False,
                "label_present": label_path.is_file(),
                "empty_label": False,
                "invalid_polygons": 0,
                "valid_polygons": 0,
                "image_sha256": "",
                "label_sha256": "",
                **{f"{name}_polygons": 0 for name in classes.values()},
            }
            try:
                with Image.open(image_path) as image:
                    row.update(width=image.width, height=image.height, image_mode=image.mode,
                               header_readable=True)
            except (OSError, ValueError) as exc:
                issues.append({"image": row["image"], "label": row["label"], "line": "",
                               "class_id": "", "reason": f"image_header_error:{exc}"})
            if not args.skip_hashes:
                row["image_sha256"] = sha256_file(image_path)
                hashes[row["image_sha256"]].append(row["image"])
            if label_path.is_file():
                polygons, invalid, counts = parse_annotations(label_path, classes)
                row.update(valid_polygons=len(polygons), invalid_polygons=len(invalid),
                           empty_label=not counts, label_sha256=sha256_file(label_path))
                row.update({f"{name}_polygons": counts[category] for category, name in classes.items()})
                issues.extend({"image": row["image"], "label": row["label"], **issue} for issue in invalid)
            rows.append(row)
            if index % 500 == 0 or index == len(images):
                print(f"Full audit {split}: {index}/{len(images)} images", flush=True)
        records = rows[start_index:]
        readable = [r for r in records if r["header_readable"]]
        summary = {
            "split": split,
            "images": len(records),
            "label_files": len(labels),
            "missing_labels": len(expected - labels),
            "orphan_labels": len(labels - expected),
            "header_errors": len(records) - len(readable),
            "empty_labels": sum(r["empty_label"] for r in records),
            "invalid_polygon_files": sum(r["invalid_polygons"] > 0 for r in records),
            "invalid_polygons": sum(r["invalid_polygons"] for r in records),
            "total_image_bytes": sum(r["image_bytes"] for r in records),
            "width": summarize_numbers(r["width"] for r in readable),
            "height": summarize_numbers(r["height"] for r in readable),
            "megapixels": summarize_numbers(r["width"] * r["height"] / 1e6 for r in readable),
            "resolution_counts": dict(Counter(f'{r["width"]}x{r["height"]}' for r in readable)),
            "class_annotation_incidence": {
                name: sum(r[f"{name}_polygons"] > 0 for r in records)
                for name in classes.values()
            },
            "class_polygon_counts_including_flagged": {
                name: sum(r[f"{name}_polygons"] for r in records)
                for name in classes.values()
            },
        }
        split_summaries.append(summary)
    duplicates = [
        {"sha256": digest, "images": paths, "cross_split": len({p.split('/')[0] for p in paths}) > 1}
        for digest, paths in sorted(hashes.items()) if len(paths) > 1
    ]
    save_csv(output / "image_inventory.csv", rows)
    save_csv(output / "annotation_issues.csv", issues,
             ["image", "label", "line", "class_id", "reason"])
    save_csv(output / "pairing_issues.csv", pairing,
             ["split", "kind", "relative_label"])
    selected = sorted(random.Random(args.seed).sample(
        [r for r in rows if r["split"] == "train" and r["header_readable"] and r["label_present"]],
        min(args.mask_samples, sum(r["split"] == "train" and r["header_readable"] and r["label_present"] for r in rows)),
    ), key=lambda r: r["image"])
    sample_rows = []
    for number, row in enumerate(selected, 1):
        polygons, invalid, _ = parse_annotations(root / row["label"], classes)
        masks = raster_masks(polygons, row["width"], row["height"], classes)
        union = np.logical_or.reduce(list(masks.values()))
        overlap = np.sum(list(masks.values()), axis=0) > 1
        pixels = row["width"] * row["height"]
        sample_rows.append({
            "image": row["image"], "label": row["label"], "pixels": pixels,
            "invalid_polygons_excluded": len(invalid),
            **{f"{name}_pixels": int(mask.sum()) for name, mask in masks.items()},
            **{f"{name}_fraction": float(mask.mean()) for name, mask in masks.items()},
            "union_fraction": float(union.mean()), "overlap_fraction": float(overlap.mean()),
        })
        if number <= args.overlay_samples:
            with Image.open(root / row["image"]) as image:
                original = image.convert("RGB")
                original.thumbnail((960, 720))
            colored = original.copy()
            palette = [(231, 76, 60), (0, 170, 190), (120, 80, 200)]
            for index, mask in enumerate(masks.values()):
                small = Image.fromarray(mask).resize(original.size, Image.Resampling.NEAREST)
                tint = Image.new("RGB", original.size, palette[index % len(palette)])
                colored.paste(Image.blend(colored, tint, 0.55), mask=small.convert("L"))
            canvas = Image.new("RGB", (original.width * 2, original.height), "white")
            canvas.paste(original, (0, 0))
            canvas.paste(colored, (original.width, 0))
            canvas.save(output / f"training_overlay_{number:02d}.png")
        print(f"Training mask sample: {number}/{len(selected)}", flush=True)
    save_csv(output / "training_mask_sample.csv", sample_rows)
    total_pixels = sum(r["pixels"] for r in sample_rows)
    manifest = "\n".join(f'{r["image"]},{r["image_sha256"]},{r["label_sha256"]}' for r in rows)
    report = {
        "dataset": "Local CUBIT-InSeg YOLO polygon release",
        "source_root": str(root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "scope": "All official split image headers and annotation files; sampled training-only masks",
        "classes_from_local_data_yaml": classes,
        "data_yaml_sha256": sha256_file(root / "data.yaml"),
        "inventory_fingerprint_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "full_scan": {"images": len(rows), "splits": split_summaries,
                      "images_header_readable": sum(r["header_readable"] for r in rows),
                      "image_sha256_checked": not args.skip_hashes,
                      "image_pixels_fully_decoded": "Only sampled overlay images, not the full image corpus",
                      "invalid_polygon_reasons": dict(Counter(i["reason"] for i in issues)),
                      "exact_file_duplicate_groups": duplicates,
                      "cross_split_exact_file_duplicate_groups": sum(g["cross_split"] for g in duplicates)},
        "training_mask_sample": {
            "n": len(sample_rows),
            "sampling": "Uniform random without replacement from all readable paired training records; seed fixed",
            "method": "Valid normalized polygons scaled to original resolution; pycocotools per-class unions; source annotations unchanged",
            "invalid_polygons_excluded": sum(r["invalid_polygons_excluded"] for r in sample_rows),
            "per_image_fraction": {
                name: summarize_numbers(r[f"{name}_fraction"] for r in sample_rows)
                for name in classes.values()
            },
            "pixel_weighted_fraction": {
                name: sum(r[f"{name}_pixels"] for r in sample_rows) / total_pixels
                if total_pixels else None for name in classes.values()
            },
            "annotated_union_pixel_weighted_fraction": sum(r["union_fraction"] * r["pixels"] for r in sample_rows) / total_pixels if total_pixels else None,
            "overlay_images": [r["image"] for r in sample_rows[:args.overlay_samples]],
        },
        "limitations": [
            "Unannotated pixels are not independently verified defect-free background.",
            "Polygon geometry checks do not establish annotation completeness or semantic correctness.",
            "Zero signed/shoelace area may reflect collinearity or cancelling self-intersecting lobes; it does not imply every flagged polygon rasterizes to zero pixels.",
            "Image headers were read; full-corpus decoded-image integrity was not tested.",
            "SHA256 checks exact file duplicates only; re-encoded copies, near-duplicates and related crops are not excluded.",
            "No verified physical structure, capture-session or source-image grouping metadata was supplied by the local layout (groups_csv: null).",
            "Official partitions are inventoried, not proven leakage-free; do not infer structure IDs from filenames.",
            "Sampled raster-mask occupancy describes only the selected training sample, not full-corpus pixel proportions.",
            "No corrosion class is defined by local data.yaml; no model training or performance claims are made.",
        ],
    }
    (output / "eda_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    derive_duplicate_evidence(output)
    print(json.dumps({"output": str(output), "full_images": len(rows), "sample_masks": len(sample_rows)}, indent=2))
    return report


def derive_duplicate_evidence(output: Path) -> None:
    """Summarize existing full hashes without rereading the 18.8 GB image corpus."""
    report_path = output / "eda_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with (output / "image_inventory.csv").open(encoding="utf-8", newline="") as handle:
        inventory = {r["image"]: r for r in csv.DictReader(handle)}
    duplicates = report["full_scan"]["exact_file_duplicate_groups"]
    rows, pair_rows, pair_counts = [], [], Counter()
    for group in duplicates:
        records = [inventory[p] for p in group["images"]]
        identical_labels = len({r["label_sha256"] for r in records}) == 1
        splits = sorted({r["split"] for r in records})
        if group["cross_split"]:
            pair_counts[" / ".join(splits)] += 1
        if len(records) == 2:
            first, second = records
            class_names = report["classes_from_local_data_yaml"].values()
            pair_rows.append({
                "image_sha256": group["sha256"],
                "image_a": first["image"], "image_b": second["image"],
                "split_a": first["split"], "split_b": second["split"],
                "cross_split": group["cross_split"], "label_bytes_identical": identical_labels,
                "label_a_sha256": first["label_sha256"],
                "label_b_sha256": second["label_sha256"],
                "class_a": ",".join(n for n in class_names if int(first[f"{n}_polygons"])),
                "class_b": ",".join(n for n in class_names if int(second[f"{n}_polygons"])),
                "polygons_a": sum(int(first[f"{n}_polygons"]) for n in class_names),
                "polygons_b": sum(int(second[f"{n}_polygons"]) for n in class_names),
            })
        for record in records:
            rows.append({
                "image_sha256": group["sha256"], "image": record["image"],
                "split": record["split"], "cross_split": group["cross_split"],
                "group_size": len(records), "label_bytes_identical_in_group": identical_labels,
                "label_sha256": record["label_sha256"],
                "annotation_classes": ",".join(
                    name for name in report["classes_from_local_data_yaml"].values()
                    if int(record[f"{name}_polygons"]) > 0
                ),
            })
    save_csv(output / "exact_image_duplicate_members.csv", rows)
    save_csv(output / "exact_image_duplicate_pairs.csv", pair_rows)
    report["duplicate_summary"] = {
        "exact_image_groups": len(duplicates),
        "image_records_in_duplicate_groups": sum(len(g["images"]) for g in duplicates),
        "unique_image_file_hashes": len(inventory) - sum(len(g["images"]) - 1 for g in duplicates)
        if report["full_scan"]["image_sha256_checked"] else None,
        "groups_with_identical_label_files": sum(
            len({inventory[p]["label_sha256"] for p in g["images"]}) == 1
            for g in duplicates
        ),
        "cross_split_group_patterns": dict(pair_counts),
        "cross_split_affected_image_records": sum(
            len(g["images"]) for g in duplicates if g["cross_split"]
        ),
        "interpretation": (
            "Cross-split exact-image duplication is confirmed; establish a separate, reviewed "
            "duplicate/group-aware internal evaluation protocol before training. Preserve source "
            "files and official split records. Check annotation differences before deduplication."
            if any(g["cross_split"] for g in duplicates)
            else "No cross-split exact-file duplication detected; near-duplicates and physical grouping remain unresolved."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def render_figures(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((output / "eda_summary.json").read_text(encoding="utf-8"))
    summaries = report["full_scan"]["splits"]
    labels = [s["split"].replace("val", "Validation").replace("train", "Training").replace("test", "Test") for s in summaries]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#25364b", "text.color": "#25364b"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout="constrained")
    bars = axes[0].bar(labels, [s["images"] for s in summaries], color=["#183b56", "#287d9a", "#65b4bc"], width=.6)
    axes[0].bar_label(bars, labels=[f'{s["images"]:,}' for s in summaries], padding=4)
    axes[0].set(ylabel="Number of images", title="A  Official split inventory", ylim=(0, max(s["images"] for s in summaries) * 1.18))
    axes[0].text(.5, .96, f'Total: {report["full_scan"]["images"]:,} images', ha="center", va="top", transform=axes[0].transAxes)
    names = list(report["classes_from_local_data_yaml"].values())
    x = np.arange(len(summaries))
    for i, name in enumerate(names):
        percentages = [100 * s["class_annotation_incidence"][name] / s["images"] for s in summaries]
        bars = axes[1].bar(x + (i - (len(names) - 1) / 2) * .3, percentages, width=.3,
                           label=name.title(), color=["#c45b4c", "#287d9a"][i % 2])
        axes[1].bar_label(bars, labels=[f"{v:.1f}%" for v in percentages], padding=3, fontsize=9)
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Images with ≥1 class annotation (%)",
                title="B  Class incidence (full annotation scan)", ylim=(0, 115))
    axes[1].legend(loc="upper right", frameon=False)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.15)
    fig.savefig(output / "eda_split_class_incidence.png", dpi=250, bbox_inches="tight")
    fig.savefig(output / "eda_split_class_incidence.svg", bbox_inches="tight")
    plt.close(fig)
    with (output / "image_inventory.csv").open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    with (output / "training_mask_sample.csv").open(encoding="utf-8", newline="") as handle:
        sample = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout="constrained")
    resolutions = Counter((int(r["width"]), int(r["height"])) for r in inventory if r["width"] and r["height"])
    common = resolutions.most_common(7)
    axes[0].barh([f"{w:,} × {h:,}" for (w, h), _ in common][::-1], [count for _, count in common][::-1], color="#287d9a")
    axes[0].set(xlabel="Number of images", title="C  Most frequent original resolutions")
    if sample:
        data = [[100 * float(r[f"{name}_fraction"]) for r in sample] for name in names]
        boxes = axes[1].boxplot(data, tick_labels=[n.title() for n in names], patch_artist=True, showfliers=True,
                                medianprops={"color": "#25364b", "linewidth": 1.5})
        for patch, color in zip(boxes["boxes"], ["#e8a99b", "#8ac9d1"]):
            patch.set_facecolor(color)
        axes[1].set(ylabel="Annotated class union / image pixels (%)",
                    title=f'D  Training-only mask sample (n={len(sample)})')
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="x" if ax == axes[0] else "y", alpha=.15)
    fig.savefig(output / "eda_resolution_mask_occupancy.png", dpi=250, bbox_inches="tight")
    fig.savefig(output / "eda_resolution_mask_occupancy.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/CUBIT-InSeg"))
    parser.add_argument("--output", type=Path, default=Path("reports/cubit_eda"))
    parser.add_argument("--mask-samples", type=int, default=64)
    parser.add_argument("--overlay-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-hashes", action="store_true", help="Omit exact-file duplicate checks")
    parser.add_argument("--render-only", action="store_true", help="Render existing audit using matplotlib")
    parser.add_argument("--derive-only", action="store_true", help="Rebuild duplicate evidence from existing audit")
    args = parser.parse_args()
    if args.mask_samples < 0 or args.overlay_samples < 0:
        parser.error("Sample counts must be non-negative")
    if args.render_only and args.derive_only:
        parser.error("Choose one of --render-only and --derive-only")
    if args.render_only:
        render_figures(args.output)
    elif args.derive_only:
        derive_duplicate_evidence(args.output)
    else:
        build_cubit_eda(args)


if __name__ == "__main__":
    main()
