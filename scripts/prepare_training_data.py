"""Build and audit crack_v1 manifests from the local CUBIT and CIF full releases.

Run from the repository root with the project environment. PyArrow is required for CIF only.
For CUBIT alone, pass --config configs/experiments/cubit_cnn_no_ph.yaml.
"""

import argparse
import hashlib
import io
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from eda_cubit_inseg import parse_annotations, save_csv
from PIL import Image
from pycocotools import mask as coco_mask

from structural_defect_detection.config import ConfigLoader
from structural_defect_detection.data.cubit_yolo import CubitYoloAdapter
from structural_defect_detection.data.manifest import pixel_hash, safe_path, write_manifest
from structural_defect_detection.utils import file_hash, write_json

SPLITS = {"train": "train", "val": "validation", "test": "test"}
CIF_COUNTS = {"train": (188, 9096), "val": (78, 1942), "test": (76, 1858)}
MAPPING = Path("configs/datasets/cubit_cif_crack.yaml")
CONFIG = Path("configs/experiments/cubit_cif_ph.yaml")
CUBIT = "cubit-inseg"
CIF = "cif-dataset"


def cubit_records(root, classes):
    adapter = CubitYoloAdapter(root, CUBIT, Path("configs/datasets/cubit_yolo.local.yaml"))
    if {k: v.casefold() for k, v in classes.items()} != adapter.names:
        raise ValueError("CUBIT class mapping differs from the verified local layout")
    for split, images, labels, paths in adapter._inputs():
        for image in paths:
            relative = image.relative_to(images)
            label = labels / relative.with_suffix(".txt")
            yield {
                "id": f"{split}/{relative.with_suffix('').as_posix()}",
                "split": split,
                "source": image.relative_to(root).as_posix(),
                "label": label,
                "image": image,
                "domain": "building_facade",
            }


def cif_records(root):
    import pyarrow.parquet as pq

    columns = [
        "image_id", "image", "file_name", "width", "height", "source_collection",
        "site", "structure", "component_id", "material", "tile_row", "tile_col",
        "file_name_original", "width_original", "height_original", "parent_image_id",
        "source_parent_image_id", "objects",
    ]
    for official, split in SPLITS.items():
        shards, rows = CIF_COUNTS[official]
        names = [f"{official}_full-{i:05d}-of-{shards:05d}.parquet" for i in range(shards)]
        if sorted(p.name for p in (root / "data").glob(f"{official}_full-*.parquet")) != names:
            raise ValueError(f"Incomplete CIF shard set for {official}")
        count, seen = 0, set()
        for name in names:
            parquet = pq.ParquetFile(root / "data" / name)
            offset = 0
            for batch in parquet.iter_batches(batch_size=4, columns=columns):
                for row in batch.to_pylist():
                    identifier = row["image_id"]
                    if type(identifier) is not int or identifier in seen:
                        raise ValueError(f"Invalid CIF image ID {identifier} in {split}")
                    seen.add(identifier)
                    if any(row[k] is not None for k in columns[10:-1]):
                        raise ValueError(f"Tiled metadata found in CIF full shard {name}:{offset}")
                    yield {
                        "id": f"{split}/{identifier}",
                        "split": split,
                        "source": f"data/{name}:{offset}",
                        "source_file_name": row["file_name"],
                        "image": row["image"]["bytes"],
                        "size": (row["width"], row["height"]),
                        "objects": row["objects"],
                        "domain": row["source_collection"] or "unknown",
                        "site": row["site"],
                        "structure": row["structure"],
                        "component_id": row["component_id"],
                        "material": row["material"],
                    }
                    offset += 1
            count += offset
        if count != rows:
            raise ValueError(f"CIF {split} has {count} rows, expected {rows}")


def annotation_instances(record, width, height, mapping):
    classes = mapping["classes"]
    if "label" in record:
        polygons, issues, counts = parse_annotations(record["label"], classes)
        if issues:
            raise ValueError(f"invalid_yolo_annotation:{issues}")
        if not counts:
            raise ValueError("empty_yolo_annotation")
        return [
            (category, False, [(points * [width, height]).reshape(-1).tolist()])
            for category, points in polygons
        ]
    objects = record["objects"]
    columns = [objects[key] for key in ("id", "category_id", "iscrowd", "segmentation")]
    if not columns[0] or len(set(columns[0])) != len(columns[0]):
        raise ValueError("empty_or_duplicate_cif_annotation_ids")
    return [
        (category, crowd, polygons)
        for _, category, crowd, polygons in zip(*columns, strict=True)
    ]


def mask_for(instances, width, height, mapping, values):
    foreground, ignored = [], []
    for category, crowd, polygons in instances:
        if category not in mapping["classes"] or crowd not in (0, 1) or not polygons:
            raise ValueError("invalid_class_crowd_or_polygon")
        for polygon in polygons:
            if (
                len(polygon) < 6 or len(polygon) % 2
                or any(not math.isfinite(x) or not -width <= x <= 2 * width for x in polygon[::2])
                or any(not math.isfinite(y) or not -height <= y <= 2 * height for y in polygon[1::2])
            ):
                raise ValueError("invalid_polygon_coordinates")
        rle = coco_mask.merge(coco_mask.frPyObjects(polygons, height, width))
        if not coco_mask.area(rle):
            raise ValueError("instance_rasterizes_to_zero_pixels")
        (foreground if category in mapping["foreground"] and not crowd else ignored).append(rle)
    mask = np.full((height, width), values["background"], dtype=np.uint8)
    if foreground:
        mask[coco_mask.decode(coco_mask.merge(foreground)).astype(bool)] = values["crack"]
    if ignored:
        mask[coco_mask.decode(coco_mask.merge(ignored)).astype(bool)] = values["ignore"]
    return mask


def prepare_record(root, prepared, source, record, mapping, values):
    audit = {
        "dataset_id": source, "sample_id": record["id"], "split": record["split"],
        "source": record["source"], "source_file_name": record.get("source_file_name", ""),
        "domain": record["domain"], "site": record.get("site"),
        "structure": record.get("structure"), "component_id": record.get("component_id"),
    }
    image_bytes = record["image"]
    if isinstance(image_bytes, Path):
        image_bytes = image_bytes.read_bytes()
        audit["label"] = record["label"].relative_to(root).as_posix()
        audit["label_sha256"] = file_hash(record["label"])
    if not image_bytes:
        return audit | {"decision": "invalid_image:missing_bytes"}, None
    audit["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            audit.update(width=width, height=height)
            if image.size != record.get("size", image.size):
                raise ValueError("image_size_differs_from_annotation")
            instances = annotation_instances(record, width, height, mapping)
            classes = Counter(category for category, _, _ in instances)
            audit.update({f"class_{category}": classes[category] for category in mapping["classes"]})
            if source == CUBIT and not set(classes).intersection(mapping["foreground"]):
                return audit | {"decision": "cubit_no_confirmed_crack_coverage"}, None
            pixels = np.asarray(image.convert("RGB"))
            audit["decoded_image_sha256"] = pixel_hash(pixels)
            del pixels
            extension = ".jpg" if image.format == "JPEG" else f".{image.format.lower()}"
    except (OSError, ValueError, TypeError, KeyError, Image.DecompressionBombError) as exc:
        return audit | {"decision": f"invalid_source:{exc}"}, None
    try:
        mask = mask_for(instances, width, height, mapping, values)
    except (ValueError, TypeError) as exc:
        return audit | {"decision": f"invalid_mask:{exc}"}, None
    positives = int(np.count_nonzero(mask == values["crack"]))
    ignored = int(np.count_nonzero(mask == values["ignore"]))
    audit.update(crack_pixels=positives, ignore_pixels=ignored, mask_sha256=pixel_hash(mask))
    if ignored == mask.size or (
        not positives and any(c in mapping["foreground"] and not crowd for c, crowd, _ in instances)
    ):
        return audit | {"decision": "no_supervision_after_ignore"}, None
    mask_path = prepared / "masks" / f"{record['id']}.png"
    if source == CIF:
        image_path = prepared / "images" / f"{record['id']}{extension}"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
    else:
        image_path = record["image"]
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(mask_path, compress_level=1)
    return audit | {"decision": "candidate"}, {
        "sample_id": record["id"], "dataset_id": source, "split": record["split"],
        "image": image_path.relative_to(root).as_posix(), "mask": mask_path.relative_to(root).as_posix(),
        "width": width, "height": height, "annotation_coverage": "full_crack",
        "source_image_id": audit["decoded_image_sha256"], "domain": record["domain"],
        "material": record.get("material") or "unknown",
        "release": mapping["release"], "license": mapping["license"],
    }


def deduplicate(records):
    groups = defaultdict(list)
    for audit, row in records:
        if row:
            groups[audit["decoded_image_sha256"]].append((audit, row))
    priority = {"test": 0, "validation": 1, "train": 2}
    for members in groups.values():
        if len({audit["mask_sha256"] for audit, _ in members}) > 1:
            for audit, _ in members:
                audit["decision"] = "exclude_conflicting_duplicate_masks"
            continue
        members.sort(key=lambda item: (
            priority[item[1]["split"]], item[1]["dataset_id"], item[1]["sample_id"],
        ))
        members[0][0]["decision"] = "keep"
        for audit, _ in members[1:]:
            audit["decision"] = "exclude_exact_decoded_duplicate"


def write_reports(output, records, mappings):
    inventory = [audit for audit, _ in records]
    save_csv(output / "inventory.csv", inventory, sorted({key for row in inventory for key in row}))
    raw_hashes, structures = defaultdict(list), defaultdict(set)
    for item in inventory:
        if item.get("image_sha256"):
            raw_hashes[item["image_sha256"]].append(item)
        if item["structure"]:
            key = (item["dataset_id"], item["domain"], item["site"], item["structure"])
            structures[key].add(item["split"])
    raw_cross_split = [members for members in raw_hashes.values()
                       if len({member["split"] for member in members}) > 1]
    save_csv(output / "raw_cross_split_duplicates.csv", [
        {key: item[key] for key in ("dataset_id", "split", "sample_id", "source", "image_sha256")}
        for group in raw_cross_split for item in group
    ])
    summary = {
        "protocol": mappings["protocol"],
        "raw_cross_split_exact_hash_groups": len(raw_cross_split),
        "structure_groups_in_multiple_splits": [
            {"dataset": key[0], "collection": key[1], "site": key[2], "structure": key[3],
             "splits": sorted(splits)}
            for key, splits in structures.items() if len(splits) > 1
        ],
        "sources": {},
    }
    for source in (CUBIT, CIF):
        by_split = {}
        for split in SPLITS.values():
            items = [r for r in inventory if r["dataset_id"] == source and r["split"] == split]
            kept = [r for r in items if r["decision"] == "keep"]
            by_split[split] = {
                "raw": len(items), "kept": len(kept),
                "decisions": dict(Counter(r["decision"].split(":")[0] for r in items)),
                "crack_positive": sum(r["crack_pixels"] > 0 for r in kept),
                "crack_pixels": sum(r["crack_pixels"] for r in kept),
                "ignored_pixels": sum(r["ignore_pixels"] for r in kept),
                "pixels": sum(r["width"] * r["height"] for r in kept),
            }
        summary["sources"][source] = by_split
    write_json(output / "audit.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audit-dir", type=Path, default=Path(".cache/cubit-cif-audit-v1"))
    args = parser.parse_args()
    mappings = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
    config = ConfigLoader.load(args.config)
    destinations = [
        (source, Path(source.root).resolve(), safe_path(Path(source.root), source.manifest))
        for source in config.data.sources
    ]
    if args.audit_dir.exists() or any(path.parent.exists() for _, _, path in destinations):
        raise FileExistsError("Choose unused preparation and audit directories")
    args.audit_dir.mkdir(parents=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for source, root, manifest in destinations:
            mapping = mappings[source.name]
            inputs = cubit_records(root, mapping["classes"]) if source.name == CUBIT else cif_records(root)

            def convert(record):
                return prepare_record(
                    root, manifest.parent, source.name, record, mapping, mappings["mask_values"],
                )

            count = 0
            for count, result in enumerate(pool.map(convert, inputs, buffersize=2 * args.workers), 1):
                records.append(result)
                if count % 500 == 0:
                    print(f"{source.name}: inspected {count:,} source images", flush=True)
            print(f"{source.name}: inspected {count:,} source images", flush=True)
    deduplicate(records)
    summary = write_reports(args.audit_dir, records, mappings)
    for source, _, manifest in destinations:
        selected = [
            row for audit, row in records
            if audit["dataset_id"] == source.name and audit["decision"] == "keep"
        ]
        if {row["split"] for row in selected} != set(SPLITS.values()):
            raise ValueError(f"{source.name} has an empty training/evaluation split")
        assert len({row["source_image_id"] for row in selected}) == len(selected)
        write_manifest(manifest, selected)
        print(f"{manifest}: {len(selected):,} samples", flush=True)
    print(f"Cross-split raw duplicate groups: {summary['raw_cross_split_exact_hash_groups']}")


if __name__ == "__main__":
    main()
