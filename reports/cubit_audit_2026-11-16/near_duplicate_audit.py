"""Read-only pHash/dHash candidate audit; run with the project .venv from any directory."""

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.fft import dctn

from structural_defect_detection.data.manifest import read_manifest, safe_path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/CUBIT-InSeg"
REPORT = Path(__file__).resolve().parent
OUTPUT = REPORT / "near_duplicates"


def features(row):
    with Image.open(safe_path(DATA, row["image"])) as image:
        image.draft("RGB", (256, 256))
        gray = image.convert("L")
        small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
        coefficients = dctn(small, norm="ortho")[:8, :8].ravel()[1:]
        phash = int.from_bytes(np.packbits(np.r_[False, coefficients > np.median(coefficients)]).tobytes(), "big")
        edges = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
        dhash = int.from_bytes(np.packbits(edges[:, 1:] > edges[:, :-1]).tobytes(), "big")
        thumbnail = np.asarray(gray.resize((96, 96), Image.Resampling.LANCZOS))
    return {"image": row["image"], "phash": f"{phash:016x}", "dhash": f"{dhash:016x}",
            "gray_std": float(thumbnail.std())}, thumbnail


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    from eda_cubit_inseg import save_csv

    OUTPUT.mkdir(exist_ok=False)
    with (REPORT / "raw_eda/image_inventory.csv").open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    assert len(inventory) == 6996 and len({r["image"] for r in inventory}) == 6996
    kept = {r["image"] for r in read_manifest(DATA / "manifest.csv")}
    assert len(kept) == 3139
    hashes, thumbnails = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (row, thumbnail) in enumerate(pool.map(features, inventory, buffersize=8), 1):
            hashes.append(row)
            thumbnails.append(thumbnail)
            if index % 1000 == 0:
                print(f"Thumbnail hashes {index}/6996", flush=True)
    save_csv(OUTPUT / "image_hashes.csv", hashes)
    phash = np.array([int(r["phash"], 16) for r in hashes], dtype=np.uint64)
    dhash = np.array([int(r["dhash"], 16) for r in hashes], dtype=np.uint64)
    assert int(np.bitwise_count(np.uint64(0) ^ np.uint64(2**64 - 1))) == 64
    for index in range(20):
        assert int(np.bitwise_count(phash[index] ^ phash[index + 1])) == (int(phash[index]) ^ int(phash[index + 1])).bit_count()
    pairs, found = [], set()
    for i, left in enumerate(inventory):
        pd = np.bitwise_count(phash[i] ^ phash[i + 1:])
        dd = np.bitwise_count(dhash[i] ^ dhash[i + 1:])
        for offset in np.flatnonzero((pd <= 8) & (dd <= 10)):
            j = i + 1 + int(offset)
            right = inventory[j]
            a, b = thumbnails[i].astype(float), thumbnails[j].astype(float)
            ac, bc = a - a.mean(), b - b.mean()
            denominator = float(np.linalg.norm(ac) * np.linalg.norm(bc))
            correlation = float((ac * bc).sum() / denominator) if denominator else float(np.array_equal(a, b))
            pairs.append({"image_a": left["image"], "image_b": right["image"],
                          "split_a": left["split"], "split_b": right["split"],
                          "cross_split": left["split"] != right["split"],
                          "both_prepared": left["image"] in kept and right["image"] in kept,
                          "exact_file_duplicate": left["image_sha256"] == right["image_sha256"],
                          "phash_hamming": int(pd[offset]), "dhash_hamming": int(dd[offset]),
                          "gray_mae_96": float(np.abs(a - b).mean()), "gray_correlation_96": correlation})
            found.add((i, j))
    groups = {}
    for i, row in enumerate(inventory):
        groups.setdefault(row["image_sha256"], []).append(i)
    exact = {pair for group in groups.values() for pair in combinations(group, 2)}
    assert exact <= found, "Known byte-identical pairs must be recovered by the candidate search"
    assert len(found) == len(pairs)
    save_csv(OUTPUT / "candidate_pairs.csv", pairs)
    cross = [p for p in pairs if p["cross_split"] and p["both_prepared"] and not p["exact_file_duplicate"]]
    save_csv(OUTPUT / "prepared_cross_split_candidates.csv", cross)
    review = sorted(cross, key=lambda p: (p["phash_hamming"], p["dhash_hamming"], p["gray_mae_96"]))[:24]
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    for start in range(0, len(review), 3):
        page = Image.new("RGB", (960, 1020), "white")
        draw = ImageDraw.Draw(page)
        for row_index, pair in enumerate(review[start:start + 3]):
            y = row_index * 340
            draw.text((8, y + 3), f"pHash={pair['phash_hamming']} dHash={pair['dhash_hamming']} corr={pair['gray_correlation_96']:.4f} MAE={pair['gray_mae_96']:.2f}", fill="black", font=font)
            for column, key in enumerate(("image_a", "image_b")):
                with Image.open(safe_path(DATA, pair[key])) as image:
                    image.draft("RGB", (480, 320))
                    image = image.convert("RGB")
                    image.thumbnail((470, 280), Image.Resampling.LANCZOS)
                page.paste(image, (column * 480 + 5, y + 30))
                draw.text((column * 480 + 5, y + 312), pair[key], fill="black", font=font)
        page.save(OUTPUT / f"prepared_cross_split_review_{start // 3 + 1:02d}.png")
    summary = {
        "images": len(hashes), "all_pair_comparisons": len(hashes) * (len(hashes) - 1) // 2,
        "method": "Pillow JPEG draft256; grayscale LANCZOS; pHash32x32 DCT-II ortho top8x8 excluding DC, median63, plus horizontal dHash9x8; candidate if Hamming(pHash)<=8 AND Hamming(dHash)<=10.",
        "candidate_pairs": len(pairs), "exact_file_pairs_recovered": len(exact),
        "nonexact_candidates": sum(not p["exact_file_duplicate"] for p in pairs),
        "prepared_cross_split_nonexact_candidates": len(cross),
        "prepared_cross_split_patterns": dict(Counter("-".join(sorted((p["split_a"], p["split_b"]))) for p in cross)),
        "review_pages": (len(review) + 2) // 3,
        "limitations": ["Heuristic candidates, not automatically confirmed duplicate labels or physical identities.",
                        "Repeated facade textures can be false positives; visual review is required.",
                        "No exhaustive rotation, crop, perspective or capture-session independence guarantee.",
                        "Low-resolution thumbnail decoding does not replace full-resolution integrity checks."],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="New output directory; existing directories are refused")
    OUTPUT = parser.parse_args().output
    main()
