"""Fixtures use actual CUBIT training images/labels; no generated data fallback."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from structural_defect_detection.config import ConfigLoader
from structural_defect_detection.data.manifest import write_manifest
from structural_defect_detection.data.probe import RealDataSampler
from structural_defect_detection.utils import write_json

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption("--data-root", default=str(ROOT / "data/CUBIT-InSeg"))
    parser.addoption("--data-layout", default=str(ROOT / "configs/datasets/cubit_yolo.local.yaml"))


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture(scope="session")
def cubit_source(request):
    config = ConfigLoader.load(ROOT / "configs/experiments/hybrid_ph.yaml")
    root = Path(request.config.getoption("--data-root")).resolve()
    layout = Path(request.config.getoption("--data-layout")).resolve()
    if not root.is_dir() or not layout.is_file():
        pytest.fail("Real CUBIT data required: set --data-root and --data-layout. No fallback.")
    source = replace(config.data.sources[0], root=str(root))
    return replace(config, data=replace(config.data, sources=(source,))), layout


@pytest.fixture(scope="session")
def annotated_crops(cubit_source):
    config, layout = cubit_source
    crops = []
    for raw in RealDataSampler(config, layout).samples(4):
        points = np.argwhere(raw.target & raw.valid)
        if not len(points):
            pytest.fail(f"No foreground annotation in fixture {raw.row['sample_id']}")
        y, x = points[len(points) // 2]
        h, w = raw.target.shape
        top, left = min(max(int(y) - 24, 0), h - 48), min(max(int(x) - 24, 0), w - 48)
        region = np.s_[top : top + 48, left : left + 48]
        crops.append(
            (
                raw.image[region].copy(),
                raw.target[region].copy(),
                raw.valid[region].copy(),
                raw.row.copy(),
                raw.evidence | {"fixture_crop": [top, left, 48, 48]},
            )
        )
    return crops


@pytest.fixture
def real_config(tmp_path, annotated_crops):
    root = tmp_path / "real-cubit-crops"
    root.mkdir()
    rows, provenance = [], []
    for index, (image, target, valid, original, evidence) in enumerate(annotated_crops):
        stem = f"sample-{index}"
        Image.fromarray(image).save(root / f"{stem}.png")
        Image.fromarray(target.astype(np.uint8) * 255).save(root / f"{stem}_mask.png")
        Image.fromarray(valid.astype(np.uint8) * 255).save(root / f"{stem}_valid.png")
        rows.append(
            original
            | {
                "image": f"{stem}.png",
                "mask": f"{stem}_mask.png",
                "valid_mask": f"{stem}_valid.png",
                "width": 48,
                "height": 48,
                # Isolated software partitions of official TRAIN; not benchmark metrics.
                "split": "train" if index < 2 else ("validation" if index == 2 else "test"),
                "source_image_id": original["sample_id"],
            }
        )
        provenance.append(evidence | {"original": original})
    write_manifest(root / "manifest.csv", rows)
    write_json(
        root / "provenance.json",
        {"scope": "official train only; software-test partitions", "samples": provenance},
    )
    value = ConfigLoader.load(ROOT / "configs/experiments/hybrid_ph.yaml").to_dict()
    value["data"].update(
        sources=[{"name": "cubit-inseg", "root": str(root), "manifest": "manifest.csv"}],
        size=[48, 48],
        augment=False,
    )
    value["model"].update(channels=[8, 16, 24], embed_dim=24, decoder_channels=8, window_size=4)
    value["training"].update(epochs=2, effective_batch=2)
    value["loss"]["ph"].update(window_size=16, max_pairs=32)
    value["evaluation"].update(overlap=8, topology_window=16)
    value["runtime"].update(device="cpu", workers=0, threads=2, micro_batch=1, amp=False)
    value["output"]["root"] = str(tmp_path / "runs")
    return ConfigLoader.from_dict(value)
