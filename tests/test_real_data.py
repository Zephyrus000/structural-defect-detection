import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from pycocotools import mask as coco_mask

from structural_defect_detection.cli import main
from structural_defect_detection.data import DataModule, ManifestBuilder
from structural_defect_detection.data.cubit import decode_segmentation
from structural_defect_detection.data.cubit_yolo import CubitYoloAdapter
from structural_defect_detection.data.dataset import collate
from structural_defect_detection.data.manifest import FIELDS, read_manifest
from structural_defect_detection.data.probe import RealDataSampler
from structural_defect_detection.evaluation import SegmentationMetrics
from structural_defect_detection.inference import Predictor
from structural_defect_detection.models import build_model
from structural_defect_detection.profiling import Profiler
from structural_defect_detection.verification import DryRunRunner, TrainingProbe


def test_raw_conversion_matches_prepared_masks(cubit_source, tmp_path):
    config, layout_path = cubit_source
    raw = next(RealDataSampler(config, layout_path).samples(1))
    root = Path(config.data.sources[0].root)
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    shutil.copy2(root / raw.row["image"], tmp_path / "images/one.jpg")
    shutil.copy2(raw.evidence["label"], tmp_path / "labels/one.txt")
    layout = yaml.safe_load(layout_path.read_text())
    layout["splits"] = {"train": {"images": "images", "labels": "labels"}}
    local = tmp_path / "layout.yaml"
    local.write_text(yaml.safe_dump(layout))
    adapter = CubitYoloAdapter(tmp_path, "cubit-inseg", local)
    report = adapter.prepare(tmp_path / "manifest.csv")
    prepared = replace(
        config,
        data=replace(config.data, sources=(replace(config.data.sources[0], root=str(tmp_path)),)),
    )
    decoded = next(RealDataSampler(prepared).samples(1))
    np.testing.assert_array_equal(raw.image, decoded.image)
    np.testing.assert_array_equal(raw.target, decoded.target)
    np.testing.assert_array_equal(raw.valid, decoded.valid)
    assert report["splits"][0]["samples"] == 1
    # Fault injection in a copy of a real annotation; never edit the source dataset.
    label = tmp_path / "labels/one.txt"
    original = label.read_text()
    label.write_text("99 " + original.split(" ", 1)[1])
    with pytest.raises(ValueError, match="unknown class"):
        adapter.rasterize(label, raw.image.shape[1], raw.image.shape[0])
    label.unlink()
    with pytest.raises(ValueError, match="pairing mismatch"):
        adapter._inputs()


def test_real_spalling_and_degenerate_annotation(cubit_source):
    config, layout = cubit_source
    root = Path(config.data.sources[0].root)
    adapter = CubitYoloAdapter(root, "cubit-inseg", layout)
    label = sorted((root / "train/labels").glob("SP*.txt"))[0]
    image = next((root / "train/images").glob(label.stem + ".*"))
    with Image.open(image) as handle:
        target, valid, counts = adapter.rasterize(label, *handle.size)
    assert counts[1] > 0 and (~valid).any() and not target[~valid].any()
    with pytest.raises(ValueError, match="degenerate polygon"):
        adapter.rasterize(root / "train/labels/CR1_2035.txt", 4500, 3000)


def test_real_mask_rle_roundtrip_and_metrics(annotated_crops):
    _, target, valid, _, _ = annotated_crops[0]
    encoded = coco_mask.encode(np.asfortranarray(target.astype(np.uint8)))
    np.testing.assert_array_equal(decode_segmentation(encoded, *target.shape), target)
    assert SegmentationMetrics.sample(target, target, valid)["dice"] == 1
    damaged = target.copy()
    y, x = np.argwhere(target & valid)[0]
    damaged[y, x] = False
    assert SegmentationMetrics.sample(damaged, target, valid)["dice"] < 1
    background = target[~target & valid][:1].reshape(1, 1)
    assert SegmentationMetrics.sample(background, background, ~background)["dice"] == 1
    assert SegmentationMetrics.sample(target, target, valid & ~valid)["dice"] is None


def test_real_crop_alignment_padding_and_leakage(real_config):
    cfg = replace(real_config, data=replace(real_config.data, augment=True, size=(64, 64)))
    data = DataModule(cfg)
    a, b = next(iter(data.train_loader(2))), next(iter(data.train_loader(2)))
    assert (
        a["image"].equal(b["image"])
        and a["target"].equal(b["target"])
        and a["valid"].equal(b["valid"])
    )
    assert not a["valid"].all()
    source = cfg.data.sources[0]
    path = Path(source.root) / source.manifest
    rows = read_manifest(path)
    rows[2]["image"] = rows[0]["image"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="content leakage"):
        ManifestBuilder(cfg.data).validate()


def test_tile_alignment_on_real_rgb(annotated_crops, real_config):
    class PixelModel(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]

    cfg = replace(
        real_config,
        data=replace(real_config.data, size=(32, 32), mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    )
    rgb = annotated_crops[0][0][:43, :47]
    result = Predictor(PixelModel(), cfg, torch.device("cpu")).predict(rgb)
    expected = torch.tensor(rgb[:, :, 0].astype(float) / 255).sigmoid().numpy()
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("variant", ["standard", "efficient", "cnn_only"])
def test_real_batch_backward_preserves_weights(real_config, variant):
    cfg = replace(
        real_config,
        model=replace(
            real_config.model,
            name="cnn_only" if variant == "cnn_only" else "hybrid",
            cnn="efficient" if variant == "efficient" else "standard",
        ),
    )
    sample = next(RealDataSampler(cfg).samples(1)).transformed(cfg)
    for key in ("image", "target", "valid"):
        sample[key] = sample[key][:, :29, :35]
    model = build_model(cfg.model)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    report, _ = TrainingProbe(model, cfg, torch.device("cpu")).check(collate([sample]))
    assert report["logits_shape"] == [1, 1, 29, 35]
    assert report["ph_gradient_norm"] > 0 and report["model_gradient_norm"] > 0
    assert report["optimizer_steps"] == 0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_dry_run_artifacts_limits_and_failure(real_config):
    result = DryRunRunner(real_config).run("real", max_samples=1)
    output = Path(result["output"])
    assert result["samples_checked"] == 1 and result["optimizer_steps"] == 0
    assert result["samples"][0]["sample"]["split"] == "train"
    assert list((output / "previews").glob("*_target.png"))
    assert not list(output.rglob("*.pt")) and not (output / "state.json").exists()
    with pytest.raises(FileExistsError):
        DryRunRunner(real_config).run("real")
    source = real_config.data.sources[0]
    missing = replace(
        real_config,
        data=replace(real_config.data, sources=(replace(source, manifest="missing.csv"),)),
    )
    with pytest.raises(FileNotFoundError, match="Missing prepared manifest"):
        DryRunRunner(missing).run("missing")
    failed = json.loads(
        (Path(real_config.output.root) / "dry-runs/missing/report.json").read_text()
    )
    assert failed["status"] == "failed"
    with pytest.raises(ValueError, match="positive"):
        DryRunRunner(real_config).run("zero", max_samples=0)


def test_dry_run_cli_guards_and_real_profiler(real_config, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(real_config.to_dict()))
    common = ["train", "--config", str(config)]
    assert main(common + ["--max-samples", "1"]) == 2
    assert main(common + ["--dry-run", "--retry"]) == 2
    assert main(common + ["--dry-run", "--max-samples", "0"]) == 2
    assert main(common + ["--dry-run", "--max-samples", "1", "--run-id", "cli"]) == 0
    sample = next(RealDataSampler(real_config).samples(1))
    result = Profiler(build_model(real_config.model), real_config, torch.device("cpu")).measure(
        sample, 1, 0
    )
    assert result["input"] == "real transformed training patch"
    assert result["evidence"]["mask_sha256"] and result["training_probe"]["optimizer_steps"] == 0


def test_ph_diagram_matches_gudhi_on_real_prediction(real_config):
    import gudhi

    from structural_defect_detection.losses.topology import GudhiTopologyBackend

    sample = next(RealDataSampler(real_config).samples(1)).transformed(real_config)
    model = build_model(real_config.model).eval()
    probability = model(sample["image"][None]).sigmoid()[0, 0, :16, :16]
    actual = GudhiTopologyBackend(min_persistence=0.01, max_pairs=1000).diagram(probability)
    reference = gudhi.CubicalComplex(top_dimensional_cells=(1 - probability).detach().numpy())
    reference.compute_persistence(homology_coeff_field=2, min_persistence=0)
    expected = reference.persistence_intervals_in_dimension(0)
    expected[np.isinf(expected)] = 1
    expected = expected[expected[:, 1] - expected[:, 0] > 0.01]
    np.testing.assert_allclose(
        sorted(actual.detach().tolist()), sorted(expected.tolist()), atol=1e-6
    )
    actual.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_no_ph_and_unsupervised_crop_policy(real_config):
    cfg = replace(
        real_config, loss=replace(real_config.loss, ph=replace(real_config.loss.ph, weight=0))
    )
    sample = next(RealDataSampler(cfg).samples(1)).transformed(cfg)
    probe = TrainingProbe(build_model(cfg.model), cfg, torch.device("cpu"))
    report, _ = probe.check(collate([sample]))
    assert report["loss"]["ph"] == 0 and report["ph_gradient_norm"] is None
    # Fault injection: remove supervision from a real crop and require a diagnostic.
    sample["valid"] = sample["valid"] & ~sample["valid"]
    with pytest.raises(ValueError, match="supervised pixels"):
        probe.check(collate([sample]))
