from pathlib import Path

import pytest

from structural_defect_detection.cli import main, parser
from structural_defect_detection.config import ConfigLoader


@pytest.mark.parametrize(
    "override",
    [
        {"model": {"unknown": 1}},
        {"training": {"epochs": True}},
        {"runtime": {"micro_batch": 3}},
        {"runtime": {"amp_dtype": "float8"}},
        {"data": {"mean": [float("nan"), 0.5, 0.5]}},
        {"model": {"embed_dim": 17}},
        {"model": {"patch_merging": True}},
        {"loss": {"ph": {"weight": -1}}},
        {"data": {"size": [8]}},
        {"loss": {"ph": {"ignore_policy": "background"}}},
    ],
)
def test_invalid_config_fails(override):
    with pytest.raises(ValueError):
        ConfigLoader.from_dict(override)


def test_config_precedence_and_identity(tmp_path):
    (tmp_path / "base.yaml").write_text("training:\n  epochs: 4\n")
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("base: base.yaml\ntraining:\n  epochs: 5\n")
    machine = tmp_path / "machine.yaml"
    machine.write_text(
        "runtime:\n  micro_batch: 1\noutput:\n  root: other\nroots:\n  cracks: another-root\n"
    )
    original = ConfigLoader.load(experiment)
    assert ConfigLoader.load(experiment, machine, ["training.epochs=6"]).training.epochs == 6
    assert ConfigLoader.load(experiment, machine).experiment_hash == original.experiment_hash
    machine.write_text("model:\n  depth: 4\n")
    with pytest.raises(ValueError):
        ConfigLoader.load(experiment, machine)
    (tmp_path / "base.yaml").write_text("base: experiment.yaml\n")
    with pytest.raises(ValueError, match="Cyclic"):
        ConfigLoader.load(experiment)


def test_cli_validation_and_removed_command(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("training:\n  epoches: 2\n")
    assert main(["train", "--config", str(path)]) == 2
    assert "Unknown fields" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parser().parse_args(["demo", "--output", str(tmp_path)])


def test_repository_configs():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs/experiments").glob("*.yaml"):
        ConfigLoader.load(path)
    for path in (root / "configs/machines").glob("*.yaml"):
        ConfigLoader.load(root / "configs/base.yaml", path)
