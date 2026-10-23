import json
from pathlib import Path

import pytest
import torch
import yaml

from structural_defect_detection.engine import CheckpointManager, Trainer
from structural_defect_detection.experiments import (
    ExperimentRunner,
    ResultAggregator,
    SweepPlanner,
    SweepRunner,
)


def test_real_crop_resume_and_evaluation(real_config, monkeypatch):
    runner = ExperimentRunner(real_config)
    runner.run("continuous")
    original = Trainer._train_epoch

    def interrupt(self, epoch):
        if epoch == 1:
            raise RuntimeError("interruption after epoch checkpoint")
        return original(self, epoch)

    with monkeypatch.context() as patch:
        patch.setattr(Trainer, "_train_epoch", interrupt)
        with pytest.raises(RuntimeError, match="interruption"):
            runner.run("interrupted")
    root = Path(real_config.output.root)
    runner.run("interrupted", resume=root / "interrupted/checkpoints/last.pt")
    a = CheckpointManager.load(root / "continuous/checkpoints/last.pt")
    b = CheckpointManager.load(root / "interrupted/checkpoints/last.pt")
    assert a["step"] == b["step"] and a["scheduler"] == b["scheduler"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    result = runner.evaluate(root / "continuous/checkpoints/best.pt", "test", root / "evaluation")
    assert result["overall"]["samples"] == 1
    assert list((root / "evaluation/predictions").glob("*_mask.png"))


@pytest.mark.parametrize("method", ["grid", "random"])
def test_real_crop_sweep_shards_collection_retry(real_config, tmp_path, monkeypatch, method):
    config = real_config.to_dict()
    config["training"]["epochs"] = 1
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(config))
    sweep = {
        "base": "base.yaml",
        "seeds": [11],
        "search": {
            "method": method,
            "seed": 7,
            "trials": 20,
            "parameters": {"loss.ph.weight": [0.0, 0.05], "loss.ph.window_size": [16, 24]},
        },
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(sweep))
    plan_path = tmp_path / "plan.json"
    plan = SweepPlanner().plan(path, plan_path)
    assert plan == SweepPlanner().plan(path, tmp_path / "same-plan.json")
    assert len(plan["trials"]) == 3
    partitions = [{t["run_id"] for t in SweepRunner.select(plan, i, 2)} for i in range(2)]
    assert not partitions[0] & partitions[1] and len(partitions[0] | partitions[1]) == 3
    roots = [tmp_path / "machine-a", tmp_path / "machine-b"]
    machines = []
    for i, root in enumerate(roots):
        machine = tmp_path / f"machine-{i}.yaml"
        machine.write_text(yaml.safe_dump({"output": {"root": str(root)}}))
        machines.append(machine)
    first = plan["trials"][0]["run_id"]
    with monkeypatch.context() as patch:

        def fail(*args):
            raise FloatingPointError("injected runtime failure")

        patch.setattr(Trainer, "_train_epoch", fail)
        assert SweepRunner().run(plan_path, machines[0], run_ids=[first])["failed"] == [first]
    failed = ResultAggregator().collect(plan_path, roots, tmp_path / "failed")
    assert failed["failed"] == 1 and not failed["rows"]
    for i, machine in enumerate(machines):
        result = SweepRunner().run(plan_path, machine, i, 2, retry_failed=True)
        assert not result["failed"]
        assert SweepRunner().run(plan_path, machine, i, 2)["skipped"] == result["completed"]
    collected = ResultAggregator().collect(plan_path, roots, tmp_path / "collected")
    assert collected["completed"] == 3 and not collected["missing_or_pending"]
    with pytest.raises(ValueError, match="Duplicate"):
        ResultAggregator().collect(plan_path, [roots[0], roots[0]], tmp_path / "duplicates")
    bad = next(roots[0].glob("*/result.json"))
    content = json.loads(bad.read_text())
    content["identity"]["split_hash"] = "tampered"
    bad.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="split mismatch"):
        ResultAggregator().collect(plan_path, roots, tmp_path / "mismatch")
    plan["trials"][0]["seed"] = 999
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="changed"):
        SweepRunner().run(plan_path)
