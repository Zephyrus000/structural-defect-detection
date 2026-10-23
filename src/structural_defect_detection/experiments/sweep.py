"""Immutable grid/random plans, disjoint shards and provenance-aware collection."""

from __future__ import annotations

import copy
import csv
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from ..config import ConfigLoader, PHConfig, set_value, stable_hash
from ..data import DataModule
from ..utils import write_json
from .runner import ExperimentRunner


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in plan.items() if k != "plan_id"}
    if plan.get("version") != 1 or stable_hash(payload) != plan.get("plan_id"):
        raise ValueError("Sweep plan changed or has an unsupported version")
    ids = [trial["run_id"] for trial in plan["trials"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate trial IDs in plan")
    return plan


class SweepPlanner:
    def plan(self, path: Path, output: Path, machine: Path | None = None) -> dict:
        import dataclasses

        specification = yaml.safe_load(path.read_text(encoding="utf-8"))
        allowed = {"name", "base", "search", "seeds", "selection", "execution"}
        if not isinstance(specification, dict) or specification.keys() - allowed:
            raise ValueError("Unknown sweep fields")
        selection = specification.get(
            "selection", {"split": "validation", "metric": "crack_dice", "mode": "max"}
        )
        if selection != {"split": "validation", "metric": "crack_dice", "mode": "max"}:
            raise ValueError("v1 selects validation crack_dice, mode=max only")
        execution = specification.get("execution", {})
        if (
            execution.keys() - {"max_parallel_per_device", "resume_completed"}
            or execution.get("max_parallel_per_device", 1) != 1
            or execution.get("resume_completed", "skip") != "skip"
        ):
            raise ValueError("v1 uses one run/device and skips completed runs")
        base = ConfigLoader.load(path.parent / specification["base"], machine=machine)
        audit = DataModule(base).report
        search = specification["search"]
        if search.keys() - {"method", "parameters", "groups", "seed", "trials"}:
            raise ValueError("Unknown search fields")
        method = search.get("method", "grid")
        if method not in ("grid", "random"):
            raise ValueError("Search method must be grid or random")
        seeds = specification.get("seeds", [42])
        if (
            not seeds
            or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("Seeds must be a nonempty list of unique nonnegative integers")
        branches = search.get("groups", [{}])
        if not isinstance(branches, list) or not branches:
            raise ValueError("search.groups must contain parameter mappings")
        spaces = []
        for branch in branches:
            parameters = search.get("parameters", {}) | branch
            for key, values in parameters.items():
                if (
                    key.startswith(("runtime.", "output.", "data.sources"))
                    or key == "training.seed"
                ):
                    raise ValueError(
                        "Sweep data sources/runtime in separate plans; training seeds use seeds"
                    )
                if not isinstance(values, list) or not values:
                    raise ValueError(f"Search parameter {key} requires a nonempty list of choices")
            spaces.append(parameters)
        rng = random.Random(search.get("seed", 42))
        combinations = []
        if method == "grid":
            for space in spaces:
                keys = sorted(space)
                combinations.extend(
                    dict(zip(keys, values, strict=True))
                    for values in itertools.product(*(space[k] for k in keys))
                )
        else:
            budget = search.get("trials")
            if type(budget) is not int or budget < 1:
                raise ValueError("Random search requires a positive trials budget")
            for _ in range(budget):
                space = rng.choice(spaces)
                combinations.append(
                    {key: rng.choice(values) for key, values in sorted(space.items())}
                )
        trials, seen = [], set()
        for combination in combinations:
            for seed in seeds:
                value = base.to_dict()
                for key, choice in combination.items():
                    set_value(value, key, choice)
                value["training"]["seed"] = seed
                if value["loss"]["ph"]["weight"] == 0:
                    value["loss"]["ph"] = dataclasses.asdict(PHConfig(weight=0))
                cfg = ConfigLoader.from_dict(value)
                if cfg.experiment_hash in seen:
                    continue
                seen.add(cfg.experiment_hash)
                trials.append(
                    {
                        "run_id": cfg.experiment_hash[:24],
                        "experiment_hash": cfg.experiment_hash,
                        "config": cfg.to_dict(),
                        "seed": seed,
                    }
                )
        plan = {
            "version": 1,
            "name": specification.get("name", path.stem),
            "selection": selection,
            "search": search,
            "trials": trials,
            "expected_data": {
                "manifest_hash": audit["manifest_hash"],
                "split_hash": audit["split_hash"],
            },
        }
        plan["plan_id"] = stable_hash(plan)
        if output.exists():
            raise FileExistsError("Sweep plans are immutable; use a new output path")
        write_json(output, plan)
        return plan


class SweepRunner:
    @staticmethod
    def select(plan: dict, shard_index: int, shard_count: int, run_ids: list[str] | None = None):
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("Invalid shard index/count")
        if run_ids:
            if shard_index != 0 or shard_count != 1:
                raise ValueError("Choose explicit run IDs or sharding, not both")
            if len(set(run_ids)) != len(run_ids) or set(run_ids) - {
                t["run_id"] for t in plan["trials"]
            }:
                raise ValueError("Unknown or duplicate run IDs")
            return [t for t in plan["trials"] if t["run_id"] in run_ids]
        return [t for i, t in enumerate(plan["trials"]) if i % shard_count == shard_index]

    def run(
        self,
        plan_path: Path,
        machine: Path | None = None,
        shard_index: int = 0,
        shard_count: int = 1,
        run_ids: list[str] | None = None,
        retry_failed: bool = False,
    ) -> dict:
        plan = read_plan(plan_path)
        summary = {"completed": [], "skipped": [], "failed": []}
        for trial in self.select(plan, shard_index, shard_count, run_ids):
            value = ConfigLoader.machine(trial["config"], machine) if machine else trial["config"]
            cfg = ConfigLoader.from_dict(value)
            if cfg.experiment_hash != trial["experiment_hash"]:
                raise ValueError("Machine override changed experiment identity")
            report = DataModule(cfg).report
            if any(report[k] != v for k, v in plan["expected_data"].items()):
                raise ValueError("Dataset content/splits changed since planning")
            state_file = Path(cfg.output.root) / trial["run_id"] / "state.json"
            if state_file.exists():
                state = json.loads(state_file.read_text())
                if (
                    state.get("plan_id") != plan["plan_id"]
                    or state.get("experiment_hash") != cfg.experiment_hash
                ):
                    raise ValueError("Existing output belongs to another sweep/configuration")
                if state["status"] == "completed":
                    summary["skipped"].append(trial["run_id"])
                    continue
                if state["status"] == "running":
                    raise RuntimeError(
                        "A selected run is still running or interrupted; inspect its process/lock"
                    )
                if state["status"] == "failed" and not retry_failed:
                    summary["failed"].append(trial["run_id"])
                    continue
            try:
                ExperimentRunner(cfg).run(
                    trial["run_id"], retry=retry_failed, plan_id=plan["plan_id"]
                )
                summary["completed"].append(trial["run_id"])
            except Exception as exc:
                summary["failed"].append(trial["run_id"])
                print(f"run={trial['run_id']} failed: {exc}", flush=True)
        return summary


class ResultAggregator:
    def collect(self, plan_path: Path, roots: list[Path], output: Path) -> dict:
        plan = read_plan(plan_path)
        trials = {t["run_id"]: t for t in plan["trials"]}
        discovered, rows, statuses = {}, [], {}
        for root in roots:
            for path in sorted(root.glob("*/state.json")):
                state = json.loads(path.read_text())
                run_id = state["run_id"]
                if run_id not in trials:
                    continue
                if run_id in discovered:
                    raise ValueError(f"Duplicate run ID across roots: {run_id}")
                discovered[run_id] = path
                trial = trials[run_id]
                if (
                    state.get("plan_id") != plan["plan_id"]
                    or state["experiment_hash"] != trial["experiment_hash"]
                ):
                    raise ValueError(f"Plan/config mismatch: {run_id}")
                statuses[run_id] = state["status"]
                if state["status"] != "completed":
                    continue
                result = json.loads((path.parent / "result.json").read_text())
                resolved = ConfigLoader.load(path.parent / "resolved.yaml")
                if (
                    resolved.experiment_hash != trial["experiment_hash"]
                    or result["config_hash"] != resolved.config_hash
                    or result.get("plan_id") != plan["plan_id"]
                    or result["run_id"] != run_id
                    or result["status"] != "completed"
                ):
                    raise ValueError(f"Result/config provenance mismatch: {run_id}")
                if any(result["identity"][k] != v for k, v in plan["expected_data"].items()):
                    raise ValueError(f"Dataset/split mismatch: {run_id}")
                family_config = copy.deepcopy(trial["config"])
                family_config["training"]["seed"] = 0
                family = ConfigLoader.from_dict(family_config).experiment_hash
                report = result["best_validation"]
                for group_type in ("datasets", "domains"):
                    for group, metrics in report[group_type].items():
                        rows.append(
                            {
                                "run_id": run_id,
                                "family": family,
                                "seed": trial["seed"],
                                "group_type": group_type,
                                "group": group,
                                **{
                                    k: metrics[k]
                                    for k in (
                                        "dice",
                                        "iou",
                                        "precision",
                                        "recall",
                                        "window_betti0_mae",
                                    )
                                },
                            }
                        )
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["family"], row["group_type"], row["group"])].append(row)
        aggregate = []
        for (family, group_type, group), values in grouped.items():
            item = {"family": family, "group_type": group_type, "group": group, "runs": len(values)}
            for metric in ("dice", "iou", "precision", "recall", "window_betti0_mae"):
                numbers = [v[metric] for v in values if v[metric] is not None]
                item[metric] = {
                    "mean": float(np.mean(numbers)) if numbers else None,
                    "std": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None,
                    "n": len(numbers),
                }
            aggregate.append(item)
        missing = sorted(trials.keys() - discovered.keys())
        report = {
            "plan_id": plan["plan_id"],
            "rows": rows,
            "aggregate": aggregate,
            "statuses": statuses,
            "missing_or_pending": missing,
            "completed": sum(v == "completed" for v in statuses.values()),
            "failed": sum(v == "failed" for v in statuses.values()),
        }
        write_json(output / "summary.json", report)
        if rows:
            with (output / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return report
