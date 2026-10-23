"""Thin command handlers. Run `sdd --help` for the public interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .config import ConfigLoader
from .data import DeepCrackAdapter, ManifestBuilder, PairedFolderAdapter, SplitBuilder
from .data.cubit import CubitInSegAdapter
from .data.cubit_yolo import CubitYoloAdapter
from .data.manifest import safe_path, write_manifest
from .data.probe import RealDataSampler
from .experiments import ExperimentRunner, ResultAggregator, SweepPlanner, SweepRunner
from .inference import PredictionWriter, Predictor
from .models import build_model
from .profiling import Profiler
from .utils import file_hash, resolve_device, runtime_metadata, seed_everything, write_json
from .verification import DryRunRunner


def _configuration(parser):
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--machine", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="sdd", description="CNN-Swin + PH structural defect segmentation"
    )
    root.add_argument("--debug", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data").add_subparsers(dest="action", required=True)
    for name in ("validate", "prepare", "split"):
        command = data.add_parser(name)
        _configuration(command)
        command.add_argument("--source", help="Source name; optional for a single-source config")
        if name == "validate":
            command.add_argument("--output", type=Path)
        elif name == "prepare":
            command.add_argument(
                "--adapter",
                choices=["cubit-yolo", "cubit-coco", "paired", "deepcrack"],
                required=True,
            )
            command.add_argument("--layout", type=Path, help="Required for CUBIT adapters")
            command.add_argument("--images", default="images")
            command.add_argument("--masks", default="masks")
            command.add_argument("--release")
            command.add_argument("--coverage", choices=["full_crack"])
            command.add_argument(
                "--split",
                choices=["train", "validation", "test", "unassigned"],
                default="unassigned",
            )
        else:
            command.add_argument(
                "--output", required=True, help="New manifest path relative to source root"
            )
            command.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    train = commands.add_parser("train")
    _configuration(train)
    train.add_argument("--run-id")
    train.add_argument("--resume", type=Path)
    train.add_argument("--retry", action="store_true")
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Check real training samples without optimizer updates",
    )
    train.add_argument("--max-samples", type=int, help="Dry-run sample limit (default: 2)")
    train.add_argument(
        "--layout", type=Path, help="Dry-run only: read raw CUBIT YOLO labels in memory"
    )
    evaluate = commands.add_parser("evaluate")
    _configuration(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=["validation", "test", "external_test"], required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("predict")
    _configuration(predict)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    sweep = commands.add_parser("sweep").add_subparsers(dest="action", required=True)
    plan = sweep.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--machine", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    run = sweep.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--machine", type=Path)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--run-id", action="append", dest="run_ids")
    run.add_argument("--retry-failed", action="store_true")
    collect = sweep.add_parser("collect")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--root", type=Path, action="append", required=True, dest="roots")
    collect.add_argument("--output", type=Path, required=True)
    profile = commands.add_parser("profile")
    _configuration(profile)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--iterations", type=int, default=10)
    profile.add_argument("--warmup", type=int, default=2)
    profile.add_argument(
        "--layout", type=Path, help="Use a raw CUBIT YOLO training sample instead of a manifest"
    )
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--device", default="cpu")
    return root


def execute(args):
    if args.command == "doctor":
        return runtime_metadata(resolve_device(args.device))
    if args.command == "sweep":
        if args.action == "plan":
            plan = SweepPlanner().plan(args.config, args.output, args.machine)
            return {
                "plan_id": plan["plan_id"],
                "trials": len(plan["trials"]),
                "path": str(args.output),
            }
        if args.action == "run":
            return SweepRunner().run(
                args.plan,
                args.machine,
                args.shard_index,
                args.shard_count,
                args.run_ids,
                args.retry_failed,
            )
        report = ResultAggregator().collect(args.plan, args.roots, args.output)
        return {
            "completed": report["completed"],
            "failed": report["failed"],
            "missing_or_pending": report["missing_or_pending"],
            "output": str(args.output),
        }
    config = ConfigLoader.load(args.config, args.machine, args.overrides)
    if args.command == "data":
        if args.action == "validate":
            _, report = ManifestBuilder(config.data).validate()
            if args.output:
                write_json(args.output, report)
            return report
        matches = [s for s in config.data.sources if args.source is None or s.name == args.source]
        if len(matches) != 1:
            raise ValueError("Select exactly one configured source with --source")
        source = matches[0]
        if args.action == "split":
            rows = SplitBuilder().build(source, config.training.seed, tuple(args.ratios))
            destination = safe_path(Path(source.root), args.output)
            write_manifest(destination, rows)
            return {
                "manifest": str(destination),
                "samples": len(rows),
                "next": "Update source.manifest to this file",
            }
        destination = safe_path(Path(source.root), source.manifest)
        if args.adapter in {"cubit-coco", "cubit-yolo"}:
            if args.layout is None:
                raise ValueError(
                    "CUBIT adapters require --layout with verified paths/class mapping"
                )
            adapter_type = CubitInSegAdapter if args.adapter == "cubit-coco" else CubitYoloAdapter
            return adapter_type(Path(source.root), source.name, args.layout).prepare(destination)
        if not args.release or args.coverage != "full_crack":
            raise ValueError(
                "Provide --release and --coverage full_crack after inspecting annotation coverage"
            )
        if args.adapter == "deepcrack":
            if source.name != "deepcrack":
                raise ValueError("DeepCrack source must be named deepcrack")
            adapter = DeepCrackAdapter(Path(source.root), args.release, seed=config.training.seed)
        else:
            adapter = PairedFolderAdapter(
                Path(source.root),
                source.name,
                args.images,
                args.masks,
                args.release,
                args.coverage,
                args.split,
            )
        adapter.write(destination)
        return {"manifest": str(destination)}
    runner = ExperimentRunner(config)
    if args.command == "train":
        if args.dry_run:
            if args.resume or args.retry:
                raise ValueError("--dry-run cannot be combined with --resume or --retry")
            result = DryRunRunner(config).run(
                args.run_id, args.layout, args.max_samples if args.max_samples is not None else 2
            )
            return {
                k: result[k]
                for k in (
                    "kind",
                    "run_id",
                    "status",
                    "samples_checked",
                    "samples_with_foreground",
                    "optimizer_steps",
                    "output",
                )
            }
        if args.layout is not None or args.max_samples is not None:
            raise ValueError("train --layout/--max-samples require --dry-run")
        result = runner.run(args.run_id, args.resume, args.retry)
        return {
            "run_id": result["run_id"],
            "status": result["status"],
            "best_validation_dice": result["best_validation"]["selection"]["crack_dice"],
        }
    if args.command == "evaluate":
        result = runner.evaluate(args.checkpoint, args.split, args.output)
        return {"split": args.split, "metrics": result["overall"], "output": str(args.output)}
    if args.command == "predict":
        model, device, _ = runner.load_model(args.checkpoint)
        predictor, writer = (
            Predictor(model, config, device),
            PredictionWriter(args.output, config.evaluation.threshold),
        )
        paths = (
            sorted(
                p
                for p in args.input.rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
            )
            if args.input.is_dir()
            else [args.input]
        )
        if not paths:
            raise ValueError("No input images")
        for path in paths:
            with Image.open(path) as handle:
                image = np.array(handle.convert("RGB"))
            sample_id = (
                path.relative_to(args.input).as_posix() if args.input.is_dir() else path.name
            )
            writer.write(
                image,
                predictor.predict(image),
                {
                    "sample_id": sample_id,
                    "dataset_id": "inference",
                    "image": str(path),
                    "mode": config.evaluation.mode,
                    "size": list(config.data.size),
                },
            )
        write_json(
            args.output / "prediction_run.json",
            {
                "checkpoint_hash": file_hash(args.checkpoint),
                "config": config.to_dict(),
                "images": len(paths),
            },
        )
        return {"images": len(paths), "output": str(args.output)}
    seed_everything(config.training.seed, config.runtime.threads)
    device = resolve_device(config.runtime.device)
    sample = next(RealDataSampler(config, args.layout).samples(1))
    report = Profiler(build_model(config.model).to(device), config, device).measure(
        sample, args.iterations, args.warmup
    )
    write_json(args.output, report)
    return report


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        result = execute(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 1 if args.command == "sweep" and args.action == "run" and result["failed"] else 0
    except (ValueError, RuntimeError, OSError, KeyError, FloatingPointError) as exc:
        if args.debug:
            raise
        print(f"sdd: {exc}", file=sys.stderr)
        return 2
