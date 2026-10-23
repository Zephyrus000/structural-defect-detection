"""Strict, immutable configuration and machine-independent experiment identities."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import types
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclasses.dataclass(frozen=True)
class SourceConfig:
    name: str = "cracks"
    root: str = "data/cracks"
    manifest: str = "manifest.csv"
    weight: float = 1.0
    foreground_values: tuple[int, ...] = (255,)
    background_values: tuple[int, ...] = (0,)
    ignore_values: tuple[int, ...] = (128,)


@dataclasses.dataclass(frozen=True)
class DataConfig:
    sources: tuple[SourceConfig, ...] = (SourceConfig(),)
    train_mode: str = "crop"
    size: tuple[int, int] = (256, 256)
    augment: bool = True
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    name: str = "hybrid"
    cnn: str = "standard"
    channels: tuple[int, ...] = (32, 64, 128)
    feature_level: int = 2
    embed_dim: int = 96
    depth: int = 2
    heads: int = 3
    window_size: int = 8
    patch_merging: bool = False
    decoder_channels: int = 32
    dropout: float = 0.0
    init_weights: str | None = None


@dataclasses.dataclass(frozen=True)
class PHConfig:
    weight: float = 0.05
    window_size: int = 32
    dimensions: tuple[int, ...] = (0,)
    min_persistence: float = 0.01
    max_pairs: int = 64
    every_n_steps: int = 1
    warmup_epochs: int = 0
    ignore_policy: str = "skip_window"
    boundary_policy: str = "closed_window"


@dataclasses.dataclass(frozen=True)
class LossConfig:
    dice_weight: float = 1.0
    focal_weight: float = 1.0
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    ph: PHConfig = PHConfig()


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 0.0003
    weight_decay: float = 0.01


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 20
    effective_batch: int = 8
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: str = "cosine"
    grad_clip: float = 1.0


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    threshold: float = 0.5
    mode: str = "tile"
    overlap: int = 32
    topology_window: int = 64
    prediction_samples: int = 4


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    workers: int = 0
    micro_batch: int = 2
    amp: bool = False
    amp_dtype: str = "bfloat16"
    threads: int = 4


@dataclasses.dataclass(frozen=True)
class OutputConfig:
    root: str = "outputs"


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    output: OutputConfig = OutputConfig()

    def to_dict(self) -> dict:
        return json.loads(json.dumps(dataclasses.asdict(self)))

    @property
    def config_hash(self) -> str:
        return stable_hash(self.to_dict())

    @property
    def experiment_hash(self) -> str:
        value = self.to_dict()
        value.pop("runtime")
        value.pop("output")
        for source in value["data"]["sources"]:
            source.pop("root")
            source.pop("manifest")
        return stable_hash(value)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if (isinstance(value, dict) and isinstance(result.get(key), dict))
            else copy.deepcopy(value)
        )
    return result


def set_value(data: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ValueError(f"Unknown configuration path: {dotted}")
        node = node[part]
    if parts[-1] not in node:
        raise ValueError(f"Unknown configuration path: {dotted}")
    node[parts[-1]] = value


def _convert(kind: Any, value: Any, path: str) -> Any:
    origin, args = get_origin(kind), get_args(kind)
    if origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        return _convert(next(t for t in args if t is not type(None)), value, path)
    if dataclasses.is_dataclass(kind):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping")
        hints = get_type_hints(kind)
        unknown = value.keys() - hints.keys()
        if unknown:
            raise ValueError(f"Unknown fields at {path}: {sorted(unknown)}")
        return kind(**{k: _convert(hints[k], v, f"{path}.{k}") for k, v in value.items()})
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path} must be a list")
        if args[-1] is not Ellipsis and len(value) != len(args):
            raise ValueError(f"{path} must have {len(args)} entries")
        return tuple(_convert(args[0], v, path) for v in value)
    if kind is float and type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return float(value)
    if type(value) is not kind:
        raise ValueError(f"{path} must be {kind.__name__}, got {value!r}")
    return value


def validate(cfg: ExperimentConfig) -> ExperimentConfig:
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    m, d, t, r, e, ph = cfg.model, cfg.data, cfg.training, cfg.runtime, cfg.evaluation, cfg.loss.ph
    require(bool(d.sources), "At least one source is required")
    require(len({s.name for s in d.sources}) == len(d.sources), "Duplicate source names")
    for source in d.sources:
        require(source.weight > 0, "Source weight must be positive")
        sets = [
            set(source.foreground_values),
            set(source.background_values),
            set(source.ignore_values),
        ]
        require(bool(sets[0]) and bool(sets[1]), "Foreground/background mappings cannot be empty")
        require(
            not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]),
            "Mask value mappings overlap",
        )
        require(all(0 <= v <= 255 for values in sets for v in values), "Mask values must be uint8")
    require(d.train_mode in ("crop", "resize"), "train_mode must be crop or resize")
    require(min(d.size) >= 8, "Input size must be at least 8")
    require(len(d.mean) == len(d.std) == 3 and min(d.std) > 0, "Invalid RGB normalization")
    require(m.name in ("hybrid", "cnn_only"), "Unknown model")
    require(m.cnn in ("standard", "efficient"), "Unknown CNN encoder")
    require(
        len(m.channels) >= 2 and min(m.channels) >= 4, "CNN requires >=2 levels with >=4 channels"
    )
    require(1 <= m.feature_level < len(m.channels), "Invalid feature level")
    require(
        m.heads > 0 and m.embed_dim > 0 and m.embed_dim % m.heads == 0,
        "Embedding must divide heads",
    )
    require(m.depth >= 2 and m.window_size >= 2, "Swin requires >=2 blocks and window >=2")
    require(not m.patch_merging, "v1 uses a single spatial Swin stage; patch_merging must be false")
    require(m.decoder_channels >= 4 and 0 <= m.dropout < 1, "Invalid decoder/dropout")
    require(
        t.epochs > 0 and t.effective_batch > 0 and t.seed >= 0, "Invalid training configuration"
    )
    require(
        r.micro_batch > 0 and t.effective_batch % r.micro_batch == 0,
        "effective_batch must divide micro_batch",
    )
    require(r.workers >= 0 and r.threads > 0, "Invalid workers/threads")
    require(r.device == "cpu" or r.device.startswith("cuda"), "Device must be cpu or cuda[:index]")
    require(not r.amp or r.device.startswith("cuda"), "AMP is supported on CUDA only")
    require(r.amp_dtype in ("float16", "bfloat16"), "AMP dtype must be float16 or bfloat16")
    require(t.scheduler in ("cosine", "constant"), "Unknown scheduler")
    require(
        t.optimizer.lr > 0 and t.optimizer.weight_decay >= 0 and t.grad_clip > 0,
        "Invalid optimizer",
    )
    require(
        cfg.loss.dice_weight >= 0 and cfg.loss.focal_weight >= 0 and ph.weight >= 0,
        "Negative loss weight",
    )
    require(
        cfg.loss.dice_weight + cfg.loss.focal_weight > 0, "A pixel supervision loss is required"
    )
    require(0 <= cfg.loss.focal_alpha <= 1 and cfg.loss.focal_gamma >= 0, "Invalid focal settings")
    require(
        ph.window_size >= 2 and ph.max_pairs > 0 and ph.every_n_steps > 0 and ph.warmup_epochs >= 0,
        "Invalid PH settings",
    )
    require(
        bool(ph.dimensions)
        and set(ph.dimensions) <= {0, 1}
        and len(set(ph.dimensions)) == len(ph.dimensions),
        "PH dimensions must contain unique 0 and/or 1",
    )
    require(0 <= ph.min_persistence < 1, "Invalid persistence threshold")
    require(
        ph.ignore_policy == "skip_window" and ph.boundary_policy == "closed_window",
        "Unsupported PH domain policy",
    )
    require(
        e.mode in ("tile", "resize") and 0 <= e.threshold <= 1, "Invalid evaluation mode/threshold"
    )
    require(
        0 <= e.overlap < min(d.size) and e.topology_window >= 2 and e.prediction_samples >= 0,
        "Invalid evaluation tiling/topology settings",
    )
    return cfg


class ConfigLoader:
    @staticmethod
    def from_dict(value: dict) -> ExperimentConfig:
        return validate(_convert(ExperimentConfig, value, "config"))

    @classmethod
    def _read(cls, path: Path, seen: tuple[Path, ...] = ()) -> dict:
        path = path.resolve()
        if path in seen:
            raise ValueError(f"Cyclic base configuration: {path}")
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Configuration must be a mapping: {path}")
        parent = value.pop("base", None)
        return merge(cls._read(path.parent / parent, (*seen, path)), value) if parent else value

    @staticmethod
    def machine(value: dict, path: str | Path) -> dict:
        machine = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(machine, dict) or machine.keys() - {"runtime", "output", "roots"}:
            raise ValueError("Machine config allows only runtime, output, roots")
        if not isinstance(machine.get("output", {}), dict) or set(machine.get("output", {})) - {
            "root"
        }:
            raise ValueError("Machine output allows only root")
        result = merge(value, {k: v for k, v in machine.items() if k != "roots"})
        roots = machine.get("roots", {})
        names = {s["name"] for s in result["data"]["sources"]}
        if not isinstance(roots, dict) or roots.keys() - names:
            raise ValueError("Machine roots reference unknown source names")
        for source in result["data"]["sources"]:
            if source["name"] in roots:
                source["root"] = roots[source["name"]]
        return result

    @classmethod
    def load(
        cls, path: str | Path, machine: str | Path | None = None, overrides: list[str] | None = None
    ) -> ExperimentConfig:
        value = merge(ExperimentConfig().to_dict(), cls._read(Path(path)))
        if machine:
            value = cls.machine(value, machine)
        for entry in overrides or []:
            if "=" not in entry:
                raise ValueError("Overrides require dotted.key=value")
            key, raw = entry.split("=", 1)
            set_value(value, key, yaml.safe_load(raw))
        return cls.from_dict(value)
