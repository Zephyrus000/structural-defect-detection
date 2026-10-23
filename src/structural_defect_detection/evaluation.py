"""Metrics on original images, grouped before dataset/domain aggregation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import ExperimentConfig
from .data.manifest import load_arrays
from .inference import PredictionWriter, Predictor
from .utils import write_json


class SegmentationMetrics:
    @staticmethod
    def sample(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict:
        if not valid.any():
            return {
                "valid_pixels": 0,
                "dice": None,
                "iou": None,
                "precision": None,
                "recall": None,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            }
        p, t = prediction[valid], target[valid]
        tp, fp, fn, tn = (int(a.sum()) for a in (p & t, p & ~t, ~p & t, ~p & ~t))
        return {
            "valid_pixels": int(valid.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "precision": tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0),
            "recall": tp / (tp + fn) if tp + fn else (1.0 if not fp else 0.0),
        }


class TopologyMetrics:
    @staticmethod
    def sample(prediction, target, valid, window: int) -> dict:
        errors, skipped = [], 0
        for top in range(0, valid.shape[0], window):
            for left in range(0, valid.shape[1], window):
                region = np.s_[top : top + window, left : left + window]
                if min(valid[region].shape) < 2 or not valid[region].all():
                    skipped += 1
                    continue
                predicted = ndimage.label(prediction[region], structure=np.ones((3, 3)))[1]
                expected = ndimage.label(target[region], structure=np.ones((3, 3)))[1]
                errors.append(abs(predicted - expected))
        return {
            "window_betti0_mae": float(np.mean(errors)) if errors else None,
            "topology_windows": len(errors),
            "topology_skipped_windows": skipped,
        }


def summarize(rows: list[dict]) -> dict:
    result = {"samples": len(rows), "valid_samples": sum(r["valid_pixels"] > 0 for r in rows)}
    for key in ("dice", "iou", "precision", "recall", "window_betti0_mae"):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    for key in (
        "tp",
        "fp",
        "fn",
        "tn",
        "valid_pixels",
        "topology_windows",
        "topology_skipped_windows",
    ):
        result[key] = sum(r[key] for r in rows)
    return result


class Evaluator:
    def __init__(self, model, config: ExperimentConfig, device):
        self.config = config
        self.predictor = Predictor(model, config, device)

    def evaluate(
        self, records, split: str, output: Path | None = None, save_predictions: bool = False
    ) -> dict:
        groups, domains, samples = defaultdict(list), defaultdict(list), []
        writer = (
            PredictionWriter(output / "predictions", self.config.evaluation.threshold)
            if output and save_predictions
            else None
        )
        for source, row in records:
            if row["split"] != split:
                raise ValueError("Evaluator received a sample from the wrong split")
            image, target, valid = load_arrays(source, row)
            probability = self.predictor.predict(image)
            if not np.isfinite(probability).all():
                raise FloatingPointError("Non-finite prediction")
            prediction = probability >= self.config.evaluation.threshold
            metrics = SegmentationMetrics.sample(prediction, target, valid)
            metrics.update(
                TopologyMetrics.sample(
                    prediction, target, valid, self.config.evaluation.topology_window
                )
            )
            metrics.update(
                sample_id=row["sample_id"],
                dataset_id=source.name,
                domain=row.get("domain") or "unknown",
            )
            groups[source.name].append(metrics)
            domains[metrics["domain"]].append(metrics)
            samples.append(metrics)
            if writer and len(samples) <= self.config.evaluation.prediction_samples:
                writer.write(
                    image,
                    probability,
                    row
                    | {
                        "transform": {
                            "mode": self.config.evaluation.mode,
                            "input_size": list(self.config.data.size),
                            "overlap": self.config.evaluation.overlap,
                        }
                    },
                )
        datasets = {key: summarize(rows) for key, rows in groups.items()}
        scores = [value["dice"] for value in datasets.values() if value["dice"] is not None]
        result = {
            "split": split,
            "seed": self.config.training.seed,
            "overall": summarize(samples),
            "datasets": datasets,
            "domains": {k: summarize(v) for k, v in domains.items()},
            "samples": samples,
            "selection": {"crack_dice": float(np.mean(scores)) if scores else None},
            "policy": {
                "selection": "macro dataset mean of per-image Dice",
                "threshold": self.config.evaluation.threshold,
                "empty_both": 1.0,
                "all_ignore": "excluded",
                "topology": "8-connected H0 on fully valid closed windows",
                "topology_window": self.config.evaluation.topology_window,
            },
        }
        if output:
            write_json(output / f"{split}_metrics.json", result)
        return result
