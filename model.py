"""Train the group's crack segmentation model and export the Studio 7 results."""

import argparse
from dataclasses import replace
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from structural_defect_detection.config import ConfigLoader
from structural_defect_detection.experiments.runner import ExperimentRunner

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs/experiments/studio7_smoke.yaml"


def write_results(run_dir: Path, result: dict, config_path: Path) -> None:
    history = result["history"]
    epochs = [entry["epoch"] + 1 for entry in history]
    losses = [entry["training"]["total"] for entry in history]
    dice = [entry["validation_dice"] for entry in history]
    validation = result["best_validation"]

    figure = Figure(figsize=(8, 3.4), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle(f"Crack segmentation · {config_path.stem}")
    loss_ax, dice_ax = figure.subplots(1, 2)
    loss_ax.plot(epochs, losses, marker="o", color="tab:blue")
    loss_ax.set(ylabel="Training loss", xlabel="Epoch")
    dice_ax.plot(epochs, dice, marker="o", color="tab:green")
    dice_ax.set(ylabel="Validation Dice", xlabel="Epoch", ylim=(0, 1))
    for axis in (loss_ax, dice_ax):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.25)
    figure.savefig(run_dir / "model_results.png", dpi=150)

    metrics = (
        f"Configuration: {config_path.name}\n"
        f"Seed: {result['seed']}\n"
        f"Epochs completed: {len(epochs)}\n"
        f"Optimizer steps: {result['optimizer_steps']}\n"
        f"Final training loss: {losses[-1]:.6f}\n"
        f"Best validation Dice: {validation['selection']['crack_dice']:.6f}\n"
        f"Best checkpoint validation IoU: {validation['overall']['iou']:.6f}\n"
        f"Validation samples: {validation['overall']['valid_samples']}\n"
    )
    (run_dir / "metrics.txt").write_text(metrics, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--machine", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    config = ConfigLoader.load(args.config, machine=args.machine)
    if args.output_root is not None:
        config = replace(config, output=replace(config.output, root=str(args.output_root)))
    result = ExperimentRunner(config).run(run_id=args.run_id)
    run_dir = Path(config.output.root) / result["run_id"]
    write_results(run_dir, result, args.config)
    print(f"Studio 7 results saved to {run_dir}")


if __name__ == "__main__":
    main()
