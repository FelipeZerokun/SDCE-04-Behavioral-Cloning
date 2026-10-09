"""Evaluate a saved model on an ordered manifest without augmentation."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from behavioral_cloning.batches import DrivingDataset
from behavioral_cloning.model import keras
from behavioral_cloning.preprocessing import load_image
from behavioral_cloning.training_data import TrainingExample, load_manifest

__all__ = ["evaluate_model", "plt"]


def evaluate_model(
    model: Any,
    examples: list[TrainingExample],
    output_dir: Path,
    batch_size: int = 32,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DrivingDataset(examples, batch_size)
    predictions = np.concatenate(
        [
            np.asarray(model.predict_on_batch(dataset[i][0])).reshape(-1)
            for i in range(len(dataset))
        ]
    ).astype(np.float64)
    targets = np.array([e.steering for e in examples], dtype=np.float64)
    if predictions.shape != targets.shape or not np.isfinite(predictions).all():
        raise ValueError("Model predictions must be finite, one per example")

    def metrics(mask: Any) -> dict[str, Any]:
        truth, predicted = targets[mask], predictions[mask]
        if not len(truth):
            return {"count": 0, "mse": None, "mae": None, "zero_mse": None}
        return {
            "count": len(truth),
            "mse": float(np.mean((predicted - truth) ** 2)),
            "mae": float(np.mean(np.abs(predicted - truth))),
            "zero_mse": float(np.mean(truth**2)),
        }

    report = {
        "overall": metrics(np.ones(len(targets), dtype=bool)),
        "by_session": {
            session: metrics(np.array([e.session == session for e in examples]))
            for session in sorted({e.session for e in examples})
        },
        "by_steering": {
            "left": metrics(targets < -0.05),
            "near_straight": metrics(np.abs(targets) <= 0.05),
            "right": metrics(targets > 0.05),
        },
        "near_straight_threshold": 0.05,
        "prediction_clipping": False,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["session", "record", "image_path", "target", "prediction"])
        for example, prediction in zip(examples, predictions, strict=True):
            writer.writerow(
                [
                    example.session,
                    example.record,
                    example.image_path,
                    example.steering,
                    float(prediction),
                ]
            )
    fig, ax = plt.subplots()
    ax.scatter(targets, predictions, s=5, alpha=0.35)
    low = float(min(targets.min(), predictions.min()))
    high = float(max(targets.max(), predictions.max()))
    ax.plot([low, high], [low, high], "k--")
    ax.set(xlabel="Recorded steering", ylabel="Predicted steering")
    fig.savefig(output_dir / "predictions.png", bbox_inches="tight")
    plt.close(fig)
    worst = np.argsort(np.abs(predictions - targets))[-6:][::-1]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, worst, strict=False):
        ax.imshow(np.clip(load_image(examples[i].image_path) + 0.5, 0, 1))
        ax.set_title(
            f"{examples[i].session}:{examples[i].record}\n"
            f"true {targets[i]:.3f}, predicted {predictions[i]:.3f}",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "largest_errors.png")
    plt.close(fig)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/dataset/validation.csv")
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    examples = load_manifest(args.manifest, args.raw_dir)
    model = keras.models.load_model(args.model, compile=False)
    report = evaluate_model(model, examples, args.output_dir, args.batch_size)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
