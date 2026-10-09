"""Audit, train a center-image baseline, export, and evaluate the best epoch."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from behavioral_cloning.batches import DrivingDataset
from behavioral_cloning.data_validation import prepare_dataset
from behavioral_cloning.evaluate import evaluate_model, plt
from behavioral_cloning.model import build_model, keras
from behavioral_cloning.training_data import check_split_overlap, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke", action="store_true", help="One epoch on small subsets"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        parser.error("epochs/batch-size must be positive; patience nonnegative")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if args.output_dir.resolve().is_relative_to(args.raw_dir.resolve()):
        parser.error("output-dir must be outside raw recordings")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_dir = args.output_dir / "dataset"
    if not prepare_dataset(args.raw_dir, args.config, manifest_dir):
        raise SystemExit("Dataset audit failed; see run dataset/report.json")
    train = load_manifest(manifest_dir / "train.csv", args.raw_dir)
    validation = load_manifest(manifest_dir / "validation.csv", args.raw_dir)
    check_split_overlap(train, validation)
    keras.utils.set_random_seed(args.seed)
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        train = [train[int(i)] for i in rng.permutation(len(train))[:32]]
        validation = [validation[int(i)] for i in rng.permutation(len(validation))[:16]]
    training = DrivingDataset(train, args.batch_size, shuffle=True, seed=args.seed)
    held_out = DrivingDataset(validation, args.batch_size)
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "effective_epochs": 1 if args.smoke else args.epochs,
        "training_count": len(train),
        "validation_count": len(validation),
        "keras_version": keras.__version__,
        "backend": keras.backend.backend(),
        "python_version": platform.python_version(),
        "preprocessing": {
            "shape": [160, 320, 3],
            "color": "RGB",
            "dtype": "float32",
            "normalization": "pixel / 255.0 - 0.5",
            "crop": None,
            "resize": None,
            "augmentation": None,
            "location": "outside model; behavioral_cloning.preprocessing",
        },
        "selected_training": [[e.session, e.record] for e in train],
        "selected_validation": [[e.session, e.record] for e in validation],
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    model = build_model(args.learning_rate)
    model.summary()
    checkpoint = args.output_dir / "model.keras"
    history = model.fit(
        training,
        validation_data=held_out,
        epochs=config["effective_epochs"],
        shuffle=False,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                checkpoint, save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.CSVLogger(args.output_dir / "history.csv"),
        ],
        verbose=2,
    )
    if not all(np.isfinite(v).all() for v in history.history.values()):
        raise RuntimeError("Nonfinite training metrics; inspect history.csv")
    model = keras.models.load_model(checkpoint, compile=False)
    fixed_images = held_out[0][0]
    expected = model.predict_on_batch(fixed_images)
    model.save(args.output_dir / "model.h5")
    for filename in ("model.keras", "model.h5"):
        reloaded = keras.models.load_model(args.output_dir / filename, compile=False)
        np.testing.assert_allclose(
            reloaded.predict_on_batch(fixed_images),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
    (args.output_dir / "export_verification.json").write_text(
        json.dumps(
            {
                "keras_reload": True,
                "h5_reload": True,
                "checked_examples": len(fixed_images),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots()
    for key in ("loss", "val_loss"):
        ax.plot(
            range(1, len(history.history[key]) + 1), history.history[key], label=key
        )
    ax.set(xlabel="Epoch", ylabel="MSE")
    ax.legend()
    fig.savefig(args.output_dir / "loss.png", bbox_inches="tight")
    plt.close(fig)
    report = evaluate_model(
        model, validation, args.output_dir / "validation", args.batch_size
    )
    print(json.dumps(report["overall"], indent=2))
    print(f"Saved run to {args.output_dir}; smoke={args.smoke}")


if __name__ == "__main__":
    main()
