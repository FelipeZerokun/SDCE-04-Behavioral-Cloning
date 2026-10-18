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
from behavioral_cloning.provenance import file_hash, save_provenance
from behavioral_cloning.training_data import (
    check_split_overlap,
    expand_cameras,
    load_manifest,
)


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
        "--architecture", choices=("baseline", "lenet", "nvidia"),
        default="baseline",
    )
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--resize", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument(
        "--flip", action="store_true", help="Add one mirrored training view per image"
    )
    parser.add_argument("--cameras", choices=("center", "all"), default="center")
    parser.add_argument("--camera-correction", type=float)
    parser.add_argument(
        "--camera-target-policy", choices=("reject", "clip"), default="reject"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Audit and save configuration without fitting",
    )
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
    if args.cameras == "all" and args.camera_correction is None:
        parser.error("--cameras all requires an explicit --camera-correction")
    if args.cameras == "center" and args.camera_correction is not None:
        parser.error("--camera-correction requires --cameras all")
    if args.cameras == "center" and args.camera_target_policy != "reject":
        parser.error("--camera-target-policy requires --cameras all")
    if args.camera_correction is not None and not (
        np.isfinite(args.camera_correction) and 0 < args.camera_correction <= 1
    ):
        parser.error("camera-correction must be finite and in (0, 1]")
    keras.utils.set_random_seed(args.seed)
    model = build_model(
        args.learning_rate,
        architecture=args.architecture,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        resize=tuple(args.resize) if args.resize else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_dir = args.output_dir / "dataset"
    if not prepare_dataset(args.raw_dir, args.config, manifest_dir):
        raise SystemExit("Dataset audit failed; see run dataset/report.json")
    train = load_manifest(manifest_dir / "train.csv", args.raw_dir)
    validation = load_manifest(manifest_dir / "validation.csv", args.raw_dir)
    check_split_overlap(train, validation)
    source_training_count = len(train)
    clipped_targets = 0
    if args.cameras == "all":
        clipped_targets = (
            sum(
                not -1 <= e.steering + offset <= 1
                for e in train
                for offset in (args.camera_correction, -args.camera_correction)
            )
            if args.camera_target_policy == "clip"
            else 0
        )
        try:
            train = expand_cameras(
                train, args.raw_dir, args.camera_correction, args.camera_target_policy
            )
            check_split_overlap(train, validation)
        except (ValueError, OSError) as exc:
            (manifest_dir / "camera-audit.json").write_text(
                json.dumps({"passed": False, "error": str(exc)}, indent=2) + "\n",
                encoding="utf-8",
            )
            raise
        (manifest_dir / "camera-audit.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "audited_views": len(train),
                    "correction": args.camera_correction,
                    "target_policy": args.camera_target_policy,
                    "clipped_targets": clipped_targets,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        train = [train[int(i)] for i in rng.permutation(len(train))[:32]]
        validation = [validation[int(i)] for i in rng.permutation(len(validation))[:16]]
    training = DrivingDataset(
        train, args.batch_size, shuffle=True, seed=args.seed, flip=args.flip
    )
    held_out = DrivingDataset(validation, args.batch_size)
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "effective_epochs": 1 if args.smoke else args.epochs,
        "training_count": len(train),
        "source_training_count": source_training_count,
        "effective_training_views": len(training.indices),
        "steps_per_epoch": len(training),
        "parameter_count": model.count_params(),
        "training_augmentation": "original plus horizontal mirror with negated steering"
        if args.flip
        else None,
        "clipped_camera_targets_before_smoke_subset": clipped_targets,
        "camera_sign_convention": "left + correction; right - correction",
        "validation_count": len(validation),
        "keras_version": keras.__version__,
        "backend": keras.backend.backend(),
        "python_version": platform.python_version(),
        "preprocessing": {
            "shape": [160, 320, 3],
            "color": "RGB",
            "dtype": "float32",
            "normalization": "pixel / 255.0 - 0.5",
            "crop": [args.crop_top, args.crop_bottom]
            if args.crop_top or args.crop_bottom
            else None,
            "resize": args.resize,
            "augmentation": None,
            "location": "normalization outside model; crop/resize inside saved model",
            "interpolation": "bilinear" if args.resize else None,
        },
        "selected_training": [[e.session, e.record] for e in train],
        "selected_validation": [[e.session, e.record] for e in validation],
    }
    save_provenance(args.output_dir, args.config, train, validation)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    model.summary()
    (args.output_dir / "architecture.json").write_text(
        model.to_json() + "\n", encoding="utf-8"
    )
    if args.prepare_only:
        print(
            f"Prepared {args.output_dir}; no training or quality evaluation performed"
        )
        return
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
    (args.output_dir / "model-hashes.json").write_text(
        json.dumps(
            {
                name: file_hash(args.output_dir / name)
                for name in ("model.keras", "model.h5")
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
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
