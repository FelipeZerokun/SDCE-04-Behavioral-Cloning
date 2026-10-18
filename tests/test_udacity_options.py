import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from behavioral_cloning.batches import DrivingDataset
from behavioral_cloning.model import build_model, keras
from behavioral_cloning.preprocessing import load_image, preprocess_rgb
from behavioral_cloning.training_data import (
    TrainingExample,
    check_split_overlap,
    expand_cameras,
)


def camera_record(root: Path, steering=0.1):
    session = root / "session"
    (session / "IMG").mkdir(parents=True)
    paths = []
    for i, camera in enumerate(("center", "left", "right")):
        path = session / "IMG" / f"{camera}.png"
        pixels = np.zeros((160, 320, 3), np.uint8)
        pixels[:, :100] = (i * 30, 70, 200)
        assert cv2.imwrite(str(path), pixels)
        paths.append(path)
    with (session / "driving_log.csv").open("w", newline="") as stream:
        csv.writer(stream).writerow([*map(str, paths), steering, 0, 0, 9])
    example = TrainingExample(
        paths[0],
        steering,
        "session",
        1,
        hashlib.sha256(paths[0].read_bytes()).hexdigest(),
    )
    return example, paths


def test_flip_pairing_coverage_and_validation(tmp_path):
    example, _ = camera_record(tmp_path)
    dataset = DrivingDataset([example], batch_size=1, flip=True)
    original, label = dataset[0]
    mirrored, mirrored_label = dataset[1]
    np.testing.assert_array_equal(mirrored, original[:, :, ::-1, :])
    np.testing.assert_array_equal(mirrored_label, -label)
    assert len(DrivingDataset([example], batch_size=1)) == 1
    shuffled = DrivingDataset([example], 1, shuffle=True, seed=7, flip=True)
    assert sorted(shuffled.indices.tolist()) == [0, 1]
    # Repeated indexed access does not draw a different augmentation.
    np.testing.assert_array_equal(shuffled[0][0], shuffled[0][0])


def test_camera_signs_and_split_identity(tmp_path):
    example, paths = camera_record(tmp_path)
    expanded = expand_cameras([example], tmp_path, 0.2)
    assert [e.camera for e in expanded] == ["center", "left", "right"]
    assert [e.steering for e in expanded] == pytest.approx([0.1, 0.3, -0.1])
    assert {e.session for e in expanded} == {example.session}
    assert [e.image_path for e in expanded] == paths
    # Side image bytes overlapping a validation example are rejected too.
    held_out = [
        TrainingExample(tmp_path / "elsewhere.png", 0, "other", 1, expanded[1].sha256)
    ]
    with pytest.raises(ValueError, match="sha256"):
        check_split_overlap(expanded, held_out)


def test_camera_invalid_shape_and_target(tmp_path):
    example, paths = camera_record(tmp_path, steering=0.9)
    with pytest.raises(ValueError, match="outside"):
        expand_cameras([example], tmp_path, 0.2)
    clipped = expand_cameras([example], tmp_path, 0.2, target_policy="clip")
    assert [e.steering for e in clipped] == pytest.approx([0.9, 1.0, 0.7])
    assert cv2.imwrite(str(paths[1]), np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError, match="shape"):
        expand_cameras([example], tmp_path, 0.05)
    with pytest.raises(ValueError, match="correction"):
        expand_cameras([example], tmp_path, float("nan"))


def test_camera_missing_and_changed_source(tmp_path):
    example, paths = camera_record(tmp_path)
    paths[2].unlink()
    with pytest.raises(ValueError, match="Could not read"):
        expand_cameras([example], tmp_path, 0.2)
    paths[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since audit"):
        expand_cameras([example], tmp_path, 0.2)


def test_rgb_training_agreement(tmp_path):
    example, _ = camera_record(tmp_path)
    bgr = cv2.imread(str(example.image_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(preprocess_rgb(rgb), load_image(example.image_path))
    with pytest.raises(ValueError, match="uint8"):
        preprocess_rgb(preprocess_rgb(rgb))  # Prevent double normalization.


def test_crop_boundaries_and_nvidia_feature_shapes():
    model = build_model(crop_top=50, crop_bottom=20)
    x = np.broadcast_to(
        np.arange(160, dtype=np.float32)[None, :, None, None], (1, 160, 320, 3)
    ).copy()
    np.testing.assert_array_equal(np.asarray(model.layers[0](x)), x[:, 50:140])
    nvidia = build_model(
        architecture="nvidia", crop_top=50, crop_bottom=20, resize=(66, 200)
    )
    conv = [layer for layer in nvidia.layers if isinstance(layer, keras.layers.Conv2D)]
    assert [tuple(layer.output.shape[1:]) for layer in conv] == [
        (31, 98, 24),
        (14, 47, 36),
        (5, 22, 48),
        (3, 20, 64),
        (1, 18, 64),
    ]
    assert nvidia.input_shape == (None, 160, 320, 3)
    for options in (
        {"crop_top": 160},
        {"crop_top": -1},
        {"architecture": "nvidia"},
        {"resize": (5, 5)},
    ):
        with pytest.raises(ValueError):
            build_model(**options)


@pytest.mark.parametrize(
    "architecture,resize", [("baseline", None), ("nvidia", (66, 200))]
)
def test_geometry_export_reload_without_training(tmp_path, architecture, resize):
    model = build_model(
        architecture=architecture, crop_top=50, crop_bottom=20, resize=resize
    )
    x = np.random.default_rng(1).uniform(-0.5, 0.5, (1, 160, 320, 3)).astype(np.float32)
    expected = np.asarray(model(x, training=False))
    assert expected.shape == (1, 1) and np.isfinite(expected).all()
    for suffix in ("keras", "h5"):
        path = tmp_path / f"model.{suffix}"
        model.save(path)
        restored = keras.models.load_model(path, compile=False)
        np.testing.assert_allclose(
            np.asarray(restored(x, training=False)), expected, rtol=1e-5, atol=1e-6
        )


def test_baseline_default_architecture():
    baseline = build_model()
    assert baseline.count_params() == 475145
    assert not any(
        isinstance(layer, (keras.layers.Cropping2D, keras.layers.Resizing))
        for layer in baseline.layers
    )


def test_prepare_only_records_full_configuration_without_fit(tmp_path, monkeypatch):
    from behavioral_cloning import train

    example, _ = camera_record(tmp_path / "raw")
    validation_path = tmp_path / "raw" / "heldout.png"
    assert cv2.imwrite(str(validation_path), np.full((160, 320, 3), 55, np.uint8))
    validation = TrainingExample(
        validation_path,
        0,
        "heldout",
        1,
        hashlib.sha256(validation_path.read_bytes()).hexdigest(),
    )
    configuration = tmp_path / "dataset.json"
    configuration.write_text('{"train": ["session"], "validation": ["heldout"]}')

    def prepare(raw, config, destination):
        destination.mkdir()
        for name in ("train", "validation"):
            (destination / f"{name}.csv").write_text("test manifest")
        return True

    monkeypatch.setattr(train, "prepare_dataset", prepare)
    monkeypatch.setattr(
        train,
        "load_manifest",
        lambda path, raw: [example] if path.stem == "train" else [validation],
    )

    def forbidden_fit(*args, **kwargs):
        pytest.fail("prepare-only must never call fit")

    monkeypatch.setattr(keras.Sequential, "fit", forbidden_fit)
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--prepare-only",
            "--flip",
            "--cameras",
            "all",
            "--camera-correction",
            "0.2",
            "--architecture",
            "nvidia",
            "--resize",
            "66",
            "200",
            "--crop-top",
            "50",
            "--crop-bottom",
            "20",
            "--raw-dir",
            str(tmp_path / "raw"),
            "--config",
            str(configuration),
            "--output-dir",
            str(output),
        ],
    )
    train.main()
    saved = json.loads((output / "config.json").read_text())
    assert saved["training_count"] == 3
    assert saved["effective_training_views"] == 6
    assert saved["validation_count"] == 1
    records = json.loads((output / "dataset/validation-effective.json").read_text())
    assert len(records) == 1 and records[0]["camera"] == "center"
    assert (output / "source/src/behavioral_cloning/model.py").exists()
    assert not (output / "model.keras").exists()
    with pytest.raises(FileExistsError):
        train.main()
