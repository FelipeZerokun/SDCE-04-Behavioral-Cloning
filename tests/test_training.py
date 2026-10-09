from pathlib import Path

import cv2
import numpy as np
import pytest

from behavioral_cloning.batches import DrivingDataset
from behavioral_cloning.evaluate import evaluate_model
from behavioral_cloning.model import build_model, keras
from behavioral_cloning.training_data import (
    TrainingExample,
    check_split_overlap,
    load_manifest,
)


def test_manifest(tmp_path: Path):
    manifest = tmp_path / "train.csv"
    manifest.write_text(
        "session,record,center,steering,sha256\na,1,a/IMG/x.png,-0.2,abc\n",
        encoding="utf-8",
    )
    examples = load_manifest(manifest, tmp_path)
    assert examples == [
        TrainingExample(
            tmp_path / "a/IMG/x.png",
            -0.2,
            "a",
            1,
            "abc",
        )
    ]
    with pytest.raises(ValueError, match="overlap"):
        check_split_overlap(examples, examples)


@pytest.mark.parametrize("center,steering", [("../x.png", "0"), ("x.png", "nan")])
def test_invalid_manifest(tmp_path: Path, center: str, steering: str):
    manifest = tmp_path / "invalid.csv"
    manifest.write_text(
        f"session,record,center,steering,sha256\na,1,{center},{steering},abc\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_manifest(manifest, tmp_path)


def test_batch_alignment_and_partial_batch(tmp_path: Path):
    examples = []
    for i, value in enumerate((0, 64, 255)):
        path = tmp_path / f"{i}.png"
        assert cv2.imwrite(str(path), np.full((160, 320, 3), value, np.uint8))
        examples.append(TrainingExample(path, value / 255 - 0.5))
    dataset = DrivingDataset(examples, 2, shuffle=True, seed=7)
    other = DrivingDataset(examples, 2, shuffle=True, seed=7)
    np.testing.assert_array_equal(dataset.indices, other.indices)
    assert len(dataset) == 2
    assert dataset[1][0].shape[0] == 1
    for i in range(len(dataset)):
        x, y = dataset[i]
        np.testing.assert_allclose(x[:, 0, 0, 0], y[:, 0], atol=1e-7)
    ordered = DrivingDataset(examples, 2)
    ordered.on_epoch_end()
    np.testing.assert_array_equal(ordered.indices, [0, 1, 2])


def test_model_learning_and_export(tmp_path: Path):
    keras.utils.set_random_seed(42)
    model = build_model()
    x = np.zeros((2, 160, 320, 3), np.float32)
    y = np.full((2, 1), 0.1, np.float32)
    before = float(np.mean((model.predict_on_batch(x) - y) ** 2))
    for _ in range(4):
        model.train_on_batch(x, y)
    prediction = model.predict_on_batch(x)
    assert prediction.shape == (2, 1)
    assert float(np.mean((prediction - y) ** 2)) < before
    for suffix in ("keras", "h5"):
        path = tmp_path / f"model.{suffix}"
        model.save(path)
        restored = keras.models.load_model(path, compile=False)
        np.testing.assert_allclose(restored.predict_on_batch(x), prediction, atol=1e-6)


def test_evaluation_metrics(tmp_path: Path):
    class ZeroModel:
        def predict_on_batch(self, x):
            return np.zeros((len(x), 1), np.float32)

    path = tmp_path / "image.png"
    assert cv2.imwrite(str(path), np.zeros((160, 320, 3), np.uint8))
    examples = [
        TrainingExample(path, v, "lap", i) for i, v in enumerate((-0.2, 0, 0.2))
    ]
    report = evaluate_model(ZeroModel(), examples, tmp_path / "evaluation", 2)
    assert report["overall"]["mse"] == pytest.approx(0.08 / 3)
    assert report["overall"]["mse"] == report["overall"]["zero_mse"]
    assert report["by_steering"]["near_straight"]["count"] == 1
