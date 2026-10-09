from pathlib import Path

import cv2
import numpy as np
import pytest

from behavioral_cloning.preprocessing import load_image, preprocess_image


def test_preprocessing():
    image_bgr = np.full(
        (160, 320, 3),
        (0, 128, 255),
        dtype=np.uint8,
    )

    expected_result = np.full(
        (160, 320, 3),
        (0.5, 0.0019607843, -0.5),
        dtype=np.float32,
    )

    prepared = preprocess_image(image_bgr)

    assert prepared.shape == (160, 320, 3)
    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared, expected_result, rtol=0, atol=1e-7)


def test_wrong_image_size():
    image_bgr = np.full(
        (160, 240, 3),
        (0, 128, 255),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="Expected image shape"):
        preprocess_image(image_bgr)


def test_wrong_image_type():
    image_bgr = np.full(
        (160, 320, 3),
        (0, 128, 255),
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="Expected image of type"):
        preprocess_image(image_bgr)


def test_load_image(tmp_path: Path):
    image_path = tmp_path / "sample.png"

    image_bgr = np.full(
        (160, 320, 3),
        (0, 128, 255),
        dtype=np.uint8,
    )

    assert cv2.imwrite(str(image_path), image_bgr)

    prepared = load_image(image_path)

    expected_result = np.full(
        (160, 320, 3),
        (0.5, 0.0019607843, -0.5),
        dtype=np.float32,
    )

    assert prepared.shape == (160, 320, 3)
    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared, expected_result, rtol=0, atol=1e-7)


def test_load_image_missing_file(tmp_path: Path):
    image_path = tmp_path / "missing.png"

    with pytest.raises(ValueError, match="Could not read image"):
        load_image(image_path)
