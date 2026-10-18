"""Small steering CNN; inputs have already been normalized outside the model."""

import os
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras as keras  # type: ignore[import-untyped]  # noqa: E402


def build_model(
    learning_rate: float = 1e-3,
    *,
    architecture: str = "baseline",
    crop_top: int = 0,
    crop_bottom: int = 0,
    resize: tuple[int, int] | None = None,
) -> Any:
    """Raw-size normalized RGB input; saved layers own all spatial transforms.

    nvidia follows the supplied Udacity diagram's explicit Dense(1164) variant,
    not the approximately 250k-parameter original-paper implementation.
    """
    if architecture not in {"baseline", "lenet", "nvidia"}:
        raise ValueError("Unknown architecture")
    if crop_top < 0 or crop_bottom < 0 or crop_top + crop_bottom >= 160:
        raise ValueError("Crop must retain positive image height")
    shape = resize or (160 - crop_top - crop_bottom, 320)
    if min(shape) < 1:
        raise ValueError("Resize dimensions must be positive")
    if architecture == "nvidia" and shape != (66, 200):
        raise ValueError("NVIDIA variant requires --resize 66 200")
    kernels = [(5, 2), (5, 2), (3, 2), (3, 2)]
    if architecture == "nvidia":
        kernels = [(5, 2)] * 3 + [(3, 1)] * 2
    elif architecture == "lenet":
        kernels = [(5, 1), (2, 2), (5, 1), (2, 2)]
    for kernel, stride in kernels:
        shape = ((shape[0] - kernel) // stride + 1, (shape[1] - kernel) // stride + 1)
        if min(shape) < 1:
            raise ValueError("Image is too small for convolution stack")
    layers = [keras.Input(shape=(160, 320, 3), dtype="float32")]
    if crop_top or crop_bottom:
        layers.append(keras.layers.Cropping2D(((crop_top, crop_bottom), (0, 0))))
    if resize:
        layers.append(keras.layers.Resizing(*resize, interpolation="bilinear"))
    if architecture == "nvidia":
        for filters, (kernel, stride) in zip(
            (24, 36, 48, 64, 64), kernels, strict=True
        ):
            layers.append(
                keras.layers.Conv2D(filters, kernel, strides=stride, activation="relu")
            )
        layers.append(keras.layers.Flatten())
        layers.extend(
            keras.layers.Dense(width, activation="relu")
            for width in (1164, 100, 50, 10)
        )
        layers.append(keras.layers.Dense(1))
    elif architecture == "lenet":
        layers.extend(
            [
                keras.layers.Conv2D(6, 5, activation="relu"),
                keras.layers.AveragePooling2D(pool_size=2, strides=2),
                keras.layers.Conv2D(16, 5, activation="relu"),
                keras.layers.AveragePooling2D(pool_size=2, strides=2),
                keras.layers.Flatten(),
                keras.layers.Dense(120, activation="relu"),
                keras.layers.Dense(84, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
    else:
        layers.extend(
            [
                keras.layers.Conv2D(16, 5, strides=2, activation="relu"),
                keras.layers.Conv2D(24, 5, strides=2, activation="relu"),
                keras.layers.Conv2D(32, 3, strides=2, activation="relu"),
                keras.layers.Conv2D(48, 3, strides=2, activation="relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
    model = keras.Sequential(
        layers,
        name=f"steering_{architecture}",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model
