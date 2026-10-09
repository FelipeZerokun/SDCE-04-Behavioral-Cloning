"""Small steering CNN; inputs have already been normalized outside the model."""

import os
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras as keras  # type: ignore[import-untyped]  # noqa: E402


def build_model(learning_rate: float = 1e-3) -> Any:
    model = keras.Sequential(
        [
            keras.Input(shape=(160, 320, 3), dtype="float32"),
            keras.layers.Conv2D(16, 5, strides=2, activation="relu"),
            keras.layers.Conv2D(24, 5, strides=2, activation="relu"),
            keras.layers.Conv2D(32, 3, strides=2, activation="relu"),
            keras.layers.Conv2D(48, 3, strides=2, activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ],
        name="steering_baseline",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model
