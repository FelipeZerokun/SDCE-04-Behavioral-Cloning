"""Load only the current batch of images, retaining image/label alignment."""

import math

import numpy as np
from numpy.typing import NDArray

from behavioral_cloning.model import keras
from behavioral_cloning.preprocessing import load_image
from behavioral_cloning.training_data import TrainingExample


class DrivingDataset(keras.utils.PyDataset):  # type: ignore[misc]
    def __init__(
        self,
        examples: list[TrainingExample],
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
        if not examples or batch_size < 1:
            raise ValueError("Examples must be nonempty and batch_size positive")
        self.examples = examples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(examples))
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.examples) / self.batch_size)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        samples = [self.examples[int(i)] for i in indices]
        images = np.stack([load_image(e.image_path) for e in samples])
        labels = np.array([e.steering for e in samples], dtype=np.float32)[:, None]
        return images, labels

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)
