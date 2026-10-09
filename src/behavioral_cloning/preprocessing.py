from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


def preprocess_image(
    image_bgr: NDArray[np.uint8],
) -> NDArray[np.float32]:

    if image_bgr.shape != (160, 320, 3):
        raise ValueError(f"Expected image shape (160, 320, 3), got {image_bgr.shape}")
    if image_bgr.dtype != np.uint8:
        raise ValueError("Expected image of type uint8")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_float = image_rgb.astype(np.float32)
    image_float = (image_float / 255.0) - 0.5

    return image_float


def load_image(image_path: Path) -> NDArray[np.float32]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    return preprocess_image(cast(NDArray[np.uint8], image_bgr))
