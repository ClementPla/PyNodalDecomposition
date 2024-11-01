from pathlib import Path
import numpy as np
from tifffile import imread
import cv2


def read_img(path: Path) -> np.ndarray:
    img = imread(path)
    img = img / img.max()
    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2RGB)
    return (img * 255).astype(np.uint8)


def resize_img(img: np.ndarray, size: int) -> np.ndarray:
    h, w, c = img.shape
    fx = size / w

    img = cv2.resize(img, dsize=None, fx=fx, fy=fx)
    return img
