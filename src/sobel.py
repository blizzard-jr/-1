from __future__ import annotations

import numpy as np
from scipy import ndimage

#ядро фильтра SOBEL_X сравнивает правую и левую части окна, поэтому реагирует на изменение яркости вдоль оси x и выделяет вертикальные
# границы. SOBEL_Y сравнивает нижнюю и верхнюю части и выделяет горизонтальные границы.
SOBEL_X = np.array(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    dtype=np.float64,
)
SOBEL_Y = np.array(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
    dtype=np.float64,
)


def _validate(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("Sobel expects a two-dimensional grayscale image")
    if array.size == 0:
        raise ValueError("The image must not be empty")
    return array.astype(np.float64, copy=False)


def normalize_magnitude(magnitude: np.ndarray) -> np.ndarray: #приводим все к нашеим привычным 255
    maximum = float(np.max(magnitude))
    if maximum == 0.0:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    return np.clip(magnitude * (255.0 / maximum), 0, 255).astype(np.uint8)


def sobel_native(image: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    source = _validate(image)
    padded = np.pad(source, 1, mode="constant")
    height, width = source.shape
    magnitude = np.empty((height, width), dtype=np.float64)

    for row in range(height):
        for column in range(width):
            window = padded[row : row + 3, column : column + 3]
            gx = 0.0
            gy = 0.0
            for kernel_row in range(3):
                for kernel_column in range(3):
                    pixel = float(window[kernel_row, kernel_column])
                    gx += pixel * float(SOBEL_X[kernel_row, kernel_column]) #изменение слева направо
                    gy += pixel * float(SOBEL_Y[kernel_row, kernel_column]) # сверху вниз
            magnitude[row, column] = (gx * gx + gy * gy) ** 0.5

    return normalize_magnitude(magnitude) if normalize else magnitude


def sobel_library(image: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    source = _validate(image)
    gx = ndimage.correlate(source, SOBEL_X, mode="constant", cval=0.0) #mode="constant", cval=0.0 - все символы за
    # границей считаются 0 как и в нативном методе
    gy = ndimage.correlate(source, SOBEL_Y, mode="constant", cval=0.0)
    magnitude = np.hypot(gx, gy) #поэлементно вычисляет резкость измнения в каждом пикселе через корень суммы квадратов
    return normalize_magnitude(magnitude) if normalize else magnitude

