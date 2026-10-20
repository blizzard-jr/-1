from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from src.sobel import sobel_library, sobel_native


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sobel edge detector")
    parser.add_argument("input", type=Path, help="input image")
    parser.add_argument("output", type=Path, help="output PNG/JPEG image")
    parser.add_argument(
        "--method",
        choices=("native", "library"),
        default="library",
        help="implementation to run (default: library)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = np.asarray(Image.open(args.input).convert("L"))#получаем матрицу яркостей
    algorithm = sobel_native if args.method == "native" else sobel_library

    started = perf_counter()
    result = algorithm(image)
    elapsed_ms = (perf_counter() - started) * 1000

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(args.output)
    print(f"method={args.method}; size={image.shape[1]}x{image.shape[0]}; time={elapsed_ms:.3f} ms")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

