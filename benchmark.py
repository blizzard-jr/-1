from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from src.sobel import sobel_library, sobel_native


def synthetic_image(size: int) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    gradient = 0.45 * x + 0.25 * y
    circle = ((x - size * 0.34) ** 2 + (y - size * 0.42) ** 2 < (size * 0.2) ** 2) * 100
    stripes = ((x // max(2, size // 24)) % 2) * 30
    return np.clip(gradient + circle + stripes, 0, 255).astype(np.uint8)


def measure(function, image: np.ndarray, repeats: int) -> float:
    function(image)
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        function(image)
        samples.append((perf_counter() - started) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.csv"))
    args = parser.parse_args()

    rows = []
    for size in args.sizes:
        image = synthetic_image(size)
        native_ms = measure(sobel_native, image, args.repeats)
        library_ms = measure(sobel_library, image, args.repeats)
        rows.append(
            {
                "size": f"{size}x{size}",
                "pixels": size * size,
                "native_ms": round(native_ms, 3),
                "library_ms": round(library_ms, 3),
                "speedup": round(native_ms / library_ms, 1),
            }
        )
        print(rows[-1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "repeats": args.repeats,
        "statistic": "median",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

