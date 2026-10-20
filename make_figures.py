from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from src.sobel import sobel_library, sobel_native


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def make_demo_image(size: int = 512) -> np.ndarray:
    image = Image.new("L", (size, size), 24)
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 55, 230, 230), fill=180, outline=245, width=7)
    draw.ellipse((285, 55, 460, 230), fill=110, outline=235, width=7)
    draw.polygon([(80, 440), (200, 270), (320, 440)], fill=215)
    draw.line((340, 285, 465, 435), fill=250, width=20)
    draw.line((465, 285, 340, 435), fill=250, width=20)
    ramp = np.tile(np.linspace(0, 100, size, dtype=np.uint8), (size, 1))
    return np.clip(np.asarray(image, dtype=np.uint16) + ramp, 0, 255).astype(np.uint8)


def save_comparison() -> None:
    source = make_demo_image()
    native = sobel_native(source)
    library = sobel_library(source)
    difference = np.abs(native.astype(np.int16) - library.astype(np.int16)).astype(np.uint8)

    Image.fromarray(source).save(RESULTS / "input.png")
    Image.fromarray(native).save(RESULTS / "sobel_native.png")
    Image.fromarray(library).save(RESULTS / "sobel_library.png")

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
    panels = [source, native, library, difference]
    titles = ["Исходное", "Нативный Python", "SciPy", "Разность"]
    for axis, panel, title in zip(axes, panels, titles):
        axis.imshow(panel, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_benchmark_chart() -> None:
    with (RESULTS / "benchmark.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    labels = [row["size"] for row in rows]
    native = [float(row["native_ms"]) for row in rows]
    library = [float(row["library_ms"]) for row in rows]
    positions = np.arange(len(labels))
    width = 0.36

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(positions - width / 2, native, width, label="Нативный Python", color="#4361ee")
    axis.bar(positions + width / 2, library, width, label="SciPy", color="#2ec4b6")
    axis.set_yscale("log")
    axis.set_ylabel("Медианное время, мс (логарифмическая шкала)")
    axis.set_xlabel("Размер изображения")
    axis.set_xticks(positions, labels)
    axis.set_title("Быстродействие реализаций фильтра Собеля")
    axis.grid(axis="y", which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    save_comparison()
    save_benchmark_chart()

