"""Draw opaque rectangles over detected boxes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from .ocr.base import BBox


def redact(image_path: Path, boxes: Iterable[BBox], out_path: Path, padding: int = 4, color: str = "#000000") -> int:
    with Image.open(image_path) as im:
        img = im.convert("RGB")
    draw = ImageDraw.Draw(img)
    n = 0
    for x0, y0, x1, y1 in boxes:
        draw.rectangle(
            (max(0, x0 - padding), max(0, y0 - padding), min(img.width, x1 + padding), min(img.height, y1 + padding)),
            fill=color,
        )
        n += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")  # PNG: re-encoded pixels, no original metadata carried over
    return n
