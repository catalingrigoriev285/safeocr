"""Stub OCR backend (used until Tesseract is installed).

It reads the ground-truth sidecar produced by `scripts/generate_samples.py`, but it is
honest about the pixels: a word whose box is mostly dark (i.e. covered by a black
rectangle) is treated as unreadable. That keeps `verify_redaction` meaningful.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import BBox, OCRError, OCRResult, Word

DARK_LEVEL = 60  # grayscale value below which a pixel counts as "dark"


def dark_fraction(gray: np.ndarray, bbox: BBox) -> float:
    h, w = gray.shape
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = gray[y0:y1, x0:x1]
    return float((region < DARK_LEVEL).mean())


class StubOCR:
    name = "stub"

    def __init__(self, dark_threshold: float = 0.85):
        self.dark_threshold = dark_threshold

    def run(self, image_path: Path, *, ground_truth: Optional[Path] = None) -> OCRResult:
        if ground_truth is None or not Path(ground_truth).exists():
            raise OCRError(
                "The stub OCR backend only works on generated samples (needs a *.gt.json "
                "sidecar). Install Tesseract and set ocr.backend=tesseract for other images."
            )
        gt = json.loads(Path(ground_truth).read_text(encoding="utf-8"))
        with Image.open(image_path) as im:
            gray = np.asarray(im.convert("L"))
        words = [
            Word(text=w["text"], bbox=tuple(w["bbox"]), line_id=w["line_id"], conf=95.0)
            for w in gt["words"]
            if dark_fraction(gray, tuple(w["bbox"])) < self.dark_threshold
        ]
        return OCRResult(width=gray.shape[1], height=gray.shape[0], backend=self.name, words=words)
