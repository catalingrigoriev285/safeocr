"""Tesseract OCR backend (local, no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image, ImageOps

from .base import OCRError, OCRResult, Word


class TesseractOCR:
    name = "tesseract"

    def __init__(self, lang: str = "ron+eng", psm: int = 6, tesseract_cmd: Optional[str] = None):
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover
            raise OCRError("pytesseract is not installed (pip install pytesseract)") from exc
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._pt = pytesseract
        self.lang = lang
        self.psm = psm

    def run(self, image_path: Path, *, ground_truth: Optional[Path] = None) -> OCRResult:
        pt = self._pt
        try:
            with Image.open(image_path) as im:
                gray = ImageOps.grayscale(im)
                data = pt.image_to_data(
                    gray, lang=self.lang, config=f"--psm {self.psm}", output_type=pt.Output.DICT
                )
                width, height = gray.size
        except pt.TesseractNotFoundError as exc:
            raise OCRError("tesseract binary not found: brew install tesseract tesseract-lang") from exc

        line_ids: Dict[Tuple[int, int, int], int] = {}
        words = []
        for i, text in enumerate(data["text"]):
            text = (text or "").strip()
            conf = float(data["conf"][i])
            if not text or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            line_id = line_ids.setdefault(key, len(line_ids))
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append(Word(text=text, bbox=(x, y, x + w, y + h), line_id=line_id, conf=conf))
        return OCRResult(width=width, height=height, backend=self.name, words=words)
