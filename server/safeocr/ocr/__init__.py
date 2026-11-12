from __future__ import annotations

from ..config import OCRSettings
from .base import OCRBackend, OCRError, OCRResult, Word
from .stub import StubOCR


def get_ocr_backend(cfg: OCRSettings) -> OCRBackend:
    if cfg.backend == "tesseract":
        from .tesseract import TesseractOCR

        return TesseractOCR(lang=cfg.lang, psm=cfg.psm, tesseract_cmd=cfg.tesseract_cmd)
    return StubOCR(dark_threshold=cfg.dark_threshold)


__all__ = ["OCRBackend", "OCRError", "OCRResult", "Word", "StubOCR", "get_ocr_backend"]
