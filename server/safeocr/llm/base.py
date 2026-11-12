"""Local SLM hook (future work).

The idea: a small local model (e.g. via Ollama) classifies OCR lines the rules missed.
It runs on the user's machine, so the raw text still never reaches a cloud model.
"""

from __future__ import annotations

from typing import List, Protocol, Tuple

from ..config import LLMSettings
from ..ocr.base import OCRResult

Group = Tuple[str, List[int]]


class LocalLLM(Protocol):
    def find_pii(self, ocr: OCRResult) -> List[Group]: ...


class NoopLLM:
    """Placeholder: returns nothing. Replace with an Ollama/llama.cpp implementation."""

    def __init__(self, cfg: LLMSettings):
        self.cfg = cfg

    def find_pii(self, ocr: OCRResult) -> List[Group]:
        return []


def get_llm(cfg: LLMSettings) -> LocalLLM | None:
    # TODO: return an OllamaLLM(cfg) when cfg.enabled
    return NoopLLM(cfg) if cfg.enabled else None
