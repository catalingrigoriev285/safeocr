"""Local text-to-speech hook (future work).

Target: a CPU-only local voice (e.g. Piper `ro_RO-mihai-medium`, English fallback).
For now the announcement is only composed as text.
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol

from ..config import TTSSettings

TYPE_NAMES_RO = {  # (singular, plural)
    "CNP": ("CNP", "CNP-uri"),
    "NAME": ("nume", "nume"),
    "ADDRESS": ("adresă", "adrese"),
    "ID_SERIES": ("serie de act", "serii de act"),
    "ID_NUMBER": ("număr de act", "numere de act"),
    "MRZ": ("linie MRZ", "linii MRZ"),
}


def _ro_count(n: int, singular: str, plural: str) -> str:
    """Romanian: 1 adresă, 2 adrese, 20 de adrese."""
    if n == 1:
        return f"1 {singular}"
    return f"{n} de {plural}" if n % 100 == 0 or n % 100 >= 20 else f"{n} {plural}"


def compose_announcement(redacted_counts: Dict[str, int], passed: Optional[bool]) -> str:
    """Romanian message with counts only, never the values."""
    total = sum(redacted_counts.values())
    parts = ", ".join(
        _ro_count(n, *TYPE_NAMES_RO.get(t, (t, t))) for t, n in sorted(redacted_counts.items())
    )
    msg = f"Anonimizare finalizată. Am acoperit {_ro_count(total, 'dată personală', 'date personale')}"
    msg += f": {parts}." if parts else "."
    if passed is True:
        msg += " Verificarea a reușit: nu a mai rămas nicio dată detectată."
    elif passed is False:
        msg += " Atenție: verificarea a găsit date rămase neacoperite."
    return msg


class TTSEngine(Protocol):
    def speak(self, text: str) -> dict: ...


class NoopTTS:
    def __init__(self, cfg: TTSSettings):
        self.cfg = cfg

    def speak(self, text: str) -> dict:
        return {"spoken": False, "engine": "none", "voice": None, "audio_path": None}


def get_tts(cfg: TTSSettings) -> TTSEngine:
    # TODO: return PiperTTS(cfg) when cfg.enabled
    return NoopTTS(cfg)
