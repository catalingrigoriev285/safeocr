"""End-of-task voice announcement.

The announcement contains only counts and the verification result, never personal values,
so it is safe to send to any TTS engine (even a remote one).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Protocol

from ..config import TTSSettings

TYPE_NAMES = {  # (singular, plural)
    "ro": {
        "CNP": ("CNP", "CNP-uri"),
        "NAME": ("nume", "nume"),
        "ADDRESS": ("adresă", "adrese"),
        "ID_SERIES": ("serie de act", "serii de act"),
        "ID_NUMBER": ("număr de act", "numere de act"),
        "MRZ": ("linie MRZ", "linii MRZ"),
    },
    "en": {
        "CNP": ("CNP", "CNPs"),
        "NAME": ("name", "names"),
        "ADDRESS": ("address", "addresses"),
        "ID_SERIES": ("ID series", "ID series"),
        "ID_NUMBER": ("ID number", "ID numbers"),
        "MRZ": ("MRZ line", "MRZ lines"),
    },
}

PHRASES = {
    "ro": {
        "done": "Anonimizare finalizată. Am acoperit {total}",
        "items": ("dată personală", "date personale"),
        "passed": " Verificarea a reușit: nu a mai rămas nicio dată detectată.",
        "failed": " Atenție: verificarea a găsit date rămase neacoperite.",
    },
    "en": {
        "done": "Anonymization finished. I covered {total}",
        "items": ("personal data item", "personal data items"),
        "passed": " Verification passed: no personal data is still detectable.",
        "failed": " Warning: verification found personal data that is still visible.",
    },
}


def _count(n: int, singular: str, plural: str, lang: str) -> str:
    if n == 1:
        return f"1 {singular}"
    if lang == "ro" and (n % 100 == 0 or n % 100 >= 20):  # Romanian: 20 de adrese
        return f"{n} de {plural}"
    return f"{n} {plural}"


def compose_announcement(redacted_counts: Dict[str, int], passed: Optional[bool], lang: str = "ro") -> str:
    """Announcement text (Romanian or English) with counts only, never the values."""
    lang = lang if lang in PHRASES else "en"
    p, names = PHRASES[lang], TYPE_NAMES[lang]
    total = sum(redacted_counts.values())
    parts = ", ".join(_count(n, *names.get(t, (t, t)), lang) for t, n in sorted(redacted_counts.items()))
    msg = p["done"].format(total=_count(total, *p["items"], lang))
    msg += f": {parts}." if parts else "."
    if passed is True:
        msg += p["passed"]
    elif passed is False:
        msg += p["failed"]
    return msg


class TTSError(RuntimeError):
    pass


class TTSEngine(Protocol):
    name: str
    voice: Optional[str]

    def synthesize(self, text: str, out_path: Path) -> None:
        """Write a WAV file to out_path or raise TTSError."""


class NoopTTS:
    name = "none"
    voice = None

    def synthesize(self, text: str, out_path: Path) -> None:
        raise TTSError("TTS is disabled (tts.enabled: false)")


def get_tts(cfg: TTSSettings) -> TTSEngine:
    if not cfg.enabled or cfg.engine == "none":
        return NoopTTS()
    from .engines import MacSayTTS, OpenAISpeechTTS, PiperTTS

    if cfg.engine == "macos":
        return MacSayTTS(cfg.macos)
    if cfg.engine == "piper":
        return PiperTTS(cfg.piper)
    if cfg.engine == "openai":
        return OpenAISpeechTTS(cfg.openai)
    raise ValueError(f"unknown tts.engine: {cfg.engine}")
