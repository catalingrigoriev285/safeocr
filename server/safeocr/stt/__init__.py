"""Speech-to-text for voice input.

* faster-whisper - Whisper running locally on CPU (pip install faster-whisper); audio never leaves the machine
* openai         - any OpenAI-compatible POST /v1/audio/transcriptions server
                   (LocalAI, whisper.cpp server, speaches, OpenAI)

Uploaded audio is written to a temp file, transcribed and deleted immediately.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import httpx

from ..config import FasterWhisperSettings, OpenAISTTSettings, Settings, STTSettings


class STTError(RuntimeError):
    pass


@dataclass
class Transcript:
    text: str
    language: Optional[str]
    duration: Optional[float]


class STTEngine(Protocol):
    name: str
    model: str

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> Transcript: ...


class FasterWhisperSTT:
    name = "faster-whisper"

    def __init__(self, cfg: FasterWhisperSettings):
        self.cfg = cfg
        self.model = cfg.model
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise STTError("faster-whisper is not installed: pip install faster-whisper") from exc
                root = Settings.resolve(self.cfg.download_root)
                root.mkdir(parents=True, exist_ok=True)
                # first use downloads the model (small ~ 480 MB) into models/whisper
                self._model = WhisperModel(
                    self.cfg.model, device=self.cfg.device, compute_type=self.cfg.compute_type, download_root=str(root)
                )
            return self._model

    @staticmethod
    def _decode(audio_path: Path):
        """Any browser recording (webm/opus, mp4/aac, ogg, wav) -> mono float32 at 16 kHz.

        Done here with PyAV instead of faster_whisper.decode_audio, which passes an argument
        (`metadata_errors`) that newer PyAV versions removed.
        """
        import av
        import numpy as np

        chunks = []
        with av.open(str(audio_path)) as container:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
        if not chunks:
            raise STTError("no audio found in the recording")
        return np.concatenate(chunks).astype(np.float32) / 32768.0

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> Transcript:
        model = self._load()
        try:
            audio = self._decode(audio_path)
            segments, info = model.transcribe(audio, language=language, beam_size=1, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as exc:  # noqa: BLE001 - decoding errors etc.
            raise STTError(f"could not transcribe audio: {exc}") from exc
        return Transcript(text=text, language=info.language, duration=round(info.duration, 2))


class OpenAITranscribeSTT:
    name = "openai"

    def __init__(self, cfg: OpenAISTTSettings, transport: Optional[httpx.BaseTransport] = None):
        self.cfg = cfg
        self.model = cfg.model
        self._transport = transport

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> Transcript:
        headers = {}
        key = os.environ.get(self.cfg.api_key_env, "") if self.cfg.api_key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = {"model": self.cfg.model, "response_format": "json"}
        if language:
            data["language"] = language
        url = self.cfg.base_url.rstrip("/") + "/audio/transcriptions"
        try:
            with httpx.Client(timeout=self.cfg.timeout, transport=self._transport) as http, open(audio_path, "rb") as f:
                resp = http.post(url, data=data, files={"file": (audio_path.name, f)}, headers=headers)
        except httpx.HTTPError as exc:
            raise STTError(f"STT server not reachable at {self.cfg.base_url}: {exc}") from exc
        if resp.status_code != 200:
            raise STTError(f"STT server error {resp.status_code}: {resp.text[:200]}")
        body = resp.json()
        return Transcript(text=str(body.get("text", "")).strip(), language=body.get("language"), duration=body.get("duration"))


class NoopSTT:
    name = "none"
    model = ""

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> Transcript:
        raise STTError("voice input is disabled (stt.enabled: false)")


def get_stt(cfg: STTSettings) -> STTEngine:
    if not cfg.enabled:
        return NoopSTT()
    if cfg.engine == "openai":
        return OpenAITranscribeSTT(cfg.openai)
    return FasterWhisperSTT(cfg.faster_whisper)


__all__ = ["STTEngine", "STTError", "Transcript", "get_stt"]
