"""Speech-to-text for voice input.

* faster-whisper - Whisper running locally on CPU (pip install faster-whisper); audio never leaves the machine
* openai         - any OpenAI-compatible POST /v1/audio/transcriptions server
                   (LocalAI, whisper.cpp server, speaches, OpenAI)

Uploaded audio is written to a temp file, transcribed and deleted immediately.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import httpx

from ..config import FasterWhisperSettings, OpenAISTTSettings, Settings, STTSettings


log = logging.getLogger("safeocr.stt")


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
        self.device: Optional[str] = None  # resolved on load
        self._model = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()  # one transcription at a time (the GPU backend is experimental)

    def _candidates(self):
        """(device, compute_type) to try, best first."""
        device, ctype = self.cfg.device, self.cfg.compute_type
        if device != "auto":
            return [(device, ctype if ctype != "auto" else ("int8" if device == "cpu" else "float16"))]
        out = []
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            out.append(("mps", "float16" if ctype == "auto" else ctype))  # Apple GPU, ~2x faster than CPU
        out.append(("cpu", "int8" if ctype == "auto" else ctype))
        return out

    def _load(self):
        with self._load_lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise STTError("faster-whisper is not installed: pip install faster-whisper") from exc
                root = Settings.resolve(self.cfg.download_root)
                root.mkdir(parents=True, exist_ok=True)
                errors = []
                # first use downloads the model (small ~ 460 MB) into models/whisper
                for device, ctype in self._candidates():
                    try:
                        self._model = WhisperModel(
                            self.cfg.model, device=device, compute_type=ctype,
                            cpu_threads=self.cfg.cpu_threads, download_root=str(root),
                        )
                        self.device = f"{device}/{ctype}"
                        log.info("faster-whisper %s loaded on %s", self.cfg.model, self.device)
                        break
                    except Exception as exc:  # noqa: BLE001 - e.g. no MPS support -> fall back to CPU
                        errors.append(f"{device}: {exc}")
                if self._model is None:
                    raise STTError("could not load the Whisper model: " + "; ".join(errors))
            return self._model

    def warmup(self) -> None:
        """Load the model and run it once on silence, so the first real request is fast."""
        import numpy as np

        model = self._load()
        with self._run_lock:
            segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1,
                                           without_timestamps=True)
            list(segments)

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
            with self._run_lock:
                segments, info = model.transcribe(
                    audio,
                    language=language,  # None = auto-detect (costs ~0.3 s); fixing it is faster
                    beam_size=self.cfg.beam_size,
                    vad_filter=self.cfg.vad_filter,
                    without_timestamps=True,  # we only need the text
                    condition_on_previous_text=False,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as exc:  # noqa: BLE001 - decoding errors etc.
            raise STTError(f"could not transcribe audio: {exc}") from exc
        return Transcript(text=text, language=info.language, duration=round(info.duration, 2))


class OpenAITranscribeSTT:
    name = "openai"
    device = "remote"

    def __init__(self, cfg: OpenAISTTSettings, transport: Optional[httpx.BaseTransport] = None):
        self.cfg = cfg
        self.model = cfg.model
        self._transport = transport

    def warmup(self) -> None:
        pass

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
    device = None

    def warmup(self) -> None:
        pass

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> Transcript:
        raise STTError("voice input is disabled (stt.enabled: false)")


def get_stt(cfg: STTSettings) -> STTEngine:
    if not cfg.enabled:
        return NoopSTT()
    if cfg.engine == "openai":
        return OpenAITranscribeSTT(cfg.openai)
    return FasterWhisperSTT(cfg.faster_whisper)


__all__ = ["STTEngine", "STTError", "Transcript", "get_stt"]
