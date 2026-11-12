"""TTS engines. Each one writes a WAV file.

* macos  - the built-in `say` command (CPU, offline; Romanian voice "Ioana")
* piper  - Piper neural TTS (CPU, offline; e.g. ro_RO-mihai-medium, en_US-lessac-medium)
* openai - any OpenAI-compatible POST /v1/audio/speech server:
           Ollama >= 0.40.11 (phonon), Kokoro-FastAPI, LocalAI, openedai-speech, or OpenAI itself
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import Optional

import httpx

from ..config import MacTTSSettings, OpenAITTSSettings, PiperTTSSettings, Settings
from .base import TTSError


class MacSayTTS:
    name = "macos"

    def __init__(self, cfg: MacTTSSettings):
        self.voice: Optional[str] = cfg.voice or None
        self.voices = dict(cfg.voices)
        self.rate = cfg.rate

    def synthesize(self, text: str, out_path: Path, language: Optional[str] = None) -> None:
        if not shutil.which("say"):
            raise TTSError("the `say` command is only available on macOS")
        cmd = ["say", "--file-format=WAVE", "--data-format=LEI16@22050", "-o", str(out_path)]
        voice = self.voices.get(language or "", self.voice)
        if voice:
            cmd += ["-v", voice]
        if self.rate:
            cmd += ["-r", str(self.rate)]
        res = subprocess.run(cmd + ["--", text], capture_output=True, text=True, timeout=60)
        if res.returncode != 0 or not out_path.exists():
            raise TTSError(f"say failed: {res.stderr.strip() or res.returncode}")


class PiperTTS:
    name = "piper"

    def __init__(self, cfg: PiperTTSSettings):
        self.model_path = Settings.resolve(cfg.model)
        self.voice: Optional[str] = self.model_path.stem
        self.voices = {lang: Settings.resolve(p) for lang, p in cfg.voices.items()}
        self._loaded: dict = {}
        self._lock = threading.Lock()

    def _load(self, path: Path):
        with self._lock:
            if path not in self._loaded:
                try:
                    from piper import PiperVoice
                except ImportError as exc:
                    raise TTSError("piper is not installed: pip install piper-tts") from exc
                if not path.exists():
                    raise TTSError(
                        f"Piper voice not found: {path}. Download it with "
                        f"`python -m piper.download_voices {path.stem} --data-dir {path.parent}`"
                    )
                self._loaded[path] = PiperVoice.load(str(path))
            return self._loaded[path]

    def synthesize(self, text: str, out_path: Path, language: Optional[str] = None) -> None:
        voice = self._load(self.voices.get(language or "", self.model_path))
        with wave.open(str(out_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)


class OpenAISpeechTTS:
    name = "openai"

    def __init__(self, cfg: OpenAITTSSettings, transport: Optional[httpx.BaseTransport] = None):
        self.cfg = cfg
        self.voice: Optional[str] = cfg.voice or None
        self._transport = transport  # for tests

    def synthesize(self, text: str, out_path: Path, language: Optional[str] = None) -> None:
        payload = {"model": self.cfg.model, "input": text[:4000], "response_format": "wav"}
        if self.voice:
            payload["voice"] = self.voice
        headers = {}
        api_key = os.environ.get(self.cfg.api_key_env, "") if self.cfg.api_key_env else ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = self.cfg.base_url.rstrip("/") + "/audio/speech"
        try:
            with httpx.Client(timeout=self.cfg.timeout, transport=self._transport) as http:
                resp = http.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise TTSError(f"TTS server not reachable at {self.cfg.base_url}: {exc}") from exc
        if resp.status_code != 200:
            raise TTSError(f"TTS server error {resp.status_code}: {resp.text[:200]}")
        out_path.write_bytes(resp.content)
