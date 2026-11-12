"""Settings: config/settings.yaml + environment overrides (SAFEOCR_<SECTION>__<KEY>)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = Path(os.environ.get("SAFEOCR_CONFIG", ROOT_DIR / "config" / "settings.yaml"))

PII_TYPES = ["CNP", "NAME", "ADDRESS", "ID_SERIES", "ID_NUMBER", "MRZ"]


class OCRSettings(BaseModel):
    backend: Literal["stub", "tesseract"] = "stub"
    lang: str = "ron+eng"
    tesseract_cmd: Optional[str] = None
    psm: int = 6
    dark_threshold: float = 0.85


class PathSettings(BaseModel):
    samples_dir: Path = Path("data/samples")
    uploads_dir: Path = Path("data/uploads")
    store_dir: Path = Path("data/store")
    allowed_input_dirs: List[Path] = Field(
        default_factory=lambda: [Path("data/samples"), Path("data/uploads")]
    )


class RedactSettings(BaseModel):
    padding: int = 4
    color: str = "#000000"


class PIISettings(BaseModel):
    enabled_types: List[str] = Field(default_factory=lambda: list(PII_TYPES))


class LLMSettings(BaseModel):
    enabled: bool = False  # extra PII classifier inside detect_pii (the chat agent always uses the model)
    provider: str = "ollama"
    model: str = "qwen3:1.7b"
    base_url: str = "http://localhost:11434"
    think: bool = False  # qwen3 reasoning mode; off = faster answers on CPU
    temperature: float = 0.3
    timeout: float = 180.0
    max_tool_rounds: int = 4


class MacTTSSettings(BaseModel):
    voice: str = "Ioana"  # `say -v '?'` lists voices; Ioana = ro_RO
    rate: Optional[int] = None  # words per minute
    voices: Dict[str, str] = Field(default_factory=dict)  # per-language voices for chat replies, e.g. {en: Samantha}


class PiperTTSSettings(BaseModel):
    model: Path = Path("models/piper/ro_RO-mihai-medium.onnx")
    voices: Dict[str, Path] = Field(default_factory=dict)  # per-language models, e.g. {en: models/piper/en_US-...onnx}


class OpenAITTSSettings(BaseModel):
    base_url: str = "http://localhost:11434/v1"  # Ollama; or Kokoro-FastAPI http://localhost:8880/v1, OpenAI ...
    model: str = "phonon"
    voice: str = ""  # empty = server default
    api_key_env: str = ""  # name of an env var holding the API key (only for remote providers)
    timeout: float = 60.0


class TTSSettings(BaseModel):
    enabled: bool = False
    engine: Literal["none", "macos", "piper", "openai"] = "macos"
    language: Literal["ro", "en"] = "ro"  # language of the spoken announcement (must match the voice)
    macos: MacTTSSettings = Field(default_factory=MacTTSSettings)
    piper: PiperTTSSettings = Field(default_factory=PiperTTSSettings)
    openai: OpenAITTSSettings = Field(default_factory=OpenAITTSSettings)


class FasterWhisperSettings(BaseModel):
    model: str = "small"  # tiny | base | small | medium | large-v3 | turbo (bigger = better, slower)
    device: str = "auto"  # auto = Apple GPU (mps) on Apple Silicon, else cpu; or cpu | mps | cuda
    compute_type: str = "auto"  # auto = float16 on mps/cuda, int8 on cpu
    cpu_threads: int = 0  # 0 = library default
    vad_filter: bool = False  # the browser already trims silence; Whisper's own VAD adds ~0.3-1 s
    beam_size: int = 1  # 1 = greedy (fastest); 5 = slightly more accurate, slower
    preload: bool = True  # load + warm up the model when the API starts (no slow first request)
    download_root: Path = Path("models/whisper")


class OpenAISTTSettings(BaseModel):
    base_url: str = "http://localhost:8080/v1"  # LocalAI / whisper.cpp server / speaches / OpenAI
    model: str = "whisper-1"
    api_key_env: str = ""
    timeout: float = 120.0


class STTSettings(BaseModel):
    """Speech-to-text for voice input. Audio is transcribed locally and never stored."""

    enabled: bool = False
    engine: Literal["faster-whisper", "openai"] = "faster-whisper"
    language: Optional[str] = None  # None = auto-detect; or e.g. "ro"
    max_seconds: int = 120
    faster_whisper: FasterWhisperSettings = Field(default_factory=FasterWhisperSettings)
    openai: OpenAISTTSettings = Field(default_factory=OpenAISTTSettings)


class APISettings(BaseModel):
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SAFEOCR_",
        env_nested_delimiter="__",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
        yaml_file=CONFIG_FILE,
    )

    ocr: OCRSettings = Field(default_factory=OCRSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    redact: RedactSettings = Field(default_factory=RedactSettings)
    pii: PIISettings = Field(default_factory=PIISettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    api: APISettings = Field(default_factory=APISettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # priority: explicit kwargs > env > .env > yaml
        return (init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls))

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def resolve(path: Path) -> Path:
        """Resolve a configured path against the repository root."""
        p = Path(path).expanduser()
        return (p if p.is_absolute() else ROOT_DIR / p).resolve()

    @property
    def samples_dir(self) -> Path:
        return self.resolve(self.paths.samples_dir)

    @property
    def uploads_dir(self) -> Path:
        return self.resolve(self.paths.uploads_dir)

    @property
    def store_dir(self) -> Path:
        return self.resolve(self.paths.store_dir)

    @property
    def allowed_input_dirs(self) -> List[Path]:
        return [self.resolve(p) for p in self.paths.allowed_input_dirs]


@lru_cache
def get_settings() -> Settings:
    return Settings()
