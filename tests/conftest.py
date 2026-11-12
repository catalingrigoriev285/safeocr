import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def env(tmp_path_factory, monkeypatch_session):
    """Isolated samples + store folders, populated with 2 generated documents."""
    base = tmp_path_factory.mktemp("safeocr")
    samples, store, uploads = base / "samples", base / "store", base / "uploads"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_samples.py"), "--count", "2", "--out", str(samples)],
        check=True,
        capture_output=True,
    )
    monkeypatch_session.setenv("SAFEOCR_OCR__BACKEND", "stub")
    # tests must not depend on the developer's local voice/LLM settings in config/settings.yaml
    monkeypatch_session.setenv("SAFEOCR_TTS__ENABLED", "false")
    monkeypatch_session.setenv("SAFEOCR_STT__ENABLED", "false")
    monkeypatch_session.setenv("SAFEOCR_LLM__ENABLED", "false")
    monkeypatch_session.setenv("SAFEOCR_PATHS__SAMPLES_DIR", str(samples))
    monkeypatch_session.setenv("SAFEOCR_PATHS__STORE_DIR", str(store))
    monkeypatch_session.setenv("SAFEOCR_PATHS__UPLOADS_DIR", str(uploads))
    monkeypatch_session.setenv("SAFEOCR_PATHS__ALLOWED_INPUT_DIRS", f'["{samples}", "{uploads}"]')

    from safeocr.config import get_settings
    from safeocr.mcp_server import service

    get_settings.cache_clear()
    service.cache_clear()
    yield {"samples": samples, "store": store}
    get_settings.cache_clear()
    service.cache_clear()


@pytest.fixture(scope="session")
def monkeypatch_session():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
