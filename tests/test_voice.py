"""Voice endpoints with fake engines (no models needed)."""

import json

import httpx
from fastapi.testclient import TestClient


def test_transcribe_and_speak_endpoints(env, monkeypatch):
    from safeocr.api import main
    from safeocr.mcp_server import service
    from safeocr.stt import Transcript

    svc = service()
    seen = {}

    class FakeSTT:
        name, model = "fake", "tiny"

        def transcribe(self, path, language=None):
            seen["suffix"], seen["exists"] = path.suffix, path.exists()
            seen["path"] = path
            return Transcript(text="Ce date poți detecta?", language="ro", duration=1.5)

    class FakeTTS:
        name, voice = "fake", "v"

        def synthesize(self, text, out_path, language=None):
            seen["spoken"], seen["lang"] = text, language
            out_path.write_bytes(b"RIFFfake")

    client = TestClient(main.app)
    # disabled -> 503
    assert client.post("/api/transcribe", files={"file": ("a.webm", b"x", "audio/webm")}).status_code == 503
    assert client.post("/api/speak", json={"text": "hi"}).status_code == 503

    monkeypatch.setattr(svc.settings.stt, "enabled", True)
    monkeypatch.setattr(svc.settings.tts, "enabled", True)
    monkeypatch.setattr(svc, "stt", FakeSTT())
    monkeypatch.setattr(svc, "tts", FakeTTS())

    r = client.post("/api/transcribe", files={"file": ("speech.webm", b"\x1aE\xdf\xa3", "audio/webm;codecs=opus")})
    assert r.status_code == 200 and r.json()["text"] == "Ce date poți detecta?" and r.json()["language"] == "ro"
    assert seen["suffix"] == ".webm" and seen["exists"] and not seen["path"].exists()  # temp audio deleted
    assert client.post("/api/transcribe", files={"file": ("e.webm", b"", "audio/webm")}).status_code == 400

    r = client.post("/api/speak", json={"text": "Pot detecta:\n- **CNP**\n- `numele`", "language": "ro"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content == b"RIFFfake"
    assert seen["spoken"] == "Pot detecta: CNP numele" and seen["lang"] == "ro"  # markdown stripped
    assert client.post("/api/speak", json={"text": "**"}).status_code == 400


def test_openai_compatible_stt(tmp_path):
    from safeocr.config import OpenAISTTSettings
    from safeocr.stt import OpenAITranscribeSTT

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:8080/v1/audio/transcriptions"
        assert b'name="model"' in request.content and b"whisper-1" in request.content
        return httpx.Response(200, json={"text": " Hello there ", "language": "en"})

    audio = tmp_path / "a.webm"
    audio.write_bytes(b"data")
    stt = OpenAITranscribeSTT(OpenAISTTSettings(), transport=httpx.MockTransport(handler))
    t = stt.transcribe(audio)
    assert t.text == "Hello there" and t.language == "en"


def test_done_event_carries_reply_language(env, monkeypatch):
    from safeocr.api import main

    class FakeOllama:
        async def chat_stream(self, messages, tools=None):
            yield {"message": {"content": "Je peux détecter le CNP."}}

    monkeypatch.setattr(main.agent, "ollama", FakeOllama())
    client = TestClient(main.app)
    with client.stream("POST", "/api/chat", json={"message": "Quelles données personnelles peux-tu détecter ?"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events[-1]["type"] == "done" and events[-1]["language"] == "fr"
