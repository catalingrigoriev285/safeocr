"""Chat agent tests with a fake Ollama (no model needed)."""

import asyncio
import json

from safeocr.agent.chat import IdTracker, ThinkFilter
from safeocr.agent.language import detect_language, language_rule


def test_id_repair_replaces_hallucinated_ids_and_maps_types():
    t = IdTracker()
    t.update({"document_id": "doc_0123456789ab"})
    t.update({"detections": [{"id": "d01", "type": "CNP"}, {"id": "d02", "type": "NAME"}, {"id": "d03", "type": "NAME"}]})
    t.update({"result_id": "res_0123456789ab"})

    assert t.repair("detect_pii", {"document_id": "ocr_document_result"}) == {"document_id": "doc_0123456789ab"}
    assert t.repair("verify_redaction", {"result_id": "latest"}) == {"result_id": "res_0123456789ab"}
    assert t.repair("redact_image", {"document_id": "x", "detection_ids": ["NAME"]})["detection_ids"] == ["d02", "d03"]
    assert t.repair("redact_image", {"document_id": "x", "detection_ids": "d01"})["detection_ids"] == ["d01"]
    assert t.repair("ocr_document", {"path": "doc_03"}) == {"path": "doc_03.png"}


def test_think_filter_strips_reasoning_across_chunks():
    f = ThinkFilter()
    out = "".join(f.feed(c) for c in ["<thi", "nk>secret plan</th", "ink>Bună", "! Gata.", " <b"]) + f.flush()
    assert out == "Bună! Gata. <b"


def test_language_detection_is_international():
    assert detect_language("Acum acoperă doar CNP-ul") == "ro"
    assert detect_language("Ce date poti detecta?") == "ro"  # no diacritics
    assert detect_language("What personal data can you detect?") == "en"  # langdetect alone says Catalan
    assert detect_language("Comment protégez-vous ma vie privée ?") == "fr"
    assert detect_language("Welche personenbezogenen Daten kannst du erkennen?") == "de"
    # too short to tell -> previous user message, then English
    assert detect_language("merci", ["[Attached image: doc_01.png]\nAnonymise ce document, s'il te plaît."]) == "fr"
    assert detect_language("ok") == "en"
    assert "română" in language_rule("ro") and "German" in language_rule("de")


class FakeOllama:
    """First call: model asks to redact only the CNP (with a bogus id). Second: final answer."""

    def __init__(self):
        self.calls = []

    async def chat_stream(self, messages, tools=None):
        self.calls.append(messages)
        if len(self.calls) == 2 and tools:
            yield {"message": {"content": "", "tool_calls": [
                {"function": {"name": "redact_image", "arguments": {"document_id": "the_doc", "detection_ids": ["CNP"]}}}
            ]}}
        else:
            for tok in ["Gata", ", am ", "acoperit."]:
                yield {"message": {"content": tok}}


def test_chat_turns_with_attachment_and_model_tool_call(env):
    from safeocr.agent import ChatAgent
    from safeocr.mcp_server import mcp, service

    agent = ChatAgent(mcp, service().settings, recall_fn=lambda r: service().recall(r))
    agent.ollama = FakeOllama()

    async def collect(message, history, attachment=None):
        return [ev async for ev in agent.run(message, history, attachment)]

    # turn 1: attachment -> deterministic pipeline, then model answer
    ev1 = asyncio.run(collect("Anonimizează", [], "doc_00.png"))
    names = [e["name"] for e in ev1 if e["type"] == "tool_call"]
    assert names == ["ocr_document", "detect_pii", "redact_image", "verify_redaction", "announce"]
    assert next(e for e in ev1 if e["name"] == "verify_redaction" and e["type"] == "tool_result")["report"]["passed"]
    assert next(e for e in ev1 if e["type"] == "recall")["recall"] == 1.0
    assert "".join(e["content"] for e in ev1 if e["type"] == "token") == "Gata, am acoperit."
    history = ev1[-1]["history"]

    # privacy: nothing the model saw contains a ground-truth PII value
    gt = json.loads((env["samples"] / "doc_00.gt.json").read_text(encoding="utf-8"))
    seen = json.dumps(agent.ollama.calls, ensure_ascii=False)
    for w in gt["words"]:
        if w["pii"] and len(w["text"]) > 2:
            assert w["text"] not in seen

    # turn 2: model calls redact_image with a bogus id + a type name -> repaired
    ev2 = asyncio.run(collect("Acum doar CNP-ul", history))
    call = next(e for e in ev2 if e["type"] == "tool_call")
    assert call["auto"] is False and call["arguments"]["document_id"].startswith("doc_")
    result = next(e for e in ev2 if e["type"] == "tool_result")
    assert result["ok"] and result["report"]["redacted_counts"] == {"CNP": 1}
    assert result["report"]["left_visible_counts"]["NAME"] == 2


def test_chat_endpoint_streams_ndjson(env, monkeypatch):
    from fastapi.testclient import TestClient

    from safeocr.api import main

    monkeypatch.setattr(main.agent, "ollama", FakeOllama())
    client = TestClient(main.app)
    with client.stream("POST", "/api/chat", json={"message": "", "attachment": {"source": "sample", "name": "doc_01.png"}}) as r:
        assert r.status_code == 200
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events[0]["type"] == "tool_call" and events[-1]["type"] == "done"
    assert client.post("/api/chat", json={"message": "  "}).status_code == 400


def test_announcement_languages():
    from safeocr.tts import compose_announcement

    counts = {"CNP": 1, "NAME": 2, "ADDRESS": 1}
    assert compose_announcement(counts, True, "ro").startswith("Anonimizare finalizată. Am acoperit 4 date personale")
    en = compose_announcement(counts, False, "en")
    assert "I covered 4 personal data items: 1 address, 1 CNP, 2 names." in en and en.endswith("still visible.")
    assert "20 de nume" in compose_announcement({"NAME": 20}, None, "ro")


def test_openai_compatible_tts_engine(tmp_path):
    import httpx

    from safeocr.config import OpenAITTSSettings
    from safeocr.tts.engines import OpenAISpeechTTS

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"RIFF....WAVEfake")

    engine = OpenAISpeechTTS(OpenAITTSSettings(model="phonon", voice="Zoey"), transport=httpx.MockTransport(handler))
    out = tmp_path / "a.wav"
    engine.synthesize("Gata.", out)
    assert seen["url"] == "http://localhost:11434/v1/audio/speech"
    assert seen["body"] == {"model": "phonon", "input": "Gata.", "response_format": "wav", "voice": "Zoey"}
    assert out.read_bytes().startswith(b"RIFF")


def test_announce_with_tts_and_audio_endpoint(env, monkeypatch):
    from fastapi.testclient import TestClient

    from safeocr.api import main
    from safeocr.mcp_server import service

    class FakeTTS:
        name, voice = "fake", "test-voice"

        def synthesize(self, text, out_path):
            out_path.write_bytes(b"RIFF" + text.encode())

    svc = service()
    monkeypatch.setattr(svc.settings.tts, "enabled", True)
    monkeypatch.setattr(svc, "tts", FakeTTS())
    doc = svc.ocr_document(str(env["samples"] / "doc_00.png"))["document_id"]
    svc.detect_pii(doc)
    res = svc.redact_image(doc)["result_id"]
    svc.verify_redaction(res)
    report = svc.announce(res, "en")
    assert report["audio"] is True and report["engine"] == "fake" and report["language"] == "en"

    client = TestClient(main.app)
    audio = client.get(f"/api/audio/{res}")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert client.get("/api/audio/res_000000000000").status_code == 404
