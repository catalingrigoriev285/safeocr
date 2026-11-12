"""Chat agent tests with a fake Ollama (no model needed)."""

import asyncio
import json

from safeocr.agent.chat import IdTracker, ThinkFilter, _language_hint


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


def test_language_hint():
    assert "română" in _language_hint("Acum acoperă doar CNP-ul")
    assert "English" in _language_hint("What can you detect?")


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
