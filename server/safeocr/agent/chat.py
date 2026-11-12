"""Chat agent: a local Ollama model (qwen3) driving the SafeOCR MCP tools.

Privacy by design: the model only ever receives the user's own messages and tool *reports*
(ids, types, counts, pass/fail), never OCR text or personal values.

Small models are weak at chaining ids between tool calls, so:
  * when the user attaches an image, the backend runs the whole pipeline deterministically
    through MCP (each call is shown in the chat and added to the model's context);
  * the model answers from those reports and may call tools itself for follow-ups
    ("redact only the CNP", "verify again");
  * invented ids in model tool calls are replaced with the latest real ones.
"""

from __future__ import annotations

import itertools
import json
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from ..config import PII_TYPES, Settings
from ..tts import compose_announcement
from .language import (
    ANONYMIZE_INTENT,
    PRIVACY_TOPIC,
    detect_language,
    language_rule,
    no_image_state,
    privacy_facts,
    summary_language,
)
from .ollama import OllamaClient, OllamaError

SYSTEM_PROMPT = """You are SafeOCR, a local assistant that anonymizes FICTIVE Romanian identity cards for the user.

Talk TO the user, in the second person: "your data", "your privacy", "your document".
You are a tool: you have no privacy and no personal data of your own, so never say "my privacy" or "my data".

Privacy rule: you never see the document text or any personal value. You only receive tool reports
(ids, data types, counts, pass/fail). Never invent names, CNPs, addresses or numbers.

When the user attaches an image, the system has already run ocr_document, detect_pii, redact_image and
verify_redaction for you; read those reports and summarize briefly: which types of personal data were
found and covered (with counts) and whether verification passed. The redacted image is shown to the user.

For follow-ups you can call tools. Use ids exactly as they appear in earlier reports:
- redact_image(document_id, detection_ids) to redact only some detections (e.g. only the CNP)
- verify_redaction(result_id), detect_pii(document_id), ocr_document(path) for samples like doc_03.png
Personal data types: CNP, NAME, ADDRESS, ID_SERIES, ID_NUMBER, MRZ.

Reply in the SAME language as the user's last message (any language).
Plain text, short: 1-4 sentences or a short list."""

MODEL_TOOLS = ("ocr_document", "detect_pii", "redact_image", "verify_redaction")
PIPELINE = ("ocr_document", "detect_pii", "redact_image", "verify_redaction", "announce")
DOC_RE = re.compile(r"^doc_[0-9a-f]{12}$")
RES_RE = re.compile(r"^res_[0-9a-f]{12}$")
DET_RE = re.compile(r"^d\d{2}$")
HISTORY_LIMIT = 40
class IdTracker:
    """Latest real ids seen in the conversation (used to repair hallucinated arguments)."""

    def __init__(self) -> None:
        self.document_id: Optional[str] = None
        self.result_id: Optional[str] = None
        self.detections: List[Dict[str, str]] = []

    def update(self, report: Dict[str, Any]) -> None:
        if DOC_RE.match(str(report.get("document_id", ""))):
            self.document_id = report["document_id"]
        if RES_RE.match(str(report.get("result_id", ""))):
            self.result_id = report["result_id"]
        if isinstance(report.get("detections"), list):
            self.detections = [{"id": d["id"], "type": d["type"]} for d in report["detections"] if "id" in d]

    @classmethod
    def from_messages(cls, messages: List[Dict[str, Any]]) -> "IdTracker":
        t = cls()
        for m in messages:
            if m.get("role") == "tool":
                try:
                    t.update(json.loads(m.get("content") or "{}"))
                except (ValueError, TypeError):
                    pass
        return t

    def repair(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = dict(args or {})
        if name in ("detect_pii", "redact_image") and not DOC_RE.match(str(args.get("document_id", ""))):
            if self.document_id:
                args["document_id"] = self.document_id
        if name == "verify_redaction" and not RES_RE.match(str(args.get("result_id", ""))):
            if self.result_id:
                args["result_id"] = self.result_id
        if name == "ocr_document":
            path = str(args.get("path", "")).strip()
            if path and "." not in path.rsplit("/", 1)[-1]:
                args["path"] = path + ".png"
        if name == "redact_image":
            raw = args.get("detection_ids")
            if isinstance(raw, str):
                raw = [raw]
            if isinstance(raw, list) and raw:
                ids: List[str] = []
                for item in raw:
                    item = str(item).strip()
                    if DET_RE.match(item):
                        ids.append(item)
                    elif item.upper() in PII_TYPES:  # model passed a type name, e.g. "CNP"
                        ids += [d["id"] for d in self.detections if d["type"] == item.upper()]
                args["detection_ids"] = sorted(set(ids)) or None
            else:
                args["detection_ids"] = None
        return args


class ThinkFilter:
    """Drop <think>...</think> blocks from streamed text (in case thinking is not disabled)."""

    def __init__(self) -> None:
        self.inside = False
        self.buf = ""

    def feed(self, text: str) -> str:
        self.buf += text
        out = ""
        while self.buf:
            if self.inside:
                end = self.buf.find("</think>")
                if end < 0:
                    self.buf = self.buf[-8:]
                    return out
                self.buf, self.inside = self.buf[end + 8 :], False
            else:
                start = self.buf.find("<think>")
                if start < 0:
                    lt = self.buf.rfind("<", max(0, len(self.buf) - 7))
                    keep = len(self.buf) - lt if lt >= 0 else 0  # possible partial "<think>" at the end
                    out += self.buf[: len(self.buf) - keep]
                    self.buf = self.buf[len(self.buf) - keep :]
                    return out
                out += self.buf[:start]
                self.buf, self.inside = self.buf[start + 7 :], True
        return out

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return rest


def _for_model(report: Dict[str, Any]) -> str:
    """Compact report for the model context (positions are useless to it)."""
    slim = dict(report)
    if isinstance(slim.get("detections"), list):
        slim["detections"] = [{"id": d.get("id"), "type": d.get("type")} for d in slim["detections"]]
    for noisy in ("image_size", "words_still_readable", "boxes_drawn", "line_count"):
        slim.pop(noisy, None)
    return json.dumps(slim, ensure_ascii=False)


def _sanitize_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean = []
    for m in history[-HISTORY_LIMIT:]:
        role = m.get("role")
        if role == "system" and str(m.get("content", "")).startswith("Pipeline summary:"):
            clean.append({"role": "system", "content": str(m["content"])})
            continue
        if role not in ("user", "assistant", "tool"):
            continue
        item: Dict[str, Any] = {"role": role, "content": str(m.get("content") or "")}
        if role == "assistant" and isinstance(m.get("tool_calls"), list):
            item["tool_calls"] = m["tool_calls"]
        if role == "tool":
            item["tool_name"] = str(m.get("tool_name") or "")
        clean.append(item)
    return clean


class ChatAgent:
    def __init__(self, mcp: FastMCP, settings: Settings, recall_fn=None):
        self.mcp = mcp
        self.settings = settings
        self.ollama = OllamaClient(settings.llm)
        self.recall_fn = recall_fn

    async def _model_tools(self, client: Client) -> List[Dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.input_schema}}
            for t in await client.list_tools()
            if t.name in MODEL_TOOLS
        ]

    async def _call(self, client: Client, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            res = await client.call_tool(name, args)
            out = {"ok": True, "report": res.structured_content or {}}
        except ToolError as exc:
            out = {"ok": False, "error": str(exc)}
        out["ms"] = round((time.perf_counter() - t0) * 1000)
        return out

    async def run(self, message: str, history: List[Dict[str, Any]], attachment: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        counter = itertools.count(1)
        next_id = lambda: f"t{next(counter)}"  # noqa: E731 - per-run, safe for concurrent chats
        past = _sanitize_history(history)
        lang = detect_language(message, [m["content"] for m in past if m["role"] == "user"])
        text = message.strip() or ("Anonimizează documentul atașat." if lang == "ro" else "Anonymize the attached document.")
        hint = language_rule(lang)
        # privacy facts only when asked about privacy; otherwise a small model pads every answer with them
        facts = f"\n\n{privacy_facts(lang)}" if PRIVACY_TOPIC.search(text) else ""
        system = {"role": "system", "content": f"{SYSTEM_PROMPT}{facts}\n\n{hint}"}
        msgs: List[Dict[str, Any]] = [system, *past]
        msgs.append({"role": "user", "content": f"[Attached image: {attachment}]\n{text}" if attachment else text})
        ids = IdTracker.from_messages(msgs)
        redacted_counts: Dict[str, int] = {}
        passed: Optional[bool] = None

        async with Client(self.mcp) as client:
            # 1) attached image -> deterministic pipeline through MCP
            if attachment:
                for name in PIPELINE:
                    args = {"path": attachment} if name == "ocr_document" else ids.repair(name, {})
                    if name == "announce":
                        args = {"result_id": ids.result_id}
                    call_id = next_id()
                    yield {"type": "tool_call", "id": call_id, "name": name, "arguments": args, "auto": True}
                    out = await self._call(client, name, args)
                    yield {"type": "tool_result", "id": call_id, "name": name, **out}
                    if out["ok"]:
                        ids.update(out["report"])
                        redacted_counts = out["report"].get("redacted_counts", redacted_counts)
                        passed = out["report"].get("passed", passed)
                    if name == "announce":
                        # pre-digested summary (in the chat language) helps a small model answer correctly;
                        # the announce tool itself speaks in the TTS voice's language
                        summary = compose_announcement(redacted_counts, passed, summary_language(lang))
                        msgs.append({"role": "system", "content": f"Pipeline summary: {summary}"})
                        break
                    msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]})
                    msgs.append({"role": "tool", "tool_name": name, "content": _for_model(out["report"]) if out["ok"] else f"ERROR: {out['error']}"})
                    if not out["ok"]:
                        break
                if ids.result_id and self.recall_fn:
                    try:
                        recall = self.recall_fn(ids.result_id)
                    except Exception:  # noqa: BLE001 - recall is optional
                        recall = None
                    if recall:
                        yield {"type": "recall", **recall}

            # 2) model turn(s); the model may call tools for follow-ups
            tools = await self._model_tools(client)
            max_rounds = self.settings.llm.max_tool_rounds
            asks_anonymize = bool(ANONYMIZE_INTENT.search(text))
            state = no_image_state(lang) + "\n" if not ids.document_id and asks_anonymize else ""
            reminder = {"role": "system", "content": state + hint}
            try:
                for round_no in range(max_rounds + 1):
                    content, calls, filt = "", [], ThinkFilter()
                    async for chunk in self.ollama.chat_stream(msgs + [reminder], tools if round_no < max_rounds else None):
                        m = chunk.get("message") or {}
                        visible = filt.feed(m.get("content") or "")
                        if visible:
                            content += visible
                            yield {"type": "token", "content": visible}
                        calls.extend(m.get("tool_calls") or [])
                    tail = filt.flush()
                    if tail:
                        content += tail
                        yield {"type": "token", "content": tail}
                    if not calls:
                        msgs.append({"role": "assistant", "content": content})
                        break
                    msgs.append({"role": "assistant", "content": content, "tool_calls": calls})
                    for call in calls:
                        fn = call.get("function") or {}
                        name = fn.get("name", "")
                        call_id = next_id()
                        if name not in MODEL_TOOLS:
                            msgs.append({"role": "tool", "tool_name": name, "content": f"ERROR: unknown tool {name}"})
                            continue
                        args = ids.repair(name, fn.get("arguments") or {})
                        yield {"type": "tool_call", "id": call_id, "name": name, "arguments": args, "auto": False}
                        out = await self._call(client, name, args)
                        yield {"type": "tool_result", "id": call_id, "name": name, **out}
                        if out["ok"]:
                            ids.update(out["report"])
                        msgs.append({"role": "tool", "tool_name": name, "content": _for_model(out["report"]) if out["ok"] else f"ERROR: {out['error']}"})
            except OllamaError as exc:
                yield {"type": "error", "message": str(exc)}

        yield {"type": "done", "history": msgs[1:], "language": lang}
