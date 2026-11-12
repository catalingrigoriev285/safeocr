"""FastAPI backend for the demo frontend.

The backend acts as the MCP *client*: it drives the pipeline by calling the SafeOCR MCP
tools through FastMCP's in-memory transport, i.e. it sees exactly what an agent would.

    uvicorn safeocr.api.main:app --reload --app-dir server
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from ..agent import ChatAgent
from ..agent.ollama import OllamaClient
from ..mcp_server import mcp, service
from ..service import IMAGE_SUFFIXES
from ..storage import NotFoundError

settings = service().settings
agent = ChatAgent(mcp, settings, recall_fn=lambda rid: service().recall(rid))
NAME_RE = re.compile(r"^[\w.-]+$")

app = FastAPI(title="SafeOCR API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    source: Literal["sample", "upload"] = "sample"
    name: str
    detection_ids: Optional[List[str]] = None


class Attachment(BaseModel):
    source: Literal["sample", "upload"]
    name: str


class ChatRequest(BaseModel):
    message: str = ""
    history: List[Dict[str, Any]] = Field(default_factory=list)
    attachment: Optional[Attachment] = None


def _safe_file(folder: Path, name: str) -> Path:
    if not NAME_RE.match(name):
        raise HTTPException(400, "invalid file name")
    p = (folder / name).resolve()
    if not p.is_relative_to(folder) or not p.is_file():
        raise HTTPException(404, "file not found")
    return p


async def _call(client: Client, tool: str, args: Dict[str, Any], steps: List[Dict]) -> Dict:
    t0 = time.perf_counter()
    try:
        res = await client.call_tool(tool, args)
    except ToolError as exc:
        steps.append({"tool": tool, "ok": False, "ms": round((time.perf_counter() - t0) * 1000), "error": str(exc)})
        raise HTTPException(422, {"message": str(exc), "steps": steps}) from exc
    report = res.structured_content or {}
    steps.append({"tool": tool, "ok": True, "ms": round((time.perf_counter() - t0) * 1000), "report": report})
    return report


@app.get("/api/health")
async def health() -> Dict:
    return {
        "status": "ok",
        "ocr_backend": settings.ocr.backend,
        "llm_enabled": settings.llm.enabled,
        "tts_enabled": settings.tts.enabled,
        "chat_model": settings.llm.model,
        "ollama": await OllamaClient(settings.llm).available(),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream NDJSON events: tool_call, tool_result, recall, token, error, done."""
    attachment = None
    if req.attachment:
        folder = settings.samples_dir if req.attachment.source == "sample" else settings.uploads_dir
        attachment = _safe_file(folder, req.attachment.name).name  # the model only sees the file name
    if not req.message.strip() and not attachment:
        raise HTTPException(400, "empty message")

    async def events() -> AsyncIterator[str]:
        async for ev in agent.run(req.message, req.history, attachment):
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/api/samples")
def list_samples() -> List[Dict]:
    out = []
    for img in sorted(settings.samples_dir.glob("*.png")):
        gt = img.with_name(img.stem + ".gt.json")
        variation = None
        if gt.exists():
            variation = json.loads(gt.read_text(encoding="utf-8")).get("variation")
        out.append({"name": img.name, "has_ground_truth": gt.exists(), "variation": variation})
    return out


@app.post("/api/documents")
async def upload(file: UploadFile) -> Dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise HTTPException(400, f"unsupported image type: {suffix or '?'}")
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    name = f"upload_{uuid.uuid4().hex[:12]}{suffix}"
    dest = settings.uploads_dir / name
    dest.write_bytes(data)
    # Stub OCR needs ground truth: if the upload is byte-identical to a generated sample
    # (e.g. dragged from data/samples in Finder), reuse that sample's sidecar.
    matched = _matching_sample(data)
    if matched is not None:
        shutil.copy2(matched.with_name(matched.stem + ".gt.json"), dest.with_name(dest.stem + ".gt.json"))
    return {"source": "upload", "name": name, "matched_sample": matched.name if matched else None}


def _matching_sample(data: bytes) -> Optional[Path]:
    digest = hashlib.sha256(data).hexdigest()
    for img in settings.samples_dir.glob("*.png"):
        if img.with_name(img.stem + ".gt.json").exists() and hashlib.sha256(img.read_bytes()).hexdigest() == digest:
            return img
    return None


@app.post("/api/pipeline/run")
async def run_pipeline(req: RunRequest) -> Dict:
    folder = settings.samples_dir if req.source == "sample" else settings.uploads_dir
    path = _safe_file(folder, req.name)
    steps: List[Dict] = []
    async with Client(mcp) as client:
        ocr = await _call(client, "ocr_document", {"path": str(path)}, steps)
        doc_id = ocr["document_id"]
        await _call(client, "detect_pii", {"document_id": doc_id}, steps)
        red = await _call(client, "redact_image", {"document_id": doc_id, "detection_ids": req.detection_ids}, steps)
        result_id = red["result_id"]
        ver = await _call(client, "verify_redaction", {"result_id": result_id}, steps)
        ann = await _call(client, "announce", {"result_id": result_id}, steps)
    return {
        "document_id": doc_id,
        "result_id": result_id,
        "passed": ver["passed"],
        "steps": steps,
        "announcement": ann,
        "recall": service().recall(result_id),  # local evaluation vs ground truth (counts only)
    }


@app.get("/api/images/{kind}/{name}")
def image(kind: Literal["sample", "upload", "result"], name: str) -> FileResponse:
    if kind == "result":
        try:
            p = service().store.redacted_path(name)
        except NotFoundError as exc:
            raise HTTPException(404, "result not found") from exc
        if not p.exists():
            raise HTTPException(404, "result not found")
        return FileResponse(p, media_type="image/png")
    folder = settings.samples_dir if kind == "sample" else settings.uploads_dir
    return FileResponse(_safe_file(folder, name))
