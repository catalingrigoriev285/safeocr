"""End-to-end: MCP tools via FastMCP's in-memory client, and the FastAPI app."""

import asyncio
import json

import pytest


def _no_values_leak(report: dict, samples) -> None:
    """No PII value from the ground truth may appear in any tool report."""
    blob = json.dumps(report)
    for gt_file in samples.glob("*.gt.json"):
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        for w in gt["words"]:
            if w["pii"] and len(w["text"]) > 2:
                assert w["text"] not in blob, f"leaked {w['pii']} value"


def test_mcp_tools_end_to_end(env):
    from fastmcp import Client

    from safeocr.mcp_server import mcp

    async def run():
        async with Client(mcp) as client:
            names = {t.name for t in await client.list_tools()}
            assert {"ocr_document", "detect_pii", "redact_image", "verify_redaction", "announce"} <= names

            sample = str(env["samples"] / "doc_01.png")
            ocr = (await client.call_tool("ocr_document", {"path": sample})).structured_content
            det = (await client.call_tool("detect_pii", {"document_id": ocr["document_id"]})).structured_content
            red = (await client.call_tool("redact_image", {"document_id": ocr["document_id"]})).structured_content
            ver = (await client.call_tool("verify_redaction", {"result_id": red["result_id"]})).structured_content
            ann = (await client.call_tool("announce", {"result_id": red["result_id"]})).structured_content
            return ocr, det, red, ver, ann

    ocr, det, red, ver, ann = asyncio.run(run())
    assert ocr["word_count"] > 20
    assert det["counts"]["CNP"] == 1 and det["counts"]["NAME"] == 2
    assert red["boxes_drawn"] >= det["total"]
    assert ver["passed"] is True
    assert "Verificarea a reușit" in ann["text"]
    for r in (ocr, det, red, ver, ann):
        _no_values_leak(r, env["samples"])


def test_mcp_rejects_paths_outside_allowed_dirs(env, tmp_path):
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from safeocr.mcp_server import mcp

    async def run():
        async with Client(mcp) as client:
            await client.call_tool("ocr_document", {"path": "/etc/hosts"})

    with pytest.raises(ToolError, match="outside the allowed"):
        asyncio.run(run())


def test_partial_redaction_fails_verification(env):
    from safeocr.mcp_server import service

    svc = service()
    doc = svc.ocr_document(str(env["samples"] / "doc_00.png"))["document_id"]
    det = svc.detect_pii(doc)
    keep = [d["id"] for d in det["detections"] if d["type"] != "CNP"]
    res = svc.redact_image(doc, keep)["result_id"]
    ver = svc.verify_redaction(res)
    assert ver["passed"] is False and ver["remaining_counts"] == {"CNP": 1}
    rec = svc.recall(res)
    assert rec["per_type"]["CNP"]["covered"] == 0 and rec["recall"] < 1


def test_api_pipeline(env):
    from fastapi.testclient import TestClient

    from safeocr.api.main import app

    client = TestClient(app)
    assert client.get("/api/health").json()["ocr_backend"] == "stub"
    samples = client.get("/api/samples").json()
    assert [s["name"] for s in samples] == ["doc_00.png", "doc_01.png"]

    body = client.post("/api/pipeline/run", json={"source": "sample", "name": "doc_00.png"}).json()
    assert body["passed"] is True
    assert [s["tool"] for s in body["steps"]] == ["ocr_document", "detect_pii", "redact_image", "verify_redaction", "announce"]
    assert body["recall"]["recall"] == 1.0
    img = client.get(f"/api/images/result/{body['result_id']}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert client.get("/api/images/sample/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)
