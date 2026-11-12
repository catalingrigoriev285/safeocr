"""SafeOCR MCP server (FastMCP).

Privacy by design: the model calling these tools only ever sees identifiers and reports
(types, counts, positions, pass/fail). OCR text and PII values stay in the local store.

Run:
    fastmcp run fastmcp.json          # stdio, from the repo root
    fastmcp dev fastmcp.json          # MCP Inspector
    python -m safeocr.mcp_server      # stdio
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .service import PipelineError, SafeOCRService

mcp = FastMCP(
    name="safeocr",
    instructions=(
        "Anonymize (fictive) identity documents. Call ocr_document(path) -> detect_pii(document_id) "
        "-> redact_image(document_id) -> verify_redaction(result_id), then announce(result_id). "
        "Tools never return document text or personal values; only ids, types, counts and positions."
    ),
)


@lru_cache
def service() -> SafeOCRService:
    return SafeOCRService()


def _guard(fn, *args, **kwargs) -> dict:
    try:
        return fn(*args, **kwargs)
    except PipelineError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(annotations={"readOnlyHint": False, "openWorldHint": False})
def ocr_document(path: str) -> dict:
    """Run local OCR on an image and store text + word boxes locally.

    `path` must be inside one of the allowed input folders (data/samples, data/uploads).
    Returns only a document_id and word/line counts.
    """
    return _guard(service().ocr_document, path)


@mcp.tool(annotations={"readOnlyHint": False, "openWorldHint": False})
def detect_pii(document_id: str) -> dict:
    """Detect personal data (CNP via regex + checksum; name, address, ID series/number, MRZ via rules).

    Returns detection ids, types, counts and bounding boxes, never the values.
    """
    return _guard(service().detect_pii, document_id)


@mcp.tool(annotations={"readOnlyHint": False, "openWorldHint": False})
def redact_image(document_id: str, detection_ids: Optional[List[str]] = None) -> dict:
    """Draw black rectangles over the given detections (all of them if omitted) and save a new image.

    Returns a result_id plus how many boxes were drawn per type.
    """
    return _guard(service().redact_image, document_id, detection_ids)


@mcp.tool(annotations={"readOnlyHint": False, "openWorldHint": False})
def verify_redaction(result_id: str) -> dict:
    """Re-run OCR + detection on the redacted image and confirm no personal data is still detectable."""
    return _guard(service().verify_redaction, result_id)


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
def announce(result_id: str) -> dict:
    """Compose (and, once a local TTS is configured, speak) the end-of-task announcement in Romanian."""
    return _guard(service().announce, result_id)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
