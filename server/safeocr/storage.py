"""Local, on-disk document store.

Everything sensitive (source image, OCR text, detections, ground truth) lives here and
never leaves the machine. MCP tools only hand out the identifiers.

Layout:
    <store>/documents/<document_id>/source.<ext>
                                    ground_truth.json   (only for generated samples)
                                    ocr.json
                                    detections.json
                                    meta.json
    <store>/results/<result_id>/redacted.png
                                meta.json
                                verification.json
                                announcement.wav    (when TTS is enabled)
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ID_RE = re.compile(r"^(doc|det|res)_[0-9a-f]{12}$")


class NotFoundError(KeyError):
    pass


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ground_truth_sidecar(image_path: Path) -> Optional[Path]:
    """`doc_00.png` -> `doc_00.gt.json` if it exists."""
    candidate = image_path.with_name(image_path.stem + ".gt.json")
    return candidate if candidate.exists() else None


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "documents").mkdir(parents=True, exist_ok=True)
        (self.root / "results").mkdir(parents=True, exist_ok=True)

    # --- paths ---------------------------------------------------------------
    def _checked(self, ident: str, prefix: str) -> str:
        if not ID_RE.match(ident or "") or not ident.startswith(prefix + "_"):
            raise NotFoundError(f"invalid id: {ident!r}")
        return ident

    def doc_dir(self, document_id: str) -> Path:
        d = self.root / "documents" / self._checked(document_id, "doc")
        if not d.exists():
            raise NotFoundError(f"unknown document: {document_id}")
        return d

    def result_dir(self, result_id: str) -> Path:
        d = self.root / "results" / self._checked(result_id, "res")
        if not d.exists():
            raise NotFoundError(f"unknown result: {result_id}")
        return d

    # --- json helpers ----------------------------------------------------------
    @staticmethod
    def write_json(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def read_json(path: Path) -> Any:
        if not path.exists():
            raise NotFoundError(f"missing file: {path.name}")
        return json.loads(path.read_text(encoding="utf-8"))

    # --- documents -------------------------------------------------------------
    def create_document(self, source: Path) -> str:
        document_id = new_id("doc")
        d = self.root / "documents" / document_id
        d.mkdir(parents=True)
        shutil.copy2(source, d / f"source{source.suffix.lower()}")
        gt = ground_truth_sidecar(source)
        if gt:
            shutil.copy2(gt, d / "ground_truth.json")
        self.write_json(
            d / "meta.json",
            {
                "document_id": document_id,
                "source_name": source.name,
                "has_ground_truth": gt is not None,
                "created_at": now_iso(),
            },
        )
        return document_id

    def source_path(self, document_id: str) -> Path:
        d = self.doc_dir(document_id)
        for p in d.glob("source.*"):
            return p
        raise NotFoundError(f"document {document_id} has no source image")

    def ground_truth_path(self, document_id: str) -> Optional[Path]:
        p = self.doc_dir(document_id) / "ground_truth.json"
        return p if p.exists() else None

    def save_doc_json(self, document_id: str, name: str, data: Any) -> None:
        self.write_json(self.doc_dir(document_id) / name, data)

    def load_doc_json(self, document_id: str, name: str) -> Any:
        return self.read_json(self.doc_dir(document_id) / name)

    # --- results ----------------------------------------------------------------
    def create_result(self, document_id: str) -> str:
        result_id = new_id("res")
        (self.root / "results" / result_id).mkdir(parents=True)
        return result_id

    def redacted_path(self, result_id: str) -> Path:
        return self.result_dir(result_id) / "redacted.png"

    def announcement_path(self, result_id: str) -> Path:
        return self.result_dir(result_id) / "announcement.wav"

    def save_result_json(self, result_id: str, name: str, data: Any) -> None:
        self.write_json(self.result_dir(result_id) / name, data)

    def load_result_json(self, result_id: str, name: str) -> Any:
        return self.read_json(self.result_dir(result_id) / name)
