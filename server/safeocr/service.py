"""The four pipeline operations behind the MCP tools.

Every public method returns a *report*: identifiers, types, counts, positions and
pass/fail. No method returns OCR text or PII values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .config import Settings, get_settings
from .evaluation import coverage_recall
from .llm import get_llm
from .ocr import OCRError, OCRResult, get_ocr_backend
from .pii import Detection, count_by_type, detect
from .redact import redact
from .storage import NotFoundError, Store, new_id, now_iso
from .tts import TTSError, compose_announcement, get_tts

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class PipelineError(ValueError):
    """User-facing error (safe to show: contains no document content)."""


class SafeOCRService:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.store = Store(self.settings.store_dir)
        self.ocr = get_ocr_backend(self.settings.ocr)
        self.llm = get_llm(self.settings.llm)
        self.tts = get_tts(self.settings.tts)

    # --- helpers ------------------------------------------------------------
    def _resolve_input(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            # relative paths: try the configured input folders first, then the repo root
            candidates = [d / p for d in self.settings.allowed_input_dirs] + [self.settings.resolve(p)]
            p = next((c for c in candidates if c.exists()), candidates[-1])
        p = p.resolve()
        if not any(p.is_relative_to(d) for d in self.settings.allowed_input_dirs):
            raise PipelineError("path is outside the allowed input folders (paths.allowed_input_dirs)")
        if not p.is_file():
            raise PipelineError(f"file not found: {p.name}")
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            raise PipelineError(f"unsupported image type: {p.suffix}")
        return p

    def _run_ocr(self, image: Path, ground_truth: Optional[Path]) -> OCRResult:
        try:
            return self.ocr.run(image, ground_truth=ground_truth)
        except OCRError as exc:
            raise PipelineError(str(exc)) from exc

    def _load_detections(self, document_id: str) -> Dict:
        try:
            return self.store.load_doc_json(document_id, "detections.json")
        except NotFoundError as exc:
            raise PipelineError("run detect_pii on this document first") from exc

    # --- 1. OCR ---------------------------------------------------------------
    def ocr_document(self, path: str) -> Dict:
        src = self._resolve_input(path)
        document_id = self.store.create_document(src)
        result = self._run_ocr(self.store.source_path(document_id), self.store.ground_truth_path(document_id))
        self.store.save_doc_json(document_id, "ocr.json", result.to_dict())
        return {
            "document_id": document_id,
            "ocr_backend": result.backend,
            "word_count": len(result.words),
            "line_count": len(result.lines()),
            "image_size": [result.width, result.height],
        }

    # --- 2. Detect --------------------------------------------------------------
    def detect_pii(self, document_id: str) -> Dict:
        try:
            ocr = OCRResult.from_dict(self.store.load_doc_json(document_id, "ocr.json"))
        except NotFoundError as exc:
            raise PipelineError(str(exc)) from exc
        detections = detect(ocr, self.settings.pii.enabled_types, self.llm)
        detection_set_id = new_id("det")
        self.store.save_doc_json(
            document_id,
            "detections.json",
            {"detection_set_id": detection_set_id, "created_at": now_iso(), "detections": [d.to_dict() for d in detections]},
        )
        return {
            "document_id": document_id,
            "detection_set_id": detection_set_id,
            "total": len(detections),
            "counts": count_by_type(detections),
            "detections": [d.public() for d in detections],
        }

    # --- 3. Redact ----------------------------------------------------------------
    def redact_image(self, document_id: str, detection_ids: Optional[List[str]] = None) -> Dict:
        saved = self._load_detections(document_id)
        all_dets = [Detection.from_dict(d) for d in saved["detections"]]
        if detection_ids:
            wanted = set(detection_ids)
            unknown = wanted - {d.id for d in all_dets}
            if unknown:
                raise PipelineError(f"unknown detection ids: {sorted(unknown)}")
            selected = [d for d in all_dets if d.id in wanted]
        else:
            selected = all_dets

        result_id = self.store.create_result(document_id)
        boxes = [b for d in selected for b in d.boxes]
        drawn = redact(
            self.store.source_path(document_id),
            boxes,
            self.store.redacted_path(result_id),
            padding=self.settings.redact.padding,
            color=self.settings.redact.color,
        )
        counts = count_by_type(selected)
        left_visible = count_by_type(d for d in all_dets if d not in selected)
        self.store.save_result_json(
            result_id,
            "meta.json",
            {
                "result_id": result_id,
                "document_id": document_id,
                "detection_set_id": saved["detection_set_id"],
                "detection_ids": [d.id for d in selected],
                "redacted_counts": counts,
                "created_at": now_iso(),
            },
        )
        return {
            "result_id": result_id,
            "document_id": document_id,
            "detections_redacted": len(selected),
            "boxes_drawn": drawn,
            "redacted_counts": counts,
            "left_visible_counts": left_visible,  # detected but deliberately not redacted
        }

    # --- 4. Verify ------------------------------------------------------------------
    def verify_redaction(self, result_id: str) -> Dict:
        try:
            meta = self.store.load_result_json(result_id, "meta.json")
        except NotFoundError as exc:
            raise PipelineError(str(exc)) from exc
        document_id = meta["document_id"]
        ocr = self._run_ocr(self.store.redacted_path(result_id), self.store.ground_truth_path(document_id))
        remaining = detect(ocr, self.settings.pii.enabled_types, self.llm)
        report = {
            "result_id": result_id,
            "passed": len(remaining) == 0,
            "remaining_total": len(remaining),
            "remaining_counts": count_by_type(remaining),
            "words_still_readable": len(ocr.words),
        }
        self.store.save_result_json(result_id, "verification.json", {**report, "checked_at": now_iso()})
        return report

    # --- extras -----------------------------------------------------------------------
    def announce(self, result_id: str, language: Optional[str] = None) -> Dict:
        try:
            meta = self.store.load_result_json(result_id, "meta.json")
        except NotFoundError as exc:
            raise PipelineError(str(exc)) from exc
        try:
            passed = self.store.load_result_json(result_id, "verification.json")["passed"]
        except NotFoundError:
            passed = None
        lang = language if language in ("ro", "en") else self.settings.tts.language
        text = compose_announcement(meta["redacted_counts"], passed, lang)
        report = {"result_id": result_id, "text": text, "language": lang, "engine": self.tts.name,
                  "voice": self.tts.voice, "audio": False}
        if self.settings.tts.enabled:
            try:
                self.tts.synthesize(text, self.store.announcement_path(result_id))
                report["audio"] = True
            except TTSError as exc:  # a missing voice must not break the pipeline
                report["tts_error"] = str(exc)
        return report

    def recall(self, result_id: str) -> Optional[Dict]:
        """Coverage recall vs ground truth (only for generated samples)."""
        meta = self.store.load_result_json(result_id, "meta.json")
        gt = self.store.ground_truth_path(meta["document_id"])
        if gt is None:
            return None
        return coverage_recall(self.store.redacted_path(result_id), gt, self.settings.ocr.dark_threshold)
