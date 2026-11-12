"""Combine regex/rule finders (+ optional local SLM) into detections."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from ..llm.base import LocalLLM
from ..ocr.base import BBox, OCRResult
from .rules import ALL_FINDERS

SOURCE_BY_TYPE = {"CNP": "regex+checksum"}


@dataclass
class Detection:
    id: str
    type: str
    word_indices: List[int]
    boxes: List[BBox]  # one box per word (rotation-friendly)
    bbox: BBox  # union, for display
    source: str = "rules"

    def public(self) -> dict:
        """What may leave the process: type + positions, never the value."""
        return {"id": self.id, "type": self.type, "bbox": list(self.bbox), "boxes": len(self.boxes), "source": self.source}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        return cls(
            id=d["id"],
            type=d["type"],
            word_indices=list(d["word_indices"]),
            boxes=[tuple(b) for b in d["boxes"]],
            bbox=tuple(d["bbox"]),
            source=d.get("source", "rules"),
        )


def _union(boxes: Iterable[BBox]) -> BBox:
    bs = list(boxes)
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def detect(ocr: OCRResult, enabled_types: Iterable[str], llm: Optional[LocalLLM] = None) -> List[Detection]:
    enabled = set(enabled_types)
    groups = [g for finder in ALL_FINDERS for g in finder(ocr)]
    if llm is not None:
        groups += llm.find_pii(ocr)

    # de-duplicate: a word belongs to at most one detection (first finder wins)
    claimed: set = set()
    detections: List[Detection] = []
    for kind, idxs in groups:
        if kind not in enabled:
            continue
        idxs = [i for i in idxs if i not in claimed]
        if not idxs:
            continue
        claimed.update(idxs)
        boxes = [ocr.words[i].bbox for i in idxs]
        detections.append(
            Detection(
                id=f"d{len(detections) + 1:02d}",
                type=kind,
                word_indices=idxs,
                boxes=boxes,
                bbox=_union(boxes),
                source=SOURCE_BY_TYPE.get(kind, "rules"),
            )
        )
    return detections


def count_by_type(detections: Iterable[Detection]) -> Dict[str, int]:
    return dict(Counter(d.type for d in detections))
