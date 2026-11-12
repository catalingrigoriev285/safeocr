"""OCR data model + backend protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1 (pixels, inclusive-exclusive)


@dataclass
class Word:
    text: str
    bbox: BBox
    line_id: int
    conf: float = 100.0


@dataclass
class OCRResult:
    width: int
    height: int
    backend: str
    words: List[Word] = field(default_factory=list)

    def lines(self) -> List[List[int]]:
        """Word indices grouped by line, lines in reading order, words left-to-right."""
        groups: Dict[int, List[int]] = {}
        for i, w in enumerate(self.words):
            groups.setdefault(w.line_id, []).append(i)
        ordered = sorted(groups.items(), key=lambda kv: kv[0])
        return [sorted(idx, key=lambda i: self.words[i].bbox[0]) for _, idx in ordered]

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "backend": self.backend,
            "words": [asdict(w) for w in self.words],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OCRResult":
        return cls(
            width=d["width"],
            height=d["height"],
            backend=d.get("backend", "unknown"),
            words=[
                Word(text=w["text"], bbox=tuple(w["bbox"]), line_id=w["line_id"], conf=w.get("conf", 100.0))
                for w in d["words"]
            ],
        )


class OCRError(RuntimeError):
    pass


class OCRBackend(Protocol):
    name: str

    def run(self, image_path: Path, *, ground_truth: Optional[Path] = None) -> OCRResult: ...
