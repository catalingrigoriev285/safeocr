"""Recall against ground truth: how many PII entities are covered on the output image."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from .ocr.stub import dark_fraction


def coverage_recall(redacted_image: Path, ground_truth: Path, dark_threshold: float = 0.85) -> Dict:
    """An entity counts as covered when every one of its words is (almost) fully dark."""
    gt = json.loads(Path(ground_truth).read_text(encoding="utf-8"))
    with Image.open(redacted_image) as im:
        gray = np.asarray(im.convert("L"))
    per_type: Dict[str, Dict[str, int]] = {}
    for ent in gt["entities"]:
        stats = per_type.setdefault(ent["type"], {"covered": 0, "total": 0})
        stats["total"] += 1
        if all(dark_fraction(gray, tuple(gt["words"][i]["bbox"])) >= dark_threshold for i in ent["word_indices"]):
            stats["covered"] += 1
    covered = sum(s["covered"] for s in per_type.values())
    total = sum(s["total"] for s in per_type.values())
    for s in per_type.values():
        s["recall"] = round(s["covered"] / s["total"], 4) if s["total"] else 1.0
    return {
        "covered": covered,
        "total": total,
        "recall": round(covered / total, 4) if total else 1.0,
        "per_type": per_type,
    }
