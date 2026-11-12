"""Rule-based PII finders working on OCR lines.

Each finder returns `(type, [word_indices])` groups. Values are never returned to callers
outside the local process — only the word indices (and later their boxes).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Sequence, Tuple

from ..ocr.base import OCRResult
from .cnp import find_cnps

Group = Tuple[str, List[int]]

LABEL_KEYWORDS = re.compile(
    r"\b(NUME|NOM|PRENUME|PRENOM|LAST NAME|FIRST NAME|CETATENIE|NATIONALIT\w*|SEX|SEXE|"
    r"LOC NASTERE|LIEU DE NAISSANCE|PLACE OF BIRTH|DOMICILIU|ADRESSE|ADDRESS|EMISA|"
    r"DELIVREE|ISSUED|VALABILITATE|VALIDIT\w*)\b"
)
NAME_LABEL = re.compile(r"^(NUME|PRENUME)\b|\b(LAST|FIRST) NAME\b")
ADDRESS_LABEL = re.compile(r"^DOMICILIU\b|\bADDRESS\b|\bADRESSE\b")
SERIES_TOKEN = re.compile(r"^[A-Z]{2}$")
NUMBER_TOKEN = re.compile(r"^\d{6,7}$")
NUMBER_LABELS = {"NR", "NO", "NUMAR"}
MRZ_LINE = re.compile(r"^[A-Z0-9<]{25,}$")


def normalize(text: str) -> str:
    """Uppercase, strip diacritics, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped).strip().upper()


def _line_text(ocr: OCRResult, line: Sequence[int]) -> str:
    return normalize(" ".join(ocr.words[i].text for i in line))


def is_label_line(norm_line: str) -> bool:
    return bool(LABEL_KEYWORDS.search(norm_line))


def is_mrz_line(norm_line: str) -> bool:
    compact = norm_line.replace(" ", "")
    return "<" in compact and bool(MRZ_LINE.match(compact))


def find_cnp(ocr: OCRResult) -> List[Group]:
    return [("CNP", [i]) for i, w in enumerate(ocr.words) if any(find_cnps(normalize(w.text)))]


def find_series_and_number(ocr: OCRResult) -> List[Group]:
    groups: List[Group] = []
    for line in ocr.lines():
        tokens = [normalize(ocr.words[i].text).rstrip(".:") for i in line]
        for pos, tok in enumerate(tokens[:-1]):
            nxt = tokens[pos + 1]
            if tok == "SERIA" and SERIES_TOKEN.match(nxt) and nxt not in NUMBER_LABELS:
                groups.append(("ID_SERIES", [line[pos + 1]]))
            elif tok in NUMBER_LABELS and NUMBER_TOKEN.match(nxt):
                groups.append(("ID_NUMBER", [line[pos + 1]]))
    return groups


def find_labeled_values(ocr: OCRResult) -> List[Group]:
    """Name / address values: the line(s) right below their label."""
    groups: List[Group] = []
    lines = ocr.lines()
    texts = [_line_text(ocr, ln) for ln in lines]
    for li, text in enumerate(texts):
        if NAME_LABEL.search(text):
            kind, max_lines = "NAME", 1
        elif ADDRESS_LABEL.search(text):
            kind, max_lines = "ADDRESS", 2
        else:
            continue
        value: List[int] = []  # a multi-line address is one entity
        for nxt in range(li + 1, min(li + 1 + max_lines, len(lines))):
            if is_label_line(texts[nxt]) or is_mrz_line(texts[nxt]):
                break
            value.extend(lines[nxt])
        if value:
            groups.append((kind, value))
    return groups


def find_mrz(ocr: OCRResult) -> List[Group]:
    return [("MRZ", list(ln)) for ln in ocr.lines() if is_mrz_line(_line_text(ocr, ln))]


ALL_FINDERS = (find_cnp, find_series_and_number, find_labeled_values, find_mrz)
