"""Romanian CNP (Cod Numeric Personal) detection and validation.

Format: S YY MM DD JJ NNN C
    S   sex/century (1-8, 9 = foreign resident)
    JJ  county code (01-52)
    C   control digit = (sum(d_i * w_i) mod 11), with 10 -> 1, weights 279146358279
"""

from __future__ import annotations

import re
from datetime import date
from typing import Iterator

CNP_RE = re.compile(r"(?<!\d)([1-9]\d{12})(?!\d)")
WEIGHTS = "279146358279"
CENTURY = {"1": 1900, "2": 1900, "3": 1800, "4": 1800, "5": 2000, "6": 2000, "7": 1900, "8": 1900, "9": 1900}


def control_digit(first12: str) -> int:
    total = sum(int(d) * int(w) for d, w in zip(first12, WEIGHTS))
    rest = total % 11
    return 1 if rest == 10 else rest


def is_valid_cnp(cnp: str) -> bool:
    if not re.fullmatch(r"[1-9]\d{12}", cnp or ""):
        return False
    year = CENTURY[cnp[0]] + int(cnp[1:3])
    try:
        date(year, int(cnp[3:5]), int(cnp[5:7]))
    except ValueError:
        return False
    county = int(cnp[7:9])
    if not (1 <= county <= 52 or county in (70, 80, 81, 82, 83)):
        return False
    return control_digit(cnp[:12]) == int(cnp[12])


def find_cnps(text: str) -> Iterator[str]:
    """Yield every valid CNP inside an arbitrary text token."""
    for m in CNP_RE.finditer(text):
        if is_valid_cnp(m.group(1)):
            yield m.group(1)


def make_cnp(sex_digit: int, birth: date, county: int, serial: int) -> str:
    """Build a checksum-valid (fictive) CNP. Used by the sample generator and tests."""
    first12 = f"{sex_digit}{birth:%y%m%d}{county:02d}{serial:03d}"
    return first12 + str(control_digit(first12))
