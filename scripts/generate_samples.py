"""Generate fictive Romanian ID-card images + ground truth.

All people, numbers and addresses are invented (CNPs are checksum-valid but random).
Each image gets a sidecar `<name>.gt.json` with every word's text, bounding box, line id
and PII label. The stub OCR backend reads it, and the recall evaluation uses it.

    python scripts/generate_samples.py --count 12 --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
from safeocr.pii.cnp import make_cnp  # noqa: E402

W, H = 1000, 640
BG = (233, 240, 236)

FONT_CANDIDATES = {
    "Arial": ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "Times New Roman": ["/System/Library/Fonts/Supplemental/Times New Roman.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
    "Georgia": ["/System/Library/Fonts/Supplemental/Georgia.ttf"],
    "Verdana": ["/System/Library/Fonts/Supplemental/Verdana.ttf"],
    "Arial Narrow": ["/System/Library/Fonts/Supplemental/Arial Narrow.ttf"],
    "Courier New": ["/System/Library/Fonts/Supplemental/Courier New.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}
MONO = "Courier New"

LAST_NAMES = ["POPESCU", "IONESCU", "POPA", "RADU", "DUMITRU", "STAN", "STOICA", "GHEORGHE", "MATEI", "CONSTANTIN", "MARIN", "TUDOR", "NISTOR", "FLOREA"]
FIRST_M = ["ANDREI", "MIHAI", "ALEXANDRU", "STEFAN", "IOAN", "VLAD", "DAVID", "MATEI", "RARES", "CRISTIAN"]
FIRST_F = ["MARIA", "ELENA", "IOANA", "ANA", "ANDREEA", "CRISTINA", "DIANA", "ALEXANDRA", "BIANCA", "ROXANA"]
STREETS = ["Str. Lalelelor", "Bd. Unirii", "Str. Morilor", "Calea Victoriei", "Str. Teilor", "Al. Castanilor", "Str. Fabricii", "Bd. Independentei"]
# (city line, county code for CNP, ID series options)
PLACES = [
    ("Mun. Bucuresti Sec. 3", 43, ["RT", "RX", "DP"]),
    ("Mun. Bucuresti Sec. 6", 46, ["RD", "RR", "DX"]),
    ("Mun. Cluj-Napoca Jud. CJ", 12, ["CJ", "KX"]),
    ("Mun. Iasi Jud. IS", 22, ["MX", "MZ"]),
    ("Mun. Timisoara Jud. TM", 35, ["TM", "TZ"]),
    ("Mun. Constanta Jud. CT", 13, ["KT", "KZ"]),
    ("Mun. Brasov Jud. BV", 8, ["BV", "ZV"]),
]


def load_font(name: str, size: int) -> Tuple[ImageFont.FreeTypeFont, str]:
    for path in FONT_CANDIDATES.get(name, []):
        if Path(path).exists():
            return ImageFont.truetype(path, size), name
    return ImageFont.load_default(size=size), "default"


class Card:
    """Draws words one by one and records their boxes, line ids and PII labels."""

    def __init__(self, img: Image.Image):
        self.img = img
        self.draw = ImageDraw.Draw(img)
        self.words: List[Dict] = []
        self.entities: List[Dict] = []
        self.line_id = 0

    def line(self, x: int, y: int, text: str, font, fill=(25, 25, 30), pii: Optional[str] = None,
             pii_tokens: Optional[List[int]] = None) -> List[int]:
        """Draw `text` word by word. `pii_tokens` limits the label to some token positions."""
        tokens = text.split()
        idxs: List[int] = []
        space = font.getlength(" ")
        for pos, tok in enumerate(tokens):
            bbox = self.draw.textbbox((x, y), tok, font=font)
            self.draw.text((x, y), tok, font=font, fill=fill)
            label = pii if pii and (pii_tokens is None or pos in pii_tokens) else None
            self.words.append({"text": tok, "bbox": list(bbox), "line_id": self.line_id, "pii": label})
            idxs.append(len(self.words) - 1)
            x += font.getlength(tok) + space
        self.line_id += 1
        return idxs

    def entity(self, kind: str, idxs: List[int]) -> None:
        self.entities.append({"type": kind, "word_indices": idxs})


def mrz_check(s: str) -> str:
    vals = {**{str(i): i for i in range(10)}, **{chr(65 + i): 10 + i for i in range(26)}, "<": 0}
    return str(sum(vals[c] * w for c, w in zip(s, [7, 3, 1] * 20)) % 10)


def rotate_box(bbox, angle_deg: float, old_size, new_size) -> List[int]:
    """Map an axis-aligned box through PIL's rotate(angle, expand=True) (counter-clockwise)."""
    a = math.radians(angle_deg)
    cx, cy = old_size[0] / 2, old_size[1] / 2
    ncx, ncy = new_size[0] / 2, new_size[1] / 2
    x0, y0, x1, y1 = bbox
    pts = []
    for px, py in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        dx, dy = px - cx, py - cy
        pts.append((ncx + dx * math.cos(a) + dy * math.sin(a), ncy - dx * math.sin(a) + dy * math.cos(a)))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys))]


def make_person(rng: random.Random) -> Dict:
    female = rng.random() < 0.5
    birth = date(1950, 1, 1) + timedelta(days=rng.randint(0, 365 * 55))
    city, county, series_opts = rng.choice(PLACES)
    sex_digit = (2 if female else 1) if birth.year < 2000 else (6 if female else 5)
    first = rng.choice(FIRST_F if female else FIRST_M)
    if rng.random() < 0.4:
        first += "-" + rng.choice([n for n in (FIRST_F if female else FIRST_M) if n != first])
    issued = date(2018, 1, 1) + timedelta(days=rng.randint(0, 365 * 6))
    return {
        "last": rng.choice(LAST_NAMES),
        "first": first,
        "female": female,
        "birth": birth,
        "cnp": make_cnp(sex_digit, birth, county, rng.randint(1, 999)),
        "series": rng.choice(series_opts),
        "number": f"{rng.randint(100000, 999999)}",
        "city": city,
        "street": f"{rng.choice(STREETS)} nr. {rng.randint(1, 120)}",
        "flat": f"bl. {rng.choice('ABCDEFGH')}{rng.randint(1, 9)} sc. {rng.randint(1, 4)} et. {rng.randint(0, 10)} ap. {rng.randint(1, 80)}",
        "issued": issued,
        "expires": issued.replace(year=issued.year + 10),
    }


def render(p: Dict, font_name: str) -> Tuple[Image.Image, Card, str]:
    img = Image.new("RGB", (W, H), BG)
    card = Card(img)
    d = card.draw
    # decorative background: soft guilloche-like waves (light colors only)
    for k in range(0, H, 14):
        d.line([(x, k + 6 * math.sin(x / 37 + k)) for x in range(0, W, 8)], fill=(214, 226, 220), width=1)
    d.rectangle((0, 0, W, 70), fill=(205, 222, 235))

    f_head, used = load_font(font_name, 26)
    f_lab, _ = load_font(font_name, 15)
    f_val, _ = load_font(font_name, 24)
    f_mrz, _ = load_font(MONO, 27)
    label_c, value_c = (70, 80, 95), (15, 15, 20)

    card.line(30, 10, "ROMANIA ROUMANIE ROMANIA", f_head, fill=(30, 50, 110))
    card.line(30, 40, "CARTE DE IDENTITATE IDENTITY CARD", f_lab, fill=(30, 50, 110))
    s = card.line(560, 22, f"SERIA {p['series']} NR {p['number']}", f_val, fill=value_c)
    card.entity("ID_SERIES", [s[1]])
    card.entity("ID_NUMBER", [s[3]])
    card.words[s[1]]["pii"], card.words[s[3]]["pii"] = "ID_SERIES", "ID_NUMBER"

    # photo placeholder
    d.rectangle((35, 105, 245, 365), fill=(196, 204, 210), outline=(150, 160, 170))
    d.ellipse((95, 140, 185, 240), fill=(160, 170, 178))
    d.pieslice((65, 250, 215, 420), 180, 360, fill=(160, 170, 178))

    c = card.line(35, 385, f"CNP {p['cnp']}", f_val, fill=value_c, pii="CNP", pii_tokens=[1])
    card.entity("CNP", [c[1]])

    x, y = 280, 92
    card.line(x, y, "Nume/Nom/Last name", f_lab, fill=label_c)
    card.entity("NAME", card.line(x, y + 20, p["last"], f_val, fill=value_c, pii="NAME"))
    card.line(x, y + 58, "Prenume/Prénom/First name", f_lab, fill=label_c)
    card.entity("NAME", card.line(x, y + 78, p["first"], f_val, fill=value_c, pii="NAME"))
    card.line(x, y + 116, "Cetățenie/Nationalité/Nationality", f_lab, fill=label_c)
    card.line(x, y + 136, "Română / ROU", f_val, fill=value_c)
    card.line(x + 380, y + 116, "Sex/Sexe/Sex", f_lab, fill=label_c)
    card.line(x + 380, y + 136, "F" if p["female"] else "M", f_val, fill=value_c)
    card.line(x, y + 174, "Loc naștere/Lieu de naissance/Place of birth", f_lab, fill=label_c)
    card.line(x, y + 194, p["city"], f_val, fill=value_c)
    card.line(x, y + 232, "Domiciliu/Adresse/Address", f_lab, fill=label_c)
    a1 = card.line(x, y + 252, p["city"], f_val, fill=value_c, pii="ADDRESS")
    a2 = card.line(x, y + 282, f"{p['street']} {p['flat']}", f_val, fill=value_c, pii="ADDRESS")
    card.entity("ADDRESS", a1 + a2)
    card.line(x, y + 320, "Emisă de/Délivrée par/Issued by", f_lab, fill=label_c)
    card.line(x, y + 340, "SPCLEP " + p["city"].split()[1], f_val, fill=value_c)
    card.line(x + 380, y + 320, "Valabilitate/Validité/Validity", f_lab, fill=label_c)
    card.line(x + 380, y + 340, f"{p['issued']:%d.%m.%y}-{p['expires']:%d.%m.%Y}", f_val, fill=value_c)

    # MRZ (two lines)
    name_field = f"{p['last']}<<{p['first'].replace('-', '<')}"
    l1 = f"IDROU{name_field}".ljust(36, "<")[:36]
    doc = f"{p['series']}{p['number']}"
    yob = f"{p['birth']:%y%m%d}"
    exp = f"{p['expires']:%y%m%d}"
    l2 = f"{doc}<{mrz_check(doc)}ROU{yob}{mrz_check(yob)}{'F' if p['female'] else 'M'}{exp}{mrz_check(exp)}{p['cnp'][0]}{p['cnp'][7:13]}"
    l2 = l2.ljust(36, "<")[:36]
    d.rectangle((0, 540, W, H), fill=(242, 244, 240))
    card.entity("MRZ", card.line(35, 552, l1, f_mrz, fill=value_c, pii="MRZ"))
    card.entity("MRZ", card.line(35, 590, l2, f_mrz, fill=value_c, pii="MRZ"))
    return img, card, used


def watermark(img: Image.Image, rng: random.Random) -> Image.Image:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    f, _ = load_font("Arial", 90)
    ImageDraw.Draw(layer).text((260, 230), "SPECIMEN", font=f, fill=(200, 60, 60, 38))
    layer = layer.rotate(rng.uniform(15, 25), resample=Image.BICUBIC)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "samples")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    fonts = list(FONT_CANDIDATES)[:-1]  # all but the mono font used for MRZ

    for i in range(args.count):
        person = make_person(rng)
        font_name = fonts[i % len(fonts)]
        # variations: doc 0 is clean, then increasing rotation / noise / blur
        angle = 0.0 if i == 0 else round(rng.uniform(-3.0, 3.0), 2)
        noise = 0.0 if i == 0 else round(rng.uniform(3, 14), 1)
        blur = 0.0 if i % 3 != 2 else round(rng.uniform(0.4, 0.9), 2)

        img, card, used_font = render(person, font_name)
        img = watermark(img, rng)
        words = card.words
        if angle:
            old = img.size
            img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(248, 248, 248))
            for w in words:
                w["bbox"] = rotate_box(w["bbox"], angle, old, img.size)
        if blur:
            img = img.filter(ImageFilter.GaussianBlur(blur))
        if noise:
            arr = np.asarray(img).astype(np.float32)
            arr += np_rng.normal(0, noise, arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        name = f"doc_{i:02d}"
        img.save(args.out / f"{name}.png")
        gt = {
            "image": f"{name}.png",
            "fictive": True,
            "width": img.width,
            "height": img.height,
            "variation": {"font": used_font, "rotation_deg": angle, "noise_sigma": noise, "blur_radius": blur},
            "words": words,
            "entities": card.entities,
        }
        (args.out / f"{name}.gt.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name}.png  font={used_font:<16} rot={angle:+.2f}°  noise={noise:<4} blur={blur}  entities={len(card.entities)}")
    print(f"\n{args.count} fictive documents written to {args.out}")


if __name__ == "__main__":
    main()
