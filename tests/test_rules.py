from datetime import date

from safeocr.ocr.base import OCRResult, Word
from safeocr.pii import detect, make_cnp

ALL = ["CNP", "NAME", "ADDRESS", "ID_SERIES", "ID_NUMBER", "MRZ"]


def ocr_from_lines(lines):
    words, y = [], 0
    for line_id, text in enumerate(lines):
        x = 0
        for tok in text.split():
            words.append(Word(tok, (x, y, x + 10 * len(tok), y + 20), line_id))
            x += 10 * len(tok) + 8
        y += 30
    return OCRResult(width=1000, height=y, backend="test", words=words)


def types(dets):
    return sorted(d.type for d in dets)


def test_full_card():
    cnp = make_cnp(1, date(1985, 3, 12), 40, 123)
    ocr = ocr_from_lines([
        "ROMANIA ROUMANIE ROMANIA",
        "SERIA RX NR 123456",
        f"CNP {cnp}",
        "Nume/Nom/Last name",
        "POPESCU",
        "Prenume/Prénom/First name",
        "ANDREI-MIHAI",
        "Cetățenie/Nationalité/Nationality",
        "Română / ROU",
        "Domiciliu/Adresse/Address",
        "Mun. Bucuresti Sec. 3",
        "Str. Lalelelor nr. 12 ap. 4",
        "Emisă de/Délivrée par/Issued by",
        "SPCLEP Bucuresti",
        "IDROUPOPESCU<<ANDREI<MIHAI<<<<<<<<<<",
        "RX123456<1ROU8503124M300312<1234567",
    ])
    dets = detect(ocr, ALL)
    assert types(dets) == sorted(["ID_SERIES", "ID_NUMBER", "CNP", "NAME", "NAME", "ADDRESS", "MRZ", "MRZ"])
    address = next(d for d in dets if d.type == "ADDRESS")
    assert len(address.boxes) == 10  # both address lines, one entity


def test_redacted_card_has_no_false_positives():
    # after redaction the values are gone; labels must not be mistaken for values
    ocr = ocr_from_lines([
        "SERIA NR",
        "CNP",
        "Nume/Nom/Last name",
        "Prenume/Prénom/First name",
        "Cetățenie/Nationalité/Nationality",
        "Română / ROU",
        "Domiciliu/Adresse/Address",
        "Emisă de/Délivrée par/Issued by",
    ])
    assert detect(ocr, ALL) == []


def test_invalid_cnp_is_ignored_and_types_can_be_disabled():
    ocr = ocr_from_lines(["CNP 1234567890123", "SERIA RX NR 123456"])
    assert types(detect(ocr, ALL)) == ["ID_NUMBER", "ID_SERIES"]
    assert types(detect(ocr, ["ID_NUMBER"])) == ["ID_NUMBER"]
