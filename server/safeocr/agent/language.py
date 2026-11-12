"""Answer-language handling for the chat agent.

The chat is international: the model replies in the language of the user's message.
Small models mirror the (English) system prompt instead, so we detect the language and
state it explicitly every turn. Romanian (the project's main language) gets hand-written
rules; any other language gets the English rules plus "reply in <language>".
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0  # deterministic results
except ImportError:  # pragma: no cover - optional dependency
    detect_langs = None

LANG_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan", "cs": "Czech",
    "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French", "gu": "Gujarati", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian", "ml": "Malayalam",
    "mr": "Marathi", "ne": "Nepali", "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sl": "Slovenian", "so": "Somali",
    "sq": "Albanian", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
    "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)",
}

RO_LETTERS = re.compile(r"[ășşțţ]", re.IGNORECASE)  # letters specific to Romanian
RO_WORDS = re.compile(
    r"\b(si|și|este|sunt|te rog|acum|doar|care|cum|pentru|ce|imi|îmi|poti|poți|datele|acest|aceasta|"
    r"buletin\w*|anonimiz\w*|acoper\w*|protejez\w*|vreau|nu|da|multumesc|mulțumesc)\b",
    re.IGNORECASE,
)
EN_WORDS = re.compile(  # langdetect sometimes mislabels plain English (e.g. as Catalan)
    r"\b(what|can|you|the|how|my|your|please|this|does|are|with|which|only|about|would|could|should|there|"
    r"hello|thanks|thank)\b",
    re.IGNORECASE,
)
# The "no image attached yet" instruction is only shown for anonymization requests, otherwise a
# small model answers every question with "please upload the card".
ANONYMIZE_INTENT = re.compile(
    r"anonym|anonim|аноним|анонім|redact|mask|acoper|ascun|cover|hide|verberg|schwärz|cacher|masquer|ocultar|"
    r"tachar|nascond|oscur|ukry|zamaz",
    re.IGNORECASE,
)
ATTACHMENT_PREFIX = re.compile(r"^\[(?:Attached image|Imagine atașată): [^\]]*\]\s*")  # older chats used RO


def _detect_one(text: str) -> Optional[str]:
    """Language code for one message, or None when not confident."""
    text = ATTACHMENT_PREFIX.sub("", text or "").strip()
    if not text:
        return None
    if RO_LETTERS.search(text):
        return "ro"
    ro_hits, en_hits = len(RO_WORDS.findall(text)), len(EN_WORDS.findall(text))
    if en_hits >= 2 and ro_hits < 2:
        return "en"
    words = len(text.split())
    best = None
    if detect_langs is not None:
        try:
            best = detect_langs(text)[0]
        except LangDetectException:
            best = None
    if best and best.prob >= 0.9 and words >= 3:
        return best.lang
    if ro_hits >= 2:
        return "ro"
    if best and best.prob >= 0.8 and words >= 3:
        return best.lang
    return None


def detect_language(text: str, previous_user_messages: Iterable[str] = ()) -> str:
    """Detect the user's language; fall back to their earlier messages, then English."""
    found = _detect_one(text)
    if found:
        return found
    for prev in reversed(list(previous_user_messages)):
        found = _detect_one(prev)
        if found:
            return found
    return "en"


def language_name(code: str) -> str:
    return LANG_NAMES.get(code, "the same language as the user's message")


def summary_language(code: str) -> str:
    """Language of the pre-built pipeline summary: Romanian or English (the model translates others)."""
    return "ro" if code == "ro" else "en"


# --- prompt fragments ------------------------------------------------------------------------

_PRIVACY_FACTS_RO = """Cum îți protejează SafeOCR confidențialitatea (folosește aceste fapte când ești întrebat):
- Totul rulează local, pe calculatorul tău: OCR-ul, detecția, mascarea și acest model (Ollama). Nimic nu este trimis în cloud.
- Asistentul nu vede niciodată textul documentului tău sau datele tale personale, ci doar rapoarte: ce tipuri de date există, câte sunt și dacă verificarea a reușit.
- Imaginea originală rămâne în stocarea locală. Rezultatul este o imagine NOUĂ, cu dreptunghiuri negre peste CNP, nume, adresă, seria și numărul actului și zona MRZ.
- După mascare, OCR-ul rulează din nou pe imaginea nouă și confirmă că nu a mai rămas nicio dată personală lizibilă."""

_PRIVACY_FACTS_EN = """How SafeOCR protects the user's privacy (use these facts when asked):
- Everything runs locally on the user's computer: OCR, detection, redaction and this model (Ollama). Nothing is sent to the cloud.
- The assistant never sees the document's text or any personal data, only reports: which data types exist, how many, and whether verification passed.
- The original image stays in the local store. The result is a NEW image with black boxes over the CNP, name, address, ID series and number, and the MRZ.
- After redaction, OCR runs again on the new image to confirm that no personal data is still readable."""

PRIVACY_TOPIC = re.compile(
    r"confiden|privac|privé|privat|protej|protect|protég|schütz|cloud|sigur|secur|sûr|sicher|original|stoc|store|"
    r"local|vede|see|trimi|send|server|gdpr|rgpd|dsgvo|datos|données|daten|dane|данн",
    re.IGNORECASE,
)


def privacy_facts(code: str) -> str:
    return _PRIVACY_FACTS_RO if code == "ro" else _PRIVACY_FACTS_EN


def language_rule(code: str) -> str:
    if code == "ro":
        return (
            "IMPORTANT: Răspunde DOAR în limba română, corect gramatical. Despre utilizator vorbești la persoana a doua "
            "(«datele tale», «confidențialitatea ta»); despre tine, la persoana întâi («pot detecta», «am acoperit»). "
            "Spune doar ce reiese din rapoarte; nu adăuga afirmații generale."
        )
    name = language_name(code)
    rule = (
        f"IMPORTANT: Reply ONLY in {name}, speaking directly to the user («you», «your data»); about yourself use "
        "the first person («I can detect»). Only state what the reports show; do not add generic claims."
    )
    if code != "en":  # small models drift back to English (the prompt language) unless told firmly
        rule += (
            f" The user writes in {name}: your WHOLE answer must be in {name}, not English. "
            f"Translate data type names into {name} too."
        )
    return rule


def no_image_state(code: str) -> str:
    if code == "ro":
        return (
            "STARE: În această conversație NU a fost atașată nicio imagine. Dacă utilizatorul cere anonimizarea unui act, "
            "răspunde exact: «Nu am primit încă nicio imagine. Atașează imaginea actului (butonul agrafă, drag & drop sau "
            "lipire) și o anonimizez imediat.»"
        )
    sentence = (
        "I haven't received an image yet. Attach the ID card image (paperclip button, drag & drop or paste) "
        "and I'll anonymize it."
    )
    if code == "en":
        return f"STATE: NO image has been attached in this conversation. If the user asks to anonymize a document, reply exactly: «{sentence}»"
    return (
        "STATE: NO image has been attached in this conversation. If the user asks to anonymize a document, reply with "
        f"this sentence translated into {language_name(code)}: «{sentence}»"
    )
