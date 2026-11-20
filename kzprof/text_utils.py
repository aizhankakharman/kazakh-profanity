"""Text helpers shared by data preparation, training and inference."""

from __future__ import annotations

import re
import unicodedata

# --- personal information ------------------------------------------------------
_PII_PATTERNS = [
    ("[URL]", re.compile(r"https?://\S+|www\.\S+", re.I)),
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("[CARD]", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
    ("[PHONE]", re.compile(r"(?:\+?[78][\s\-(]*)?\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b")),
    ("[ID]", re.compile(r"\b\d{12}\b")),                     # IIN
    ("[USER]", re.compile(r"(?<!\w)@[A-Za-z0-9_.]{2,}")),
]


def scrub_pii(text: str) -> str:
    """Replace URLs, e-mails, card/phone/ID numbers and @handles with placeholders."""
    for token, pat in _PII_PATTERNS:
        text = pat.sub(token, text)
    return text


def clean_text(text: str) -> str:
    """Unicode-normalise and squeeze whitespace. Punctuation and emoji are kept:
    they carry signal (masks like 'қ*тақ', angry emoji) and the model's
    tokenizer handles them."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "").replace("﻿", "")
    return re.sub(r"\s+", " ", text).strip()


def dedup_key(text: str) -> str:
    """Key for exact-duplicate detection: case, punctuation, emoji and letter
    repetitions do not make two reviews different."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"(.)\1+", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()
