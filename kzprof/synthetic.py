"""Synthetic training examples (TRAIN split only, marked source="synthetic").

Two kinds:
  * market:     a real product review from the train split with one real obscene
                word (taken from train positives) inserted. Real profanity is almost
                absent from KazSAnDRA product reviews (18 of 29k), so without this
                the model would only see profanity in app reviews.
  * obfuscated: a real train positive whose obscene words are rewritten the way
                people dodge filters: "*" masks, Latin transliteration, Russian
                keyboard letters, stretched letters.
  * phonetic:   (added in v1.1, scripts/augment_extra.py) a real train positive
                whose obscene words are written "as heard": voiced consonants
                devoiced (б→п, д→т, з→с, г→к, ж→ш, в→ф), unstressed vowels swapped
                (о→а, е→и), and sometimes glued to the next word ("ептвоемать").
  * caps:       (v1.1) an ALL-CAPS copy of a real train comment, profane or clean,
                so "ЕБАТЬ ОЙЫН" and "ҚОТАҚ" look like their lower-case forms.
Every synthetic example keeps the group id of the review it was built from, so it
can never leak into val/test.
"""

from __future__ import annotations

import random
import re

LATIN = {
    "а": "a", "ә": "a", "б": "b", "в": "v", "г": "g", "ғ": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "қ": "q", "л": "l", "м": "m", "н": "n",
    "ң": "n", "о": "o", "ө": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "u",
    "ү": "u", "ф": "f", "х": "h", "һ": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "",
    "ы": "y", "і": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
RU_KEYBOARD = {"қ": "к", "ғ": "г", "ң": "н", "ә": "а", "ө": "о", "ұ": "у", "ү": "у", "і": "и", "һ": "х"}
VOWELS = set("аәеёиоөуұүыіэюяaeiouy")


def to_latin(word: str) -> str:
    out = "".join(LATIN.get(ch.lower(), ch) for ch in word)
    return out.capitalize() if word[:1].isupper() else out


def mask(word: str, rng: random.Random) -> str:
    inner = [i for i in range(1, len(word) - 1) if word[i].lower() in VOWELS] or list(range(1, max(2, len(word) - 1)))
    i = rng.choice(inner)
    return word[:i] + "*" + word[i + 1:]


def stretch(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word))
    return word[: i + 1] + word[i] * rng.randint(2, 4) + word[i + 1:]


def ru_keyboard(word: str) -> str:
    return "".join(RU_KEYBOARD.get(ch, RU_KEYBOARD.get(ch.lower(), ch)) for ch in word)


def obfuscate(word: str, rng: random.Random) -> str:
    kind = rng.choice(["mask", "latin", "stretch", "ru_keyboard"])
    new = {"mask": lambda w: mask(w, rng), "latin": to_latin,
           "stretch": lambda w: stretch(w, rng), "ru_keyboard": ru_keyboard}[kind](word)
    return new if new != word else mask(word, rng)


def _replace_spans(text: str, spans: list[list[int]], fn) -> tuple[str, list[list[int]]]:
    out, new_spans, pos = [], [], 0
    for s, e in sorted(spans):
        out.append(text[pos:s])
        start = sum(len(p) for p in out)
        repl = fn(text[s:e])
        out.append(repl)
        new_spans.append([start, start + len(repl)])
        pos = e
    out.append(text[pos:])
    return "".join(out), new_spans


def _insert(text: str, word: str, rng: random.Random) -> tuple[str, list[list[int]]]:
    words = list(re.finditer(r"\S+", text))
    where = rng.choice(["start", "end", "middle"]) if len(words) > 2 else rng.choice(["start", "end"])
    if where == "start":
        sep = rng.choice([" ", ", ", "! "])
        new = word.capitalize() + sep + text[:1].lower() + text[1:]
        return new, [[0, len(word)]]
    if where == "end":
        sep = rng.choice([" ", ", ", ". "])
        base = text.rstrip(".!? ")
        new = base + sep + word + rng.choice(["", "!", "!!", "."])
        start = len(base) + len(sep)
        return new, [[start, start + len(word)]]
    cut = words[rng.randrange(1, len(words) - 1)].end()
    new = text[:cut] + " " + word + text[cut:]
    return new, [[cut + 1, cut + 1 + len(word)]]


def make_synthetic(train: list[dict], rng: random.Random, n_market: int, n_obfuscated: int) -> list[dict]:
    real_pos = [r for r in train if r["label"] == 1 and r["source"] == "kazsandra"]
    vocab = [r["text"][s:e] for r in real_pos for s, e in r["spans"]]
    market_neg = [r for r in train if r["label"] == 0 and r["domain"] == "market"]
    out = []

    for i, base in enumerate(rng.sample(market_neg, min(n_market, len(market_neg)))):
        word = rng.choice(vocab).lower()
        if rng.random() < 0.3:
            word = obfuscate(word, rng)
        text, spans = _insert(base["text"], word, rng)
        out.append({**base, "id": f"synth_market_{i}", "text": text, "spans": spans, "label": 1,
                    "source": "synthetic", "hard_negative": False})

    for i, base in enumerate(rng.sample(real_pos, min(n_obfuscated, len(real_pos)))):
        text, spans = _replace_spans(base["text"], base["spans"], lambda w: obfuscate(w, rng))
        out.append({**base, "id": f"synth_obf_{i}", "text": text, "spans": spans, "label": 1,
                    "source": "synthetic", "hard_negative": False})
    return out


# --- v1.1: "as heard" spellings --------------------------------------------------
SOUND = {"б": "п", "д": "т", "з": "с", "г": "к", "ж": "ш", "в": "ф", "о": "а", "е": "и"}


def phonetic(word: str, rng: random.Random) -> str:
    """Rewrite 1-2 letters the way the word sounds: заебал -> заепал, пиздец -> писдец."""
    pos = [i for i, ch in enumerate(word) if ch.lower() in SOUND]
    if not pos:
        return word
    chars = list(word)
    for i in rng.sample(pos, min(len(pos), rng.choice([1, 1, 2]))):
        new = SOUND[chars[i].lower()]
        chars[i] = new.upper() if chars[i].isupper() else new
    return "".join(chars)


def _glue_next(text: str, span: list[int]) -> tuple[str, list[int]] | None:
    """Glue the profane word to the next word: "еп твою" -> "ептвою"; the span covers both."""
    s, e = span
    m = re.match(r"\s+(\w+)", text[e:])
    if not m:
        return None
    glued = text[:e] + text[e + len(m.group(0)) - len(m.group(1)):]
    return glued, [s, e + len(m.group(1))]


def make_phonetic(train: list[dict], rng: random.Random, n: int, glue_rate: float = 0.3) -> list[dict]:
    real_pos = [r for r in train if r["label"] == 1 and r["source"] == "kazsandra"]
    out = []
    for base in rng.sample(real_pos, min(n, len(real_pos))):
        text, spans = _replace_spans(base["text"], base["spans"], lambda w: phonetic(w, rng))
        if rng.random() < glue_rate and len(spans) == 1:
            glued = _glue_next(text, spans[0])
            if glued:
                text, spans = glued[0], [glued[1]]
        if text == base["text"]:
            continue
        out.append({**base, "id": f"synth_phon_{len(out)}", "text": text, "spans": spans, "label": 1,
                    "source": "synthetic", "hard_negative": False})
    return out


def make_caps(train: list[dict], rng: random.Random, n_pos: int, n_neg: int) -> list[dict]:
    """ALL-CAPS copies of real comments (both classes), spans unchanged."""
    real = [r for r in train if r["source"] == "kazsandra"]
    pos = [r for r in real if r["label"] == 1]
    neg = [r for r in real if r["label"] == 0]
    out = []
    for base in rng.sample(pos, min(n_pos, len(pos))) + rng.sample(neg, min(n_neg, len(neg))):
        text = base["text"].upper()
        if len(text) != len(base["text"]) or text == base["text"]:
            continue
        out.append({**base, "id": f"synth_caps_{len(out)}", "text": text, "source": "synthetic"})
    return out
