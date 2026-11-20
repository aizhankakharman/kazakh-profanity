"""Synthetic training examples (TRAIN split only, marked source="synthetic").

Two kinds:
  * market:     a real product review from the train split with one real obscene
                word (taken from train positives) inserted. Real profanity is almost
                absent from KazSAnDRA product reviews (18 of 29k), so without this
                the model would only see profanity in app reviews.
  * obfuscated: a real train positive whose obscene words are rewritten the way
                people dodge filters: "*" masks, Latin transliteration, Russian
                keyboard letters, stretched letters.
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
