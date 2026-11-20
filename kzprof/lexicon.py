"""Rule-based pre-labelling of Kazakh profanity (used ONLY to build the dataset).

The trained GLiNER model does not use these rules at inference time. They propose
labels and character spans for real reviews; word forms the rules cannot decide
(mostly "*"-masked words) are decided by hand in data/review/word_decisions.csv.

Labelling rule (README, "Labelling rules"):
  1 = the comment contains an OBSCENE word: genitals / sexual acts / anus
      vocabulary and its derivatives, or a sexual slur. Kazakh words (қотақ, ам,
      сік-/сіг-, көт, жәлеп, сайқал, қаншық) and Russian mat used inside a Kazakh
      or mixed comment (хуй, пизд-, еб-, бля, сука, долбоёб, ...) both count,
      because code-switching is normal in Kazakh reviews.
  0 = everything else, including rude-but-not-obscene words (боқ, ақымақ, мал,
      есек, қатын, сасық, лох, говно, блин, далбаса) and plain negative reviews.

Matching is per word. The word is lower-cased and runs of repeated letters are
collapsed ("қотаааақ" -> "қотақ") before the patterns run. Letter classes cover the
usual substitutions: Kazakh letters typed on a Russian keyboard (қ->к, ғ->г,
і->и/ы, ә->а), Latin transliteration and "*" masks.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

WORD_RE = re.compile(r"[\w*]+", re.UNICODE)
DECISIONS_CSV = Path(__file__).resolve().parent.parent / "data" / "review" / "word_decisions.csv"

# Ordinary words that the obscene patterns below would otherwise catch.
NOT_OBSCENE = re.compile(
    r"^(?:"
    r"амир\w*|амин\w*|ами|амиго|амия\w*|амилия|амиран"           # names, 'amen'
    r"|сигнал\w*|сигма\w*|сигар\w*|сикрет\w*|сикунд\w*|сикунт\w*|сикл\w*|сиквел|сикх\w*|сике|сигек"
    r"|скин\w*|сукно|бляшк\w*|мандарин\w*|педикюр\w*|хернандес|huyndai|huirshah"
    r"|котер\w*|котед\w*|котир\w*|котик\w*|котел\w*|котыр\w*|котлет\w*|кот|коты|котын|коти|котим\w*|котым|коте"
    r"|далбаса\w*|далбан\w*|далба|далбаев"                         # euphemisms -> 0
    r"|еби|ебин|ебі|ебін|ебіне|ебини|ебедей\w*|ебебі"             # Kazakh 'ебі' = knack
    r")$"
)

# Word-start patterns (re.match on the collapsed word) -> obscene.
OBSCENE_START = [
    # --- Kazakh ---------------------------------------------------------------
    ("qotaq", r"[қк][о*][тд][а*][қкғг]"),             # қотақ, котак, қотағым, котакбас
    ("qotaq_lat", r"[qk]ota[qkg]"),                    # qotaq, kotak
    ("am", r"ам(?:$|[ыі*]|и(?:[нң]|м|с|ң?ды))"),       # ам, амы, амың, амыңды, амыс(ігейін)
    ("am_compound", r"ам(?:жала|бас|пас|дар|шы[қк])"),  # амжалақ, амбас, амдар, амшық
    ("sik", r"с[іи*][кг](?!нал|ма|ар|рет|унд|унт|л|ир|ер|вел|х|натур|ански)"),  # сік-, сіг-, сигейн
    ("sik_y", r"сы[гк](?!ыр|тывкар|ык$)"),             # сыгейн (ы typed for і); сығу/сықыр use ғ/қ
    ("sik_novowel", r"с{1,2}[г](?:ин|ейн|ейін)"),       # сгин, сгейн
    ("ru_lat", r"s[uy]ka|blya|blyat|pizd|ebat|dalba[iy]?[oe]b|dolboeb|nahuy"),
    ("sik_lat", r"s[iy][kg](?:ey|ei|il|ip|ti|ken|pe|is|ir)"),
    ("kot", r"көт(?!е(?!к)|ір)"),                      # көт, көтің, көтіңді, көтек; not көтеру, көтен
    ("kot_ru", r"кот(?:ине|инди|ынды|ыңды|ыне|инде|ек)"),  # көт typed with о (+ possessive)
    ("zhalep", r"ж[әа]л[әае]п"),                        # жәлеп, жалап(тар)
    ("saiqal", r"са[йи][қк]ал"),                       # сайқал
    ("qanshyq", r"[қк]аншы[қк]"),                      # қаншық
    # --- Russian mat inside Kazakh / mixed comments ---------------------------
    ("ru_eb", r"(?:за|вы|на|у|до|от|по|про|раз|из|под|при|ъ)?[её]б(?:а|ал|ан|ат|ай|аи|ну|ут|уч|ыр|ок|л|ись)"),
    ("ru_eb_mat", r"[еёя]пт?ва[йяию]"),                 # ептвая, яптваймай (ёб твою)
    ("ru_zaebal", r"за[йи]б[аи]л"),                     # зайбал (заебал)
    ("ru_dolboeb", r"д[ао]лб[ао](?:[её]|[йи]о|ю|йё)[бп]"),  # далбаёб, далбайоп
    ("ru_nahu", r"наху$"),
    ("ru_blya", r"(?:бля|бл[яәə](?:т|д|$)|б[іи]л[әяə](?:т|$)|бл\*)"),  # бля, блә, біләт, блə
    ("ru_suka", r"с[уy]к[аиеу](?!н)|сучк|сучар"),
    ("ru_mudak", r"муда[кч]|мудил"),
    ("ru_pidor", r"пид[оа]р|пидр|пидар|педик"),
    ("ru_gandon", r"(?:ка)?г[ао]ндон"),
    ("ru_shlyukha", r"шлюх"),
    ("ru_zalupa", r"залуп"),
    ("ru_zhopa", r"жоп[аеуы]"),
    ("ru_her", r"(?:на|по)?хер(?:$|ов|ня|н)"),
]

# Patterns that may occur anywhere inside a word (compounds such as "нихуя").
OBSCENE_ANYWHERE = [
    ("ru_hui", r"ху[йеёяию](?!д)"),
    ("ru_hui_lat", r"(?:^|[^a-z])(?:huy|hui|xuy|xyu)"),
    ("ru_pizd", r"п[иеі*]?[зс]д|пзд"),
    ("ru_eb_mid", r"(?:долбо|за|вы|на)[её]б(?:а|ан|ал|ат|ок|ищ)"),
]

# Anything with a "*" that the rules did not already decide goes to a human.
REVIEW_START = [("mask", r"\w*\*\w*")]


@dataclass(frozen=True)
class WordMatch:
    start: int
    end: int
    word: str
    rule: str


def collapse(word: str) -> str:
    """Lower-case and collapse runs of the same character (қотааақ -> қотақ)."""
    return re.sub(r"(.)\1+", r"\1", word.lower())


@lru_cache(maxsize=1)
def manual_decisions() -> dict[str, int]:
    """word (lower-case) -> 1 obscene / 0 not obscene, decided by hand."""
    if not DECISIONS_CSV.exists():
        return {}
    with DECISIONS_CSV.open(encoding="utf-8") as fh:
        return {r["word"].lower(): int(r["obscene"]) for r in csv.DictReader(fh)}


def classify_word(word: str) -> tuple[str | None, str | None]:
    """Return (decision, rule): decision is 'obscene', 'review' or None."""
    lw = word.lower()
    manual = manual_decisions().get(lw)
    if manual is not None:
        return ("obscene", "manual") if manual else (None, None)
    if not re.search(r"[^\W\d_]", lw):          # digits / masks only ("5*", "***")
        return None, None
    c = collapse(word)
    if NOT_OBSCENE.match(c) or NOT_OBSCENE.match(lw):
        return None, None
    if re.match(r"көтт", lw):                    # plural көттер: check before collapsing "тт"
        return "obscene", "kot_plural"
    for name, pat in OBSCENE_START:
        if re.match(pat, c):
            return "obscene", name
    for name, pat in OBSCENE_ANYWHERE:
        if re.search(pat, c):
            return "obscene", name
    for name, pat in REVIEW_START:
        if re.match(pat, c):
            return "review", name
    return None, None


def find_candidates(text: str) -> list[tuple[WordMatch, str]]:
    """Words in `text` that the rules mark as obscene or send for review."""
    out = []
    for m in WORD_RE.finditer(text):
        decision, rule = classify_word(m.group(0))
        if decision:
            out.append((WordMatch(m.start(), m.end(), m.group(0), rule), decision))
    return out


def obscene_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of obscene words (review items count as not decided -> skipped)."""
    return [(m.start, m.end) for m, d in find_candidates(text) if d == "obscene"]
