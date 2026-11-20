"""Build the Kazakh profanity dataset from KazSAnDRA.

Steps (each is explained in README, "Dataset"):
  1. download KazSAnDRA (CC BY 4.0) from GitHub if it is not in data/raw/
  2. clean text, replace personal information with placeholders
  3. pre-label with the lexicon + manual word decisions  -> label and spans
  4. sample negatives: product (market) reviews, 1-2 star reviews and hard
     negatives (rude-but-not-obscene words, look-alike words)
  5. drop exact duplicates; group near-duplicates (MinHash, Jaccard >= 0.7)
  6. split train/val/test 70/15/15 by group, stratified by label, seed 42
  7. add synthetic examples to TRAIN ONLY (val/test stay 100% real)
  8. mask person names inside the texts with the base GLiNER model (optional)

Usage:  python scripts/prepare_data.py [--no-name-masking]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kzprof.lexicon import find_candidates, obscene_spans  # noqa: E402
from kzprof.synthetic import make_synthetic                 # noqa: E402
from kzprof.text_utils import clean_text, dedup_key, scrub_pii  # noqa: E402

RAW = ROOT / "data" / "raw" / "kazsandra"
OUT = ROOT / "data" / "processed"
KAZSANDRA_URL = "https://raw.githubusercontent.com/IS2AI/KazSAnDRA/main/dataset/{}.zip"
KAZSANDRA_FILES = ["06_sc_train_ib", "09_sc_valid", "10_sc_test"]  # all 175k reviews, 1-5 stars

SEED = 42
N_NEG = {"market": 3000, "appstore": 2600, "mapping": 700, "bookstore": 400}
N_HARD = 1500
N_SYNTH_MARKET = 500      # synthetic positives built from real product reviews (train only)
N_SYNTH_OBFUSCATED = 400  # obfuscated copies of real train positives (train only)
MAX_WORDS = 120           # longer reviews (<0.5%) are dropped from training data

# Words that are rude or look like profanity but are NOT obscene under our rules.
HARD_NEGATIVE_RE = re.compile(
    r"\b(?:боқ|бок|мал|малдар|ақымақ|акымак|есек|ешак|ишак|қатын|катын|сасық|сасык|лох|"
    r"говно|гавно|тупой|дурак|идиот|дебил|блин|капец|далбаса|далбан|шешең|шешен|секс|"
    r"көтер\w*|котер\w*|сигнал\w*|ебі\w*|ебедей\w*|сиқыр\w*|жапырақ\w*|жаппай\w*|коты|код\w*|"
    r"скин\w*|сығ\w*|сықыр\w*|амал\w*|аман\w*|жалақ\w*|сұмдық|жаман)\b",
    re.I,
)


def download_kazsandra() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in KAZSANDRA_FILES:
        if (RAW / f"{name}.csv").exists():
            continue
        print(f"downloading {name} ...")
        data = urllib.request.urlopen(KAZSANDRA_URL.format(name), timeout=120).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(RAW)


def load_reviews() -> list[dict]:
    rows = []
    for name in KAZSANDRA_FILES:
        with open(RAW / f"{name}.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                text = scrub_pii(clean_text(r["text"]))
                if not re.search(r"[^\W\d_]", text):
                    continue
                rows.append({
                    "id": f"kazsandra_{r['custom_id']}",
                    "text": text,
                    "domain": r["domain"],
                    "stars": int(r["label"]) + 1,   # SC labels are 0..4
                    "source": "kazsandra",
                })
    return rows


OVERRIDES_CSV = ROOT / "data" / "review" / "example_overrides.csv"


def load_overrides() -> dict[str, int]:
    """Whole-comment decisions where context flips the word-level rule."""
    with OVERRIDES_CSV.open(encoding="utf-8") as fh:
        return {r["id"]: int(r["label"]) for r in csv.DictReader(fh)}


def label_rows(rows: list[dict]) -> None:
    overrides = load_overrides()
    for r in rows:
        spans = obscene_spans(r["text"])
        if overrides.get(r["id"]) == 0:
            spans = []
        r["spans"] = [list(s) for s in spans]
        r["label"] = int(bool(spans))
        r["hard_negative"] = bool(not spans and HARD_NEGATIVE_RE.search(r["text"]))


def sample(rows: list[dict], rng: random.Random) -> list[dict]:
    rows = [r for r in rows if len(r["text"].split()) <= MAX_WORDS]
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    hard = [r for r in neg if r["hard_negative"]]
    chosen = rng.sample(hard, min(N_HARD, len(hard)))
    taken = {r["id"] for r in chosen}
    for domain, n in N_NEG.items():
        pool = [r for r in neg if r["domain"] == domain and r["id"] not in taken]
        low = [r for r in pool if r["stars"] <= 2]   # half of each quota from 1-2 star reviews
        high = [r for r in pool if r["stars"] > 2]
        part = rng.sample(low, min(n // 2, len(low)))
        part += rng.sample(high, min(n - len(part), len(high)))
        chosen += part
        taken |= {r["id"] for r in part}
    return pos + chosen


def drop_exact_duplicates(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        k = dedup_key(r["text"])
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def near_duplicate_groups(rows: list[dict], threshold: float = 0.7) -> list[int]:
    """Union-find over MinHash-LSH candidates on character 3-grams."""
    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    hashes = []
    for i, r in enumerate(rows):
        k = dedup_key(r["text"])
        m = MinHash(num_perm=64, seed=SEED)
        for g in {k[j:j + 3] for j in range(max(1, len(k) - 2))}:
            m.update(g.encode("utf-8"))
        hashes.append(m)
        lsh.insert(str(i), m)
    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, m in enumerate(hashes):
        for j in lsh.query(m):
            a, b = find(i), find(int(j))
            if a != b:
                parent[a] = b
    # Canonical id = smallest member index. The root that union-find ends up with
    # depends on LSH query order (i.e. on PYTHONHASHSEED); the smallest index does not,
    # so the split is identical on every run.
    roots = [find(i) for i in range(len(rows))]
    smallest: dict[int, int] = {}
    for i, r in enumerate(roots):
        smallest.setdefault(r, i)
    return [smallest[r] for r in roots]


def split(rows: list[dict], groups: list[int]) -> dict[str, list[dict]]:
    """70/15/15 split that keeps every near-duplicate group inside one split."""
    y = np.array([r["label"] for r in rows])
    g = np.array(groups)
    idx = np.arange(len(rows))
    # 1st cut: test = 1 of 7 folds (~14-15%)
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    rest, test = next(outer.split(idx, y, g))
    # 2nd cut on the rest: val = 1 of 6 folds (~15% of the total)
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
    tr, va = next(inner.split(rest, y[rest], g[rest]))
    parts = {"train": rest[tr], "val": rest[va], "test": test}
    return {name: [rows[i] for i in ids] for name, ids in parts.items()}


# Capitalised words the zero-shot "person" label tends to grab that are not names.
NOT_A_NAME = re.compile(
    r"^(?:шеше|шешен|шешең|ана|әке|аке|ата|аға|апа|ене|енең|алла|аллах|машалла|иншалла|құдай|кудай|"
    r"қазақ|казак|адам|мен|сен|біз|сіз|ол|бұл|осы|бала|қыз|жігіт|каспи|kaspi|халық|халык|казком|"
    r"мына|анау|бірақ|рахмет|сәлем|салем|ассалам)",
    re.I,
)


def mask_person_names(splits: dict[str, list[dict]]) -> int:
    """Replace person names with [NAME] using zero-shot GLiNER (base model).
    Obscene spans are never touched; spans are shifted after each replacement."""
    from gliner import GLiNER
    import torch

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to(device)
    n = 0
    for rows in splits.values():
        texts = [r["text"] for r in rows]
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            preds = model.batch_predict_entities(batch, ["person"], threshold=0.7)
            for r, ents in zip(rows[start:start + 32], preds):
                for e in sorted(ents, key=lambda e: -e["start"]):
                    s, t = e["start"], e["end"]
                    if any(not (t <= a or s >= b) for a, b in r["spans"]):
                        continue
                    if not re.search(r"[A-ZА-ЯӘҒҚҢӨҰҮҺІЁ]", r["text"][s:t]) or NOT_A_NAME.match(r["text"][s:t]):
                        continue   # real names are capitalised; kinship words, 'Алла', banks are not names
                    shift = len("[NAME]") - (t - s)
                    r["text"] = r["text"][:s] + "[NAME]" + r["text"][t:]
                    r["spans"] = [[a + shift, b + shift] if a >= t else [a, b] for a, b in r["spans"]]
                    n += 1
    return n


def write_jsonl(path: Path, rows: list[dict]) -> None:
    keep = ["id", "text", "label", "spans", "domain", "stars", "source", "group", "hard_negative"]
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-name-masking", action="store_true")
    ap.add_argument("--out", default=str(OUT), help="output folder (default data/processed)")
    args = ap.parse_args()
    rng = random.Random(SEED)

    download_kazsandra()
    rows = load_reviews()
    label_rows(rows)
    print(f"loaded {len(rows)} reviews, lexicon positives: {sum(r['label'] for r in rows)}")

    rows = sample(rows, rng)
    n_sampled = len(rows)
    rows = drop_exact_duplicates(rows)
    groups = near_duplicate_groups(rows)
    for r, gid in zip(rows, groups):
        r["group"] = f"g{gid}"
    splits = split(rows, groups)

    synth = make_synthetic(splits["train"], rng, n_market=N_SYNTH_MARKET, n_obfuscated=N_SYNTH_OBFUSCATED)
    splits["train"] += synth
    rng.shuffle(splits["train"])

    n_names = 0 if args.no_name_masking else mask_person_names(splits)

    # sanity: spans must still point at the words we labelled
    for rows_ in splits.values():
        for r in rows_:
            assert r["label"] == int(bool(r["spans"])), r
            if r["source"] == "kazsandra" and r["label"]:
                for s, e in r["spans"]:
                    assert find_candidates(r["text"][s:e]), (r["text"], r["text"][s:e])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"seed": SEED, "sampled": n_sampled, "after_exact_dedup": len(rows),
             "near_duplicate_groups": len(set(groups)), "names_masked": n_names, "splits": {}}
    for name, rows_ in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows_)
        stats["splits"][name] = {
            "total": len(rows_),
            "profane": sum(r["label"] for r in rows_),
            "clean": sum(1 - r["label"] for r in rows_),
            "synthetic": sum(r["source"] == "synthetic" for r in rows_),
            "hard_negatives": sum(bool(r["hard_negative"]) for r in rows_),
            "by_domain": dict(Counter(r["domain"] for r in rows_)),
            "profane_by_domain": dict(Counter(r["domain"] for r in rows_ if r["label"])),
        }
    # no group may appear in two splits
    gs = {n: {r["group"] for r in rows_} for n, rows_ in splits.items()}
    assert not (gs["train"] & gs["val"] or gs["train"] & gs["test"] or gs["val"] & gs["test"])
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
