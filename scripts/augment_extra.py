"""v1.1: add extra synthetic examples to the TRAIN split (val/test untouched).

  * phonetic: 600 real positives rewritten "as heard" (заебал -> заепал, glued words)
  * caps:     150 profane + 150 clean real comments copied in ALL CAPS

Run after scripts/prepare_data.py:
    python scripts/augment_extra.py               # -> data/processed/train.jsonl (+ stats.json)

Idempotent: earlier synth_phon_* / synth_caps_* rows are dropped before new ones are added,
so running it twice gives the same file. Seed 42.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kzprof.synthetic import make_caps, make_phonetic  # noqa: E402

SEED = 42
N_PHONETIC = 600
N_CAPS_POS = 150
N_CAPS_NEG = 150
DATA = ROOT / "data" / "processed"


def main() -> None:
    path = DATA / "train.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if not r["id"].startswith(("synth_phon_", "synth_caps_"))]
    rng = random.Random(SEED)
    base = sorted(rows, key=lambda r: r["id"])            # fixed order -> same sample on every run
    new = make_phonetic(base, rng, n=N_PHONETIC)
    caps = make_caps(base, rng, n_pos=N_CAPS_POS, n_neg=N_CAPS_NEG)
    rows = base + new + caps
    rng.shuffle(rows)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    stats_path = DATA / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    tr = stats["splits"]["train"]
    tr["total"] = len(rows)
    tr["profane"] = sum(r["label"] == 1 for r in rows)
    tr["clean"] = sum(r["label"] == 0 for r in rows)
    tr["synthetic"] = sum(r["source"] == "synthetic" for r in rows)
    tr["synthetic_phonetic"] = len(new)
    tr["synthetic_caps"] = len(caps)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"added {len(new)} phonetic + {len(caps)} caps examples -> train {len(rows)} rows")
    for r in new[:8]:
        s, e = r["spans"][0]
        print("  ", r["text"][max(0, s - 15):e + 15].replace("\n", " "), "|", r["text"][s:e])


if __name__ == "__main__":
    main()
