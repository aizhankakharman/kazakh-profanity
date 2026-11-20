"""Predict on new Kazakh comments without retraining.

  python scripts/predict.py "Мына телефон қотақ екен, бір аптада сынды"
  python scripts/predict.py -i                 # interactive: type comments, empty line = exit
  python scripts/predict.py --file comments.txt

Output: label (0 clean / 1 profane), score (max span probability) and the words
that triggered it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kzprof.inference import ProfanityDetector  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "kazakh-profanity-gliner"


def show(res: dict) -> None:
    words = ", ".join(f"{s['text']} ({s['score']:.2f})" for s in res["spans"]) or "-"
    print(f"label={res['label']}  score={res['score']:.3f}  words: {words}\n  {res['text']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--file")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="local dir or Hugging Face repo id")
    ap.add_argument("--threshold", type=float, default=None, help="override decision.json")
    args = ap.parse_args()

    det = ProfanityDetector(args.model, threshold=args.threshold)
    print(f"model={args.model}  threshold={det.threshold}  device={det.device}")
    texts = [" ".join(args.text)] if args.text else []
    if args.file:
        texts += [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    for res in det.predict_batch(texts):
        show(res)
    if args.interactive:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            show(det.predict(line))


if __name__ == "__main__":
    main()
