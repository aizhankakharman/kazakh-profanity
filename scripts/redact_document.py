"""Black out profanity in a PDF / DOCX / TXT file.

  python scripts/redact_document.py examples/sample.pdf            # -> examples/sample.redacted.pdf
  python scripts/redact_document.py in.docx -o out.docx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kzprof.inference import ProfanityDetector  # noqa: E402
from kzprof.redact import redact_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--model", default=str(ROOT / "models" / "kazakh-profanity-gliner"))
    args = ap.parse_args()
    det = ProfanityDetector(args.model)
    dst, n = redact_file(det, args.input, args.output)
    print(f"redacted {n} profane span(s) -> {dst}")


if __name__ == "__main__":
    main()
