"""Bleep profanity in a video or audio file (ffmpeg + GigaAM ASR + the GLiNER model).

  python scripts/censor_video.py examples/sample.mp4     # -> examples/sample.censored.mp4 (+ .json report)
  python scripts/censor_video.py talk.mp3 -o talk.censored.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kzprof.inference import ProfanityDetector  # noqa: E402
from kzprof.video import Asr, censor_media  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--model", default=str(ROOT / "models" / "kazakh-profanity-gliner"))
    args = ap.parse_args()
    report = censor_media(ProfanityDetector(args.model), Asr(), args.input, args.output)
    print("transcript:", report["transcript"])
    print("censored:  ", report["censored_transcript"])
    print("beeps (s): ", report["beeps"])
    print("output:    ", report["output"])


if __name__ == "__main__":
    main()
