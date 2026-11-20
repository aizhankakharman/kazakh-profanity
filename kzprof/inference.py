"""Load the fine-tuned model and predict on new text.

How one comment becomes a prediction:
  1. the text is split into words (GLiNER's word splitter, punctuation and "*"
     become separate tokens);
  2. GLiNER builds the prompt  [ENT] profanity [SEP] w1 w2 ... wn , runs mDeBERTa
     and scores every span of up to 12 words against the "profanity" label
     (sigmoid -> probability);
  3. spans with probability >= threshold are the profane words (used for
     redaction / beeping);
  4. comment score = the highest span probability (0 if there is none);
     label = 1 if score >= threshold. The threshold is chosen on the
     validation set (results/decision.json).
Long texts are cut into overlapping windows of words so nothing is truncated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

LABEL = "profanity"
SCORE_FLOOR = 0.01          # spans below this are not returned at all
WINDOW_WORDS = 200          # GLiNER sees at most ~384 subword tokens; reviews are short
WINDOW_OVERLAP = 20


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Span:
    start: int
    end: int
    text: str
    score: float


def _windows(text: str) -> list[tuple[int, str]]:
    words = list(re.finditer(r"\S+", text))
    if len(words) <= WINDOW_WORDS:
        return [(0, text)]
    out, step = [], WINDOW_WORDS - WINDOW_OVERLAP
    for i in range(0, len(words), step):
        chunk = words[i:i + WINDOW_WORDS]
        s, e = chunk[0].start(), chunk[-1].end()
        out.append((s, text[s:e]))
        if i + WINDOW_WORDS >= len(words):
            break
    return out


class ProfanityDetector:
    def __init__(self, model_dir: str | Path, threshold: float | None = None, device: str | None = None):
        from gliner import GLiNER

        model_dir = Path(model_dir)
        self.device = device or pick_device()
        self.model = GLiNER.from_pretrained(str(model_dir), local_files_only=model_dir.exists())
        self.model.to(self.device)
        self.model.eval()
        if threshold is None:
            if model_dir.exists():                      # local folder
                decision = model_dir / "decision.json"
                threshold = json.loads(decision.read_text())["threshold"] if decision.exists() else 0.5
            else:                                       # Hugging Face repo id
                from huggingface_hub import hf_hub_download
                try:
                    path = hf_hub_download(str(model_dir), "decision.json")
                    threshold = json.loads(Path(path).read_text())["threshold"]
                except Exception:
                    threshold = 0.5
        self.threshold = float(threshold)

    def spans(self, text: str, min_score: float | None = None) -> list[Span]:
        """Profane spans with probability >= min_score (default: the decision threshold)."""
        return self.spans_batch([text], min_score)[0]

    def spans_batch(self, texts: list[str], min_score: float | None = None, batch_size: int = 16) -> list[list[Span]]:
        floor = self.threshold if min_score is None else min_score
        jobs = [(ti, off, chunk) for ti, t in enumerate(texts) for off, chunk in _windows(t) if chunk.strip()]
        found: list[dict[tuple[int, int], Span]] = [dict() for _ in texts]
        for b in range(0, len(jobs), batch_size):
            part = jobs[b:b + batch_size]
            preds = self.model.batch_predict_entities([c for _, _, c in part], [LABEL], threshold=max(floor, SCORE_FLOOR))
            for (ti, off, _), ents in zip(part, preds):
                for e in ents:
                    s, t = e["start"] + off, e["end"] + off
                    old = found[ti].get((s, t))
                    if old is None or e["score"] > old.score:
                        found[ti][(s, t)] = Span(s, t, texts[ti][s:t], float(e["score"]))
        return [sorted(d.values(), key=lambda sp: sp.start) for d in found]

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        out = []
        for text, spans in zip(texts, self.spans_batch(texts, min_score=SCORE_FLOOR)):
            score = max((sp.score for sp in spans), default=0.0)
            out.append({
                "text": text,
                "label": int(score >= self.threshold),
                "score": round(score, 4),
                "spans": [sp.__dict__ for sp in spans if sp.score >= self.threshold],
            })
        return out
