"""Bleep profanity in video/audio: ffmpeg -> GigaAM ASR -> GLiNER -> ffmpeg beep.

  1. ffmpeg extracts the audio track as 16 kHz mono WAV;
  2. GigaAM-Multilingual (CTC, supports Kazakh) transcribes it with word
     timestamps; audio longer than ~22 s is cut at quiet moments first;
  3. the transcript goes through the same ProfanityDetector as the text task;
     every recognised word that overlaps a profane span gives a time interval;
  4. ffmpeg mutes the original audio inside those intervals and mixes in a
     1 kHz sine beep there; the video stream is copied without re-encoding.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .inference import ProfanityDetector

SR = 16000
MAX_CHUNK_S = 22.0        # GigaAM.transcribe refuses > 25 s
PAD_BEFORE, PAD_AFTER = 0.08, 0.12   # CTC timestamps are peaky; widen a little
ASR_REPO, ASR_REVISION = "ai-sage/GigaAM-Multilingual", "ctc"


@dataclass
class TimedWord:
    text: str
    start: float
    end: float
    profane: bool = False
    score: float = 0.0


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def has_video(path: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_type",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return "video" in out


def extract_audio(src: Path, wav: Path) -> None:
    ffmpeg("-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav))


def split_points(audio: np.ndarray) -> list[tuple[int, int]]:
    """Cut long audio into <= MAX_CHUNK_S pieces at the quietest 50 ms frame
    within the last 5 s before each limit, so words are rarely cut in half."""
    n, limit = len(audio), int(MAX_CHUNK_S * SR)
    if n <= limit:
        return [(0, n)]
    frame = int(0.05 * SR)
    out, start = [], 0
    while n - start > limit:
        lo, hi = start + limit - 5 * SR, start + limit
        frames = audio[lo:hi][: (hi - lo) // frame * frame].reshape(-1, frame)
        cut = lo + int(np.argmin((frames ** 2).mean(axis=1))) * frame + frame // 2
        out.append((start, cut))
        start = cut
    out.append((start, n))
    return out


def _load_gigaam():
    """GigaAM's remote code imports pyannote only for its own long-audio VAD
    (transcribe_longform), which we do not use - we cut long audio ourselves.
    transformers scans imports statically and refuses to load without it, so the
    scan is told to ignore that one optional package."""
    from transformers import AutoModel
    from transformers import dynamic_module_utils as dmu

    original = dmu.get_imports

    def get_imports_without_pyannote(filename):
        return [imp for imp in original(filename) if not imp.startswith("pyannote")]

    dmu.get_imports = get_imports_without_pyannote
    try:
        return AutoModel.from_pretrained(ASR_REPO, revision=ASR_REVISION, trust_remote_code=True)
    finally:
        dmu.get_imports = original


class Asr:
    def __init__(self, device: str | None = None):
        import torch

        self.model = _load_gigaam()
        # MPS autocast in GigaAM is flaky; CPU is fast enough for a 220M model.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def transcribe(self, wav: Path) -> list[TimedWord]:
        audio, sr = sf.read(str(wav), dtype="float32")
        assert sr == SR
        words: list[TimedWord] = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, (a, b) in enumerate(split_points(audio)):
                chunk = Path(tmp) / f"chunk{i}.wav"
                sf.write(str(chunk), audio[a:b], SR)
                res = self.model.transcribe(str(chunk), word_timestamps=True)
                for w in res.words or []:
                    words.append(TimedWord(w.text, w.start + a / SR, w.end + a / SR))
        return words


def mark_profane(det: ProfanityDetector, words: list[TimedWord]) -> str:
    """Join words into one transcript, run the text model, map spans back to words."""
    text, offsets = "", []
    for w in words:
        if text:
            text += " "
        offsets.append((len(text), len(text) + len(w.text)))
        text += w.text
    for sp in det.spans(text):
        for w, (a, b) in zip(words, offsets):
            if a < sp.end and b > sp.start:
                w.profane, w.score = True, max(w.score, sp.score)
    return text


def intervals(words: list[TimedWord], duration: float) -> list[tuple[float, float]]:
    spans = sorted((max(0.0, w.start - PAD_BEFORE), min(duration, w.end + PAD_AFTER)) for w in words if w.profane)
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in merged]


def apply_beeps(src: Path, dst: Path, spans: list[tuple[float, float]], duration: float, video: bool) -> None:
    if not spans:
        shutil.copy(src, dst)
        return
    cond = "+".join(f"between(t,{s},{e})" for s, e in spans)
    graph = (
        f"[0:a]volume=0:enable='{cond}'[muted];"
        f"sine=frequency=1000:sample_rate=48000:duration={duration:.3f},volume=0.35,"
        f"volume=0:enable='not({cond})'[beep];"
        f"[muted][beep]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
    )
    args = ["-i", str(src), "-filter_complex", graph, "-map", "[aout]"]
    if video:
        args += ["-map", "0:v:0", "-c:v", "copy"]
    codec = ["-c:a", "pcm_s16le"] if dst.suffix.lower() == ".wav" else ["-c:a", "aac", "-b:a", "192k"]
    ffmpeg(*args, *codec, str(dst))


def censor_media(det: ProfanityDetector, asr: Asr, src: str | Path, dst: str | Path | None = None) -> dict:
    src = Path(src)
    video = has_video(src)
    if dst is None:
        dst = src.with_name(f"{src.stem}.censored{src.suffix if video else '.wav'}")
    dst = Path(dst)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_audio(src, wav)
        duration = sf.info(str(wav)).duration
        words = asr.transcribe(wav)
    transcript = mark_profane(det, words)
    spans = intervals(words, duration)
    apply_beeps(src, dst, spans, duration, video)
    report = {
        "input": str(src), "output": str(dst), "duration_s": round(duration, 2),
        "transcript": transcript,
        "censored_transcript": " ".join("█" * len(w.text) if w.profane else w.text for w in words),
        "beeps": spans, "words": [asdict(w) for w in words],
    }
    dst.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
