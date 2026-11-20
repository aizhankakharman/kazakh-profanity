"""Web demo: check a comment, redact a document, bleep a video.

  python app.py            # opens http://127.0.0.1:7860
  python app.py --share    # temporary public Gradio link (e.g. for the defense)
"""

from __future__ import annotations

import argparse
import html
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr

from kzprof.inference import ProfanityDetector
from kzprof.redact import redact_file

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "kazakh-profanity-gliner"
EXAMPLES = [
    "Сапасы нашар, бір аптада сынып қалды. Ақшаңызды босқа шығармаңыз.",
    "Қотақ телефон екен, зарядты ұстамайды",
    "Курьер ақымақ екен, тапсырысты кешіктірді, бірақ тауар жақсы",
    "бл*ть тағы да жаңарту, ештеңе ашылмайды",
    "Көтеру механизмі мықты, жүкті оңай көтереді",
]


@lru_cache(maxsize=1)
def detector() -> ProfanityDetector:
    return ProfanityDetector(MODEL_PATH)


@lru_cache(maxsize=1)
def asr():
    from kzprof.video import Asr

    return Asr()


def check_text(text: str):
    if not text or not text.strip():
        return "", ""
    res = detector().predict(text)
    text = res["text"]          # text after normalization (offsets match it)
    out, pos = [], 0
    for sp in res["spans"]:
        out.append(html.escape(text[pos:sp["start"]]))
        out.append(f'<mark style="background:#111;color:#111;border-radius:3px" title="{sp["score"]:.2f}">'
                   f'{html.escape(sp["text"])}</mark>')
        pos = sp["end"]
    out.append(html.escape(text[pos:]))
    if res["label"]:
        verdict = "1 — мат бар (profane)"
    elif res["score"] >= 0.5:
        verdict = "0 — ⚠️ күмәнді, модераторға (borderline: send to a moderator)"
    else:
        verdict = "0 — таза (clean)"
    words = ", ".join(f'{s["text"]} ({s["score"]:.2f})' for s in res["spans"]) or "—"
    summary = (f"**Label:** {verdict}  \n**Score:** {res['score']:.3f} "
               f"(threshold {detector().threshold})  \n**Words:** {words}")
    return summary, f'<div style="font-size:1.1em;line-height:1.7">{"".join(out)}</div>'


def redact(doc):
    if doc is None:
        return None, ""
    src = Path(doc)
    dst = Path(tempfile.mkdtemp()) / f"{src.stem}.redacted{src.suffix}"
    _, n = redact_file(detector(), src, dst)
    return str(dst), f"Redacted spans: **{n}**"


def bleep(media):
    if media is None:
        return None, None, ""
    from kzprof.video import censor_media

    src = Path(media)
    out_dir = Path(tempfile.mkdtemp())
    report = censor_media(detector(), asr(), src, out_dir / f"{src.stem}.censored{src.suffix}")
    out = report["output"]
    is_video = out.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi"))
    beeps = ", ".join(f"{s:.2f}–{e:.2f}s" for s, e in report["beeps"]) or "—"
    md = (f"**Transcript (GigaAM):** {report['transcript']}\n\n"
          f"**Censored:** {report['censored_transcript']}\n\n**Beeps:** {beeps}")
    return (out if is_video else None), (None if is_video else out), md


def build() -> gr.Blocks:
    with gr.Blocks(title="Kazakh profanity detector") as demo:
        gr.Markdown("# Қазақша балағат сөздерді анықтау / Kazakh profanity detection\n"
                    "Fine-tuned GLiNER (mDeBERTa-v3). Label 1 = the comment contains Kazakh (or Russian mat in a "
                    "Kazakh comment) obscene words.")
        with gr.Tab("Text"):
            inp = gr.Textbox(label="Comment / review", lines=3)
            btn = gr.Button("Check", variant="primary")
            summary = gr.Markdown()
            shown = gr.HTML()
            btn.click(check_text, inp, [summary, shown])
            inp.submit(check_text, inp, [summary, shown])
            gr.Examples(EXAMPLES, inp)
        with gr.Tab("Document (PDF / DOCX / TXT)"):
            doc_in = gr.File(label="Upload document", file_types=[".pdf", ".docx", ".txt"], type="filepath")
            doc_btn = gr.Button("Redact", variant="primary")
            doc_out = gr.File(label="Redacted document")
            doc_info = gr.Markdown()
            doc_btn.click(redact, doc_in, [doc_out, doc_info])
        with gr.Tab("Video / audio"):
            media_in = gr.File(label="Upload video or audio", type="filepath",
                               file_types=[".mp4", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".m4a", ".ogg"])
            media_btn = gr.Button("Bleep profanity", variant="primary")
            video_out = gr.Video(label="Censored video")
            audio_out = gr.Audio(label="Censored audio", type="filepath")
            media_info = gr.Markdown()
            media_btn.click(bleep, media_in, [video_out, audio_out, media_info])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    MODEL_PATH = a.model
    build().launch(share=a.share, server_port=a.port)
