"""Black out profanity in documents (PDF, DOCX, TXT).

The text under a redaction is really removed, not just covered:
  * PDF  - PyMuPDF redaction annotations + apply_redactions() delete the glyphs
           under each black box; copying text or running pdftotext afterwards
           does not reveal the word.
  * DOCX - the profane characters are replaced with "█" and highlighted black,
           so the original letters are no longer in the file.
  * TXT  - profane characters are replaced with "█".
"""

from __future__ import annotations

import re
from pathlib import Path

from .inference import ProfanityDetector

BLOCK = "█"


def _blackout(text: str, spans) -> str:
    chars = list(text)
    for sp in spans:
        for i in range(sp.start, sp.end):
            if not chars[i].isspace():
                chars[i] = BLOCK
    return "".join(chars)


def redact_txt(det: ProfanityDetector, src: Path, dst: Path) -> int:
    text = src.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    spans = det.spans_batch(lines)
    dst.write_text("\n".join(_blackout(ln, sp) for ln, sp in zip(lines, spans)), encoding="utf-8")
    return sum(len(s) for s in spans)


def redact_docx(det: ProfanityDetector, src: Path, dst: Path) -> int:
    from docx import Document
    from docx.enum.text import WD_COLOR_INDEX

    doc = Document(str(src))
    paragraphs = list(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs.extend(cell.paragraphs)
    paragraphs = [p for p in paragraphs if p.text.strip()]
    all_spans = det.spans_batch([p.text for p in paragraphs])
    n = 0
    for p, spans in zip(paragraphs, all_spans):
        if not spans:
            continue
        n += len(spans)
        bad = set()
        for sp in spans:
            bad.update(range(sp.start, sp.end))
        pos = 0
        # Walk the runs; split a run wherever it crosses a profane/clean border.
        for run in list(p.runs):
            text = run.text
            if not text:
                continue
            pieces, cur, cur_bad = [], "", None
            for k, ch in enumerate(text):
                is_bad = (pos + k) in bad and not ch.isspace()
                if cur and is_bad != cur_bad:
                    pieces.append((cur, cur_bad))
                    cur = ""
                cur += BLOCK if is_bad else ch
                cur_bad = is_bad
            if cur:
                pieces.append((cur, cur_bad))
            pos += len(text)
            if len(pieces) == 1 and not pieces[0][1]:
                continue
            run.text = pieces[0][0]
            if pieces[0][1]:
                run.font.highlight_color = WD_COLOR_INDEX.BLACK
            anchor = run._r
            for piece, is_bad in pieces[1:]:
                new_r = _clone_run(run, piece)
                anchor.addnext(new_r)
                anchor = new_r
                if is_bad:
                    _set_highlight(new_r)
                else:
                    _clear_highlight(new_r)
    doc.save(str(dst))
    return n


def _clone_run(run, text: str):
    """Copy a run with its formatting (font, size, bold...) but new text."""
    import copy

    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    new_r = copy.deepcopy(run._r)
    for child in list(new_r):
        if child.tag != qn("w:rPr"):          # keep formatting, drop text/tabs/breaks
            new_r.remove(child)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    new_r.append(t)
    return new_r


def _set_highlight(r) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    rpr = r.find(qn("w:rPr"))
    if rpr is None:
        rpr = OxmlElement("w:rPr")
        r.insert(0, rpr)
    for h in rpr.findall(qn("w:highlight")):
        rpr.remove(h)
    h = OxmlElement("w:highlight")
    h.set(qn("w:val"), "black")
    rpr.append(h)


def _clear_highlight(r) -> None:
    from docx.oxml.ns import qn

    rpr = r.find(qn("w:rPr"))
    if rpr is not None:
        for h in rpr.findall(qn("w:highlight")):
            rpr.remove(h)


def _inner_rect(r, n_chars: int):
    """MuPDF removes every glyph whose box touches the redaction area, and the
    boxes of neighbouring glyphs touch. Shrinking the word box by a quarter of a
    character on each side still covers the first/last letter of the word but no
    longer touches the next word."""
    cw = r.width / max(1, n_chars)
    return type(r)(r.x0 + 0.25 * cw, r.y0 + 0.1 * r.height, r.x1 - 0.25 * cw, r.y1 - 0.1 * r.height)


def _new_x0(after: list, word: tuple) -> float | None:
    """Where `word` (a word that was NOT redacted) ended up after redaction."""
    same = [a for a in after if a[4] == word[4] and abs(a[1] - word[1]) < 1.0 and a[0] <= word[0] + 0.5]
    return max(same, key=lambda a: a[0])[0] if same else None


def redact_pdf(det: ProfanityDetector, src: Path, dst: Path) -> int:
    """1) find profane words per text line, 2) let MuPDF delete their glyphs,
    3) paint black boxes into the gaps.

    Boxes are drawn AFTER deletion, from the final layout: MuPDF 1.26 shifts
    the rest of the line left by the width of the first deleted glyph, so a box
    drawn at the original position would cover a letter of the next word."""
    import pymupdf as fitz

    doc = fitz.open(str(src))
    n = 0
    for page in doc:
        words = page.get_text("words", sort=True)   # (x0, y0, x1, y1, text, block, line, word_no)
        lines: dict[tuple[int, int], list[int]] = {}
        for i, w in enumerate(words):
            lines.setdefault((w[5], w[6]), []).append(i)
        texts, offsets = [], []
        for idxs in lines.values():
            pos, offs = 0, []
            for i in idxs:
                offs.append((pos, pos + len(words[i][4])))
                pos += len(words[i][4]) + 1
            texts.append(" ".join(words[i][4] for i in idxs))
            offsets.append(offs)
        bad: set[int] = set()
        for idxs, offs, spans in zip(lines.values(), offsets, det.spans_batch(texts)):
            n += len(spans)
            for sp in spans:
                bad.update(i for i, (a, b) in zip(idxs, offs) if a < sp.end and b > sp.start)
        if not bad:
            continue
        for i in bad:
            page.add_redact_annot(_inner_rect(fitz.Rect(words[i][:4]), len(words[i][4])), fill=False)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        after = page.get_text("words")
        for idxs in lines.values():
            for k, i in enumerate(idxs):
                if i not in bad:
                    continue
                r = fitz.Rect(words[i][:4])
                prev = next((idxs[j] for j in range(k - 1, -1, -1) if idxs[j] not in bad), None)
                nxt = next((idxs[j] for j in range(k + 1, len(idxs)) if idxs[j] not in bad), None)
                shift = 0.0
                if prev is not None and (px := _new_x0(after, words[prev])) is not None:
                    shift = words[prev][0] - px            # how far this part of the line moved
                x0, x1 = r.x0 - shift, r.x1 - shift
                if nxt is not None and (nx := _new_x0(after, words[nxt])) is not None:
                    x1 = min(x1, nx - 1.5)                 # never paint over the next word
                page.draw_rect(fitz.Rect(x0, r.y0 + 0.1 * r.height, max(x0 + 2, x1), r.y1 - 0.1 * r.height),
                               color=None, fill=(0, 0, 0))
    doc.save(str(dst), garbage=4, deflate=True)
    return n


def redact_file(det: ProfanityDetector, src: str | Path, dst: str | Path | None = None) -> tuple[Path, int]:
    src = Path(src)
    suffix = src.suffix.lower()
    dst = Path(dst) if dst else src.with_name(f"{src.stem}.redacted{src.suffix}")
    if suffix == ".pdf":
        n = redact_pdf(det, src, dst)
    elif suffix == ".docx":
        n = redact_docx(det, src, dst)
    elif suffix in {".txt", ".md", ".csv"}:
        n = redact_txt(det, src, dst)
    else:
        raise ValueError(f"unsupported file type: {suffix} (use .pdf, .docx or .txt)")
    return dst, n


def is_blacked(text: str) -> bool:
    return bool(re.search(BLOCK, text))
