"""Create small test documents (TXT, DOCX, PDF) for the redaction pipeline.

The reviews below were written for this demo (not from the dataset). They mix
profane comments with traps that must NOT be blacked out: insults without mat
(ақымақ, боқ) and look-alike words (көтереді 'lifts').

  python examples/make_examples.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"   # any TTF with Kazakh letters works

TITLE = "Тауарлар туралы пікірлер (тест құжаты)"
REVIEWS = [
    "Шаңсорғыш өте жақсы, қуаты мықты, үйді тез тазалайды. Ұсынамын!",
    "Телефон қотақ екен, екі аптада экраны сынды, ақшамды қайтарыңдар.",
    "Курьер ақымақ, тапсырысты үш күн кешіктірді, бірақ тауардың өзі жақсы.",
    "Бл*ть, тағы да жаңарту, қосымша мүлдем ашылмайды!",
    "Көтергіш жүкті оңай көтереді, сапасы керемет, бағасы да қолайлы.",
    "Мына құлаққап пиздец сапасыз, бір жағы бір аптадан кейін істемей қалды.",
    "Кітап қызықты, балаларға арналған, суреттері әдемі.",
    "Сатушы котакбас, хабарламаға жауап бермейді, енді бұл дүкеннен алмаймын.",
    "Боқ сияқты сапа, ақшаңызды босқа шығармаңыз.",
    "Qotaq zat eken, alyp ókindim.",
]


def make_txt(path: Path) -> None:
    path.write_text(TITLE + "\n\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(REVIEWS, 1)) + "\n", encoding="utf-8")


def make_docx(path: Path) -> None:
    from docx import Document

    doc = Document()
    doc.add_heading(TITLE, level=1)
    for i, r in enumerate(REVIEWS[:6], 1):
        p = doc.add_paragraph(f"{i}. ")
        # split one review into several runs (bold middle) to test run-boundary handling
        if i == 2:
            p.add_run("Телефон ")
            p.add_run("қотақ екен").bold = True
            p.add_run(", екі аптада экраны сынды.")
        else:
            p.add_run(r)
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "№", "Пікір"
    for i, r in enumerate(REVIEWS[6:], 7):
        row = table.add_row().cells
        row[0].text, row[1].text = str(i), r
    doc.save(str(path))


def make_pdf(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_font(fontname="arial", fontfile=FONT)
    y = 60
    page.insert_text((50, y), TITLE, fontname="arial", fontsize=15)
    y += 34
    for i, r in enumerate(REVIEWS, 1):
        rect = pymupdf.Rect(50, y, 545, y + 44)
        page.insert_textbox(rect, f"{i}. {r}", fontname="arial", fontsize=11)
        y += 46
    doc.subset_fonts()                       # embed only the glyphs used (~800 KB -> ~30 KB)
    doc.save(str(path), garbage=4, deflate=True)


if __name__ == "__main__":
    make_txt(HERE / "sample.txt")
    make_docx(HERE / "sample.docx")
    make_pdf(HERE / "sample.pdf")
    print("wrote", *(HERE / f"sample.{e}" for e in ("txt", "docx", "pdf")))
