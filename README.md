# Convertitore Universale

A local file converter for Windows, with a desktop GUI and a command line. Its main job is turning documents, including scanned
PDFs and handwritten notes, into clean Markdown that an LLM can read. It also converts between document, data, image and media
formats. Everything runs on your own machine, with no paid APIs and no uploads, and it does not need a GPU.

The code comments, messages and interface are in Italian.

## What it does

- **Any format to Markdown** (`tomd2.py`): keeps text, tables, LaTeX formulas and images. It picks an engine for each file, with
  `pymupdf4llm` as the main one for PDFs that have a text layer.
- **Format to format** (`convert.py`, `router.py`): a router sends each source and target pair to the right module. If a pair is
  not supported, you get a clear error listing the targets available for that file. Run `python convert.py --list` to see them all.
- **OCR with local vision models** (`ocr_vlm.py`): runs on CPU through `llama.cpp`, and can resume after an interruption.
- **Readable Markdown to PDF** (`tomdclaude/`): report-style output with math rendered as SVG. A validator discards the output and
  logs the reason if the conversion lost data.
- **Image descriptions** (`img_describe.py`): adds a local AI description of each image in a Markdown file, and turns formulas into
  LaTeX.

## The hybrid OCR pipeline

Each open model has different strengths, so `vlm_ibrido` combines several and has them check each other:

| Model | Role |
|---|---|
| PaddleOCR-VL 1.6 | Reads every page; fast and strong on printed text, formulas and tables |
| Chandra OCR 2 | Handles pages you mark with `--hq-pages` and pages where Paddle fails (too little text for the ink on the page, repetition loops, unreadable text); strongest on handwriting and describes figures |
| Qwen3.5-9B | Describes figures using the page text as context, and rereads handwritten formula blocks |
| GLM-4.6V-Flash | A different model family, used as referee when readings disagree and to check figure descriptions |

Anything still uncertain goes into `<name>_da_controllare.md` for a person to review, so doubtful text is flagged rather than
silently accepted.

## Getting started

Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
```

On Windows, Markdown to PDF also needs the GTK3 runtime:

```bash
winget install --id tschoonj.GTKForWindows -e
```

Open the GUI by double-clicking `Avvia_Convertitore.bat`, or run `python tomd2_gui.py`.

From the command line:

```bash
python tomd2.py file.pdf
```

```bash
python convert.py --list
```

```bash
python tomd2.py file.pdf --engine vlm_ibrido --hq-pages 379-453
```

The OCR models are not in this repository because they take about 18 GB. Download them and `llama.cpp` once:

```bash
python ocr_vlm.py setup tutti
```

Then try it on a few pages. This writes separate test files and leaves existing `.md` files alone:

```bash
python ocr_vlm.py prova file.pdf --pages 12,381 --engine tutti
```

Original files are never changed or deleted. More detail is in `USAGE.txt`, `LOCAL_OCR.txt` and `tomdclaude/README.md`.

## Tests

```bash
python -m tomdclaude.tests.test_md_to_pdf
```

These 6 tests cover the `tomdclaude` Markdown to PDF path and its validators. The other modules have no automated tests yet.

## Known limits

- The math renderer uses matplotlib mathtext, which does not support `\begin{matrix|align|cases}` environments.
- The first conversion with math in a new process takes about 90 seconds to warm up the font cache, then about 3 seconds.
- Local OCR on CPU is slow. It is meant for accuracy on documents you care about, not for bulk throughput.
- It is developed and used on Windows; other systems are untested.
