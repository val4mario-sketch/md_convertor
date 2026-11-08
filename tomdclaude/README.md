# tomdclaude

A **modular** file conversion system with **gatekeeper validation**.
First converter implemented: **readable Markdown → PDF** (report/article style).

It sits alongside the existing universal converter (`tomd2.py` + router) without
changing it: it lives in the `tomdclaude/` subpackage.

## Architecture

```
tomdclaude/
├── core/            contract, result, registry, pipeline (gatekeeper)
│   ├── base_converter.py   BaseConverter (ABC): each converter is a class
│   ├── result.py           ConversionResult, ValidationIssue
│   ├── registry.py         list of converters + lookup by format
│   └── pipeline.py         orchestration: convert → validate → commit/abort
├── converters/      one file = one independent converter
│   └── md_to_pdf.py        MarkdownToPdfConverter (markdown → HTML → WeasyPrint)
├── validators/      the gatekeeper: compares metadata
│   ├── md_validator.py     heading hierarchy (MD)
│   ├── pdf_validator.py    PDF integrity (header, pages, text)
│   └── xlsx_validator.py   rows/columns per sheet (XLSX), ready to use
├── utils/           logging, GTK runtime, metadata extraction
├── assets/report.css       report stylesheet
├── examples/report_demo.md demo document
├── tests/test_md_to_pdf.py first test case
├── cli.py                  command line
└── logs/                   detailed failure logs
```

## The gatekeeper (key requirement)

Each conversion is written to a **temporary file**. It becomes final (atomic
commit) **only** if it passes every validator. If a validator detects data loss
or an inconsistency (for example a lost heading or a nearly empty PDF), the
temporary file is **discarded**, the final file is **not** saved, and a
**detailed log** is written to `tomdclaude/logs/`.

## MD → PDF: what it produces

- Cover with title/author/date (front matter `title:` / `author:`)
- **Automatic table of contents** with page numbers (CSS `target-counter`)
- Automatic chapter numbering (1, 1.1, 1.1.1)
- Running header and page-number footer
- Formatted tables, code with syntax highlighting (Pygments), note boxes
- **LaTeX math** `$...$` / `$$...$$` rendered as SVG (Computer Modern font, a
  LaTeX look) without needing a TeX toolchain (see below)
- Automatic removal of the citation markers `[cite_start]`/`[cite: N]`
- Readable typography (serif, justified text, hyphenation)

## Math without LaTeX

Formulas are isolated by `pymdownx.arithmatex` (code blocks are left alone) and
rendered as **SVG** by matplotlib's *mathtext* engine (`mathtext.fontset = cm`).
Inline formulas align with the baseline; display formulas are centered.
A sanitizing step converts small syntax differences (`\le`→`\leq`,
`\dfrac`→`\frac`, removal of `\Big`, `\displaystyle`, ...).

Tested coverage: 236/236 formulas on a real numerical methods document.
Expressions that cannot be rendered do not block by default: they stay visible
as raw text and go into the log; with `--strict-math` they become blocking.

Limit: mathtext does not support `\begin{matrix|align|cases}` environments
(absent from the target documents). Those would need a TeX toolchain
(MiKTeX + dvisvgm).

## Usage

```bash
# from the folder that contains the package
python -m tomdclaude examples/report_demo.md --to pdf
python -m tomdclaude notes.md -o build/notes.pdf --title "Numerical Methods"
python -m tomdclaude --list

# useful options
#   --no-math          do not render the math
#   --strict-math      reject the PDF if a formula cannot be rendered
#   --keep-citations   do not remove the [cite_start]/[cite: N] markers

# tests (6/6)
python -m tomdclaude.tests.test_md_to_pdf
```

## Dependencies

`pip install -r tomdclaude/requirements.txt` plus the GTK3 runtime on Windows
(`winget install --id tschoonj.GTKForWindows -e`). The `utils/gtk.py` module
hooks GTK automatically at run time.

## Math modules

```
converters/math_render.py     MathRenderer: LaTeX -> SVG via matplotlib mathtext
validators/math_validator.py  counts the formulas that were not rendered (gatekeeper)
```
