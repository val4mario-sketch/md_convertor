"""Metadata extraction used by the validators.

Pure, stateless functions: given a Markdown text / a PDF file / an XLSX,
they return comparable structures (heading tree, counts, sizes).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
# Separator row of a Markdown table: |---|:--:|  etc.
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


def normalize_text(s: str) -> str:
    """Normalizes for robust comparisons: NFKC, unified quotes/dashes,
    compressed spaces, lowercase."""
    s = unicodedata.normalize("NFKC", s or "")
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("—", "-"), ("–", "-"), (" ", " ")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip().lower()


def strip_inline_md(s: str) -> str:
    """Removes inline formatting (`code`, **bold**, _it_, [link](url)) and
    `$...$` math formulas, to compare the 'pure' text of headings.

    Formulas must be removed because in the PDF they become SVG images (with no
    extractable text): keeping them in the comparison would produce false 'missing heading' errors."""
    s = re.sub(r"\$\$.+?\$\$", " ", s)
    s = re.sub(r"\$[^$]+\$", " ", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    return s.strip()


def markdown_heading_tree(text: str) -> list[tuple[int, str]]:
    """Hierarchical tree of the ATX headings (#..######), ignoring code blocks."""
    out: list[tuple[int, str]] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _ATX.match(line)
        if m:
            out.append((len(m.group(1)), strip_inline_md(m.group(2))))
    return out


def html_heading_tree(html: str) -> list[tuple[int, str]]:
    """Heading tree from HTML (<h1>..<h6>)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out: list[tuple[int, str]] = []
    for tag in soup.find_all(re.compile(r"^h[1-6]$")):
        out.append((int(tag.name[1]), tag.get_text(" ", strip=True)))
    return out


def count_md_tables(text: str) -> int:
    """Number of Markdown tables (counting the header separator rows)."""
    in_fence = False
    n = 0
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and _TABLE_SEP.match(line):
            n += 1
    return n


def md_plain_length(text: str) -> int:
    """Approximate length of the 'visible' text (without syntax symbols).
    Used as a reference to check that the PDF is not almost empty."""
    plain = re.sub(r"[#>*`_|~]", "", text)
    plain = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", plain)
    return len(re.sub(r"\s+", " ", plain).strip())


def pdf_info(path) -> dict:
    """Opens a PDF and returns its number of pages + extractable text (via PyMuPDF)."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    try:
        text = "\n".join(page.get_text("text") for page in doc)
        return {"pages": doc.page_count, "text": text, "text_len": len(text.strip())}
    finally:
        doc.close()


def xlsx_dimensions(path) -> dict:
    """Sizes (rows x columns) of each sheet of an XLSX (via openpyxl).
    Used by the XLSX validator to check that no row/column was lost."""
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    try:
        return {ws.title: {"rows": ws.max_row or 0, "cols": ws.max_column or 0}
                for ws in wb.worksheets}
    finally:
        wb.close()
