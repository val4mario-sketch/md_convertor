"""Estrazione metadati usata dai validatori.

Funzioni pure, senza stato: dato un testo Markdown / un file PDF / un XLSX,
ritornano strutture confrontabili (albero dei titoli, conteggi, dimensioni).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
# Riga separatrice di una tabella Markdown: |---|:--:|  ecc.
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


def normalize_text(s: str) -> str:
    """Normalizza per confronti robusti: NFKC, virgolette/trattini unificati,
    spazi compressi, minuscolo."""
    s = unicodedata.normalize("NFKC", s or "")
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("—", "-"), ("–", "-"), (" ", " ")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip().lower()


def strip_inline_md(s: str) -> str:
    """Rimuove la formattazione inline (`code`, **bold**, _it_, [link](url)) e le
    formule matematiche `$...$`, per confrontare il testo 'puro' dei titoli.

    Le formule vanno rimosse perche' nel PDF diventano immagini SVG (senza testo
    estraibile): tenerle nel confronto produrrebbe falsi 'titolo mancante'."""
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
    """Albero gerarchico dei titoli ATX (#..######), ignorando i blocchi di codice."""
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
    """Albero dei titoli da HTML (<h1>..<h6>)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out: list[tuple[int, str]] = []
    for tag in soup.find_all(re.compile(r"^h[1-6]$")):
        out.append((int(tag.name[1]), tag.get_text(" ", strip=True)))
    return out


def count_md_tables(text: str) -> int:
    """Numero di tabelle Markdown (contando le righe-separatore di header)."""
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
    """Lunghezza approssimata del testo 'visibile' (senza simboli di sintassi).
    Serve come riferimento per verificare che il PDF non sia quasi vuoto."""
    plain = re.sub(r"[#>*`_|~]", "", text)
    plain = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", plain)
    return len(re.sub(r"\s+", " ", plain).strip())


def pdf_info(path) -> dict:
    """Apre un PDF e ne ritorna n. pagine + testo estraibile (via PyMuPDF)."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    try:
        text = "\n".join(page.get_text("text") for page in doc)
        return {"pages": doc.page_count, "text": text, "text_len": len(text.strip())}
    finally:
        doc.close()


def xlsx_dimensions(path) -> dict:
    """Dimensioni (righe x colonne) per ogni foglio di un XLSX (via openpyxl).
    Usato dal validatore XLSX per verificare che nessuna riga/colonna sia persa."""
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    try:
        return {ws.title: {"rows": ws.max_row or 0, "cols": ws.max_column or 0}
                for ws in wb.worksheets}
    finally:
        wb.close()
