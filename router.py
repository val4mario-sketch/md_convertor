#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
router.py — Router centrale del convertitore universale.

Mantiene la MATRICE DELLE CONVERSIONI POSSIBILI come unica fonte di verità.
Sceglie il modulo giusto in base a (estensione_sorgente, estensione_destinazione).
Se la coppia non è supportata, restituisce un errore CHIARO con suggerimenti.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent))

# Importa i moduli di conversione
import convert_documents
import convert_office_pdf
import convert_data
import convert_images
import convert_av
import convert_markdown
import convert_md_pdf

# Lista ordinata di moduli (l'ordine determina priorità in caso di sovrapposizione)
_MODULES = [
    ("markdown",   convert_markdown),   # priorità alta per .md (output -> .md)
    ("md_pdf",     convert_md_pdf),      # md -> PDF leggibile (tomdclaude); prima di documents
    ("office_pdf", convert_office_pdf),
    ("documents",  convert_documents),
    ("data",       convert_data),
    ("images",     convert_images),
    ("av",         convert_av),
]


# ----------------------------------------------------------------------------
# Categorizzazione (per messaggi d'errore amichevoli)
# ----------------------------------------------------------------------------
CATEGORIE = {
    "Documenti testo":  {".md",".markdown",".docx",".doc",".odt",".rtf",".txt",
                          ".html",".htm",".epub",".tex",".latex"},
    "Documenti Office": {".pptx",".ppt",".xlsx",".xls",".ods",".odp"},
    "PDF":              {".pdf"},
    "Dati tabellari":   {".csv",".tsv",".json",".parquet"},
    "Immagini":         {".png",".jpg",".jpeg",".webp",".tiff",".tif",".bmp",
                          ".gif",".svg",".heic",".ico"},
    "Audio":            {".mp3",".wav",".flac",".aac",".ogg",".m4a",".opus",".wma"},
    "Video":            {".mp4",".mov",".avi",".mkv",".webm",".flv",".wmv",".m4v"},
}


def _categoria_di(ext: str) -> str:
    e = ext.lower()
    for nome, exts in CATEGORIE.items():
        if e in exts:
            return nome
    return "Altro"


def _norm_ext(s: str) -> str:
    if not s: return ""
    s = s.lower().strip()
    return s if s.startswith(".") else "." + s


# ----------------------------------------------------------------------------
# API: matrice e routing
# ----------------------------------------------------------------------------
def supported_outputs_for(src_ext: str) -> dict[str, list[str]]:
    """Per una data estensione di SORGENTE, ritorna {modulo: [estensioni out]}."""
    src = _norm_ext(src_ext)
    out: dict[str, list[str]] = {}
    for name, mod in _MODULES:
        ins  = getattr(mod, "SUPPORTED_INPUTS",  set())
        outs = getattr(mod, "SUPPORTED_OUTPUTS", set())
        if src in ins:
            out[name] = sorted(outs)
    return out


def find_module(src_ext: str, dst_ext: str):
    """Ritorna (nome_modulo, modulo) che gestisce la coppia src->dst, oppure (None, None)."""
    src = _norm_ext(src_ext); dst = _norm_ext(dst_ext)
    for name, mod in _MODULES:
        try:
            if mod.supports(src, dst):
                return name, mod
        except Exception:
            pass
    return None, None


def convert(src: Path, dst: Path, opts: dict | None = None) -> dict:
    """Dispatch alla routine giusta. Ritorna {ok, note, elapsed, module}."""
    src = Path(src); dst = Path(dst)
    s_ext = src.suffix.lower()
    d_ext = dst.suffix.lower()

    if not src.exists():
        return {"ok": False, "note": f"File sorgente non trovato: {src}",
                "elapsed": 0.0, "module": None}

    name, mod = find_module(s_ext, d_ext)
    if mod is None:
        return {"ok": False,
                "note": _messaggio_conversione_impossibile(s_ext, d_ext),
                "elapsed": 0.0, "module": None}

    dst.parent.mkdir(parents=True, exist_ok=True)
    r = mod.convert(src, dst, opts or {})
    r["module"] = name

    # ---- Fallback automatici quando il modulo primario fallisce ----
    if not r.get("ok"):
        fb = _try_fallback(src, dst, s_ext, d_ext, opts or {})
        if fb is not None:
            return fb
    return r


def _try_fallback(src: Path, dst: Path, s_ext: str, d_ext: str, opts: dict) -> dict | None:
    """Catena di fallback per casi noti: md->pdf passa per md->docx->pdf, ecc."""
    import tempfile, shutil
    # CASE: <doc_testo> -> .pdf via Pandoc fallisce per mancanza LaTeX
    # Fallback: testo->docx (Pandoc) -> pdf (LibreOffice)
    if d_ext == ".pdf" and s_ext in {".md",".markdown",".txt",".html",".htm",
                                      ".epub",".rtf",".tex",".latex"}:
        if convert_office_pdf.supports(".docx", ".pdf"):
            tmp = Path(tempfile.mkdtemp(prefix="conv_fb_"))
            try:
                inter = tmp / (src.stem + ".docx")
                r1 = convert_documents.convert(src, inter, opts)
                if r1.get("ok"):
                    r2 = convert_office_pdf.convert(inter, dst, opts)
                    if r2.get("ok"):
                        r2["module"] = "documents+office_pdf (fallback)"
                        r2["note"] = (f"PDF generato via fallback "
                                       f"{s_ext}->.docx->.pdf "
                                       f"(LaTeX assente). {r2.get('note','')}")
                        return r2
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    # CASE: office legacy/ODF -> .md
    # Il motore Markdown (markitdown/tomd2) non legge i vecchi formati binari
    # (.doc/.ppt/.xls) né l'ODF (.odt/.ods/.odp). Fallback: converti prima nel
    # formato OOXML moderno equivalente via LibreOffice, poi in Markdown.
    if d_ext in {".md", ".markdown"}:
        _MODERN = {".doc": ".docx", ".odt": ".docx",
                   ".ppt": ".pptx", ".odp": ".pptx",
                   ".xls": ".xlsx", ".ods": ".xlsx"}
        modern = _MODERN.get(s_ext)
        if modern and convert_office_pdf.supports(s_ext, modern):
            tmp = Path(tempfile.mkdtemp(prefix="conv_md_fb_"))
            try:
                inter = tmp / (src.stem + modern)
                r1 = convert_office_pdf.convert(src, inter, opts)
                if r1.get("ok"):
                    r2 = convert_markdown.convert(inter, dst, opts)
                    if r2.get("ok"):
                        r2["module"] = "office_pdf+markdown (fallback)"
                        r2["note"] = (f"Markdown generato via fallback "
                                       f"{s_ext}->{modern}->.md. {r2.get('note','')}")
                        return r2
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    return None


def list_all_conversions() -> str:
    """Stringa leggibile con TUTTE le conversioni supportate, raggruppate per categoria."""
    lines = []
    lines.append("=" * 72)
    lines.append("  CONVERSIONI SUPPORTATE — convertitore universale")
    lines.append("=" * 72)
    for cat_name, cat_exts in CATEGORIE.items():
        lines.append(f"\n## {cat_name}")
        for s_ext in sorted(cat_exts):
            mods = supported_outputs_for(s_ext)
            if not mods: continue
            tutti_out = sorted({o for outs in mods.values() for o in outs} - {s_ext})
            if not tutti_out: continue
            lines.append(f"  {s_ext:>10}  ->  {', '.join(tutti_out)}")
    lines.append("\n" + "=" * 72)
    lines.append("Output verso Markdown (.md) ad alta fedeltà: TUTTI i formati sopra")
    lines.append("                                              + .pdf .ipynb .eml .zip .py .js ...")
    lines.append("Totale estensioni accettate in input verso .md: ~170")
    lines.append("=" * 72)
    return "\n".join(lines)


def _messaggio_conversione_impossibile(s_ext: str, d_ext: str) -> str:
    """Messaggio chiaro per coppia non supportata, con suggerimenti."""
    s_cat = _categoria_di(s_ext)
    d_cat = _categoria_di(d_ext)
    out_disp = supported_outputs_for(s_ext)
    tutti_out = sorted({o for outs in out_disp.values() for o in outs} - {s_ext})
    if not tutti_out:
        msg = (f"Conversione {s_ext} -> {d_ext} NON supportata.\n"
               f"   Non conosco nessuna conversione partendo da {s_ext}.\n"
               f"   Usa --list per vedere tutte le coppie supportate.")
    else:
        msg = (f"Conversione {s_ext} -> {d_ext} NON supportata.\n"
               f"   Categoria sorgente: {s_cat}   |   destinazione: {d_cat}\n"
               f"   Da {s_ext} puoi convertire verso: "
               f"{', '.join(tutti_out)}.\n"
               f"   Esempio: --to {tutti_out[0][1:]}")
    return msg
