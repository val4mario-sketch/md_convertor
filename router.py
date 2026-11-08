#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
router.py — Central router of the universal converter.

Keeps the MATRIX OF POSSIBLE CONVERSIONS as the single source of truth.
Chooses the right module based on (source_extension, target_extension).
If the pair is not supported, returns a CLEAR error with suggestions.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent))

# Import the conversion modules
import convert_documents
import convert_office_pdf
import convert_data
import convert_images
import convert_av
import convert_markdown
import convert_md_pdf

# Ordered list of modules (the order sets the priority when they overlap)
_MODULES = [
    ("markdown",   convert_markdown),   # high priority for .md (output -> .md)
    ("md_pdf",     convert_md_pdf),      # md -> readable PDF (tomdclaude); before documents
    ("office_pdf", convert_office_pdf),
    ("documents",  convert_documents),
    ("data",       convert_data),
    ("images",     convert_images),
    ("av",         convert_av),
]


# ----------------------------------------------------------------------------
# Categorization (for friendly error messages)
# ----------------------------------------------------------------------------
CATEGORIE = {
    "Text documents":  {".md",".markdown",".docx",".doc",".odt",".rtf",".txt",
                          ".html",".htm",".epub",".tex",".latex"},
    "Office documents": {".pptx",".ppt",".xlsx",".xls",".ods",".odp"},
    "PDF":              {".pdf"},
    "Tabular data":   {".csv",".tsv",".json",".parquet"},
    "Images":         {".png",".jpg",".jpeg",".webp",".tiff",".tif",".bmp",
                          ".gif",".svg",".heic",".ico"},
    "Audio":            {".mp3",".wav",".flac",".aac",".ogg",".m4a",".opus",".wma"},
    "Video":            {".mp4",".mov",".avi",".mkv",".webm",".flv",".wmv",".m4v"},
}


def _categoria_di(ext: str) -> str:
    e = ext.lower()
    for nome, exts in CATEGORIE.items():
        if e in exts:
            return nome
    return "Other"


def _norm_ext(s: str) -> str:
    if not s: return ""
    s = s.lower().strip()
    return s if s.startswith(".") else "." + s


# ----------------------------------------------------------------------------
# API: matrix and routing
# ----------------------------------------------------------------------------
def supported_outputs_for(src_ext: str) -> dict[str, list[str]]:
    """For a given SOURCE extension, returns {module: [output extensions]}."""
    src = _norm_ext(src_ext)
    out: dict[str, list[str]] = {}
    for name, mod in _MODULES:
        ins  = getattr(mod, "SUPPORTED_INPUTS",  set())
        outs = getattr(mod, "SUPPORTED_OUTPUTS", set())
        if src in ins:
            out[name] = sorted(outs)
    return out


def find_module(src_ext: str, dst_ext: str):
    """Returns (module_name, module) handling the src->dst pair, or (None, None)."""
    src = _norm_ext(src_ext); dst = _norm_ext(dst_ext)
    for name, mod in _MODULES:
        try:
            if mod.supports(src, dst):
                return name, mod
        except Exception:
            pass
    return None, None


def convert(src: Path, dst: Path, opts: dict | None = None) -> dict:
    """Dispatches to the right routine. Returns {ok, note, elapsed, module}."""
    src = Path(src); dst = Path(dst)
    s_ext = src.suffix.lower()
    d_ext = dst.suffix.lower()

    if not src.exists():
        return {"ok": False, "note": f"Source file not found: {src}",
                "elapsed": 0.0, "module": None}

    name, mod = find_module(s_ext, d_ext)
    if mod is None:
        return {"ok": False,
                "note": _messaggio_conversione_impossibile(s_ext, d_ext),
                "elapsed": 0.0, "module": None}

    dst.parent.mkdir(parents=True, exist_ok=True)
    r = mod.convert(src, dst, opts or {})
    r["module"] = name

    # ---- Automatic fallbacks when the primary module fails ----
    if not r.get("ok"):
        fb = _try_fallback(src, dst, s_ext, d_ext, opts or {})
        if fb is not None:
            return fb
    return r


def _try_fallback(src: Path, dst: Path, s_ext: str, d_ext: str, opts: dict) -> dict | None:
    """Fallback chain for known cases: md->pdf goes through md->docx->pdf, etc."""
    import tempfile, shutil
    # CASE: <text_doc> -> .pdf via Pandoc fails because LaTeX is missing
    # Fallback: text->docx (Pandoc) -> pdf (LibreOffice)
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
                        r2["note"] = (f"PDF generated via fallback "
                                       f"{s_ext}->.docx->.pdf "
                                       f"(LaTeX missing). {r2.get('note','')}")
                        return r2
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    # CASE: office legacy/ODF -> .md
    # The Markdown engine (markitdown/tomd2) does not read the old binary formats
    # (.doc/.ppt/.xls) or ODF (.odt/.ods/.odp). Fallback: first convert to the
    # equivalent modern OOXML format via LibreOffice, then to Markdown.
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
                        r2["note"] = (f"Markdown generated via fallback "
                                       f"{s_ext}->{modern}->.md. {r2.get('note','')}")
                        return r2
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    return None


def list_all_conversions() -> str:
    """Readable string with ALL the supported conversions, grouped by category."""
    lines = []
    lines.append("=" * 72)
    lines.append("  SUPPORTED CONVERSIONS — universal converter")
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
    lines.append("High-fidelity output to Markdown (.md): ALL the formats above")
    lines.append("                                              + .pdf .ipynb .eml .zip .py .js ...")
    lines.append("Total extensions accepted as input to .md: ~170")
    lines.append("=" * 72)
    return "\n".join(lines)


def _messaggio_conversione_impossibile(s_ext: str, d_ext: str) -> str:
    """Clear message for an unsupported pair, with suggestions."""
    s_cat = _categoria_di(s_ext)
    d_cat = _categoria_di(d_ext)
    out_disp = supported_outputs_for(s_ext)
    tutti_out = sorted({o for outs in out_disp.values() for o in outs} - {s_ext})
    if not tutti_out:
        msg = (f"Conversion {s_ext} -> {d_ext} NOT supported.\n"
               f"   I know of no conversion starting from {s_ext}.\n"
               f"   Use --list to see all the supported pairs.")
    else:
        msg = (f"Conversion {s_ext} -> {d_ext} NOT supported.\n"
               f"   Source category: {s_cat}   |   target: {d_cat}\n"
               f"   From {s_ext} can be converted to: "
               f"{', '.join(tutti_out)}.\n"
               f"   Example: --to {tutti_out[0][1:]}")
    return msg
