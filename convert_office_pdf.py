#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_office_pdf.py — Office/PDF conversion module using headless LibreOffice
(`soffice --headless --convert-to ...`).

Exposes the uniform API required by the universal converter:

    convert(src: Path, dst: Path, opts: dict | None = None) -> dict
        returns {'ok': bool, 'note': str, 'elapsed': float}
        Raises no predictable exceptions. If `soffice` is not installed,
        ok=False with a message explaining how to install it.

    SUPPORTED_INPUTS  : frozenset of accepted extensions
    SUPPORTED_OUTPUTS : frozenset of producible extensions
    supports(src_ext, dst_ext) -> bool

Strategy:
  - Locates `soffice` (PATH or `C:\\Program Files\\LibreOffice\\program\\soffice.exe`,
    x86 too, or on Linux/macOS).
  - Runs `soffice --headless --convert-to <ext[:filter]> --outdir <tmpdir> <src>`
    with `subprocess.run`, then moves/renames the produced file to `dst`.
  - Input PDFs to .docx/.html/.odt use the `writer_pdf_import` filter
    (LibreOffice 7+).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

SUPPORTED_INPUTS = frozenset({
    ".docx", ".doc", ".odt", ".rtf",
    ".xlsx", ".xls", ".ods",
    ".pptx", ".ppt", ".odp",
    ".pdf",
})

SUPPORTED_OUTPUTS = frozenset({
    ".pdf", ".docx", ".odt", ".html", ".txt",
    ".xlsx", ".csv", ".pptx",
})

# Categories for validating the src->dst pair
_WORD_LIKE  = {".docx", ".doc", ".odt", ".rtf"}
_CALC_LIKE  = {".xlsx", ".xls", ".ods"}
_IMPRESS    = {".pptx", ".ppt", ".odp"}
_PDF        = {".pdf"}

# Outputs that make sense for each category.
# (LibreOffice can produce PDF from anything; the others need matching types.)
_WORD_OUT   = {".pdf", ".docx", ".odt", ".html", ".txt", ".rtf"}
_CALC_OUT   = {".pdf", ".xlsx", ".ods", ".csv", ".html"}
_IMPRESS_OUT = {".pdf", ".pptx", ".odp", ".html"}
# Input PDF: the writer_pdf_import filter imports it as a text document.
_PDF_OUT    = {".docx", ".odt", ".html", ".txt"}


def _category_outputs(src_ext: str) -> set[str]:
    e = src_ext.lower()
    if e in _WORD_LIKE:
        return _WORD_OUT
    if e in _CALC_LIKE:
        return _CALC_OUT
    if e in _IMPRESS:
        return _IMPRESS_OUT
    if e in _PDF:
        return _PDF_OUT
    return set()


def supports(src_ext: str, dst_ext: str) -> bool:
    """True if this module can convert src_ext -> dst_ext."""
    se = src_ext.lower()
    de = dst_ext.lower()
    if not se.startswith("."):
        se = "." + se
    if not de.startswith("."):
        de = "." + de
    if se not in SUPPORTED_INPUTS or de not in SUPPORTED_OUTPUTS:
        return False
    return de in _category_outputs(se)


# ---------------------------------------------------------------------------
# Locating the soffice binary
# ---------------------------------------------------------------------------

def _candidate_soffice_paths() -> Iterable[str]:
    # An explicit variable takes precedence
    env = os.environ.get("SOFFICE_BIN") or os.environ.get("LIBREOFFICE_BIN")
    if env:
        yield env

    # PATH
    for name in ("soffice", "soffice.com", "soffice.exe"):
        found = shutil.which(name)
        if found:
            yield found

    # Typical Windows paths
    yield r"C:\Program Files\LibreOffice\program\soffice.exe"
    yield r"C:\Program Files\LibreOffice\program\soffice.com"
    yield r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"
    yield r"C:\Program Files (x86)\LibreOffice\program\soffice.com"

    # macOS
    yield "/Applications/LibreOffice.app/Contents/MacOS/soffice"

    # Usual Linux paths
    yield "/usr/bin/soffice"
    yield "/usr/local/bin/soffice"


def _find_soffice() -> str | None:
    for p in _candidate_soffice_paths():
        if p and Path(p).is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Maps the output extension to the LibreOffice `--convert-to` argument
# ---------------------------------------------------------------------------

# Specific filters when the format must be forced (e.g. xlsx->csv UTF-8, PDF in).
def _convert_to_arg(src_ext: str, dst_ext: str) -> str:
    se = src_ext.lower()
    de = dst_ext.lower()

    # CSV: use the "csv" filter with tokens (44=, 34=", 76=UTF-8)
    if de == ".csv":
        return "csv:Text - txt - csv (StarCalc):44,34,76,1,,0"

    # HTML from Calc / Impress / Writer: use a dedicated filter if needed
    if de == ".html":
        if se in _CALC_LIKE:
            return "html:HTML (StarCalc)"
        if se in _IMPRESS:
            return "html:impress_html_Export"
        # writer and pdf->html
        return "html:HTML (StarWriter)"

    # TXT: Writer text
    if de == ".txt":
        return "txt:Text"

    # ODT/DOCX/RTF from PDF: the target uses the normal Writer export filter.
    # WARNING: `--convert-to` only states the OUTPUT FORMAT. For the INPUT
    # the `writer_pdf_import` filter must be forced (see `_infilter_arg`), otherwise
    # LibreOffice opens the PDF in Draw and saving as .docx fails with
    # "SfxBaseModel::impl_store ... failed".
    if se == ".pdf" and de in {".docx", ".odt", ".rtf"}:
        if de == ".docx":
            return "docx:MS Word 2007 XML"
        if de == ".odt":
            return "odt"
        if de == ".rtf":
            return "rtf"

    # PPTX from .ppt/.odp
    if de == ".pptx":
        return "pptx:Impress MS PowerPoint 2007 XML"

    # XLSX from .xls/.ods
    if de == ".xlsx":
        return "xlsx:Calc MS Excel 2007 XML"

    # DOCX from .doc/.odt/.rtf
    if de == ".docx":
        return "docx:MS Word 2007 XML"

    # Generic ODT
    if de == ".odt":
        return "odt"

    # Generic ODP
    if de == ".odp":
        return "odp"

    # Generic ODS
    if de == ".ods":
        return "ods"

    # PDF: let LO choose the right filter based on the source
    if de == ".pdf":
        return "pdf"

    return de.lstrip(".")


def _infilter_arg(src_ext: str, dst_ext: str) -> str | None:
    """IMPORT filter to force with `--infilter`.

    Needed for PDFs: unless `writer_pdf_import` is forced, LibreOffice opens
    the PDF in Draw (the default association for .pdf) and saving to
    a Writer format (.docx/.odt/.rtf/.txt/.html) fails with
    "SfxBaseModel::impl_store ... failed (Error Area:Io Class:Write)".
    With the filter forced, the PDF opens as a Writer document and the export succeeds.
    """
    if src_ext.lower() == ".pdf":
        return "writer_pdf_import"
    return None


def _list_alternatives(src_ext: str) -> list[str]:
    outs = sorted(_category_outputs(src_ext) & SUPPORTED_OUTPUTS)
    return outs


def _pdf_to_txt_pymupdf(src: Path, dst: Path, t0: float) -> dict | None:
    """Extracts plain text from a PDF with PyMuPDF (fitz).

    Needed because LibreOffice, when importing the PDF into Writer, places text in
    positioned boxes that the `txt:Text` filter does NOT export: the result
    would be an almost empty .txt. PyMuPDF extracts the PDF's real text layer.

    Returns a result dict if it produced text, `None` if PyMuPDF is not
    available or the PDF has no extractable text (so it falls back to LibreOffice).
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None
    try:
        parts: list[str] = []
        with fitz.open(str(src)) as doc:
            for page in doc:
                parts.append(page.get_text("text"))
        text = "\n".join(parts).strip()
        if not text:
            return None  # Scanned PDF/image: let LibreOffice try
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text + "\n", encoding="utf-8")
        return {"ok": True,
                "note": f"converted .pdf -> .txt via PyMuPDF ({len(text)} characters)",
                "elapsed": time.perf_counter() - t0}
    except Exception:
        return None  # if anything goes wrong, fall back to LibreOffice


# ---------------------------------------------------------------------------
# Conversione
# ---------------------------------------------------------------------------

INSTALL_HINT = (
    "LibreOffice is not installed or `soffice` cannot be reached. "
    "Install it with: winget install --id TheDocumentFoundation.LibreOffice -e "
    "(or download it from https://www.libreoffice.org). "
    "Alternatively set the SOFFICE_BIN environment variable to the full path."
)


def convert(src: Path, dst: Path, opts: dict | None = None) -> dict:
    """Converts `src` -> `dst` using headless LibreOffice.

    Always returns a dict {'ok': bool, 'note': str, 'elapsed': float}.
    Raises no predictable exceptions (FileNotFoundError, subprocess errors,
    timeouts, etc.): they are caught and reported in 'note'.
    """
    t0 = time.perf_counter()
    opts = opts or {}

    try:
        src = Path(src)
        dst = Path(dst)
    except Exception as ex:  # pragma: no cover - defensive
        return {"ok": False, "note": f"invalid path: {ex}", "elapsed": 0.0}

    se = src.suffix.lower()
    de = dst.suffix.lower()

    if not src.exists():
        return {"ok": False, "note": f"source file not found: {src}",
                "elapsed": time.perf_counter() - t0}

    if se not in SUPPORTED_INPUTS:
        return {"ok": False,
                "note": f"input extension '{se}' not supported by this module. "
                        f"Supported: {sorted(SUPPORTED_INPUTS)}",
                "elapsed": time.perf_counter() - t0}

    if de not in SUPPORTED_OUTPUTS:
        return {"ok": False,
                "note": f"output extension '{de}' not supported by this module. "
                        f"Supported: {sorted(SUPPORTED_OUTPUTS)}",
                "elapsed": time.perf_counter() - t0}

    if not supports(se, de):
        alt = _list_alternatives(se)
        return {"ok": False,
                "note": (f"conversion {se} -> {de} not supported. "
                         f"Alternatives for {se}: {alt}"),
                "elapsed": time.perf_counter() - t0}

    # PDF -> TXT: use PyMuPDF (real text). If the PDF has no extractable text
    # (scan) or PyMuPDF is missing, continue with LibreOffice below.
    if se == ".pdf" and de == ".txt":
        r = _pdf_to_txt_pymupdf(src, dst, t0)
        if r is not None:
            return r

    soffice = _find_soffice()
    if not soffice:
        return {"ok": False, "note": INSTALL_HINT,
                "elapsed": time.perf_counter() - t0}

    # XLSX -> PDF: pre-process the sheet with openpyxl to set
    # "fit to width" and landscape orientation (wide columns not cut off).
    # Works on a temporary COPY: the original file stays intact.
    fit_note = ""
    if se == ".xlsx" and de == ".pdf" and not opts.get("no_fit_page"):
        try:
            from openpyxl import load_workbook
            import tempfile as _tf
            _fit_tmp = Path(_tf.mkdtemp(prefix="xlsx_fit_")) / src.name
            shutil.copyfile(src, _fit_tmp)
            from openpyxl.utils import get_column_letter
            wb = load_workbook(str(_fit_tmp))
            for ws in wb.worksheets:
                ws.page_setup.fitToWidth = 1
                ws.page_setup.fitToHeight = 0          # free height (several pages)
                ws.sheet_properties.pageSetUpPr.fitToPage = True
                # landscape if the sheet is wide (> 7 columns used)
                if ws.max_column and ws.max_column > 7:
                    ws.page_setup.orientation = "landscape"
                # automatic column widths: no text clipped when printing
                for col_idx in range(1, (ws.max_column or 0) + 1):
                    letter = get_column_letter(col_idx)
                    if ws.column_dimensions[letter].width:   # already set by hand
                        continue
                    longest = 0
                    for row in ws.iter_rows(min_col=col_idx, max_col=col_idx,
                                            values_only=True):
                        v = row[0]
                        if v is not None:
                            longest = max(longest, len(str(v)))
                    if longest:
                        ws.column_dimensions[letter].width = min(longest + 2, 60)
            wb.save(str(_fit_tmp))
            src = _fit_tmp                              # convert the copy
            fit_note = " (fit-to-width applied)"
        except Exception:
            pass  # if the preprocessing fails, the original is converted as it is

    # Prepare a temporary output folder, so we can pick up the file
    # generated by LibreOffice (which uses the src name + new extension).
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except Exception as ex:
        return {"ok": False, "note": f"cannot create the destination folder: {ex}",
                "elapsed": time.perf_counter() - t0}

    timeout = float(opts.get("timeout", 180))
    convert_to = _convert_to_arg(se, de)
    infilter = _infilter_arg(se, de)

    with tempfile.TemporaryDirectory(prefix="lo_conv_") as tmp:
        tmpdir = Path(tmp)

        # Use an isolated user profile so it does not interfere with any
        # LibreOffice instances opened by the user.
        user_profile = tmpdir / "lo_profile"
        user_profile_uri = user_profile.as_uri()

        cmd = [
            soffice,
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--norestore",
            f"-env:UserInstallation={user_profile_uri}",
        ]
        # Force the import filter when needed (essential for PDFs).
        if infilter:
            cmd.append(f"--infilter={infilter}")
        cmd += [
            "--convert-to", convert_to,
            "--outdir", str(tmpdir),
            str(src),
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return {"ok": False, "note": INSTALL_HINT,
                    "elapsed": time.perf_counter() - t0}
        except subprocess.TimeoutExpired:
            return {"ok": False,
                    "note": f"timeout ({timeout:.0f}s) during the conversion with soffice",
                    "elapsed": time.perf_counter() - t0}
        except Exception as ex:
            return {"ok": False, "note": f"error running soffice: {ex}",
                    "elapsed": time.perf_counter() - t0}

        # LibreOffice names the output <stem(src)><.de> in the outdir
        produced = tmpdir / (src.stem + de)
        if not produced.exists():
            # Sometimes LO produces a slightly different name (e.g. with special
            # filters). Take the first file with the requested extension.
            candidates = list(tmpdir.glob(f"*{de}"))
            if candidates:
                produced = candidates[0]

        if not produced.exists():
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            tail = (stderr or stdout or "no output from soffice").splitlines()
            tail_txt = " | ".join(tail[-3:])
            return {"ok": False,
                    "note": (f"soffice did not produce the expected file "
                             f"(rc={proc.returncode}). {tail_txt}"),
                    "elapsed": time.perf_counter() - t0}

        # Move/rename to dst (overwriting if it exists)
        try:
            if dst.exists():
                dst.unlink()
            shutil.move(str(produced), str(dst))
        except Exception as ex:
            return {"ok": False,
                    "note": f"cannot move the output to {dst}: {ex}",
                    "elapsed": time.perf_counter() - t0}

    elapsed = time.perf_counter() - t0
    return {"ok": True,
            "note": f"converted {se} -> {de} via LibreOffice ({Path(soffice).name}){fit_note}",
            "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Minimal CLI for direct use / debugging
# ---------------------------------------------------------------------------

def _main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Convert Office/PDF via LibreOffice")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args(argv)
    res = convert(args.src, args.dst, {"timeout": args.timeout})
    print(res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))
