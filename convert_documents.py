"""
convert_documents.py
====================

Universal converter module for **text documents**.

Handles conversions between document formats (md, docx, odt, rtf, html, epub,
txt, tex) using **Pandoc** as the engine. The `pypandoc` library is used
if available, otherwise `pandoc.exe` is run directly via subprocess.

Public API
----------
- ``SUPPORTED_INPUTS``  : set of accepted input extensions.
- ``SUPPORTED_OUTPUTS`` : set of producible output extensions.
- ``supports(src_ext, dst_ext) -> bool`` : tells whether the pair can be handled.
- ``convert(src, dst, opts=None) -> dict`` : runs the conversion.

``convert`` does not raise exceptions for predictable cases:
it returns a dictionary ``{'ok': bool, 'note': str, 'elapsed': float}``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------

# Accepted input extensions (lowercase, with a leading dot).
SUPPORTED_INPUTS: set[str] = {
    ".md", ".markdown",
    ".docx",
    ".odt",
    ".rtf",
    ".html", ".htm",
    ".epub",
    ".txt",
    ".tex", ".latex",
}

# Producible output extensions.
SUPPORTED_OUTPUTS: set[str] = {
    ".md",
    ".docx",
    ".odt",
    ".rtf",
    ".html",
    ".epub",
    ".txt",
    ".pdf",
    ".tex",
    ".pptx",   # presentation: each H1/H2 becomes a slide
}

# Maps extension -> pandoc format (input).
_PANDOC_FROM: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".docx": "docx",
    ".odt": "odt",
    ".rtf": "rtf",
    ".html": "html",
    ".htm": "html",
    ".epub": "epub",
    ".txt": "markdown",   # plain text: treated as simple markdown
    ".tex": "latex",
    ".latex": "latex",
}

# Maps extension -> pandoc format (output).
_PANDOC_TO: dict[str, str] = {
    ".md": "markdown",
    ".docx": "docx",
    ".odt": "odt",
    ".rtf": "rtf",
    ".html": "html",
    ".epub": "epub",
    ".txt": "plain",
    ".pdf": "pdf",
    ".tex": "latex",
    ".pptx": "pptx",
}


# ---------------------------------------------------------------------------
# Utility: finding pandoc
# ---------------------------------------------------------------------------

def _find_pandoc() -> Optional[str]:
    """Returns the path of pandoc.exe if found, otherwise None."""
    # 1) Try the PATH
    path = shutil.which("pandoc")
    if path:
        return path

    # 2) Typical installation folder on Windows (winget)
    candidates = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Pandoc" / "pandoc.exe")
    program = os.environ.get("ProgramFiles")
    if program:
        candidates.append(Path(program) / "Pandoc" / "pandoc.exe")
    program_x86 = os.environ.get("ProgramFiles(x86)")
    if program_x86:
        candidates.append(Path(program_x86) / "Pandoc" / "pandoc.exe")

    for c in candidates:
        if c.is_file():
            return str(c)

    return None


def _pandoc_available() -> bool:
    """True if we can run pandoc (binary or via pypandoc)."""
    if _find_pandoc() is not None:
        return True
    try:
        import pypandoc  # type: ignore
        pypandoc.get_pandoc_version()
        return True
    except Exception:
        return False


_PANDOC_NOT_INSTALLED_MSG = (
    "Pandoc is not installed. Install it with: "
    "winget install --id Pandoc.Pandoc -e"
)


# ---------------------------------------------------------------------------
# API: supports
# ---------------------------------------------------------------------------

def supports(src_ext: str, dst_ext: str) -> bool:
    """Tells whether the pair (src_ext -> dst_ext) can be handled by this module.

    Extensions may have a leading dot or not.
    A conversion is supported if both extensions are in the sets
    and Pandoc recognizes both the source and the target format.
    """
    s = _norm_ext(src_ext)
    d = _norm_ext(dst_ext)
    if s not in SUPPORTED_INPUTS or d not in SUPPORTED_OUTPUTS:
        return False
    return s in _PANDOC_FROM and d in _PANDOC_TO


def _norm_ext(ext: str) -> str:
    if not ext:
        return ""
    ext = ext.lower().strip()
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _possible_targets_from(src_ext: str) -> list[str]:
    """Sorted list of the output extensions available from src_ext."""
    s = _norm_ext(src_ext)
    if s not in _PANDOC_FROM:
        return []
    return sorted(SUPPORTED_OUTPUTS)


# ---------------------------------------------------------------------------
# API: convert
# ---------------------------------------------------------------------------

def convert(src: Path, dst: Path, opts: Optional[dict] = None) -> dict:
    """Converts ``src`` to ``dst`` using Pandoc.

    Parameters
    ----------
    src : Path
        Source file.
    dst : Path
        Destination file (the extension sets the output format).
    opts : dict, optional
        Additional options. Recognized keys:
          - ``extra_args`` : list of extra pandoc arguments.
          - ``standalone`` : bool (default True) for the formats that need it.

    Returns
    -------
    dict with the keys:
      - ``ok`` (bool)
      - ``note`` (str)
      - ``elapsed`` (float, seconds)
    """
    t0 = time.perf_counter()
    opts = opts or {}

    src = Path(src)
    dst = Path(dst)

    # --- Basic validation -----------------------------------------------------
    if not _pandoc_available():
        return _result(False, _PANDOC_NOT_INSTALLED_MSG, t0)

    if not src.exists():
        return _result(False, f"Source file not found: {src}", t0)
    if not src.is_file():
        return _result(False, f"The source is not a regular file: {src}", t0)

    src_ext = _norm_ext(src.suffix)
    dst_ext = _norm_ext(dst.suffix)

    if src_ext not in SUPPORTED_INPUTS:
        return _result(
            False,
            f"Source extension '{src_ext}' not handled by this module.",
            t0,
        )
    if dst_ext not in SUPPORTED_OUTPUTS:
        return _result(
            False,
            f"Target extension '{dst_ext}' not handled by this module.",
            t0,
        )

    if not supports(src_ext, dst_ext):
        targets = _possible_targets_from(src_ext)
        return _result(
            False,
            (
                f"Conversione {src_ext}->{dst_ext} not supported by Pandoc. "
                f"Target formats available from {src_ext}: {targets}"
            ),
            t0,
        )

    # --- Create the destination folder if needed ------------------------------
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return _result(False, f"Cannot create the destination folder: {e}", t0)

    fmt_from = _PANDOC_FROM[src_ext]
    fmt_to = _PANDOC_TO[dst_ext]

    # --- Extra arguments ------------------------------------------------------
    extra_args: list[str] = list(opts.get("extra_args", []))

    # Standalone: needed for docx/odt/html/epub/pdf/tex to get a complete
    # file (not a fragment). Markdown/txt: left to pandoc.
    standalone = opts.get("standalone", True)
    if standalone and dst_ext in {".html", ".epub", ".tex", ".docx", ".odt", ".rtf", ".pdf"}:
        if "--standalone" not in extra_args and "-s" not in extra_args:
            extra_args.append("--standalone")

    # --- Execution: try pypandoc first, then fall back to subprocess -----------
    note_warn = ""

    used_pypandoc = False
    try:
        import pypandoc  # type: ignore
        used_pypandoc = True
    except Exception:
        used_pypandoc = False

    try:
        if used_pypandoc:
            try:
                pypandoc.convert_file(
                    str(src),
                    fmt_to,
                    format=fmt_from,
                    outputfile=str(dst),
                    extra_args=extra_args,
                )
            except Exception as e:
                # PDF: typically a missing LaTeX engine
                msg = str(e)
                if dst_ext == ".pdf" and _looks_like_pdf_engine_error(msg):
                    return _result(
                        False,
                        (
                            "Generating PDFs needs a LaTeX engine (e.g. MiKTeX or TeX Live). "
                            f"Dettagli: {msg.strip()}"
                        ),
                        t0,
                    )
                # Other errors: re-raise for the subprocess fallback
                raise
        else:
            _run_pandoc_subprocess(src, dst, fmt_from, fmt_to, extra_args)

    except FileNotFoundError:
        return _result(False, _PANDOC_NOT_INSTALLED_MSG, t0)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip() if hasattr(e, "stderr") else str(e)
        if dst_ext == ".pdf" and _looks_like_pdf_engine_error(err):
            return _result(
                False,
                (
                    "Generating PDFs needs a LaTeX engine (e.g. MiKTeX or TeX Live). "
                    f"Dettagli: {err}"
                ),
                t0,
            )
        return _result(False, f"Pandoc error: {err}", t0)
    except Exception as e:
        # Last attempt: direct subprocess (in case pypandoc fell for it).
        try:
            _run_pandoc_subprocess(src, dst, fmt_from, fmt_to, extra_args)
        except subprocess.CalledProcessError as e2:
            err = (e2.stderr or e2.stdout or "").strip()
            if dst_ext == ".pdf" and _looks_like_pdf_engine_error(err):
                return _result(
                    False,
                    (
                        "Generating PDFs needs a LaTeX engine (e.g. MiKTeX or TeX Live). "
                        f"Dettagli: {err}"
                    ),
                    t0,
                )
            return _result(False, f"Pandoc error: {err}", t0)
        except FileNotFoundError:
            return _result(False, _PANDOC_NOT_INSTALLED_MSG, t0)
        except Exception as e3:
            return _result(False, f"Unexpected error: {e3}", t0)

    # --- Output check ---------------------------------------------------------
    if not dst.exists():
        return _result(False, "Conversion completed but the output file does not exist.", t0)
    try:
        size = dst.stat().st_size
    except OSError as e:
        return _result(False, f"Cannot read the output file: {e}", t0)
    if size <= 0:
        return _result(False, "The output file is empty.", t0)

    note = f"Conversione {src_ext}->{dst_ext} riuscita ({size} byte)."
    if note_warn:
        note += " " + note_warn
    return _result(True, note, t0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _result(ok: bool, note: str, t0: float) -> dict:
    return {"ok": ok, "note": note, "elapsed": time.perf_counter() - t0}


def _run_pandoc_subprocess(
    src: Path,
    dst: Path,
    fmt_from: str,
    fmt_to: str,
    extra_args: list[str],
) -> None:
    """Runs pandoc as an external process. Raises CalledProcessError on error."""
    exe = _find_pandoc()
    if exe is None:
        raise FileNotFoundError("pandoc not found")

    cmd = [exe, "-f", fmt_from, "-t", fmt_to, str(src), "-o", str(dst), *extra_args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )


def _looks_like_pdf_engine_error(msg: str) -> bool:
    """Heuristic: does the error indicate a missing LaTeX engine for PDF?"""
    if not msg:
        return False
    m = msg.lower()
    needles = (
        "pdflatex", "xelatex", "lualatex", "wkhtmltopdf",
        "no pdf-engine", "could not find pdf",
        "pdf-engine",
        "latex engine",
        "not found",
    )
    return any(n in m for n in needles)


# ---------------------------------------------------------------------------
# Direct execution (debug)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python convert_documents.py <src> <dst>")
        sys.exit(2)
    res = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(res)
    sys.exit(0 if res["ok"] else 1)
