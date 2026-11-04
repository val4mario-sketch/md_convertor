"""
convert_documents.py
====================

Modulo del convertitore universale per **documenti testuali**.

Gestisce conversioni tra formati documentali (md, docx, odt, rtf, html, epub,
txt, tex) usando **Pandoc** come motore. La libreria `pypandoc` viene usata
se disponibile, altrimenti si usa direttamente `pandoc.exe` via subprocess.

API pubblica
------------
- ``SUPPORTED_INPUTS``  : set di estensioni accettate in ingresso.
- ``SUPPORTED_OUTPUTS`` : set di estensioni producibili in uscita.
- ``supports(src_ext, dst_ext) -> bool`` : indica se la coppia e' gestibile.
- ``convert(src, dst, opts=None) -> dict`` : esegue la conversione.

La funzione ``convert`` non solleva eccezioni per casi prevedibili:
restituisce un dizionario ``{'ok': bool, 'note': str, 'elapsed': float}``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Formati supportati
# ---------------------------------------------------------------------------

# Estensioni di input accettate (in minuscolo, con punto iniziale).
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

# Estensioni di output producibili.
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
    ".pptx",   # presentazione: ogni H1/H2 diventa una slide
}

# Mappa estensione -> formato pandoc (input).
_PANDOC_FROM: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".docx": "docx",
    ".odt": "odt",
    ".rtf": "rtf",
    ".html": "html",
    ".htm": "html",
    ".epub": "epub",
    ".txt": "markdown",   # plain text: trattato come markdown semplice
    ".tex": "latex",
    ".latex": "latex",
}

# Mappa estensione -> formato pandoc (output).
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
# Utility: scoperta di pandoc
# ---------------------------------------------------------------------------

def _find_pandoc() -> Optional[str]:
    """Restituisce il percorso a pandoc.exe se trovato, altrimenti None."""
    # 1) Tentativo via PATH
    path = shutil.which("pandoc")
    if path:
        return path

    # 2) Cartella di installazione tipica su Windows (winget)
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
    """True se possiamo eseguire pandoc (binario o via pypandoc)."""
    if _find_pandoc() is not None:
        return True
    try:
        import pypandoc  # type: ignore
        pypandoc.get_pandoc_version()
        return True
    except Exception:
        return False


_PANDOC_NOT_INSTALLED_MSG = (
    "Pandoc non installato. Installalo con: "
    "winget install --id Pandoc.Pandoc -e"
)


# ---------------------------------------------------------------------------
# API: supports
# ---------------------------------------------------------------------------

def supports(src_ext: str, dst_ext: str) -> bool:
    """Indica se la coppia (src_ext -> dst_ext) e' gestibile da questo modulo.

    Le estensioni possono essere con o senza punto iniziale.
    Una conversione e' supportata se entrambe le estensioni sono nei set
    e Pandoc riconosce sia il formato sorgente sia quello destinazione.
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
    """Lista (ordinata) delle estensioni di output disponibili da src_ext."""
    s = _norm_ext(src_ext)
    if s not in _PANDOC_FROM:
        return []
    return sorted(SUPPORTED_OUTPUTS)


# ---------------------------------------------------------------------------
# API: convert
# ---------------------------------------------------------------------------

def convert(src: Path, dst: Path, opts: Optional[dict] = None) -> dict:
    """Converte ``src`` in ``dst`` usando Pandoc.

    Parametri
    ---------
    src : Path
        File sorgente.
    dst : Path
        File di destinazione (l'estensione determina il formato di uscita).
    opts : dict, opzionale
        Opzioni aggiuntive. Chiavi riconosciute:
          - ``extra_args`` : lista di argomenti pandoc extra.
          - ``standalone`` : bool (default True) per i formati che lo richiedono.

    Ritorna
    -------
    dict con chiavi:
      - ``ok`` (bool)
      - ``note`` (str)
      - ``elapsed`` (float, secondi)
    """
    t0 = time.perf_counter()
    opts = opts or {}

    src = Path(src)
    dst = Path(dst)

    # --- Validazioni di base --------------------------------------------------
    if not _pandoc_available():
        return _result(False, _PANDOC_NOT_INSTALLED_MSG, t0)

    if not src.exists():
        return _result(False, f"File sorgente non trovato: {src}", t0)
    if not src.is_file():
        return _result(False, f"La sorgente non e' un file regolare: {src}", t0)

    src_ext = _norm_ext(src.suffix)
    dst_ext = _norm_ext(dst.suffix)

    if src_ext not in SUPPORTED_INPUTS:
        return _result(
            False,
            f"Estensione sorgente '{src_ext}' non gestita da questo modulo.",
            t0,
        )
    if dst_ext not in SUPPORTED_OUTPUTS:
        return _result(
            False,
            f"Estensione destinazione '{dst_ext}' non gestita da questo modulo.",
            t0,
        )

    if not supports(src_ext, dst_ext):
        targets = _possible_targets_from(src_ext)
        return _result(
            False,
            (
                f"Conversione {src_ext}->{dst_ext} non supportata da Pandoc. "
                f"Formati di destinazione disponibili da {src_ext}: {targets}"
            ),
            t0,
        )

    # --- Crea la cartella di destinazione se serve ----------------------------
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return _result(False, f"Impossibile creare la cartella di destinazione: {e}", t0)

    fmt_from = _PANDOC_FROM[src_ext]
    fmt_to = _PANDOC_TO[dst_ext]

    # --- Argomenti extra ------------------------------------------------------
    extra_args: list[str] = list(opts.get("extra_args", []))

    # Standalone: serve per docx/odt/html/epub/pdf/tex per ottenere un file
    # completo (non un frammento). Markdown/txt: lasciamo a pandoc.
    standalone = opts.get("standalone", True)
    if standalone and dst_ext in {".html", ".epub", ".tex", ".docx", ".odt", ".rtf", ".pdf"}:
        if "--standalone" not in extra_args and "-s" not in extra_args:
            extra_args.append("--standalone")

    # --- Esecuzione: prima tenta via pypandoc, poi fallback subprocess --------
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
                # PDF: tipicamente mancanza di engine LaTeX
                msg = str(e)
                if dst_ext == ".pdf" and _looks_like_pdf_engine_error(msg):
                    return _result(
                        False,
                        (
                            "Per generare PDF serve un engine LaTeX (es. MiKTeX o TeX Live). "
                            f"Dettagli: {msg.strip()}"
                        ),
                        t0,
                    )
                # Altri errori: rilancia per fallback subprocess
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
                    "Per generare PDF serve un engine LaTeX (es. MiKTeX o TeX Live). "
                    f"Dettagli: {err}"
                ),
                t0,
            )
        return _result(False, f"Errore Pandoc: {err}", t0)
    except Exception as e:
        # Ultimo tentativo: subprocess diretto (in caso pypandoc abbia abboccato).
        try:
            _run_pandoc_subprocess(src, dst, fmt_from, fmt_to, extra_args)
        except subprocess.CalledProcessError as e2:
            err = (e2.stderr or e2.stdout or "").strip()
            if dst_ext == ".pdf" and _looks_like_pdf_engine_error(err):
                return _result(
                    False,
                    (
                        "Per generare PDF serve un engine LaTeX (es. MiKTeX o TeX Live). "
                        f"Dettagli: {err}"
                    ),
                    t0,
                )
            return _result(False, f"Errore Pandoc: {err}", t0)
        except FileNotFoundError:
            return _result(False, _PANDOC_NOT_INSTALLED_MSG, t0)
        except Exception as e3:
            return _result(False, f"Errore inatteso: {e3}", t0)

    # --- Verifica output ------------------------------------------------------
    if not dst.exists():
        return _result(False, "Conversione completata ma il file di output non esiste.", t0)
    try:
        size = dst.stat().st_size
    except OSError as e:
        return _result(False, f"Impossibile leggere il file di output: {e}", t0)
    if size <= 0:
        return _result(False, "Il file di output e' vuoto.", t0)

    note = f"Conversione {src_ext}->{dst_ext} riuscita ({size} byte)."
    if note_warn:
        note += " " + note_warn
    return _result(True, note, t0)


# ---------------------------------------------------------------------------
# Helpers privati
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
    """Esegue pandoc come processo esterno. Solleva CalledProcessError su errore."""
    exe = _find_pandoc()
    if exe is None:
        raise FileNotFoundError("pandoc non trovato")

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
    """Euristica: l'errore segnala mancanza di engine LaTeX per il PDF?"""
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
# Esecuzione diretta (debug)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: python convert_documents.py <src> <dst>")
        sys.exit(2)
    res = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(res)
    sys.exit(0 if res["ok"] else 1)
