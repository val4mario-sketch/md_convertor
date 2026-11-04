#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_office_pdf.py — Modulo di conversione Office/PDF tramite LibreOffice
headless (`soffice --headless --convert-to ...`).

Espone l'API uniforme richiesta dal convertitore universale:

    convert(src: Path, dst: Path, opts: dict | None = None) -> dict
        ritorna {'ok': bool, 'note': str, 'elapsed': float}
        Non solleva eccezioni prevedibili. Se `soffice` non è installato,
        ok=False con un messaggio che spiega come installarlo.

    SUPPORTED_INPUTS  : frozenset di estensioni accettate
    SUPPORTED_OUTPUTS : frozenset di estensioni producibili
    supports(src_ext, dst_ext) -> bool

Strategia:
  - Localizza `soffice` (PATH oppure `C:\\Program Files\\LibreOffice\\program\\soffice.exe`,
    anche x86, oppure su Linux/macOS).
  - Lancia `soffice --headless --convert-to <ext[:filter]> --outdir <tmpdir> <src>`
    con `subprocess.run`, poi sposta/rinomina il file prodotto in `dst`.
  - I PDF in input verso .docx/.html/.odt usano il filter `writer_pdf_import`
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
# Estensioni supportate
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

# Categorie per validare la coppia src->dst
_WORD_LIKE  = {".docx", ".doc", ".odt", ".rtf"}
_CALC_LIKE  = {".xlsx", ".xls", ".ods"}
_IMPRESS    = {".pptx", ".ppt", ".odp"}
_PDF        = {".pdf"}

# Output che hanno senso a partire da ogni categoria.
# (LibreOffice può produrre PDF da tutto; gli altri richiedono coerenza tipo).
_WORD_OUT   = {".pdf", ".docx", ".odt", ".html", ".txt", ".rtf"}
_CALC_OUT   = {".pdf", ".xlsx", ".ods", ".csv", ".html"}
_IMPRESS_OUT = {".pdf", ".pptx", ".odp", ".html"}
# PDF in input: il filter writer_pdf_import lo importa come documento testo.
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
    """True se questo modulo può convertire src_ext -> dst_ext."""
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
# Localizzazione del binario soffice
# ---------------------------------------------------------------------------

def _candidate_soffice_paths() -> Iterable[str]:
    # Variabile esplicita ha la precedenza
    env = os.environ.get("SOFFICE_BIN") or os.environ.get("LIBREOFFICE_BIN")
    if env:
        yield env

    # PATH
    for name in ("soffice", "soffice.com", "soffice.exe"):
        found = shutil.which(name)
        if found:
            yield found

    # Percorsi tipici Windows
    yield r"C:\Program Files\LibreOffice\program\soffice.exe"
    yield r"C:\Program Files\LibreOffice\program\soffice.com"
    yield r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"
    yield r"C:\Program Files (x86)\LibreOffice\program\soffice.com"

    # macOS
    yield "/Applications/LibreOffice.app/Contents/MacOS/soffice"

    # Linux usuali
    yield "/usr/bin/soffice"
    yield "/usr/local/bin/soffice"


def _find_soffice() -> str | None:
    for p in _candidate_soffice_paths():
        if p and Path(p).is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Mappa estensione di output -> argomento `--convert-to` di LibreOffice
# ---------------------------------------------------------------------------

# Filtri specifici quando serve forzare il formato (es. xlsx->csv UTF-8, PDF in).
def _convert_to_arg(src_ext: str, dst_ext: str) -> str:
    se = src_ext.lower()
    de = dst_ext.lower()

    # CSV: usa il filter "csv" con tokens (44=, 34=", 76=UTF-8)
    if de == ".csv":
        return "csv:Text - txt - csv (StarCalc):44,34,76,1,,0"

    # HTML da Calc / Impress / Writer: usa filter dedicato se serve
    if de == ".html":
        if se in _CALC_LIKE:
            return "html:HTML (StarCalc)"
        if se in _IMPRESS:
            return "html:impress_html_Export"
        # writer e pdf->html
        return "html:HTML (StarWriter)"

    # TXT: testo Writer
    if de == ".txt":
        return "txt:Text"

    # ODT/DOCX/RTF da PDF: il target usa il normale filtro di export Writer.
    # ATTENZIONE: il `--convert-to` indica solo il FORMATO DI USCITA. Per l'INGRESSO
    # bisogna forzare il filtro `writer_pdf_import` (vedi `_infilter_arg`), altrimenti
    # LibreOffice apre il PDF in Draw e il salvataggio come .docx fallisce con
    # "SfxBaseModel::impl_store ... failed".
    if se == ".pdf" and de in {".docx", ".odt", ".rtf"}:
        if de == ".docx":
            return "docx:MS Word 2007 XML"
        if de == ".odt":
            return "odt"
        if de == ".rtf":
            return "rtf"

    # PPTX da .ppt/.odp
    if de == ".pptx":
        return "pptx:Impress MS PowerPoint 2007 XML"

    # XLSX da .xls/.ods
    if de == ".xlsx":
        return "xlsx:Calc MS Excel 2007 XML"

    # DOCX da .doc/.odt/.rtf
    if de == ".docx":
        return "docx:MS Word 2007 XML"

    # ODT generico
    if de == ".odt":
        return "odt"

    # ODP generico
    if de == ".odp":
        return "odp"

    # ODS generico
    if de == ".ods":
        return "ods"

    # PDF: lascia che LO scelga il filter giusto in base al sorgente
    if de == ".pdf":
        return "pdf"

    return de.lstrip(".")


def _infilter_arg(src_ext: str, dst_ext: str) -> str | None:
    """Filtro di IMPORT da forzare con `--infilter`.

    Necessario per i PDF: se non si forza `writer_pdf_import`, LibreOffice apre
    il PDF in Draw (l'associazione predefinita per i .pdf) e il salvataggio verso
    un formato Writer (.docx/.odt/.rtf/.txt/.html) fallisce con
    "SfxBaseModel::impl_store ... failed (Error Area:Io Class:Write)".
    Forzando il filtro, il PDF viene aperto come documento Writer e l'export riesce.
    """
    if src_ext.lower() == ".pdf":
        return "writer_pdf_import"
    return None


def _list_alternatives(src_ext: str) -> list[str]:
    outs = sorted(_category_outputs(src_ext) & SUPPORTED_OUTPUTS)
    return outs


def _pdf_to_txt_pymupdf(src: Path, dst: Path, t0: float) -> dict | None:
    """Estrae testo semplice da un PDF con PyMuPDF (fitz).

    Serve perché LibreOffice, importando il PDF in Writer, mette il testo in
    caselle posizionate che il filtro `txt:Text` NON esporta: il risultato
    sarebbe un .txt quasi vuoto. PyMuPDF estrae il vero livello testo del PDF.

    Ritorna un dict risultato se ha prodotto testo, `None` se PyMuPDF non è
    disponibile o il PDF non ha testo estraibile (così si ricade su LibreOffice).
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
            return None  # PDF scansionato/immagine: lascia provare LibreOffice
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text + "\n", encoding="utf-8")
        return {"ok": True,
                "note": f"convertito .pdf -> .txt via PyMuPDF ({len(text)} caratteri)",
                "elapsed": time.perf_counter() - t0}
    except Exception:
        return None  # in caso di problemi, fallback a LibreOffice


# ---------------------------------------------------------------------------
# Conversione
# ---------------------------------------------------------------------------

INSTALL_HINT = (
    "LibreOffice non e' installato o `soffice` non e' raggiungibile. "
    "Installalo con: winget install --id TheDocumentFoundation.LibreOffice -e "
    "(oppure scaricalo da https://www.libreoffice.org). "
    "In alternativa imposta la variabile d'ambiente SOFFICE_BIN con il percorso completo."
)


def convert(src: Path, dst: Path, opts: dict | None = None) -> dict:
    """Converte `src` -> `dst` tramite LibreOffice headless.

    Ritorna sempre un dict {'ok': bool, 'note': str, 'elapsed': float}.
    Non solleva eccezioni prevedibili (FileNotFoundError, subprocess errors,
    timeouts, ecc.): vengono catturate e riportate in 'note'.
    """
    t0 = time.perf_counter()
    opts = opts or {}

    try:
        src = Path(src)
        dst = Path(dst)
    except Exception as ex:  # pragma: no cover - difensivo
        return {"ok": False, "note": f"path non valido: {ex}", "elapsed": 0.0}

    se = src.suffix.lower()
    de = dst.suffix.lower()

    if not src.exists():
        return {"ok": False, "note": f"file sorgente non trovato: {src}",
                "elapsed": time.perf_counter() - t0}

    if se not in SUPPORTED_INPUTS:
        return {"ok": False,
                "note": f"estensione di input '{se}' non supportata da questo modulo. "
                        f"Supportate: {sorted(SUPPORTED_INPUTS)}",
                "elapsed": time.perf_counter() - t0}

    if de not in SUPPORTED_OUTPUTS:
        return {"ok": False,
                "note": f"estensione di output '{de}' non supportata da questo modulo. "
                        f"Supportate: {sorted(SUPPORTED_OUTPUTS)}",
                "elapsed": time.perf_counter() - t0}

    if not supports(se, de):
        alt = _list_alternatives(se)
        return {"ok": False,
                "note": (f"conversione {se} -> {de} non supportata. "
                         f"Alternative per {se}: {alt}"),
                "elapsed": time.perf_counter() - t0}

    # PDF -> TXT: usa PyMuPDF (testo reale). Se il PDF non ha testo estraibile
    # (scansione) o PyMuPDF manca, si prosegue con LibreOffice qui sotto.
    if se == ".pdf" and de == ".txt":
        r = _pdf_to_txt_pymupdf(src, dst, t0)
        if r is not None:
            return r

    soffice = _find_soffice()
    if not soffice:
        return {"ok": False, "note": INSTALL_HINT,
                "elapsed": time.perf_counter() - t0}

    # XLSX -> PDF: pre-processa il foglio con openpyxl per impostare
    # "fit to width" e orientamento orizzontale (colonne larghe non tagliate).
    # Lavora su una COPIA temporanea: il file originale resta intatto.
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
                ws.page_setup.fitToHeight = 0          # altezza libera (piu' pagine)
                ws.sheet_properties.pageSetUpPr.fitToPage = True
                # orizzontale se il foglio e' largo (> 7 colonne usate)
                if ws.max_column and ws.max_column > 7:
                    ws.page_setup.orientation = "landscape"
                # auto-larghezza colonne: niente testo clippato in stampa
                for col_idx in range(1, (ws.max_column or 0) + 1):
                    letter = get_column_letter(col_idx)
                    if ws.column_dimensions[letter].width:   # gia' impostata a mano
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
            src = _fit_tmp                              # converti la copia
            fit_note = " (fit-to-width applicato)"
        except Exception:
            pass  # se il preprocess fallisce, si converte l'originale com'e'

    # Prepara cartella di output temporanea, così possiamo prendere il file
    # generato da LibreOffice (che usa il nome di src + nuova estensione).
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except Exception as ex:
        return {"ok": False, "note": f"impossibile creare cartella di destinazione: {ex}",
                "elapsed": time.perf_counter() - t0}

    timeout = float(opts.get("timeout", 180))
    convert_to = _convert_to_arg(se, de)
    infilter = _infilter_arg(se, de)

    with tempfile.TemporaryDirectory(prefix="lo_conv_") as tmp:
        tmpdir = Path(tmp)

        # Usa un profilo utente isolato così non interferisce con eventuali
        # istanze di LibreOffice aperte dall'utente.
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
        # Forza il filtro di import quando serve (indispensabile per i PDF).
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
                    "note": f"timeout ({timeout:.0f}s) durante la conversione con soffice",
                    "elapsed": time.perf_counter() - t0}
        except Exception as ex:
            return {"ok": False, "note": f"errore eseguendo soffice: {ex}",
                    "elapsed": time.perf_counter() - t0}

        # LibreOffice nomina l'output come <stem(src)><.de> nella outdir
        produced = tmpdir / (src.stem + de)
        if not produced.exists():
            # Talvolta LO produce un nome leggermente diverso (es. con filtri
            # particolari). Prendi il primo file con l'estensione richiesta.
            candidates = list(tmpdir.glob(f"*{de}"))
            if candidates:
                produced = candidates[0]

        if not produced.exists():
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            tail = (stderr or stdout or "nessun output da soffice").splitlines()
            tail_txt = " | ".join(tail[-3:])
            return {"ok": False,
                    "note": (f"soffice non ha prodotto il file atteso "
                             f"(rc={proc.returncode}). {tail_txt}"),
                    "elapsed": time.perf_counter() - t0}

        # Sposta/rinomina in dst (sovrascrivendo se esiste)
        try:
            if dst.exists():
                dst.unlink()
            shutil.move(str(produced), str(dst))
        except Exception as ex:
            return {"ok": False,
                    "note": f"impossibile spostare l'output in {dst}: {ex}",
                    "elapsed": time.perf_counter() - t0}

    elapsed = time.perf_counter() - t0
    return {"ok": True,
            "note": f"convertito {se} -> {de} via LibreOffice ({Path(soffice).name}){fit_note}",
            "elapsed": elapsed}


# ---------------------------------------------------------------------------
# CLI minimale per uso diretto / debug
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
