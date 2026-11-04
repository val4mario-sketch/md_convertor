"""
convert_data.py
----------------
Modulo di conversione per dati tabellari.

Estensioni supportate in input : .csv .tsv .xlsx .xls .ods .json
Estensioni supportate in output: .csv .tsv .xlsx .json .parquet

Funzione principale:
    convert(src: Path, dst: Path, opts: dict | None = None) -> dict

Ritorna sempre un dict con chiavi:
    ok      -> bool
    note    -> str (messaggio informativo o di errore)
    elapsed -> float (secondi)

Le eccezioni prevedibili vengono catturate e tradotte in ok=False con messaggio
amichevole.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Capacita' dichiarate
# ---------------------------------------------------------------------------

SUPPORTED_INPUTS: tuple[str, ...] = (
    ".csv", ".tsv", ".xlsx", ".xls", ".ods", ".json",
)

SUPPORTED_OUTPUTS: tuple[str, ...] = (
    ".csv", ".tsv", ".xlsx", ".json", ".parquet",
)

# Coppie (src_ext, dst_ext) effettivamente realizzate. Qualsiasi combinazione
# fra gli input/output dichiarati e' in pratica gestibile passando sempre per
# un DataFrame pandas, quindi popoliamo dinamicamente.
_SUPPORTED_PAIRS: set[tuple[str, str]] = set()
for _i in SUPPORTED_INPUTS:
    for _o in SUPPORTED_OUTPUTS:
        _SUPPORTED_PAIRS.add((_i, _o))


def supports(src_ext: str, dst_ext: str) -> bool:
    """Ritorna True se la coppia (src_ext, dst_ext) e' supportata.

    Gli argomenti possono includere o meno il punto iniziale e qualunque
    capitalizzazione.
    """
    s = _norm_ext(src_ext)
    d = _norm_ext(dst_ext)
    return (s, d) in _SUPPORTED_PAIRS


def _norm_ext(ext: str) -> str:
    if not ext:
        return ""
    ext = ext.strip().lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext


# ---------------------------------------------------------------------------
# Lettura
# ---------------------------------------------------------------------------

def _read_csv_like(src: Path, sep: str, opts: dict) -> tuple[pd.DataFrame, str]:
    encoding = opts.get("encoding")
    extra: dict = {}
    if encoding:
        extra["encoding"] = encoding
    df = pd.read_csv(src, sep=sep, **extra)
    return df, ""


def _read_excel(src: Path, opts: dict) -> tuple[pd.DataFrame, str]:
    """Legge xlsx/xls/ods. Per .ods serve odfpy."""
    ext = src.suffix.lower()
    engine = None
    if ext == ".xlsx":
        engine = "openpyxl"
    elif ext == ".xls":
        engine = "xlrd"  # potrebbe non essere installato
    elif ext == ".ods":
        engine = "odf"

    sheet_req = opts.get("sheet")
    note_extra = ""

    try:
        if sheet_req is not None:
            df = pd.read_excel(src, sheet_name=sheet_req, engine=engine)
        else:
            # Leggi tutti i fogli per poter avvisare quanti ce ne sono.
            sheets = pd.read_excel(src, sheet_name=None, engine=engine)
            if not sheets:
                raise ValueError("Il file non contiene fogli leggibili.")
            first_name = next(iter(sheets))
            df = sheets[first_name]
            if len(sheets) > 1:
                note_extra = (
                    f"Foglio usato: '{first_name}'. Trovati {len(sheets)} fogli "
                    f"totali; passare opts['sheet'] per sceglierne uno diverso."
                )
            else:
                note_extra = f"Foglio usato: '{first_name}'."
    except ImportError as e:
        # Engine mancante
        if ext == ".ods":
            raise RuntimeError(
                "Per leggere file .ods serve 'odfpy'. Installa con: pip install odfpy"
            ) from e
        if ext == ".xls":
            raise RuntimeError(
                "Per leggere file .xls serve 'xlrd' (versione 1.2.0). "
                "Installa con: pip install xlrd==1.2.0"
            ) from e
        raise

    return df, note_extra


def _read_json(src: Path, opts: dict) -> tuple[pd.DataFrame, str]:
    """Legge sia json 'lista di dict' sia 'dict di colonne'."""
    with open(src, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, list):
        df = pd.DataFrame(raw)
        note = "JSON letto come lista di record."
    elif isinstance(raw, dict):
        # Distinguiamo: dict di colonne (valori = liste della stessa lunghezza)
        # da un singolo record (valori scalari).
        values = list(raw.values())
        if values and all(isinstance(v, list) for v in values):
            df = pd.DataFrame(raw)
            note = "JSON letto come dict di colonne."
        else:
            df = pd.DataFrame([raw])
            note = "JSON letto come singolo record."
    else:
        raise ValueError("Struttura JSON non riconosciuta: attesa lista o dict.")

    return df, note


def _read_any(src: Path, opts: dict) -> tuple[pd.DataFrame, str]:
    ext = src.suffix.lower()
    if ext == ".csv":
        sep = opts.get("sep") or ","
        return _read_csv_like(src, sep=sep, opts=opts)
    if ext == ".tsv":
        sep = opts.get("sep") or "\t"
        return _read_csv_like(src, sep=sep, opts=opts)
    if ext in (".xlsx", ".xls", ".ods"):
        return _read_excel(src, opts)
    if ext == ".json":
        return _read_json(src, opts)
    raise ValueError(f"Estensione di input non supportata: {ext}")


# ---------------------------------------------------------------------------
# Scrittura
# ---------------------------------------------------------------------------

def _write_any(df: pd.DataFrame, dst: Path, opts: dict) -> str:
    ext = dst.suffix.lower()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if ext == ".csv":
        sep = opts.get("sep") or ","
        df.to_csv(dst, sep=sep, index=False)
        return f"Scritto CSV ({len(df)} righe)."

    if ext == ".tsv":
        df.to_csv(dst, sep="\t", index=False)
        return f"Scritto TSV ({len(df)} righe)."

    if ext == ".xlsx":
        sheet_name = opts.get("sheet") or "Sheet1"
        df.to_excel(dst, index=False, sheet_name=sheet_name, engine="openpyxl")
        return f"Scritto XLSX ({len(df)} righe, foglio '{sheet_name}')."

    if ext == ".json":
        orient = opts.get("orient", "records")
        df.to_json(dst, orient=orient, force_ascii=False, indent=2)
        return f"Scritto JSON orient='{orient}' ({len(df)} righe)."

    if ext == ".parquet":
        try:
            df.to_parquet(dst, index=False)
        except ImportError as e:
            raise RuntimeError(
                "Per scrivere file .parquet serve 'pyarrow' o 'fastparquet'. "
                "Installa con: pip install pyarrow"
            ) from e
        except ValueError as e:
            # pandas alza ValueError se manca l'engine parquet
            msg = str(e).lower()
            if "pyarrow" in msg or "fastparquet" in msg or "parquet" in msg:
                raise RuntimeError(
                    "Per scrivere file .parquet serve 'pyarrow' o 'fastparquet'. "
                    "Installa con: pip install pyarrow"
                ) from e
            raise
        return f"Scritto Parquet ({len(df)} righe)."

    raise ValueError(f"Estensione di output non supportata: {ext}")


# ---------------------------------------------------------------------------
# API pubblica
# ---------------------------------------------------------------------------

def convert(src: Path, dst: Path, opts: dict | None = None) -> dict:
    """Converte un file tabellare da src a dst.

    Vedi modulo docstring per dettagli sulle estensioni supportate.
    """
    t0 = time.perf_counter()
    opts = dict(opts or {})

    try:
        src = Path(src)
        dst = Path(dst)

        if not src.exists():
            return {
                "ok": False,
                "note": f"File sorgente non trovato: {src}",
                "elapsed": time.perf_counter() - t0,
            }

        src_ext = src.suffix.lower()
        dst_ext = dst.suffix.lower()

        if src_ext not in SUPPORTED_INPUTS:
            return {
                "ok": False,
                "note": (
                    f"Estensione di input '{src_ext}' non supportata. "
                    f"Supportate: {', '.join(SUPPORTED_INPUTS)}."
                ),
                "elapsed": time.perf_counter() - t0,
            }

        if dst_ext not in SUPPORTED_OUTPUTS:
            return {
                "ok": False,
                "note": (
                    f"Estensione di output '{dst_ext}' non supportata. "
                    f"Supportate: {', '.join(SUPPORTED_OUTPUTS)}."
                ),
                "elapsed": time.perf_counter() - t0,
            }

        if not supports(src_ext, dst_ext):
            return {
                "ok": False,
                "note": (
                    f"Coppia {src_ext} -> {dst_ext} non supportata. "
                    f"Prova un formato intermedio fra: "
                    f"{', '.join(SUPPORTED_OUTPUTS)}."
                ),
                "elapsed": time.perf_counter() - t0,
            }

        # --- Lettura ------------------------------------------------------
        try:
            df, read_note = _read_any(src, opts)
        except RuntimeError as e:
            # Errori amichevoli (engine mancanti, ecc.)
            return {
                "ok": False,
                "note": str(e),
                "elapsed": time.perf_counter() - t0,
            }
        except Exception as e:
            return {
                "ok": False,
                "note": f"Errore durante la lettura di {src.name}: {e}",
                "elapsed": time.perf_counter() - t0,
            }

        # --- Scrittura ----------------------------------------------------
        try:
            write_note = _write_any(df, dst, opts)
        except RuntimeError as e:
            return {
                "ok": False,
                "note": str(e),
                "elapsed": time.perf_counter() - t0,
            }
        except Exception as e:
            return {
                "ok": False,
                "note": f"Errore durante la scrittura di {dst.name}: {e}",
                "elapsed": time.perf_counter() - t0,
            }

        note_parts = [p for p in (read_note, write_note) if p]
        return {
            "ok": True,
            "note": " ".join(note_parts) if note_parts else "Conversione riuscita.",
            "elapsed": time.perf_counter() - t0,
        }

    except Exception as e:  # rete di sicurezza
        return {
            "ok": False,
            "note": f"Errore imprevisto: {e}",
            "elapsed": time.perf_counter() - t0,
        }
