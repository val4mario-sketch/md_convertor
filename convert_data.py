"""
convert_data.py
----------------
Conversion module for tabular data.

Supported input extensions: .csv .tsv .xlsx .xls .ods .json
Supported output extensions: .csv .tsv .xlsx .json .parquet

Main function:
    convert(src: Path, dst: Path, opts: dict | None = None) -> dict

Always returns a dict with the keys:
    ok      -> bool
    note    -> str (information or error message)
    elapsed -> float (seconds)

Predictable exceptions are caught and turned into ok=False with a
friendly message.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Declared capabilities
# ---------------------------------------------------------------------------

SUPPORTED_INPUTS: tuple[str, ...] = (
    ".csv", ".tsv", ".xlsx", ".xls", ".ods", ".json",
)

SUPPORTED_OUTPUTS: tuple[str, ...] = (
    ".csv", ".tsv", ".xlsx", ".json", ".parquet",
)

# (src_ext, dst_ext) pairs actually implemented. Any combination
# among the declared inputs/outputs is in practice manageable by always going through
# a pandas DataFrame, so we populate it dynamically.
_SUPPORTED_PAIRS: set[tuple[str, str]] = set()
for _i in SUPPORTED_INPUTS:
    for _o in SUPPORTED_OUTPUTS:
        _SUPPORTED_PAIRS.add((_i, _o))


def supports(src_ext: str, dst_ext: str) -> bool:
    """Returns True if the pair (src_ext, dst_ext) is supported.

    The arguments may or may not include the leading dot, in any
    capitalization.
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
    """Reads xlsx/xls/ods. .ods needs odfpy."""
    ext = src.suffix.lower()
    engine = None
    if ext == ".xlsx":
        engine = "openpyxl"
    elif ext == ".xls":
        engine = "xlrd"  # may not be installed
    elif ext == ".ods":
        engine = "odf"

    sheet_req = opts.get("sheet")
    note_extra = ""

    try:
        if sheet_req is not None:
            df = pd.read_excel(src, sheet_name=sheet_req, engine=engine)
        else:
            # Read all the sheets to be able to report how many there are.
            sheets = pd.read_excel(src, sheet_name=None, engine=engine)
            if not sheets:
                raise ValueError("The file contains no readable sheets.")
            first_name = next(iter(sheets))
            df = sheets[first_name]
            if len(sheets) > 1:
                note_extra = (
                    f"Sheet used: '{first_name}'. Found {len(sheets)} sheets "
                    f"in total; pass opts['sheet'] to choose a different one."
                )
            else:
                note_extra = f"Sheet used: '{first_name}'."
    except ImportError as e:
        # Missing engine
        if ext == ".ods":
            raise RuntimeError(
                "Reading .ods files needs 'odfpy'. Install with: pip install odfpy"
            ) from e
        if ext == ".xls":
            raise RuntimeError(
                "Reading .xls files needs 'xlrd' (version 1.2.0). "
                "Install with: pip install xlrd==1.2.0"
            ) from e
        raise

    return df, note_extra


def _read_json(src: Path, opts: dict) -> tuple[pd.DataFrame, str]:
    """Reads both 'list of dicts' and 'dict of columns' JSON."""
    with open(src, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, list):
        df = pd.DataFrame(raw)
        note = "JSON read as a list of records."
    elif isinstance(raw, dict):
        # Distinguish: dict of columns (values = lists of the same length)
        # from a single record (scalar values).
        values = list(raw.values())
        if values and all(isinstance(v, list) for v in values):
            df = pd.DataFrame(raw)
            note = "JSON read as a dict of columns."
        else:
            df = pd.DataFrame([raw])
            note = "JSON read as a single record."
    else:
        raise ValueError("JSON structure not recognized: expected a list or a dict.")

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
    raise ValueError(f"Unsupported input extension: {ext}")


# ---------------------------------------------------------------------------
# Scrittura
# ---------------------------------------------------------------------------

def _write_any(df: pd.DataFrame, dst: Path, opts: dict) -> str:
    ext = dst.suffix.lower()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if ext == ".csv":
        sep = opts.get("sep") or ","
        df.to_csv(dst, sep=sep, index=False)
        return f"Wrote CSV ({len(df)} rows)."

    if ext == ".tsv":
        df.to_csv(dst, sep="\t", index=False)
        return f"Wrote TSV ({len(df)} rows)."

    if ext == ".xlsx":
        sheet_name = opts.get("sheet") or "Sheet1"
        df.to_excel(dst, index=False, sheet_name=sheet_name, engine="openpyxl")
        return f"Wrote XLSX ({len(df)} rows, sheet '{sheet_name}')."

    if ext == ".json":
        orient = opts.get("orient", "records")
        df.to_json(dst, orient=orient, force_ascii=False, indent=2)
        return f"Wrote JSON orient='{orient}' ({len(df)} rows)."

    if ext == ".parquet":
        try:
            df.to_parquet(dst, index=False)
        except ImportError as e:
            raise RuntimeError(
                "Writing .parquet files needs 'pyarrow' or 'fastparquet'. "
                "Install with: pip install pyarrow"
            ) from e
        except ValueError as e:
            # pandas raises ValueError if the parquet engine is missing
            msg = str(e).lower()
            if "pyarrow" in msg or "fastparquet" in msg or "parquet" in msg:
                raise RuntimeError(
                    "Writing .parquet files needs 'pyarrow' or 'fastparquet'. "
                    "Install with: pip install pyarrow"
                ) from e
            raise
        return f"Wrote Parquet ({len(df)} rows)."

    raise ValueError(f"Unsupported output extension: {ext}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert(src: Path, dst: Path, opts: dict | None = None) -> dict:
    """Converts a tabular file from src to dst.

    See the module docstring for details on the supported extensions.
    """
    t0 = time.perf_counter()
    opts = dict(opts or {})

    try:
        src = Path(src)
        dst = Path(dst)

        if not src.exists():
            return {
                "ok": False,
                "note": f"Source file not found: {src}",
                "elapsed": time.perf_counter() - t0,
            }

        src_ext = src.suffix.lower()
        dst_ext = dst.suffix.lower()

        if src_ext not in SUPPORTED_INPUTS:
            return {
                "ok": False,
                "note": (
                    f"Input extension '{src_ext}' not supported. "
                    f"Supported: {', '.join(SUPPORTED_INPUTS)}."
                ),
                "elapsed": time.perf_counter() - t0,
            }

        if dst_ext not in SUPPORTED_OUTPUTS:
            return {
                "ok": False,
                "note": (
                    f"Output extension '{dst_ext}' not supported. "
                    f"Supported: {', '.join(SUPPORTED_OUTPUTS)}."
                ),
                "elapsed": time.perf_counter() - t0,
            }

        if not supports(src_ext, dst_ext):
            return {
                "ok": False,
                "note": (
                    f"Pair {src_ext} -> {dst_ext} not supported. "
                    f"Try an intermediate format among: "
                    f"{', '.join(SUPPORTED_OUTPUTS)}."
                ),
                "elapsed": time.perf_counter() - t0,
            }

        # --- Lettura ------------------------------------------------------
        try:
            df, read_note = _read_any(src, opts)
        except RuntimeError as e:
            # Friendly errors (missing engines, etc.)
            return {
                "ok": False,
                "note": str(e),
                "elapsed": time.perf_counter() - t0,
            }
        except Exception as e:
            return {
                "ok": False,
                "note": f"Error while reading {src.name}: {e}",
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
                "note": f"Error while writing {dst.name}: {e}",
                "elapsed": time.perf_counter() - t0,
            }

        note_parts = [p for p in (read_note, write_note) if p]
        return {
            "ok": True,
            "note": " ".join(note_parts) if note_parts else "Conversion successful.",
            "elapsed": time.perf_counter() - t0,
        }

    except Exception as e:  # safety net
        return {
            "ok": False,
            "note": f"Unexpected error: {e}",
            "elapsed": time.perf_counter() - t0,
        }
