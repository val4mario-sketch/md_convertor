"""Logging e dump dettagliato degli errori.

Requisito di progetto: quando una conversione fallisce o perde dati, il sistema
NON salva il file ma produce un log dettagliato. ``dump_failure_log`` scrive un
file di testo in ``tomdclaude/logs/`` con tutto il contesto utile al debug:
metadati sorgente/prodotto, elenco dei rilievi, eventuale traceback.
"""
from __future__ import annotations

import datetime as _dt
import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def get_logger(name: str = "tomdclaude", verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # gia' configurato
        if verbose:
            logger.handlers[0].setLevel(logging.DEBUG)
        return logger
    logger.setLevel(logging.DEBUG)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG if verbose else logging.INFO)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def dump_failure_log(logger, result, *, traceback_text: str | None = None,
                     converter_note: str = "") -> Path:
    """Scrive un log dettagliato del fallimento e ne ritorna il percorso."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = LOG_DIR / f"FAIL_{_safe(result.src.stem)}_{ts}.log"

    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add(f"tomdclaude — LOG DI FALLIMENTO   {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    add("=" * 78)
    add(f"convertitore : {result.converter}")
    add(f"sorgente     : {result.src}")
    add(f"destinazione : {result.dst}  (NON salvata)")
    add(f"nota         : {result.note}")
    if converter_note:
        add(f"nota motore  : {converter_note}")
    add("")
    add("--- RILIEVI DEL VALIDATOR ---")
    if result.issues:
        for i in result.issues:
            add(f"  {i}")
    else:
        add("  (nessun rilievo registrato)")
    add("")
    add("--- METADATI SORGENTE ---")
    for k, v in (result.src_metadata or {}).items():
        add(f"  {k}: {_short(v)}")
    add("")
    add("--- METADATI PRODOTTO (scartato) ---")
    for k, v in (result.dst_metadata or {}).items():
        add(f"  {k}: {_short(v)}")
    if traceback_text:
        add("")
        add("--- TRACEBACK ---")
        add(traceback_text.rstrip())
    add("=" * 78)

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.error("Log dettagliato: %s", path)
    return path


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:60]


def _short(v, limit: int = 300) -> str:
    s = repr(v)
    return s if len(s) <= limit else s[:limit] + f"... <{len(s)} char>"
