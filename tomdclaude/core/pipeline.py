"""Pipeline = orchestrazione + GATEKEEPER.

Flusso di ogni conversione:

    1. estrae i metadati del sorgente
    2. converte su un file TEMPORANEO (mai direttamente sulla destinazione)
    3. estrae i metadati del prodotto
    4. esegue i validatori: confronta i metadati (integrita')
    5a. tutti ok          -> commit atomico (rename) sulla destinazione
    5b. errore bloccante  -> scarta il temporaneo, NON salva, scrive un log dettagliato

In questo modo il requisito "se una conversione perde dati il file non viene
salvato" e' garantito strutturalmente: il file finale esiste solo dopo il
superamento del gatekeeper.
"""
from __future__ import annotations

import os
import tempfile
import time
import traceback
from pathlib import Path

from ..utils.logging_setup import dump_failure_log, get_logger
from .base_converter import norm_ext
from .registry import find_converter, targets_for
from .result import ConversionResult, ValidationIssue


def convert(src, dst=None, opts=None, *, logger=None) -> ConversionResult:
    opts = dict(opts or {})
    log = logger or get_logger()
    src = Path(src)

    # --- risoluzione destinazione ---
    if dst is None:
        to = norm_ext(opts.get("to", ""))
        if not to:
            return _fail_simple(src, src, "", "specificare 'dst' oppure opts['to']", log)
        dst = src.with_suffix("." + to)
    dst = Path(dst)

    conv = find_converter(src.suffix, dst.suffix)
    if conv is None:
        avail = targets_for(src.suffix)
        nota = (
            f"Conversione {norm_ext(src.suffix)} -> {norm_ext(dst.suffix)} non supportata. "
            + (f"Destinazioni disponibili per questo file: {avail}." if avail
               else "Nessun convertitore registrato accetta questo formato di input.")
        )
        return _fail_simple(src, dst, "", nota, log)

    result = ConversionResult(ok=False, src=src, dst=dst, converter=conv.name)
    t0 = time.perf_counter()

    if not src.exists():
        result.note = f"File sorgente inesistente: {src}"
        result.issues.append(ValidationIssue("error", "src_missing", result.note))
        result.elapsed = time.perf_counter() - t0
        log.error("KO %s", result.note)
        return result

    # --- 1. metadati sorgente ---
    try:
        result.src_metadata = conv.extract_source_metadata(src, opts)
    except Exception as e:  # metadati non estraibili: non blocchiamo, ma annotiamo
        log.warning("Metadati sorgente non estratti: %s", e)
        result.src_metadata = {}

    # --- 2. conversione su file temporaneo (stessa cartella -> rename atomico) ---
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tomdclaude_", suffix=dst.suffix, dir=str(dst.parent))
    os.close(fd)
    tmp = Path(tmp_name)

    conv_note = ""
    extra_meta: dict = {}
    try:
        out = conv.convert(src, tmp, opts)
        if isinstance(out, dict):
            conv_note = out.get("note", "")
            extra_meta = out.get("metadata", {}) or {}
        elif out:
            conv_note = str(out)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        tb = traceback.format_exc()
        result.note = f"Conversione fallita ({type(e).__name__}): {e}"
        result.issues.append(ValidationIssue("error", "convert_exception", str(e)))
        result.log_path = dump_failure_log(log, result, traceback_text=tb, converter_note=conv_note)
        result.elapsed = time.perf_counter() - t0
        log.error("KO  %s  -> file NON salvato", result.note)
        return result

    # --- 3. metadati prodotto ---
    try:
        result.dst_metadata = {**conv.extract_target_metadata(tmp), **extra_meta}
    except Exception as e:
        result.dst_metadata = dict(extra_meta)
        log.warning("Metadati prodotto non estratti: %s", e)

    # --- 4. validazione (gatekeeper) ---
    ctx = {"src": src, "tmp": tmp, "dst": dst, "opts": opts}
    for v in conv.validators():
        try:
            result.issues.extend(v.validate(result.src_metadata, result.dst_metadata, ctx))
        except Exception as e:  # un validatore rotto non deve far passare dati sospetti
            result.issues.append(ValidationIssue(
                "error", "validator_crash", f"{type(v).__name__} ha sollevato {type(e).__name__}: {e}"))

    # --- 5. commit o abort ---
    if result.errors:
        tmp.unlink(missing_ok=True)
        result.ok = False
        result.committed = False
        result.note = "Conversione RIFIUTATA dal validator (perdita/incoerenza dati). File NON salvato."
        result.log_path = dump_failure_log(log, result, converter_note=conv_note)
        result.elapsed = time.perf_counter() - t0
        log.error("KO  %s", result.note)
        for i in result.errors:
            log.error("    %s", i)
        return result

    if dst.exists() and not opts.get("overwrite", False):
        tmp.unlink(missing_ok=True)
        result.ok = False
        result.note = f"La destinazione esiste gia': {dst.name} (usa overwrite=True per sovrascrivere)."
        result.elapsed = time.perf_counter() - t0
        log.error("KO  %s", result.note)
        return result

    os.replace(tmp, dst)  # commit ATOMICO (stesso filesystem)
    result.ok = True
    result.committed = True
    result.note = conv_note or "ok"
    result.elapsed = time.perf_counter() - t0
    log.info("OK  %s -> %s  (%.2fs)  %s", src.name, dst.name, result.elapsed, result.note)
    for w in result.warnings:
        log.warning("    %s", w)
    return result


def _fail_simple(src, dst, conv_name, note, log) -> ConversionResult:
    r = ConversionResult(ok=False, src=Path(src), dst=Path(dst), converter=conv_name, note=note)
    r.issues.append(ValidationIssue("error", "unsupported", note))
    log.error("KO  %s", note)
    return r
