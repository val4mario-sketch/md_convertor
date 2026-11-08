"""Pipeline = orchestration + GATEKEEPER.

Flow of each conversion:

    1. extracts the source metadata
    2. converts to a TEMPORARY file (never directly to the destination)
    3. extracts the product metadata
    4. runs the validators: compares the metadata (integrity)
    5a. all ok            -> atomic commit (rename) to the destination
    5b. blocking error    -> discards the temporary file, does NOT save, writes a detailed log

In this way the requirement "if a conversion loses data the file is not
saved" is guaranteed structurally: the final file exists only after
passing the gatekeeper.
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

    # --- resolving the destination ---
    if dst is None:
        to = norm_ext(opts.get("to", ""))
        if not to:
            return _fail_simple(src, src, "", "give 'dst' or opts['to']", log)
        dst = src.with_suffix("." + to)
    dst = Path(dst)

    conv = find_converter(src.suffix, dst.suffix)
    if conv is None:
        avail = targets_for(src.suffix)
        nota = (
            f"Conversione {norm_ext(src.suffix)} -> {norm_ext(dst.suffix)} not supported. "
            + (f"Destinations available for this file: {avail}." if avail
               else "No registered converter accepts this input format.")
        )
        return _fail_simple(src, dst, "", nota, log)

    result = ConversionResult(ok=False, src=src, dst=dst, converter=conv.name)
    t0 = time.perf_counter()

    if not src.exists():
        result.note = f"Source file does not exist: {src}"
        result.issues.append(ValidationIssue("error", "src_missing", result.note))
        result.elapsed = time.perf_counter() - t0
        log.error("KO %s", result.note)
        return result

    # --- 1. source metadata ---
    try:
        result.src_metadata = conv.extract_source_metadata(src, opts)
    except Exception as e:  # metadata cannot be extracted: do not block, but note it
        log.warning("Source metadata not extracted: %s", e)
        result.src_metadata = {}

    # --- 2. conversion to a temporary file (same folder -> atomic rename) ---
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
        result.note = f"Conversion failed ({type(e).__name__}): {e}"
        result.issues.append(ValidationIssue("error", "convert_exception", str(e)))
        result.log_path = dump_failure_log(log, result, traceback_text=tb, converter_note=conv_note)
        result.elapsed = time.perf_counter() - t0
        log.error("KO  %s  -> file NOT saved", result.note)
        return result

    # --- 3. product metadata ---
    try:
        result.dst_metadata = {**conv.extract_target_metadata(tmp), **extra_meta}
    except Exception as e:
        result.dst_metadata = dict(extra_meta)
        log.warning("Product metadata not extracted: %s", e)

    # --- 4. validation (gatekeeper) ---
    ctx = {"src": src, "tmp": tmp, "dst": dst, "opts": opts}
    for v in conv.validators():
        try:
            result.issues.extend(v.validate(result.src_metadata, result.dst_metadata, ctx))
        except Exception as e:  # a broken validator must not let suspicious data through
            result.issues.append(ValidationIssue(
                "error", "validator_crash", f"{type(v).__name__} raised {type(e).__name__}: {e}"))

    # --- 5. commit or abort ---
    if result.errors:
        tmp.unlink(missing_ok=True)
        result.ok = False
        result.committed = False
        result.note = "Conversion REJECTED by the validator (data loss/inconsistency). File NOT saved."
        result.log_path = dump_failure_log(log, result, converter_note=conv_note)
        result.elapsed = time.perf_counter() - t0
        log.error("KO  %s", result.note)
        for i in result.errors:
            log.error("    %s", i)
        return result

    if dst.exists() and not opts.get("overwrite", False):
        tmp.unlink(missing_ok=True)
        result.ok = False
        result.note = f"The destination already exists: {dst.name} (use overwrite=True to overwrite)."
        result.elapsed = time.perf_counter() - t0
        log.error("KO  %s", result.note)
        return result

    os.replace(tmp, dst)  # ATOMIC commit (same filesystem)
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
