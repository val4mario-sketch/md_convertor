#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_markdown.py — HIGH-FIDELITY conversion to Markdown.

Wrapper around `tomd2.py` (the existing engine): reuses pymupdf4llm for PDFs,
markitdown for Office/web, python-pptx for .pptx, openpyxl for .xlsx, auto-detected
OCR-scan for scanned PDFs, optional marker, and all the other specialized
engines. Do NOT degrade the existing quality.

Interface uniform with the other modules of the universal converter:
    convert(src, dst, opts) -> {ok, note, elapsed}
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from types import SimpleNamespace

# Import the existing engine
sys.path.insert(0, str(Path(__file__).parent))
import tomd2  # noqa: E402

# Extensions: anything supported by tomd2 is fine as input to .md.
SUPPORTED_INPUTS = set(tomd2.ALL_SUPPORTED)  # ~170 estensioni
SUPPORTED_OUTPUTS = {".md", ".markdown"}


def supports(src_ext: str, dst_ext: str) -> bool:
    s = src_ext.lower()
    d = dst_ext.lower()
    if d not in SUPPORTED_OUTPUTS:
        return False
    # tomd2 already knows everything: Office extensions, PDF, code, etc.
    return s in SUPPORTED_INPUTS or s == ""  # empty extensions (Dockerfile, ...) handled


def convert(src: Path, dst: Path, opts: dict | None = None) -> dict:
    """Converts any supported file to high-fidelity Markdown.

    `opts` can contain:
      - overwrite (bool, default True)
      - page_image_dpi (int, default 150)
      - page_images (bool, default True)
      - vector_threshold (int, default 3)
      - media_dir (str, default 'media')
      - engine (str, default 'auto') — forces a specific engine
      - whisper_model (str, default 'base')
      - block_size (int, default 0)
      - force_ocr (bool, default False)
    """
    opts = opts or {}
    t0 = time.time()

    src = Path(src); dst = Path(dst)
    if not src.exists():
        return {"ok": False, "note": f"File not found: {src}", "elapsed": 0.0}

    dst_ext = dst.suffix.lower()
    if dst_ext not in SUPPORTED_OUTPUTS:
        return {"ok": False,
                "note": (f"convert_markdown only supports .md / .markdown output "
                         f"(received {dst_ext})."),
                "elapsed": 0.0}

    # Build an args object compatible with the tomd2 CLI
    args = SimpleNamespace(
        overwrite        = opts.get("overwrite", True),
        page_images      = opts.get("page_images", True),
        page_image_dpi   = opts.get("page_image_dpi", 150),
        vector_threshold = opts.get("vector_threshold", 3),
        engine           = opts.get("engine", "auto"),
        media_dir        = opts.get("media_dir", "media"),
        force_ocr        = opts.get("force_ocr", False),
        block_size       = opts.get("block_size", 0),
        whisper_model    = opts.get("whisper_model", "base"),
        describe_images  = opts.get("describe_images", False),
        describe_model   = opts.get("describe_model", None),
        pages            = opts.get("pages", None),
        hq_pages         = opts.get("hq_pages", None),
        vlm_threads      = opts.get("vlm_threads", None),
        vlm_gpu          = opts.get("vlm_gpu", False),
    )

    # tomd2.convert_one always writes <src.stem>.md in the source folder.
    # To support an arbitrary dst, a trick:
    # 1) call convert_one(src, args) -> creates <src.stem>.md
    # 2) if dst is different, move/rename it
    result = tomd2.convert_one(src, args)
    elapsed = time.time() - t0

    if not result["ok"]:
        return {"ok": False, "note": result.get("note", "unknown error"),
                "elapsed": elapsed, "report": result.get("report", "")}

    produced_md = Path(result["out"])
    if produced_md.resolve() != dst.resolve():
        # Move to the requested dst
        try:
            if dst.exists() and args.overwrite:
                dst.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            produced_md.replace(dst)
        except Exception as e:
            return {"ok": False,
                    "note": f"Conversion OK but error moving to {dst}: {e}",
                    "elapsed": elapsed}

    return {"ok": True,
            "note": result.get("report", "conversion completed"),
            "elapsed": elapsed,
            "engine": result.get("engine", "auto")}


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _self_test():
    """Creates a small .py and a .csv and converts them to .md. Cleans up at the end."""
    import shutil
    tmp = Path(__file__).parent / "_test_markdown"
    tmp.mkdir(exist_ok=True)
    try:
        # 1) py -> md
        py = tmp / "demo.py"
        py.write_text('def hello():\n    """Saluta tutti."""\n    return "ciao"\n',
                      encoding="utf-8")
        r1 = convert(py, tmp / "demo.md", {})
        ok1 = r1["ok"] and (tmp / "demo.md").stat().st_size > 50
        print(f"  [{'OK' if ok1 else 'KO'}] demo.py -> demo.md  ({r1['note']})")

        # 2) csv -> md
        csv = tmp / "tab.csv"
        csv.write_text("nome,eta,citta\nAnna,30,Roma\nLuca,25,Milano\n",
                       encoding="utf-8")
        r2 = convert(csv, tmp / "tab.md", {})
        ok2 = r2["ok"] and (tmp / "tab.md").stat().st_size > 50
        print(f"  [{'OK' if ok2 else 'KO'}] tab.csv -> tab.md  ({r2['note']})")

        return ok1 and ok2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("convert_markdown.py self-test:")
    ok = _self_test()
    sys.exit(0 if ok else 1)
