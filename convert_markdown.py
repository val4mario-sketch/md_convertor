#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_markdown.py — Conversione AD ALTA FEDELTÀ verso Markdown.

Wrapper sopra `tomd2.py` (il motore esistente): riusa pymupdf4llm per i PDF,
markitdown per Office/web, python-pptx per .pptx, openpyxl per .xlsx, OCR-scan
auto-rilevato per PDF scansionati, marker opzionale, e tutti gli altri engine
specializzati. NON peggiorare la qualità preesistente.

Interfaccia uniforme col resto dei moduli del convertitore universale:
    convert(src, dst, opts) -> {ok, note, elapsed}
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from types import SimpleNamespace

# Importa il motore esistente
sys.path.insert(0, str(Path(__file__).parent))
import tomd2  # noqa: E402

# Estensioni: ogni cosa supportata da tomd2 va bene in input verso .md.
SUPPORTED_INPUTS = set(tomd2.ALL_SUPPORTED)  # ~170 estensioni
SUPPORTED_OUTPUTS = {".md", ".markdown"}


def supports(src_ext: str, dst_ext: str) -> bool:
    s = src_ext.lower()
    d = dst_ext.lower()
    if d not in SUPPORTED_OUTPUTS:
        return False
    # tomd2 sa già tutto: estensioni Office, PDF, codice, ecc.
    return s in SUPPORTED_INPUTS or s == ""  # estensioni vuote (Dockerfile, ...) gestite


def convert(src: Path, dst: Path, opts: dict | None = None) -> dict:
    """Converte qualunque file supportato in Markdown ad alta fedeltà.

    `opts` può contenere:
      - overwrite (bool, default True)
      - page_image_dpi (int, default 150)
      - page_images (bool, default True)
      - vector_threshold (int, default 3)
      - media_dir (str, default 'media')
      - engine (str, default 'auto') — forza un engine specifico
      - whisper_model (str, default 'base')
      - block_size (int, default 0)
      - force_ocr (bool, default False)
    """
    opts = opts or {}
    t0 = time.time()

    src = Path(src); dst = Path(dst)
    if not src.exists():
        return {"ok": False, "note": f"File non trovato: {src}", "elapsed": 0.0}

    dst_ext = dst.suffix.lower()
    if dst_ext not in SUPPORTED_OUTPUTS:
        return {"ok": False,
                "note": (f"convert_markdown supporta solo output .md / .markdown "
                         f"(ricevuto {dst_ext})."),
                "elapsed": 0.0}

    # Costruisco un oggetto args compatibile col CLI di tomd2
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

    # tomd2.convert_one scrive sempre in <src.stem>.md nella cartella del sorgente.
    # Per supportare un dst arbitrario, faccio un trucco:
    # 1) chiamo convert_one(src, args) -> crea <src.stem>.md
    # 2) se dst è diverso, lo sposto/rinomino
    result = tomd2.convert_one(src, args)
    elapsed = time.time() - t0

    if not result["ok"]:
        return {"ok": False, "note": result.get("note", "errore sconosciuto"),
                "elapsed": elapsed, "report": result.get("report", "")}

    produced_md = Path(result["out"])
    if produced_md.resolve() != dst.resolve():
        # Sposto nel dst richiesto
        try:
            if dst.exists() and args.overwrite:
                dst.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            produced_md.replace(dst)
        except Exception as e:
            return {"ok": False,
                    "note": f"Conversione OK ma errore nello spostamento in {dst}: {e}",
                    "elapsed": elapsed}

    return {"ok": True,
            "note": result.get("report", "conversione completata"),
            "elapsed": elapsed,
            "engine": result.get("engine", "auto")}


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _self_test():
    """Crea un piccolo .py + un .csv e li converte in .md. Pulisce al termine."""
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
