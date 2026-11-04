"""Interfaccia a riga di comando di tomdclaude.

Esempi:
    python -m tomdclaude documento.md --to pdf
    python -m tomdclaude relazione.md -o build/relazione.pdf --title "Relazione Q2"
    python -m tomdclaude --list
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .core.pipeline import convert
from .core.registry import all_converters
from .utils.logging_setup import get_logger


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="tomdclaude",
        description="Convertitore modulare con validazione (gatekeeper).",
    )
    p.add_argument("src", nargs="?", help="file sorgente")
    p.add_argument("--to", help="estensione di destinazione (es. pdf)")
    p.add_argument("-o", "--output", help="percorso del file di output")
    p.add_argument("--overwrite", action="store_true", help="sovrascrivi se esiste")
    p.add_argument("--title", help="titolo del documento (copertina/intestazione)")
    p.add_argument("--author", help="autore del documento")
    p.add_argument("--no-math", action="store_true", help="non renderizzare la matematica LaTeX")
    p.add_argument("--strict-math", action="store_true",
                   help="rifiuta il PDF se anche una sola formula non viene resa")
    p.add_argument("--keep-citations", action="store_true",
                   help="non rimuovere i marcatori [cite_start]/[cite: N]")
    p.add_argument("--list", action="store_true", help="elenca i convertitori disponibili")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    log = get_logger(verbose=a.verbose)

    if a.list:
        print("Convertitori registrati:")
        for c in all_converters():
            print(f"  - {c.name}: {sorted(c.source_formats)} -> {sorted(c.target_formats)}")
        return 0

    if not a.src:
        p.error("specificare un file sorgente (oppure --list)")

    src = Path(a.src)
    if a.output:
        dst = Path(a.output)
    elif a.to:
        dst = src.with_suffix("." + a.to.lstrip("."))
    else:
        p.error("specificare --to <ext> oppure -o <file>")

    opts = {
        "overwrite": a.overwrite,
        "math": not a.no_math,
        "strict_math": a.strict_math,
        "strip_cite": not a.keep_citations,
    }
    if a.title:
        opts["title"] = a.title
    if a.author:
        opts["author"] = a.author

    res = convert(src, dst, opts, logger=log)

    if res.ok:
        if res.warnings:
            log.info("Completato con %d avvisi (file salvato).", len(res.warnings))
        return 0

    # fallimento: il dettaglio e' gia' a log; ricordiamo dove sta il log file
    if res.log_path:
        log.error("Dettagli completi nel log: %s", res.log_path)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
