"""tomdclaude command-line interface.

Examples:
    python -m tomdclaude document.md --to pdf
    python -m tomdclaude report.md -o build/report.pdf --title "Q2 Report"
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
        description="Modular converter with validation (gatekeeper).",
    )
    p.add_argument("src", nargs="?", help="source file")
    p.add_argument("--to", help="target extension (e.g. pdf)")
    p.add_argument("-o", "--output", help="output file path")
    p.add_argument("--overwrite", action="store_true", help="overwrite if it exists")
    p.add_argument("--title", help="document title (cover/header)")
    p.add_argument("--author", help="document author")
    p.add_argument("--no-math", action="store_true", help="do not render LaTeX math")
    p.add_argument("--strict-math", action="store_true",
                   help="reject the PDF if even one formula is not rendered")
    p.add_argument("--keep-citations", action="store_true",
                   help="do not remove the [cite_start]/[cite: N] markers")
    p.add_argument("--list", action="store_true", help="list the available converters")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    log = get_logger(verbose=a.verbose)

    if a.list:
        print("Registered converters:")
        for c in all_converters():
            print(f"  - {c.name}: {sorted(c.source_formats)} -> {sorted(c.target_formats)}")
        return 0

    if not a.src:
        p.error("give a source file (or --list)")

    src = Path(a.src)
    if a.output:
        dst = Path(a.output)
    elif a.to:
        dst = src.with_suffix("." + a.to.lstrip("."))
    else:
        p.error("give --to <ext> or -o <file>")

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
            log.info("Completed with %d warnings (file saved).", len(res.warnings))
        return 0

    # failure: the detail is already logged; remind where the log file is
    if res.log_path:
        log.error("Full details in the log: %s", res.log_path)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
