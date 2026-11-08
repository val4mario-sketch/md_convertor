#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert.py — CLI entry point of the universal converter.

Examples:
    python convert.py input.docx --to pdf
    python convert.py input.docx output.pdf
    python convert.py *.docx --to pdf
    python convert.py file.pdf --to md           # high fidelity via tomd2
    python convert.py --list                     # shows ALL possible pairs
    python convert.py input.xlsx --to mp3        # shows a clear error

Output always goes in the SAME folder as the original file (unless an output is given).
Originals are never deleted or changed.
"""
from __future__ import annotations
import argparse, glob, os, sys, time
from pathlib import Path

# Force UTF-8 on the console (Windows uses cp1252 by default)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import router


def _propaga_path_strumenti():
    """Adds LibreOffice/Pandoc/ImageMagick/FFmpeg to the session PATH,
       so the modules find them right after a winget install."""
    candidati = [
        r"C:\Program Files\LibreOffice\program",
    ]
    # Pandoc, ffmpeg, magick: look under WinGet
    import glob as _g
    base = Path(os.environ.get("LOCALAPPDATA","")) / "Microsoft/WinGet/Packages"
    if base.exists():
        for pat in ("JohnMacFarlane.Pandoc_*", "Gyan.FFmpeg_*", "ImageMagick.ImageMagick_*"):
            for d in _g.glob(str(base / pat / "**/bin"), recursive=True):
                candidati.append(d)
            # pandoc without /bin
            for d in _g.glob(str(base / pat / "**"), recursive=True):
                if Path(d, "pandoc.exe").exists():
                    candidati.append(d)
    for c in candidati:
        if c and Path(c).exists() and c not in os.environ["PATH"]:
            os.environ["PATH"] = c + os.pathsep + os.environ["PATH"]


def _normalize_target(token: str) -> str:
    t = token.lower().strip()
    if not t.startswith("."):
        t = "." + t
    return t


def _resolve_files(tokens: list[str]) -> list[Path]:
    files: list[Path] = []
    for tk in tokens:
        p = Path(tk)
        if any(c in tk for c in "*?[]"):
            for m in glob.glob(tk):
                if Path(m).is_file():
                    files.append(Path(m).resolve())
        elif p.exists() and p.is_file():
            files.append(p.resolve())
        else:
            print(f"[!] not found: {tk}", file=sys.stderr)
    # dedupe preserving order
    seen = set(); uniq = []
    for f in files:
        if f not in seen:
            seen.add(f); uniq.append(f)
    return uniq


def _opts_from_args(args) -> dict:
    return {
        "overwrite":       args.overwrite,
        "page_image_dpi":  args.page_image_dpi,
        "page_images":     args.page_images,
        "vector_threshold": args.vector_threshold,
        "media_dir":       args.media_dir,
        "engine":          args.engine,
        "whisper_model":   args.whisper_model,
        "force_ocr":       args.force_ocr,
        "block_size":      args.block_size,
        "quality":         args.quality,
        "width":           args.width,
        "height":          args.height,
        "fps":             args.fps,
        "bitrate":         args.bitrate,
        "image":           args.image,
        "sheet":           args.sheet,
        "sep":             args.sep,
    }


def main():
    _propaga_path_strumenti()

    ap = argparse.ArgumentParser(
        description="Universal converter: every format to every other (where that makes sense).",
        epilog="For the full list of conversions: python convert.py --list")
    ap.add_argument("inputs", nargs="*",
                    help="Files to convert (glob patterns too). If you pass 2 arguments "
                         "and the second is a file with an extension, it is used as the output.")
    ap.add_argument("--to", dest="target",
                    help="Target extension (e.g. 'pdf', 'mp3', 'md'). "
                         "Alternative: pass 'input output.ext' directly.")
    ap.add_argument("--list", action="store_true",
                    help="Print ALL supported conversions and exit.")
    ap.add_argument("--overwrite", action="store_true", default=True,
                    help="Overwrite existing output files (default).")
    ap.add_argument("--no-overwrite", dest="overwrite", action="store_false")

    # Markdown options (passed to tomd2)
    g_md = ap.add_argument_group("Markdown options (to .md)")
    g_md.add_argument("--page-image-dpi", type=int, default=150)
    g_md.add_argument("--page-images", action="store_true", default=True)
    g_md.add_argument("--no-page-images", dest="page_images", action="store_false")
    g_md.add_argument("--vector-threshold", type=int, default=3)
    g_md.add_argument("--media-dir", default="media")
    g_md.add_argument("--engine", default="auto",
                      help="Force a tomd2 engine (auto/pymupdf4llm/marker/ocr_scan/...)")
    g_md.add_argument("--force-ocr", action="store_true")
    g_md.add_argument("--block-size", type=int, default=0)
    g_md.add_argument("--whisper-model", default="base",
                      choices=["tiny","base","small","medium","large"])

    # media options
    g_img = ap.add_argument_group("Media options (images/audio/video)")
    g_img.add_argument("--quality", type=int, default=None,
                      help="jpg/webp quality (1-100, default 85)")
    g_img.add_argument("--width",  type=int, default=None)
    g_img.add_argument("--height", type=int, default=None)
    g_img.add_argument("--fps",    type=int, default=None, help="gif/video only")
    g_img.add_argument("--bitrate", default=None, help="Audio/video only, e.g. '192k'")
    g_img.add_argument("--image",  default=None,
                      help="Audio->video only: path of a still background image")

    # data options
    g_dt = ap.add_argument_group("Tabular data options")
    g_dt.add_argument("--sheet", default=None, help="xlsx sheet to use")
    g_dt.add_argument("--sep",   default=None, help="csv separator (default auto)")

    args = ap.parse_args()

    if args.list:
        print(router.list_all_conversions())
        return 0

    if not args.inputs:
        ap.print_help(); return 2

    # 'input output.ext' form without --to
    explicit_out: Path | None = None
    if not args.target and len(args.inputs) == 2:
        p2 = Path(args.inputs[1])
        if p2.suffix and not p2.exists():
            # The second is the output file name
            explicit_out = p2
            args.inputs = [args.inputs[0]]

    if not args.target and not explicit_out:
        print("Error: give --to <extension> or 'input output.ext'",
              file=sys.stderr)
        return 2

    files = _resolve_files(args.inputs)
    if not files:
        print("No input files found.", file=sys.stderr); return 1

    target_ext = explicit_out.suffix.lower() if explicit_out else _normalize_target(args.target)
    opts = _opts_from_args(args)

    results = []
    print(f"Files to convert: {len(files)}  ->  {target_ext}\n")
    for f in files:
        dst = explicit_out if explicit_out else f.with_suffix(target_ext)
        # ---- ORIGINALS PROTECTION ----
        # Dangerous case: converting 2.md -> 2.pdf in the folder where the ORIGINAL
        # 2.pdf exists (from which the .md was generated). If the destination
        # exists and is OLDER than the source, it is very probably
        # the original: do NOT touch it and write "<name>_convertito.<ext>".
        # (If it is newer, it is the output of an earlier conversion
        # and overwriting it is the expected behaviour.)
        if (not explicit_out and dst.exists() and dst.resolve() != f.resolve()
                and dst.stat().st_mtime < f.stat().st_mtime):
            safe = f.with_name(f"{f.stem}_convertito{target_ext}")
            print(f"    [!] {dst.name} exists and is older than the source: "
                  f"probably the ORIGINAL. Writing {safe.name} so as not to touch it.")
            dst = safe
        print(f">>> {f.name}  ->  {dst.name}")
        t0 = time.time()
        r = router.convert(f, dst, opts)
        r["src"] = str(f); r["dst"] = str(dst); r["t"] = time.time() - t0
        results.append(r)
        if r["ok"]:
            print(f"    [OK] OK  [{r.get('module')}]  {r.get('note','')[:120]}")
        else:
            print(f"    [KO] {r.get('note','error').splitlines()[0]}")
            for ln in r.get("note","").splitlines()[1:6]:
                print(f"       {ln}")

    print("\n========== RIEPILOGO ==========")
    ok = sum(1 for r in results if r["ok"])
    for r in results:
        flag = "OK " if r["ok"] else "ERR"
        print(f"  [{flag}] {Path(r['src']).name} -> {Path(r['dst']).name}  "
              f"({r.get('module') or '-'})  {r['t']:.1f}s")
    print(f"\nCompletati {ok}/{len(results)}.")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
