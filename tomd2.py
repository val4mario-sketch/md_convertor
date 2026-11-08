#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tomd2.py — Converts files to high-fidelity Markdown, FREE and LOCAL.

PRIMARY engine for PDFs: pymupdf4llm (reads the PDF text layer, very fast
on CPU, no AI models to download). Suited to "native" PDFs such as exported
PowerPoint slides (selectable text, vector).

Automatic routing:
  - PDF (.pdf)                              -> pymupdf4llm  [default]
  - Office/Web (.pptx .docx .xlsx .html ..) -> markitdown
  - Images (.png .jpg ...)                  -> marker (the only sensible engine for images)
  - .md .txt                                -> copied as they are

Optional:
  --engine marker      forces marker (only for scanned PDFs or edge cases).
                       Slow on CPU without a GPU; use --block-size for large files.

Output:
  <filename>.md in the SAME folder as the original file.
  Figures and page images in media/<filename>/.
  (the 'media/' folder collects the images of ALL converted files)

Quick use:
    python tomd2.py 2.pdf --overwrite                # pymupdf4llm (fast)
    python tomd2.py "*.pdf" --overwrite              # batch of PDFs
    python tomd2.py slide.pdf --engine marker        # force marker (slow)
    python tomd2.py scan.pdf --engine marker --force-ocr --block-size 15
    python tomd2.py 2.pdf --no-page-images           # no page render

Dependencies:
    pip install -U markitdown marker-pdf pymupdf pymupdf4llm
"""

from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys, time
from pathlib import Path

# ----------------------------------------------------------------------------
# Handled extensions
# ----------------------------------------------------------------------------
OFFICE_WEB_EXT = {".pptx", ".docx", ".xlsx", ".xls", ".doc", ".ppt",
                  ".html", ".htm", ".csv", ".tsv", ".json", ".xml", ".epub", ".rtf",
                  ".odt", ".ods", ".odp"}
IMAGE_EXT      = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
PDF_EXT        = {".pdf"}
PASSTHROUGH    = {".md", ".markdown", ".txt"}

# Source code: maps extension -> language for the markdown block
CODE_LANG = {
    ".py":"python", ".pyw":"python", ".pyi":"python",
    ".js":"javascript", ".mjs":"javascript", ".cjs":"javascript",
    ".jsx":"jsx", ".ts":"typescript", ".tsx":"tsx",
    ".vue":"vue", ".svelte":"svelte", ".astro":"astro",
    ".c":"c", ".h":"c",
    ".cc":"cpp", ".cpp":"cpp", ".cxx":"cpp", ".hpp":"cpp", ".hxx":"cpp", ".hh":"cpp",
    ".m":"objectivec", ".mm":"objectivec",
    ".cs":"csharp", ".vb":"vb", ".fs":"fsharp",
    ".java":"java", ".kt":"kotlin", ".kts":"kotlin",
    ".scala":"scala", ".groovy":"groovy", ".gradle":"groovy",
    ".swift":"swift",
    ".go":"go", ".rs":"rust", ".zig":"zig",
    ".rb":"ruby", ".erb":"erb", ".rake":"ruby",
    ".php":"php", ".phtml":"php",
    ".pl":"perl", ".pm":"perl",
    ".sh":"bash", ".bash":"bash", ".zsh":"zsh", ".fish":"fish", ".ksh":"ksh",
    ".ps1":"powershell", ".psm1":"powershell", ".psd1":"powershell",
    ".bat":"batch", ".cmd":"batch",
    ".lua":"lua", ".r":"r",
    ".sql":"sql", ".psql":"sql",
    ".dart":"dart",
    ".ex":"elixir", ".exs":"elixir",
    ".clj":"clojure", ".cljs":"clojure", ".cljc":"clojure",
    ".lisp":"lisp", ".scm":"scheme", ".rkt":"racket",
    ".tcl":"tcl", ".awk":"awk",
    ".css":"css", ".scss":"scss", ".sass":"sass", ".less":"less",
    ".tex":"latex", ".latex":"latex", ".ltx":"latex", ".sty":"latex", ".cls":"latex",
    ".bib":"bibtex",
    ".asm":"asm", ".s":"asm", ".S":"asm",
    ".vhd":"vhdl", ".vhdl":"vhdl", ".v":"verilog", ".sv":"systemverilog",
    ".f":"fortran", ".f90":"fortran", ".f95":"fortran",
    ".pas":"pascal",
    ".lisp":"lisp",
}
CODE_EXT = set(CODE_LANG.keys())

# Config / text data files (also without a specific language)
TEXT_DATA_LANG = {
    ".yaml":"yaml", ".yml":"yaml",
    ".toml":"toml",
    ".ini":"ini", ".cfg":"ini", ".conf":"ini", ".properties":"properties",
    ".env":"dotenv",
    ".log":"", ".nfo":"",
    ".diff":"diff", ".patch":"diff",
    ".gitignore":"gitignore", ".gitattributes":"", ".dockerignore":"",
    ".editorconfig":"editorconfig",
    ".cmake":"cmake",
    ".makefile":"makefile",
    ".dockerfile":"dockerfile",
    ".gradle":"groovy", ".sbt":"scala",
    ".srt":"srt", ".vtt":"webvtt",
}
TEXT_DATA_EXT = set(TEXT_DATA_LANG.keys())

# Notebook Jupyter
IPYNB_EXT = {".ipynb"}

# Email
EMAIL_EXT = {".eml"}

# Archives (zip, tar, etc.)
ARCHIVE_EXT = {".zip", ".tar", ".tgz", ".tar.gz", ".tbz", ".tar.bz2"}

# Proprietary e-books (need external tools such as calibre)
EBOOK_EXT = {".mobi", ".azw", ".azw3", ".kfx", ".fb2", ".djvu", ".djv"}

# Audio/Video (need whisper for transcription)
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv", ".wmv", ".m4v"}

ALL_SUPPORTED  = (OFFICE_WEB_EXT | IMAGE_EXT | PDF_EXT | PASSTHROUGH |
                  CODE_EXT | TEXT_DATA_EXT | IPYNB_EXT | EMAIL_EXT |
                  ARCHIVE_EXT | EBOOK_EXT | AUDIO_EXT | VIDEO_EXT)


# ----------------------------------------------------------------------------
# File search (names, paths, globs)
# ----------------------------------------------------------------------------
def resolve_inputs(tokens, root: Path, recursive: bool):
    found, missing = [], []
    for tok in tokens:
        p = Path(tok).expanduser()
        if p.exists() and p.is_file():
            found.append(p.resolve()); continue
        if any(c in tok for c in "*?[]"):
            hits = list(root.rglob(tok)) if recursive else list(root.glob(tok))
            hits = [h for h in hits if h.is_file()]
            found.extend(h.resolve() for h in hits)
            if not hits: missing.append(tok)
            continue
        stem = Path(tok).stem.lower()
        want_ext = Path(tok).suffix.lower()
        hits = []
        walker = root.rglob("*") if recursive else root.glob("*")
        for f in walker:
            if not f.is_file() or f.suffix.lower() not in ALL_SUPPORTED:
                continue
            if want_ext:
                if f.name.lower() == tok.lower(): hits.append(f)
            elif f.stem.lower() == stem:
                hits.append(f)
        if hits: found.extend(h.resolve() for h in hits)
        else:    missing.append(tok)
    seen, uniq = set(), []
    for f in found:
        if f not in seen:
            seen.add(f); uniq.append(f)
    return uniq, missing


# ----------------------------------------------------------------------------
# Image utility: correct relative links
# ----------------------------------------------------------------------------
def _fix_charset_mojibake(text: str) -> str:
    """Fixes 'mojibake' characters (UTF-8 read as CP1252/CP437).
       Examples found by the LLM reviews:
         ┬«  ->  «     (CP437/cp850 misread of U+00AB)
         ┬»  ->  »
         тАФ  ->  —     (CP1252 misread of U+2014 em-dash)
         тАЪ  ->  ‚
         тАЛ  ->  ‌
         тАЭ  ->  ”
         тАЮ  ->  „
         ┬·  ->  ·
         тДв  ->  ™
    """
    # 1) "smart" attempt: try to re-encode if we see the typical markers
    if "тАФ" in text or "┬«" in text or "тАЭ" in text or "├¿" in text:
        for src_enc, dst_enc in (("cp1252","utf-8"), ("cp437","utf-8"),
                                  ("latin-1","utf-8")):
            try:
                fixed = text.encode(src_enc, errors="strict").decode(dst_enc, errors="strict")
                # Heuristic: the fix reduces the suspicious byte sequences
                if (fixed.count("тА") + fixed.count("┬") + fixed.count("├")
                        < text.count("тА") + text.count("┬") + text.count("├")):
                    text = fixed
                    break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    # 2) direct replacement of residual sequences
    replacements = {
        "тАФ": "—", "тАУ": "–", "тАЪ": "‚", "тАЛ": "‌",
        "тАЭ": "”", "тАЮ": "„", "тАЬ": "“", "тАЩ": "’", "тАШ": "‘",
        "┬«": "«", "┬»": "»", "┬·": "·", "┬║": "º", "┬°": "°", "┬±": "±",
        "тДв": "™", "тДЦ": "№",
        "ÔÇô": "–", "ÔÇö": "—", "ÔÇ£": "“", "ÔÇЭ": "”", "ÔÇЩ": "’", "ÔÇШ": "‘",
        "├Ñ": "à", "├¿": "è", "├Ё": "Ð", "├и": "и",
    }
    for bad, good in replacements.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def _postprocess_math(md: str) -> str:
    """Conservative clean-ups typical of mathematical slide PDFs.
       Protects the Markdown links ![alt](path) and [text](url) so as not to corrupt
       image paths (e.g. 'Matlab1_AA2526_assets').
    """
    # 0) protect markdown links (both images ![..](..) and links [..](..))
    stash: list[str] = []
    def _hide(m):
        stash.append(m.group(0))
        return f"\x00LNK{len(stash)-1}\x00"
    md = re.sub(r"!?\[[^\]\n]*\]\([^)\n]+\)", _hide, md)

    # 1) <lettera>[<esponente>] -> <lettera>^{<esponente>}
    md = re.sub(
        r"([A-Za-z])\[([A-Za-z0-9 ]{1,6})\](?!\()",
        lambda m: f"{m.group(1)}^{{{m.group(2)}}}", md)

    # 2) 'xn-1' -> 'x_{n-1}' only inside _..._ italics (not multi-line)
    def _pedice_in_ital(m):
        body = m.group(1)
        body = re.sub(r"([A-Za-z])(\d+|n-\d+|n\+\d+)", r"\1_{\2}", body)
        return f"_{body}_"
    md = re.sub(r"_([^_\n]{1,160})_", _pedice_in_ital, md)

    # 3) restore the links
    md = re.sub(r"\x00LNK(\d+)\x00", lambda m: stash[int(m.group(1))], md)
    return md


def _rewrite_image_paths(md: str, prefix: str) -> str:
    """Points the image links to the `prefix` subfolder.
       Skips absolute links, URLs and links already relative to `prefix`."""
    def repl(m):
        path = m.group(1).strip()
        if path.startswith(("http://", "https://", "/")) or path.startswith(prefix + "/") or path.startswith(prefix):
            return m.group(0)
        # already relative? extract only the basename to put it under prefix
        return f"]({prefix}/{Path(path).name})"
    return re.sub(r"\]\(([^)]+\.(?:png|jpg|jpeg|webp|gif|tif|tiff|bmp))\)",
                  repl, md, flags=re.IGNORECASE)


# Link/image destination: allows ONE pair of balanced inner
# parentheses (e.g. 'media/DocScanner (2)/img.png'), as in the CommonMark spec.
_MD_LINK_RE = re.compile(
    r"(!?\[[^\]\n]*\])\(([^()\n]*(?:\([^()\n]*\)[^()\n]*)*)\)")


def _encode_md_link_urls(md: str) -> str:
    """Percent-encoding of the paths in Markdown links/images.

       A destination with bare spaces ('media/MY NOTES/f.png')
       is NOT valid CommonMark: VS Code, GitHub, Obsidian etc. do not
       render it and the image (formula included) "disappears". With
       'media/MY%20NOTES/f.png' it works everywhere.
       Does not touch external URLs, anchors, <...> or already clean destinations;
       '%' stays safe so already encoded paths are not re-encoded."""
    from urllib.parse import quote

    def repl(m):
        label, dest = m.group(1), m.group(2).strip()
        title = ""
        tm = re.match(r"^(.*?)(\s+\"[^\"]*\"|\s+'[^']*')\s*$", dest)
        if tm:
            dest, title = tm.group(1), tm.group(2)
        if not dest or (dest.startswith("<") and dest.endswith(">")):
            return m.group(0)
        if "://" in dest or dest.startswith(("mailto:", "data:", "#")):
            return m.group(0)
        if not re.search(r"[ ()]", dest):
            return m.group(0)
        enc = quote(dest, safe="/:.@!$&'*+,;=_~%-")
        return f"{label}({enc}{title})"

    return _MD_LINK_RE.sub(repl, md)


# ----------------------------------------------------------------------------
# Engine 1: markitdown (DOCX / HTML / EPUB) — free, instant
# Used as a fallback for Office/Web files that are NOT xlsx/pptx (those have an engine
# dedicated one, see below).
# ----------------------------------------------------------------------------
def convert_with_markitdown(src: Path) -> str:
    from markitdown import MarkItDown
    return MarkItDown(enable_plugins=False).convert(str(src)).text_content


# ----------------------------------------------------------------------------
# Excel engine: dedicated openpyxl (excellent quality, no NaN, no Unnamed)
# ----------------------------------------------------------------------------
def convert_with_xlsx(src: Path) -> str:
    """Converts .xlsx to Markdown an AI can read:
       - 1 H2 section per sheet
       - real markdown tables (no NaN, no 'Unnamed: X')
       - respects merged cells (the first cell holds the value)
       - numeric cells formatted compactly (no endless decimals)"""
    from openpyxl import load_workbook

    def _fmt(v):
        if v is None: return ""
        if isinstance(v, float):
            if v != v: return ""  # NaN
            if v.is_integer(): return str(int(v))
            return f"{v:.6g}"
        s = str(v).replace("\r\n", "<br>").replace("\n", "<br>").replace("|", "\\|")
        return s.strip()

    wb = load_workbook(str(src), data_only=True, read_only=True)
    parts: list[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        parts.append(f"\n## {sheet_name}\n")
        rows = [[_fmt(c.value) for c in row] for row in ws.iter_rows()]
        if not rows:
            parts.append("_(empty sheet)_\n"); continue
        # cut completely empty columns on the right
        max_cols = 0
        for r in rows:
            for i in range(len(r) - 1, -1, -1):
                if r[i] != "":
                    max_cols = max(max_cols, i + 1); break
        if max_cols == 0: parts.append("_(empty sheet)_\n"); continue
        rows = [r[:max_cols] + [""] * (max_cols - len(r[:max_cols])) for r in rows]
        # cut completely empty rows at the bottom
        while rows and all(c == "" for c in rows[-1]):
            rows.pop()
        if not rows:
            parts.append("_(empty sheet)_\n"); continue
        # look for the first "header" row: first non-empty row with at least 2 cells
        hdr_idx = 0
        for i, r in enumerate(rows):
            if sum(1 for c in r if c) >= 2:
                hdr_idx = i; break
        # if there are title/note rows above the header, write them as text
        for i in range(hdr_idx):
            tx = " ".join(c for c in rows[i] if c)
            if tx: parts.append(f"_{tx}_\n")
        # markdown table
        hdr = rows[hdr_idx]
        # if the header is all empty, use Col_1, Col_2...
        if all(c == "" for c in hdr):
            hdr = [f"Col_{i+1}" for i in range(len(hdr))]
        parts.append("| " + " | ".join(hdr) + " |")
        parts.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for r in rows[hdr_idx + 1:]:
            # skip completely empty rows
            if all(c == "" for c in r): continue
            parts.append("| " + " | ".join(r) + " |")
        parts.append("")
    wb.close()
    return "\n".join(parts).strip() + "\n"


# ----------------------------------------------------------------------------
# PowerPoint engine: dedicated python-pptx (text + ALL the extracted images)
# ----------------------------------------------------------------------------
def convert_with_pptx(src: Path, media_dirname: str = "media") -> tuple[str, Path]:
    """Converts .pptx to Markdown, extracting:
       - an H2 section per slide (title text + body + notes)
       - ALL the embedded images (Picture + Shape with image fill)
       - tables, in real markdown
       - speaker notes
    """
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    stem = src.stem
    assets = src.parent / media_dirname / stem
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)
    asset_prefix = f"{media_dirname}/{stem}"

    pres = Presentation(str(src))
    parts: list[str] = []
    img_idx = 0

    def _walk_shapes(shapes, slide_no: int, depth: int = 0):
        """Recursively visits the shapes (also handles groups)."""
        nonlocal img_idx
        out: list[str] = []
        for shape in shapes:
            try:
                stype = shape.shape_type
            except Exception:
                stype = None

            # group: recursion
            if stype == MSO_SHAPE_TYPE.GROUP:
                out.extend(_walk_shapes(shape.shapes, slide_no, depth + 1))
                continue

            # image: save and link it
            if stype == MSO_SHAPE_TYPE.PICTURE or (
                hasattr(shape, "image") and getattr(shape, "image", None)):
                try:
                    img = shape.image
                    img_idx += 1
                    ext = (img.ext or "png").lower()
                    fname = f"slide_{slide_no:03d}_img_{img_idx:03d}.{ext}"
                    (assets / fname).write_bytes(img.blob)
                    out.append(f"![]({asset_prefix}/{fname})")
                except Exception:
                    pass
                continue

            # tabella
            if getattr(shape, "has_table", False):
                tbl = shape.table
                rows = [[cell.text.strip().replace("\n", "<br>").replace("|", "\\|")
                         for cell in row.cells] for row in tbl.rows]
                if not rows: continue
                out.append("| " + " | ".join(rows[0]) + " |")
                out.append("|" + "|".join(["---"] * len(rows[0])) + "|")
                for r in rows[1:]:
                    out.append("| " + " | ".join(r) + " |")
                continue

            # text frame
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if not text: continue
                    # heuristic: the first text frame of the slide is often the title
                    out.append(text)
        return out

    for i, slide in enumerate(pres.slides, start=1):
        parts.append(f"\n## Slide {i}\n")
        # title if present
        title = None
        try:
            if slide.shapes.title and slide.shapes.title.has_text_frame:
                title = slide.shapes.title.text_frame.text.strip()
        except Exception:
            pass
        if title:
            parts.append(f"### {title}\n")
        # content (all shapes except the title)
        content_shapes = [s for s in slide.shapes
                          if not (slide.shapes.title is not None and s == slide.shapes.title)]
        body = _walk_shapes(content_shapes, i)
        if body:
            parts.append("\n".join(body))
        # speaker notes
        try:
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    parts.append(f"\n**Speaker notes:** {notes}")
        except Exception:
            pass

    return ("\n".join(parts).strip() + "\n", assets)


# ----------------------------------------------------------------------------
# CSV / TSV engine: clean markdown table + basic statistics
# ----------------------------------------------------------------------------
def _is_number(s: str) -> bool:
    if s is None or s == "": return False
    try: float(s.replace(",", "."))
    except ValueError: return False
    return True


def convert_with_csv(src: Path, max_preview_rows: int = 50) -> str:
    """Converts .csv/.tsv to Markdown with good heuristics for "technical" CSVs:
       - if the first row is a TITLE (single cell), makes it an H1
       - if there are rows of UNITS under the header, recognizes them
       - statistics of the numeric columns (n, min, mean, max)
       - shows at most N rows so as not to bloat the .md
    """
    import csv
    # detect the delimiter
    with open(src, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            class _D: delimiter = "\t" if src.suffix.lower() == ".tsv" else ","
            dialect = _D()
        reader = csv.reader(f, delimiter=dialect.delimiter)
        rows = [[c.strip() for c in r] for r in reader]

    if not rows:
        return f"# {src.name}\n\n_(empty file)_\n"

    # maximum columns used in the whole file
    max_cols = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    parts: list[str] = [f"# {src.name}\n"]

    # --- 1) look for the initial "title" rows (a single filled cell) ---
    title_rows = []
    i = 0
    while i < len(rows):
        filled = sum(1 for c in rows[i] if c)
        if filled <= 1 and any(rows[i]):
            title_rows.append(next((c for c in rows[i] if c), ""))
            i += 1
        else:
            break

    for tr in title_rows:
        parts.append(f"_(title: **{tr}**)_\n")

    # --- 2) header = next row with all cells filled and NOT numeric ---
    hdr_idx = i
    while hdr_idx < len(rows):
        r = rows[hdr_idx]
        non_empty = [c for c in r if c]
        if non_empty and not any(_is_number(c) for c in non_empty):
            break
        hdr_idx += 1
    if hdr_idx >= len(rows):
        # no header: invent Col_X and start from row "i"
        hdr_idx = i
        headers = [f"Col_{k+1}" for k in range(max_cols)]
        data_start = i
        unit_row = None
    else:
        headers = list(rows[hdr_idx])
        # if the next row is "all non-numeric strings" -> UNITS
        unit_row = None
        if hdr_idx + 1 < len(rows):
            r2 = rows[hdr_idx + 1]
            ne2 = [c for c in r2 if c]
            if ne2 and not any(_is_number(c) for c in ne2):
                unit_row = list(r2)
                data_start = hdr_idx + 2
            else:
                data_start = hdr_idx + 1
        else:
            data_start = hdr_idx + 1

    # combine header + units if present
    headers_disp = []
    for k in range(max_cols):
        h = headers[k] if k < len(headers) else ""
        u = unit_row[k] if unit_row and k < len(unit_row) else ""
        if not h: h = f"Col_{k+1}"
        if u: headers_disp.append(f"{h} [{u}]")
        else: headers_disp.append(h)

    data_rows = rows[data_start:]
    # cut completely empty rows at the end
    while data_rows and not any(data_rows[-1]):
        data_rows.pop()

    parts.append(f"**Data rows**: {len(data_rows)}  "
                 f"**Columns**: {max_cols}  "
                 f"**Separator**: `'{dialect.delimiter}'`\n")

    # --- 3) numeric column statistics ---
    stats = []
    for k in range(max_cols):
        nums = []
        for r in data_rows:
            if k < len(r) and _is_number(r[k]):
                nums.append(float(r[k].replace(",", ".")))
        if len(nums) >= max(3, len(data_rows) * 0.5):
            stats.append((headers_disp[k], len(nums), min(nums),
                          sum(nums) / len(nums), max(nums)))
    if stats:
        parts.append("\n## Numeric column statistics\n")
        parts.append("| Column | n | min | mean | max |")
        parts.append("|---|---|---|---|---|")
        for h, n, mn, av, mx in stats:
            parts.append(f"| {h} | {n} | {mn:.6g} | {av:.6g} | {mx:.6g} |")

    # --- 4) data preview ---
    parts.append("\n## Data")
    if len(data_rows) > max_preview_rows:
        parts.append(f"_(showing the first {max_preview_rows} rows of "
                     f"{len(data_rows)})_\n")
        show = data_rows[:max_preview_rows]
    else:
        show = data_rows

    safe_h = [h.replace("|", "\\|") for h in headers_disp]
    parts.append("| " + " | ".join(safe_h) + " |")
    parts.append("|" + "|".join(["---"] * max_cols) + "|")
    for r in show:
        r = [str(c).replace("\n", "<br>").replace("|", "\\|") for c in r[:max_cols]]
        parts.append("| " + " | ".join(r) + " |")
    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------------
# CODE engine: programming sources (and text config)
# Puts the content in a ``` markdown block with the correct language.
# ----------------------------------------------------------------------------
def convert_with_code(src: Path) -> str:
    """Converts a source file to Markdown:
       - H1 heading with the file name and size/lines
       - fenced code block (```lang) with the full content
    """
    ext = src.suffix.lower()
    # special names without an extension (Dockerfile, Makefile, ...)
    name_l = src.name.lower()
    lang = (CODE_LANG.get(ext) or TEXT_DATA_LANG.get(ext) or "")
    if not lang:
        if "dockerfile" in name_l: lang = "dockerfile"
        elif "makefile"  in name_l: lang = "makefile"
        elif "cmakelists" in name_l: lang = "cmake"
    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = src.read_text(encoding="latin-1", errors="replace")
    # protection: if the file contains ``` it is escaped with a suitable `
    fence = "```"
    while fence in text:
        fence += "`"
    n_lines = text.count("\n") + 1
    sz = src.stat().st_size
    parts = [
        f"# {src.name}",
        f"",
        f"**Language**: `{lang or '(text)'}`  "
        f"**Lines**: {n_lines}  **Size**: {sz/1024:.1f} KB",
        f"",
        f"{fence}{lang}",
        text.rstrip(),
        fence,
        "",
    ]
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Motore IPYNB: Jupyter notebook -> markdown (markdown + code + output)
# ----------------------------------------------------------------------------
def convert_with_ipynb(src: Path) -> str:
    import json
    nb = json.loads(src.read_text(encoding="utf-8"))
    lang = (nb.get("metadata", {}).get("kernelspec", {}).get("language")
            or nb.get("metadata", {}).get("language_info", {}).get("name")
            or "python")
    parts = [f"# {src.name}", "", f"**Notebook** (language: {lang})", ""]
    for i, cell in enumerate(nb.get("cells", []), 1):
        ctype = cell.get("cell_type", "")
        src_lines = cell.get("source", [])
        src_text = "".join(src_lines) if isinstance(src_lines, list) else str(src_lines)
        if ctype == "markdown":
            parts.append(f"<!-- Cell {i}: markdown -->\n{src_text}\n")
        elif ctype == "code":
            parts.append(f"<!-- Cell {i}: code -->")
            parts.append(f"```{lang}")
            parts.append(src_text.rstrip())
            parts.append("```")
            # output
            outs = cell.get("outputs", []) or []
            for j, out in enumerate(outs, 1):
                otype = out.get("output_type", "")
                if otype == "stream":
                    txt = "".join(out.get("text", [])) if isinstance(out.get("text"), list) else out.get("text", "")
                    if txt.strip():
                        parts.append(f"\n*Output stdout/stderr:*\n```\n{txt.rstrip()}\n```")
                elif otype in ("execute_result", "display_data"):
                    data = out.get("data", {})
                    if "text/plain" in data:
                        tp = data["text/plain"]
                        tp = "".join(tp) if isinstance(tp, list) else str(tp)
                        if tp.strip():
                            parts.append(f"\n*Output:*\n```\n{tp.rstrip()}\n```")
                    if "image/png" in data:
                        parts.append(f"\n*(output: embedded PNG image — not rendered in Markdown)*")
                elif otype == "error":
                    parts.append(f"\n*Error:* `{out.get('ename','')}: {out.get('evalue','')}`")
            parts.append("")
        elif ctype == "raw":
            parts.append(f"<!-- Cell {i}: raw -->\n```\n{src_text.rstrip()}\n```\n")
    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------------
# .eml EMAIL engine -> markdown (headers + body + extracted attachments)
# ----------------------------------------------------------------------------
def convert_with_eml(src: Path, media_dirname: str = "media") -> tuple[str, Path]:
    from email import policy
    from email.parser import BytesParser

    stem = src.stem
    assets = src.parent / media_dirname / stem
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)
    asset_prefix = f"{media_dirname}/{stem}"

    with open(src, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    parts = [f"# {src.name}", ""]
    headers = ["From", "To", "Cc", "Bcc", "Subject", "Date", "Reply-To"]
    parts.append("| Field | Value |")
    parts.append("|---|---|")
    for h in headers:
        v = msg.get(h)
        if v:
            v = str(v).replace("\n", " ").replace("|", "\\|")
            parts.append(f"| **{h}** | {v} |")
    parts.append("")

    # corpo
    body_plain = ""; body_html = ""
    for part in (msg.iter_parts() if msg.is_multipart() else [msg]):
        ctype = part.get_content_type()
        if ctype == "text/plain" and not body_plain:
            try: body_plain = part.get_content()
            except Exception: pass
        elif ctype == "text/html" and not body_html:
            try: body_html = part.get_content()
            except Exception: pass

    if body_plain:
        parts.append("## Body (text)\n")
        parts.append(body_plain.strip())
        parts.append("")
    elif body_html:
        parts.append("## Body (converted HTML)\n")
        # mini HTML -> text converter
        try:
            from markitdown import MarkItDown
            import tempfile
            tmp = Path(tempfile.mktemp(suffix=".html"))
            tmp.write_text(body_html, encoding="utf-8")
            parts.append(MarkItDown(enable_plugins=False).convert(str(tmp)).text_content)
            tmp.unlink(missing_ok=True)
        except Exception:
            parts.append(re.sub(r"<[^>]+>", "", body_html))

    # allegati
    attachments = []
    for part in msg.iter_attachments():
        fname = part.get_filename() or f"attachment_{len(attachments)+1}"
        # safe name
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", fname)
        try:
            payload = part.get_payload(decode=True)
            (assets / safe).write_bytes(payload or b"")
            attachments.append((fname, safe, len(payload or b"")))
        except Exception:
            pass

    if attachments:
        parts.append("\n## Attachments\n")
        parts.append("| Name | Size | Link |")
        parts.append("|---|---|---|")
        for orig, safe, sz in attachments:
            parts.append(f"| {orig} | {sz/1024:.1f} KB | [{safe}]({asset_prefix}/{safe}) |")

    return ("\n".join(parts) + "\n", assets)


# ----------------------------------------------------------------------------
# .zip ARCHIVE engine -> markdown (file list + recursive conversion of the contents)
# ----------------------------------------------------------------------------
def convert_with_archive(src: Path, args=None,
                         media_dirname: str = "media",
                         max_depth: int = 3, _depth: int = 0) -> tuple[str, Path]:
    import zipfile, tempfile

    stem = src.stem
    assets = src.parent / media_dirname / stem
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)

    parts = [f"# {src.name}", "", f"**Archive**: ZIP ({src.stat().st_size/1024/1024:.2f} MB)", ""]
    if not zipfile.is_zipfile(src):
        parts.append("_(not a valid ZIP)_")
        return ("\n".join(parts), assets)

    with zipfile.ZipFile(src) as zf:
        infolist = zf.infolist()
        parts.append(f"**Contained files**: {len([i for i in infolist if not i.is_dir()])}")
        parts.append("")
        parts.append("| Path | Size | Compressed |")
        parts.append("|---|---|---|")
        for info in infolist:
            if info.is_dir(): continue
            parts.append(f"| `{info.filename}` | {info.file_size/1024:.1f} KB "
                         f"| {info.compress_size/1024:.1f} KB |")
        # extract EVERYTHING into assets/_extracted/
        extract_dir = assets / "_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_dir)

    if _depth < max_depth and args is not None:
        parts.append("\n## File contents (converted recursively)\n")
        for sub in sorted(extract_dir.rglob("*")):
            if not sub.is_file(): continue
            if sub.suffix.lower() not in ALL_SUPPORTED: continue
            try:
                sub_eng = choose_engine(sub, "auto")
                parts.append(f"\n### {sub.relative_to(extract_dir).as_posix()}  "
                             f"`[{sub_eng}]`\n")
                # reuses convert_one (simple) — it would write an .md alongside
                # here we extract the text "inline" without writing an .md for each sub-file
                inner_text = _inline_convert(sub, args)
                # indent by 1 heading level so as not to overlap
                inner_text = re.sub(r"^(#+) ", r"#\1 ", inner_text, flags=re.M)
                parts.append(inner_text.strip())
            except Exception as e:
                parts.append(f"_(not converted: {type(e).__name__}: {e})_")

    return ("\n".join(parts) + "\n", assets)


def _inline_convert(sub: Path, args) -> str:
    """Converts a sub-file to a Markdown string without writing the .md to disk.
       Used by the archive engine to insert the contents into the parent md."""
    eng = choose_engine(sub, "auto")
    if eng == "code":     return convert_with_code(sub)
    if eng == "text_data": return convert_with_code(sub)
    if eng == "ipynb":    return convert_with_ipynb(sub)
    if eng == "csv":      return convert_with_csv(sub)
    if eng == "xlsx":     return convert_with_xlsx(sub)
    if eng == "markitdown": return convert_with_markitdown(sub)
    if eng == "passthrough":
        return f"```\n{sub.read_text(encoding='utf-8', errors='replace')}\n```"
    return f"_(extension {sub.suffix} not handled inline)_"


# ----------------------------------------------------------------------------
# URL engine -> download and convert
# ----------------------------------------------------------------------------
def convert_with_url(url: str, dest_dir: Path, args) -> tuple[str, Path, Path]:
    """Downloads a URL and converts it. Returns (markdown, assets, output_md_path).
       The downloaded file is saved in dest_dir/<safe_name>.<ext>."""
    import urllib.request, urllib.parse, tempfile, mimetypes
    parsed = urllib.parse.urlparse(url)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", parsed.netloc + parsed.path) or "download"
    safe = safe[:120]
    req = urllib.request.Request(url, headers={"User-Agent": "tomd2/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
        ctype = r.headers.get("Content-Type", "").split(";")[0].strip()
    ext = mimetypes.guess_extension(ctype) or ".bin"
    # if the URL already has an explicit extension, prefer it
    url_ext = Path(parsed.path).suffix
    if url_ext.lower() in ALL_SUPPORTED:
        ext = url_ext
    # avoid double extensions if 'safe' already ends with 'ext'
    if safe.lower().endswith(ext.lower()):
        out_file = dest_dir / safe
    else:
        out_file = dest_dir / f"{safe}{ext}"
    out_file.write_bytes(data)
    # now convert the file
    if ext.lower() in ALL_SUPPORTED:
        eng = choose_engine(out_file, "auto")
        # use convert_one but so as NOT to overwrite the downloaded file
        # (calling the functions directly)
        if eng == "html" or out_file.suffix.lower() in {".html", ".htm"}:
            text = convert_with_markitdown(out_file)
            return text, dest_dir, out_file.with_suffix(".md")
        # delegate to convert_one through dummy args
        from types import SimpleNamespace
        new_args = SimpleNamespace(**vars(args)) if hasattr(args, "__dict__") else args
        new_args.overwrite = True
        info = convert_one(out_file, new_args)
        return (Path(info["out"]).read_text(encoding="utf-8") if info["ok"] else
                f"_(conversion failed: {info.get('note')})_",
                dest_dir, Path(info["out"]))
    else:
        # unknown content: code block
        try: txt = data.decode("utf-8", errors="replace")
        except Exception: txt = f"_(binary content, {len(data)} bytes)_"
        return (f"# Downloaded from {url}\n\n```\n{txt[:5000]}\n```\n",
                dest_dir, out_file.with_suffix(".md"))


# ----------------------------------------------------------------------------
# Proprietary EBOOK engine (.mobi, .azw3, .djvu, .fb2): requires calibre
# ----------------------------------------------------------------------------
def convert_with_ebook(src: Path) -> str:
    """Converts proprietary e-books via Calibre's 'ebook-convert', if installed."""
    exe = shutil.which("ebook-convert")
    if not exe:
        return (f"# {src.name}\n\n"
                f"**Type**: e-book ({src.suffix})\n\n"
                f"⚠ Cannot convert it automatically: **Calibre** is needed "
                f"(https://calibre-ebook.com/download). Install it and try again.\n"
                f"Alternative manual command:\n"
                f"```\n"
                f"ebook-convert \"{src.name}\" \"{src.stem}.epub\"\n"
                f"```\n"
                f"then convert the .epub with tomd2.\n")
    import tempfile, subprocess as sp
    tmp = Path(tempfile.mkdtemp(prefix="ebook_"))
    try:
        epub = tmp / f"{src.stem}.epub"
        sp.run([exe, str(src), str(epub)], check=True, capture_output=True)
        return convert_with_markitdown(epub)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------
# AUDIO / VIDEO engine: transcription with openai-whisper (local, free)
# ----------------------------------------------------------------------------
def convert_with_media(src: Path, model: str = "base") -> str:
    """Transcribes audio/video with openai-whisper (local).
       Requires: pip install openai-whisper + ffmpeg installed."""
    try:
        import whisper
    except ImportError:
        return (f"# {src.name}\n\n"
                f"**Type**: {('audio' if src.suffix.lower() in AUDIO_EXT else 'video')} "
                f"({src.suffix})\n\n"
                f"⚠ Cannot transcribe: **openai-whisper** is missing. Install:\n"
                f"```\n"
                f"pip install -U openai-whisper\n"
                f"```\n"
                f"and make sure **ffmpeg** is on the PATH (https://ffmpeg.org).\n")
    if not shutil.which("ffmpeg"):
        return (f"# {src.name}\n\n"
                f"⚠ openai-whisper found, but **ffmpeg** is missing from the PATH "
                f"(https://ffmpeg.org/download.html).\n")
    print(f"  Whisper: loading model '{model}' (the first time it downloads ~150MB)...")
    m = whisper.load_model(model)
    print(f"  Whisper: transcribing {src.name} ...")
    res = m.transcribe(str(src), verbose=False)
    parts = [f"# {src.name}", "",
             f"**Automatic transcription** (Whisper, model `{model}`, "
             f"language: {res.get('language','?')})", ""]
    for seg in res.get("segments", []):
        t0 = seg["start"]; t1 = seg["end"]
        def _ts(s): return f"{int(s//60):02d}:{int(s%60):02d}"
        parts.append(f"`[{_ts(t0)}-{_ts(t1)}]` {seg['text'].strip()}")
    parts.append("")
    parts.append("## Full text\n")
    parts.append(res.get("text", "").strip())
    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------------
# Engine 2 (PRIMARY for PDFs): pymupdf4llm
#
# Strategia:
#  - to_markdown(page_chunks=True): we get text+tables in markdown and
#    per-page metadata (extracted images, counts, etc.).
#  - write_images=True: pymupdf4llm saves the embedded images and detected
#    charts as PNGs inside assets/.
#  - In addition we render as high-resolution PNGs ALL the pages
#    with vector graphics (drawings): so flow diagrams, even if
#    pymupdf4llm approximates them poorly in Markdown, remain available as
#    image (a multimodal AI can read them).
#  - The page render is INSERTED in the .md right in the section of that
#    page, so that text + page image live close together.
# ----------------------------------------------------------------------------
def _pages_with_vector_graphics(src: Path, min_drawings: int = 3) -> set[int]:
    """Returns the set of pages (1-based) with >= min_drawings vector
       drawings (candidates for containing diagrams/flowcharts)."""
    import fitz
    doc = fitz.open(str(src))
    out = set()
    for i, page in enumerate(doc):
        try:
            if len(page.get_drawings()) >= min_drawings:
                out.add(i + 1)
        except Exception:
            pass
    doc.close()
    return out


def _render_pages_to_png(src: Path, assets: Path, pages_1based: set[int],
                         dpi: int = 200) -> dict[int, str]:
    """Renders the given pages as PNGs inside assets/_pages/.
       Returns a mapping page_1based -> file name (basename) inside _pages."""
    import fitz
    if not pages_1based:
        return {}
    pages_dir = assets / "_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(src))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    result = {}
    for n in sorted(pages_1based):
        try:
            pix = doc[n - 1].get_pixmap(matrix=mat, alpha=False)
            fn = f"page_{n:04d}.png"
            pix.save(str(pages_dir / fn))
            result[n] = fn
        except Exception as e:
            print(f"  [WARN] render pagina {n} failed: {e}", file=sys.stderr)
    doc.close()
    return result


def convert_with_pymupdf4llm(src: Path, page_images: bool,
                             page_image_dpi: int,
                             vector_threshold: int,
                             media_dirname: str = "media",
                             lean_md: bool = False) -> tuple[str, Path]:
    """Converts a PDF to Markdown with pymupdf4llm.
       Creates media/<stem>/ with extracted images and (optionally) page images.

       lean_md=True: no <details><summary>, compact image links.
                     Reduces boilerplate for AI (suggested by Gemini)."""
    import pymupdf4llm, tempfile, pymupdf

    stem = src.stem
    assets = src.parent / media_dirname / stem
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)

    # --- pymupdf4llm BUG workaround: image names are normalized
    # (spaces -> underscores) but then looked up with the original name. To avoid this,
    # we work in a temporary directory with the PDF renamed without spaces and
    # without "exotic" characters, then move the PNGs to the real assets folder.
    # In addition we use an open Document, so pymupdf4llm sees the
    # canonical name (e.g. 'doc.pdf') when building image paths.
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "doc"
    tmpdir = Path(tempfile.mkdtemp(prefix="pmpdf4llm_"))
    try:
        safe_pdf = tmpdir / f"{safe_stem}.pdf"
        shutil.copyfile(src, safe_pdf)
        tmp_assets = tmpdir / "assets"
        tmp_assets.mkdir(parents=True, exist_ok=True)

        doc = pymupdf.open(str(safe_pdf))
        try:
            page_chunks = pymupdf4llm.to_markdown(
                doc,
                page_chunks=True,
                write_images=True,
                embed_images=False,
                image_path=str(tmp_assets),
                image_format="png",
                dpi=page_image_dpi,
                show_progress=True,
                table_strategy="lines_strict",  # avoids false positives on footers/headers
            )
        finally:
            doc.close()

        # Move all the produced images to `assets`, renaming them
        # to use the real stem (with spaces reintroduced as '_').
        for f in tmp_assets.iterdir():
            if f.is_file():
                shutil.move(str(f), str(assets / f.name))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    asset_prefix = f"{media_dirname}/{stem}"

    # --- 2) "diagram guarantee" page images for pages with vector graphics ---
    page_img_map: dict[int, str] = {}
    if page_images:
        vec_pages = _pages_with_vector_graphics(src, min_drawings=vector_threshold)
        if vec_pages:
            print(f"  Rendering vector pages ({len(vec_pages)}/{_pdf_pages(src)}) "
                  f"as PNG (dpi={page_image_dpi}) ...")
            page_img_map = _render_pages_to_png(
                src, assets, vec_pages, dpi=page_image_dpi)

    # --- 3) open the PDF with pymupdf too, for the text fallback ---
    # (for pages where pymupdf4llm extracts little text but the PDF has some)
    import pymupdf as _pdf
    raw_doc = _pdf.open(str(src))

    # lean_md is already a function parameter (default False)
    opt_lean_md = lean_md

    # --- 4) rebuild the Markdown page by page ---
    parts: list[str] = []
    n_pages = len(page_chunks)
    for idx, chunk in enumerate(page_chunks, start=1):
        meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        page_no = meta.get("page", idx)
        try:
            page_no = int(page_no) + (0 if page_no == idx else 1)
        except Exception:
            page_no = idx
        page_no = idx if not isinstance(meta.get("page", None), int) else (meta["page"] + 1)

        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        text = _rewrite_image_paths(text, asset_prefix)

        # ===== TEXT FALLBACK (Gemini fix: formulas turned into images) =====
        # If the page of the original PDF has much more text than
        # pymupdf4llm gave us, add the raw extraction alongside as a backup.
        # Threshold: if the md chunk has less than 50% of the raw text (of a page with
        # at least 200 raw characters), we add an "original text" section.
        try:
            if 0 < page_no - 1 < raw_doc.page_count:
                raw_text = raw_doc[page_no - 1].get_text("text").strip()
            else:
                raw_text = ""
        except Exception:
            raw_text = ""
        # remove the markdown image blocks from the text to measure its "true" content
        text_only = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
        text_only_chars = len(re.sub(r"\s+", "", text_only))
        raw_chars = len(re.sub(r"\s+", "", raw_text))
        needs_fallback = (raw_chars >= 200 and
                          text_only_chars < raw_chars * 0.5)

        parts.append(f"\n\n<!-- ===== Page {page_no} of {n_pages} ===== -->\n")
        # link to the page render (if present) BEFORE the text
        if page_no in page_img_map:
            rel = f"{asset_prefix}/_pages/{page_img_map[page_no]}"
            if opt_lean_md:
                # "lean" version: compact link, no <details>
                parts.append(f"\n![page {page_no}]({rel})\n")
            else:
                parts.append(
                    f"\n<details><summary>"
                    f"Page {page_no} (high-resolution image — diagram guarantee)"
                    f"</summary>\n\n"
                    f"![page {page_no}]({rel})\n\n"
                    f"</details>\n"
                )
        parts.append("\n" + text.strip() + "\n")
        # If the fallback is needed, add the raw PDF text (so as not to lose
        # formulas/numbers trapped in images)
        if needs_fallback:
            parts.append(f"\n<!-- original PDF text (formula/data fallback): -->\n")
            parts.append(f"\n<details><summary>Raw text of page {page_no} "
                         f"(for AI: contains the formulas pymupdf4llm captured as images)"
                         f"</summary>\n\n```\n{raw_text}\n```\n</details>\n")
    raw_doc.close()

    full_md = "".join(parts).strip() + "\n"
    full_md = _postprocess_math(full_md)
    full_md = _fix_charset_mojibake(full_md)
    return full_md, assets


def _pdf_pages(src: Path) -> int:
    import fitz
    d = fitz.open(str(src))
    n = d.page_count
    d.close()
    return n


# ----------------------------------------------------------------------------
# Detection of "scanned + OCR" PDFs (e.g. books from online archives)
# ----------------------------------------------------------------------------
def _is_scanned_ocr_pdf(src: Path,
                        page_coverage_threshold: float = 0.5,
                        min_pages_scanned_ratio: float = 0.6,
                        min_text_chars_per_page: int = 100) -> bool:
    """Returns True if the PDF is a SCAN with underlying OCR:
       - most pages have a raster image covering
         >= page_coverage_threshold of the page area;
       - but there is also selectable text (> min_text_chars_per_page).
       In that case pymupdf4llm is ineffective (it sees each page as
       'one big figure' and saves the page as a PNG without extracting the text).
       It is better to extract the text directly via fitz.get_text.
    """
    import fitz
    doc = fitz.open(str(src))
    try:
        n = doc.page_count
        if n == 0:
            return False
        scanned = 0
        has_text = 0
        for page in doc:
            page_area = page.rect.width * page.rect.height
            if page_area <= 0: continue
            # total area of the raster images on the page
            img_area = 0
            for img in page.get_images(full=True):
                xref = img[0]
                # bbox via get_image_rects (more reliable)
                try:
                    rects = page.get_image_rects(xref)
                    for r in rects:
                        img_area += abs(r.width * r.height)
                except Exception:
                    pass
            if img_area / page_area >= page_coverage_threshold:
                scanned += 1
            if len(page.get_text("text").strip()) >= min_text_chars_per_page:
                has_text += 1
        ratio = scanned / n
        return ratio >= min_pages_scanned_ratio and has_text / n >= 0.3
    finally:
        doc.close()


# ----------------------------------------------------------------------------
# "OCR-scan PDF" engine: keeps the OCR TEXT and shows each page as a PNG
# ----------------------------------------------------------------------------
def convert_with_ocr_scan(src: Path, page_image_dpi: int = 150,
                          media_dirname: str = "media") -> tuple[str, Path]:
    """For scanned+OCR PDFs (books from online archives, archive.org, etc.):
       - text extracted directly with fitz.get_text (keeps the OCR)
       - each page rendered as a PNG (visual backup, for figures/formulas)
       - does NOT extract the embedded images (they are just the pages themselves,
         take hundreds of MB and are redundant with the page images).
    """
    import fitz

    stem = src.stem
    assets = src.parent / media_dirname / stem
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)
    pages_dir = assets / "_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    asset_prefix = f"{media_dirname}/{stem}"

    doc = fitz.open(str(src))
    parts: list[str] = []
    mat = fitz.Matrix(page_image_dpi / 72, page_image_dpi / 72)
    n = doc.page_count
    for i in range(n):
        page = doc[i]
        # 1) OCR text extracted from the PDF
        text = page.get_text("text").strip()
        # 2) page image as visual backup
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_name = f"page_{i+1:04d}.png"
        pix.save(str(pages_dir / png_name))
        parts.append(f"\n\n<!-- ===== Page {i+1} of {n} ===== -->\n")
        parts.append(
            f"\n<details><summary>Pagina {i+1} (original scan image)"
            f"</summary>\n\n"
            f"![page {i+1}]({asset_prefix}/_pages/{png_name})\n\n"
            f"</details>\n")
        if text:
            parts.append("\n" + text + "\n")
        else:
            parts.append("\n_(text not extracted from this page — use the image above)_\n")
    doc.close()

    full_md = "".join(parts).strip() + "\n"
    return full_md, assets


# ----------------------------------------------------------------------------
# Tesseract engine: FAST local OCR for images and scanned PDFs
# WITHOUT text. Seconds instead of the hours of marker (which stays as an option
# for maximum quality on complex formulas/tables).
# ----------------------------------------------------------------------------
def _find_tesseract() -> tuple[str | None, str | None]:
    """Returns (tesseract.exe path, tessdata_dir) or (None, None).
       tessdata_dir points to the user folder if it contains ita.traineddata
       (installed there when Program Files is not writable)."""
    import os
    exe = shutil.which("tesseract")
    if not exe:
        cand = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if cand.exists():
            exe = str(cand)
    if not exe:
        return None, None
    # tessdata: prefer the user folder if it has Italian
    user_td = Path(os.environ.get("LOCALAPPDATA", "")) / "tessdata"
    if (user_td / "ita.traineddata").exists():
        return exe, str(user_td)
    return exe, None


def _tesseract_ocr_pil(img, lang: str = "ita+eng") -> str:
    """OCR of a PIL image via pytesseract. Returns text (or '')."""
    import os
    import pytesseract
    exe, tessdata = _find_tesseract()
    if not exe:
        raise RuntimeError(
            "Tesseract is not installed. Install with: "
            "winget install --id UB-Mannheim.TesseractOCR -e")
    pytesseract.pytesseract.tesseract_cmd = exe
    if tessdata:
        # TESSDATA_PREFIX is more reliable than --tessdata-dir (fragile quoting)
        os.environ["TESSDATA_PREFIX"] = tessdata
    # if ita is missing, degrade to eng without errors
    try:
        return pytesseract.image_to_string(img, lang=lang)
    except Exception:
        return pytesseract.image_to_string(img, lang="eng")


def convert_with_ocr_image(src: Path, media_dirname: str = "media",
                           lang: str = "ita+eng") -> tuple[str, Path]:
    """Image -> Markdown: fast OCR with Tesseract + linked image.
       If Tesseract is missing, still produces an .md with the image and a warning
       (no crash)."""
    from PIL import Image

    stem = src.stem
    assets = src.parent / media_dirname / stem
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)
    asset_prefix = f"{media_dirname}/{stem}"

    # copy the image to the assets (so the .md is self-sufficient)
    dest_img = assets / src.name
    shutil.copyfile(src, dest_img)

    parts = [f"# {src.name}", "",
             f"![{src.stem}]({asset_prefix}/{src.name})", ""]
    try:
        img = Image.open(src)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        text = _tesseract_ocr_pil(img, lang=lang).strip()
        if text:
            parts += ["## Extracted text (Tesseract OCR)", "", text, ""]
        else:
            parts += ["_(no text detected in the image)_", ""]
    except RuntimeError as e:
        parts += [f"⚠ OCR not available: {e}", ""]
    except Exception as e:
        parts += [f"⚠ OCR failed ({type(e).__name__}: {e}) — the image above remains.", ""]

    return "\n".join(parts), assets


def _is_scanned_no_text_pdf(src: Path,
                            page_coverage_threshold: float = 0.5,
                            min_pages_scanned_ratio: float = 0.6) -> bool:
    """True if the PDF is a scan WITHOUT selectable text:
       pages covered by raster but almost no extractable text.
       (The 'scan WITH OCR' case is handled by _is_scanned_ocr_pdf.)"""
    import fitz
    doc = fitz.open(str(src))
    try:
        n = doc.page_count
        if n == 0:
            return False
        scanned = 0
        text_chars = 0
        for page in doc:
            area = page.rect.width * page.rect.height
            if area <= 0:
                continue
            img_area = 0
            for img in page.get_images(full=True):
                try:
                    for r in page.get_image_rects(img[0]):
                        img_area += abs(r.width * r.height)
                except Exception:
                    pass
            if img_area / area >= page_coverage_threshold:
                scanned += 1
            text_chars += len(page.get_text("text").strip())
        return (scanned / n >= min_pages_scanned_ratio
                and text_chars < n * 50)  # almost no text
    finally:
        doc.close()


def convert_with_ocr_tesseract_pdf(src: Path, page_image_dpi: int = 200,
                                   media_dirname: str = "media",
                                   lang: str = "ita+eng") -> tuple[str, Path]:
    """Scanned PDF WITHOUT OCR -> Markdown, with Tesseract page by page.
       Much faster than marker on CPU (seconds/page instead of minutes).
       Each page: OCR text + page image as visual backup."""
    import fitz
    from PIL import Image
    import io as _io

    stem = src.stem
    assets = src.parent / media_dirname / stem
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)
    pages_dir = assets / "_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    asset_prefix = f"{media_dirname}/{stem}"

    doc = fitz.open(str(src))
    n = doc.page_count
    mat = fitz.Matrix(page_image_dpi / 72, page_image_dpi / 72)
    parts: list[str] = []
    print(f"  Tesseract OCR on {n} pages (dpi={page_image_dpi}) ...")
    for i in range(n):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_name = f"page_{i+1:04d}.png"
        pix.save(str(pages_dir / png_name))
        # OCR on the render just made (without reading from disk again)
        img = Image.open(_io.BytesIO(pix.tobytes("png")))
        try:
            text = _tesseract_ocr_pil(img, lang=lang).strip()
        except RuntimeError as e:
            doc.close()
            raise RuntimeError(str(e))
        parts.append(f"\n\n<!-- ===== Page {i+1} of {n} ===== -->\n")
        parts.append(
            f"\n<details><summary>Pagina {i+1} (scan image)"
            f"</summary>\n\n![page {i+1}]({asset_prefix}/_pages/{png_name})\n\n"
            f"</details>\n")
        parts.append("\n" + (text if text else
                     "_(no text detected on this page)_") + "\n")
        # progress readable by CLI and GUI
        print(f"  OCR: {int((i+1)*100/n)}% ({i+1}/{n})", flush=True)
    doc.close()

    full_md = "".join(parts).strip() + "\n"
    full_md = _fix_charset_mojibake(full_md)
    return full_md, assets


# ----------------------------------------------------------------------------
# AI-PACK engine: image-heavy PDFs -> COMPACT .md optimized for AI.
#
# Problem: photographic slides (astronomy, medicine, art...) produce md +
# giant PNGs (hundreds of MB) that no AI chat accepts.
# Solution: ONE WebP image per page, quality ~70, longest side 1568 px
# (the maximum resolution Claude/GPT use internally: larger =
# wasted weight, the AI shrinks it anyway). The page text (PDF layer
# + optional Tesseract OCR on silent pages) sits next to the image.
# Typical reduction: 10-20x compared with the original, no loss of
# information useful to the AI.
# ----------------------------------------------------------------------------
def convert_with_ai_pack(src: Path, media_dirname: str = "media",
                         quality: int = 70, max_side: int = 1568,
                         ocr_fallback: bool = True,
                         lang: str = "ita+eng") -> tuple[str, Path]:
    import fitz
    from PIL import Image
    import io as _io

    stem = src.stem
    assets = src.parent / media_dirname / stem
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)
    pages_dir = assets / "_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    asset_prefix = f"{media_dirname}/{stem}"

    doc = fitz.open(str(src))
    n = doc.page_count
    parts: list[str] = [
        f"<!-- AI-PACK: 1 WebP image q{quality} max{max_side}px per page, "
        f"full text alongside. Optimized for reading by AI. -->\n"
    ]
    tot_bytes = 0
    print(f"  AI-pack: {n} pages -> WebP q{quality}, longest side {max_side}px ...")
    for i in range(n):
        page = doc[i]
        # Dynamic DPI: render the page so the long side ~ max_side
        rect = page.rect
        long_side_pt = max(rect.width, rect.height)     # in points (1/72")
        zoom = (max_side / long_side_pt) if long_side_pt > 0 else 2.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.open(_io.BytesIO(pix.tobytes("png")))
        webp_name = f"page_{i+1:04d}.webp"
        webp_path = pages_dir / webp_name
        img.save(str(webp_path), "WEBP", quality=quality, method=6)
        tot_bytes += webp_path.stat().st_size

        # text: PDF layer, with backup OCR on silent pages
        text = page.get_text("text").strip()
        if not text and ocr_fallback:
            try:
                text = _tesseract_ocr_pil(img, lang=lang).strip()
                if text:
                    text = "*(text via OCR)*\n\n" + text
            except Exception:
                pass

        parts.append(f"\n\n<!-- ===== Page {i+1} of {n} ===== -->\n")
        parts.append(f"\n![page {i+1}]({asset_prefix}/_pages/{webp_name})\n")
        if text:
            parts.append("\n" + text + "\n")
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  AI-pack: {int((i+1)*100/n)}% ({i+1}/{n}, "
                  f"{tot_bytes/1024/1024:.1f} MB)", flush=True)
    doc.close()

    full_md = "".join(parts).strip() + "\n"
    full_md = _fix_charset_mojibake(full_md)
    print(f"  AI-pack complete: total images {tot_bytes/1024/1024:.1f} MB")
    return full_md, assets


# ----------------------------------------------------------------------------
# Engine 3 (OPTIONAL): marker — only for scanned PDFs or edge cases.
# Slow on CPU without a GPU. Supports --block-size to avoid OOM.
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Keep-awake: stops Windows from sleeping during a long conversion.
# Uses the SetThreadExecutionState API (Windows only). A no-op on other OSes.
# ----------------------------------------------------------------------------
class _KeepAwake:
    """Context manager: asks Windows NOT to sleep while the block
       is active. The screen can turn off freely (to save energy).

       Usage:
           with _KeepAwake():
               ...long work...
    """
    # flag dell'API SetThreadExecutionState
    _ES_CONTINUOUS       = 0x80000000
    _ES_SYSTEM_REQUIRED  = 0x00000001
    _ES_AWAYMODE_REQUIRED = 0x00000040
    # (we do NOT use ES_DISPLAY_REQUIRED, so the screen can turn off)

    def __enter__(self):
        if sys.platform != "win32":
            self._active = False
            return self
        try:
            import ctypes
            self._k = ctypes.windll.kernel32
            flags = self._ES_CONTINUOUS | self._ES_SYSTEM_REQUIRED
            # ES_AWAYMODE_REQUIRED: maximum resilience, even on a laptop on battery
            flags |= self._ES_AWAYMODE_REQUIRED
            self._k.SetThreadExecutionState(flags)
            self._active = True
            print("  [keep-awake] Windows sleep disabled "
                  "(the screen can still turn off)")
        except Exception as e:
            print(f"  [keep-awake] not available: {e}", file=sys.stderr)
            self._active = False
        return self

    def __exit__(self, *a):
        if getattr(self, "_active", False):
            try:
                self._k.SetThreadExecutionState(self._ES_CONTINUOUS)
                print("  [keep-awake] Windows sleep state restored")
            except Exception:
                pass


def _run_marker_single(src: Path, out_dir: Path, force_ocr: bool):
    if not shutil.which("marker_single"):
        raise RuntimeError("marker not found (pip install marker-pdf).")
    cmd = ["marker_single", str(src),
           "--output_dir", str(out_dir),
           "--output_format", "markdown"]
    if force_ocr:
        cmd.append("--force_ocr")
    subprocess.run(cmd, check=True)


def _split_pdf_in_chunks(src: Path, tmpdir: Path, chunk_size: int):
    import fitz
    doc = fitz.open(str(src))
    n = doc.page_count
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out = tmpdir / f"{src.stem}_p{start+1:04d}-{end:04d}.pdf"
        new = fitz.open()
        new.insert_pdf(doc, from_page=start, to_page=end - 1)
        new.save(str(out))
        new.close()
        chunks.append((start + 1, end, out))
    doc.close()
    return chunks


def convert_with_marker(src: Path, force_ocr: bool,
                        block_size: int = 0,
                        media_dirname: str = "media",
                        resume: bool = True) -> tuple[str, Path]:
    """Converts a PDF with marker (slow but quality OCR).
       - block_size>0: splits the PDF into blocks and joins them
       - resume=True (default): skips blocks already finished on disk,
         so you can stop and resume without losing work
    """
    assets = src.parent / media_dirname / src.stem
    is_pdf = src.suffix.lower() in PDF_EXT
    chunked = is_pdf and block_size and block_size > 0

    if chunked:
        # in block mode: do NOT delete assets if resume is active
        if not resume and assets.exists():
            shutil.rmtree(assets)
        assets.mkdir(parents=True, exist_ok=True)
        import tempfile, pymupdf
        tmpdir = Path(tempfile.mkdtemp(prefix=f"{src.stem}_chunks_"))
        try:
            chunks = _split_pdf_in_chunks(src, tmpdir, block_size)
            merged: list[str] = []
            done = 0; skipped = 0
            with _KeepAwake():
                for idx, (ps, pe, chunk_pdf) in enumerate(chunks, 1):
                    chunk_out = assets / f"_chunk_{idx:03d}"
                    existing_mds = list(chunk_out.rglob("*.md")) if chunk_out.exists() else []
                    # RESUME: block already finished? Reuse it without redoing marker.
                    if resume and existing_mds:
                        produced = existing_mds[0]
                        try:
                            # minimal validation: the .md must be non-empty
                            txt = produced.read_text(encoding="utf-8")
                            if len(txt.strip()) > 50:
                                print(f"  [block {idx}/{len(chunks)}] pages {ps}-{pe} "
                                      f"ALREADY DONE, skipping (resume)")
                                rel = produced.parent.relative_to(src.parent).as_posix()
                                txt = _rewrite_image_paths(txt, rel)
                                merged.append(
                                    f"\n\n<!-- ===== Block {idx} - pages {ps}-{pe} "
                                    f"(resume) ===== -->\n\n" + txt.strip() + "\n")
                                skipped += 1
                                continue
                        except Exception:
                            pass
                    # block to do from scratch (delete any partial leftovers)
                    if chunk_out.exists():
                        shutil.rmtree(chunk_out)
                    chunk_out.mkdir(parents=True, exist_ok=True)
                    print(f"  [block {idx}/{len(chunks)}] pages {ps}-{pe} ...")
                    _run_marker_single(chunk_pdf, chunk_out, force_ocr)
                    mds = list(chunk_out.rglob("*.md"))
                    if not mds:
                        raise RuntimeError(f"marker did not produce an .md for block {idx}")
                    produced = mds[0]
                    text = produced.read_text(encoding="utf-8")
                    rel = produced.parent.relative_to(src.parent).as_posix()
                    text = _rewrite_image_paths(text, rel)
                    merged.append(
                        f"\n\n<!-- ===== Block {idx} - pages {ps}-{pe} ===== -->\n\n"
                        + text.strip() + "\n")
                    done += 1
            if skipped:
                print(f"  [resume] skipped {skipped} blocks already finished, "
                      f"redone {done}")
            full_md = "".join(merged)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        # single mode (whole file, no blocks): resuming makes no sense
        if assets.exists():
            shutil.rmtree(assets)
        assets.mkdir(parents=True, exist_ok=True)
        with _KeepAwake():
            _run_marker_single(src, assets, force_ocr)
        mds = list(assets.rglob("*.md"))
        if not mds:
            raise RuntimeError("marker did not produce any .md")
        produced = mds[0]
        full_md = produced.read_text(encoding="utf-8")
        rel = produced.parent.relative_to(src.parent).as_posix()
        full_md = _rewrite_image_paths(full_md, rel)
    return full_md, assets


# ----------------------------------------------------------------------------
# Quality report
# ----------------------------------------------------------------------------
def quality_report(md_text: str, assets: Path | None, elapsed: float | None = None,
                   n_pages: int | None = None) -> str:
    n_chars   = len(md_text)
    # formulas: $$..$$ blocks + inline $..$
    n_formule = md_text.count("$$") // 2 + len(re.findall(r"(?<!\$)\$[^\$\n]+\$", md_text))
    n_tabelle = len(re.findall(r"^\s*\|.+\|\s*$", md_text, flags=re.M))
    n_figure  = 0
    if assets and assets.exists():
        n_figure = sum(1 for f in assets.rglob("*")
                       if f.suffix.lower() in IMAGE_EXT)
    bits = []
    if n_pages is not None: bits.append(f"{n_pages} pages")
    bits.append(f"{n_chars} characters")
    bits.append(f"~{n_tabelle} table rows")
    bits.append(f"~{n_formule} formulas")
    bits.append(f"{n_figure} figures")
    if elapsed is not None: bits.append(f"{elapsed:.1f}s")
    return " | ".join(bits)


# ----------------------------------------------------------------------------
# Routing + conversion of a single file
# ----------------------------------------------------------------------------
CSV_EXT = {".csv", ".tsv"}

def choose_engine(src: Path, forced: str | None) -> str:
    ext = src.suffix.lower()
    name_l = src.name.lower()
    if forced and forced != "auto":
        # 'marker' valid only on PDF/images; 'pymupdf4llm' valid only on PDF.
        if forced == "pymupdf4llm" and ext not in PDF_EXT:
            return "markitdown" if ext in OFFICE_WEB_EXT else "marker"
        # AI-OCR engines (ocr_vlm.py): PDF and images only
        if forced.startswith("vlm_") and ext not in PDF_EXT and ext not in IMAGE_EXT:
            return choose_engine(src, "auto")
        return forced
    # special names without an extension: Dockerfile, Makefile, CMakeLists.txt, ...
    if name_l in ("dockerfile", "makefile", "cmakelists.txt", "rakefile", "gemfile"):
        return "code"
    # dedicated routing for each format (excellent quality)
    if ext in PDF_EXT:
        # detection of scanned+OCR PDFs (books from online archives, etc.)
        try:
            if _is_scanned_ocr_pdf(src):
                return "ocr_scan"
            # scan WITHOUT text -> automatic Tesseract OCR (fast);
            # marker stays the manual choice for maximum quality
            if _is_scanned_no_text_pdf(src):
                return "ocr_tesseract"
        except Exception:
            pass
        return "pymupdf4llm"
    if ext in {".xlsx", ".xls"}: return "xlsx"
    if ext in {".pptx", ".ppt"}: return "pptx"
    if ext in CSV_EXT:        return "csv"
    if ext in IPYNB_EXT:      return "ipynb"
    if ext in EMAIL_EXT:      return "eml"
    if ext in ARCHIVE_EXT:    return "archive"
    if ext in EBOOK_EXT:      return "ebook"
    if ext in AUDIO_EXT:      return "audio"
    if ext in VIDEO_EXT:      return "video"
    if ext in CODE_EXT:       return "code"
    if ext in TEXT_DATA_EXT:  return "code"  # same engine with a specific language
    if ext in OFFICE_WEB_EXT: return "markitdown"  # docx, html, epub, json, odt, ...
    if ext in IMAGE_EXT:      return "ocr_image"   # Tesseract: seconds, not hours
    if ext in PASSTHROUGH:    return "passthrough"
    return "markitdown"


def convert_one(src: Path, args) -> dict:
    out_md = src.with_suffix(".md")
    info = {"src": str(src), "out": str(out_md), "engine": None,
            "ok": False, "note": "", "report": ""}

    if out_md.exists() and not args.overwrite and src.suffix.lower() not in PASSTHROUGH:
        info.update(ok=True, note="already exists (use --overwrite)")
        return info

    engine = choose_engine(src, args.engine)
    info["engine"] = engine
    assets = None
    n_pages = None
    t0 = time.time()
    try:
        if engine == "passthrough":
            text = src.read_text(encoding="utf-8", errors="replace")
        elif engine == "markitdown":
            text = convert_with_markitdown(src)
        elif engine == "xlsx":
            text = convert_with_xlsx(src)
        elif engine == "pptx":
            text, assets = convert_with_pptx(src, media_dirname=args.media_dir)
        elif engine == "csv":
            text = convert_with_csv(src)
        elif engine == "code":
            text = convert_with_code(src)
        elif engine == "ipynb":
            text = convert_with_ipynb(src)
        elif engine == "eml":
            text, assets = convert_with_eml(src, media_dirname=args.media_dir)
        elif engine == "archive":
            text, assets = convert_with_archive(
                src, args, media_dirname=args.media_dir)
        elif engine == "ebook":
            text = convert_with_ebook(src)
        elif engine in ("audio", "video"):
            text = convert_with_media(src, model=getattr(args, "whisper_model", "base"))
        elif engine == "pymupdf4llm":
            n_pages = _pdf_pages(src)
            text, assets = convert_with_pymupdf4llm(
                src,
                page_images=args.page_images,
                page_image_dpi=args.page_image_dpi,
                vector_threshold=args.vector_threshold,
                media_dirname=args.media_dir,
                lean_md=getattr(args, "lean_md", False))
        elif engine == "ocr_scan":
            n_pages = _pdf_pages(src)
            text, assets = convert_with_ocr_scan(
                src,
                page_image_dpi=args.page_image_dpi,
                media_dirname=args.media_dir)
        elif engine == "ocr_tesseract":
            n_pages = _pdf_pages(src)
            text, assets = convert_with_ocr_tesseract_pdf(
                src,
                page_image_dpi=max(args.page_image_dpi, 200),  # OCR wants high dpi
                media_dirname=args.media_dir,
                lang=getattr(args, "ocr_lang", "ita+eng"))
        elif engine == "ocr_image":
            text, assets = convert_with_ocr_image(
                src, media_dirname=args.media_dir,
                lang=getattr(args, "ocr_lang", "ita+eng"))
        elif engine == "ai_pack":
            n_pages = _pdf_pages(src)
            text, assets = convert_with_ai_pack(
                src, media_dirname=args.media_dir,
                quality=getattr(args, "ai_quality", 70),
                max_side=getattr(args, "ai_max_side", 1568),
                lang=getattr(args, "ocr_lang", "ita+eng"))
        elif engine in ("vlm_paddle", "vlm_chandra", "vlm_ibrido"):
            # OCR with LOCAL vision models via llama.cpp (ocr_vlm.py):
            # PaddleOCR-VL 1.6 (fast) / Chandra OCR 2 (handwriting, figures)
            import ocr_vlm
            if src.suffix.lower() in PDF_EXT:
                n_pages = _pdf_pages(src)
            text, assets = ocr_vlm.convert_file(src, engine[4:], args)
        elif engine == "marker":
            if src.suffix.lower() in PDF_EXT:
                n_pages = _pdf_pages(src)
            text, assets = convert_with_marker(
                src, args.force_ocr, block_size=args.block_size,
                media_dirname=args.media_dir)
        else:
            raise ValueError(f"invalid engine: {engine}")

        # Spaces/parentheses in image paths break Markdown rendering:
        # percent-encoding here, so it applies to ALL engines.
        text = _encode_md_link_urls(text)

        header = (f"<!-- Generated by tomd2.py | source: {src.name} | "
                  f"engine: {engine} -->\n\n")
        out_md.write_text(header + text.strip() + "\n", encoding="utf-8")

        # AI image descriptions (optional, --describe-images):
        # LOCAL vision model via Ollama, see img_describe.py.
        # It must NEVER make the conversion fail: the .md is already written.
        if getattr(args, "describe_images", False):
            try:
                import img_describe
                # NB: getattr with 'or' — argparse sets None if the flag is missing
                _vmodel = (getattr(args, "describe_model", None)
                           or img_describe.DEFAULT_MODEL)
                print(f"  AI image descriptions (model {_vmodel}) ...")
                st = img_describe.enrich_markdown(out_md, model=_vmodel)
                info["note"] = (f"AI descriptions: {st['done']} new, "
                                f"{st['cached']} from cache, {st['failed']} failed")
            except Exception as e:
                info["note"] = f"AI descriptions NOT added: {e}"
                print(f"  [WARN] {info['note']}", file=sys.stderr)

        elapsed = time.time() - t0
        info.update(ok=True,
                    report=quality_report(text, assets, elapsed, n_pages),
                    elapsed=elapsed)
    except subprocess.CalledProcessError as e:
        info["note"] = f"marker ERROR (code {e.returncode}): check RAM/models."
    except Exception as e:
        info["note"] = f"ERROR: {type(e).__name__}: {e}"
    return info


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Converts files to high-fidelity Markdown, free and local. "
                    "Default PDF engine: pymupdf4llm (very fast, CPU).")
    ap.add_argument("inputs", nargs="+",
                    help="Names, paths, glob patterns or http(s)://... URLs")
    ap.add_argument("--url-dest", default=".", type=lambda s: Path(s).expanduser(),
                    help="Destination folder for files downloaded from URLs (default: .).")
    ap.add_argument("--root", default=".", type=lambda s: Path(s).expanduser(),
                    help="Folder to search for names/globs (default: current).")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false",
                    help="Do not search subfolders.")
    ap.add_argument("--engine", choices=["auto", "pymupdf4llm", "ocr_scan",
                                          "ocr_tesseract", "ocr_image", "ai_pack",
                                          "marker", "markitdown",
                                          "vlm_paddle", "vlm_chandra", "vlm_ibrido",
                                          "xlsx", "pptx", "csv",
                                          "code", "ipynb", "eml", "archive",
                                          "ebook", "audio", "video"],
                    default="auto",
                    help="Forces an engine. Default: auto (chosen from the extension). "
                         "'ai_pack' = photographic PDFs -> COMPACT md for AI "
                         "(WebP q70, 10-20x lighter).")
    ap.add_argument("--ai-quality", type=int, default=70,
                    help="AI-pack: WebP quality 1-100 (default 70).")
    ap.add_argument("--ai-max-side", type=int, default=1568,
                    help="AI-pack: maximum image side in px (default 1568, "
                         "the maximum AIs use internally).")
    ap.add_argument("--whisper-model", default="base",
                    choices=["tiny", "base", "small", "medium", "large"],
                    help="Whisper model for audio/video (default 'base'). "
                         "'tiny' = very fast, not very accurate; 'large' = slow, very accurate.")
    ap.add_argument("--ocr-lang", default="ita+eng",
                    help="Tesseract languages for ocr_image/ocr_tesseract (default 'ita+eng').")
    ap.add_argument("--describe-images", action="store_true",
                    help="After the conversion describes EVERY image of the .md with "
                         "a LOCAL vision AI (Ollama): formulas in LaTeX, "
                         "charts and diagrams explained. Slow but resumable "
                         "(cache). Requires Ollama installed.")
    ap.add_argument("--describe-model", default=None,
                    help="Ollama vision model for --describe-images "
                         "(default: qwen2.5vl:3b).")
    # ---- OCR with local vision AI (ocr_vlm.py) ----
    ap.add_argument("--pages", default=None,
                    help="vlm_* engines: pages to convert, e.g. '1-10,15,379-383' "
                         "(default: all).")
    ap.add_argument("--hq-pages", default=None,
                    help="vlm_ibrido: pages to do with Chandra (handwriting, "
                         "figures), e.g. '379-453'. The others go to PaddleOCR-VL.")
    ap.add_argument("--vlm-threads", type=int, default=None,
                    help="CPU threads for llama.cpp (default: number of physical cores).")
    ap.add_argument("--vlm-gpu", action="store_true",
                    help="Try integrated graphics via Vulkan (experimental).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing .md files.")
    ap.add_argument("--media-dir", default="media",
                    help="Name of the folder that collects the images of ALL "
                         "the files (default: 'media'). Structure: <folder>/<media-dir>/<name>/.")
    # page-image options (default ON: needed for vector diagrams)
    ap.add_argument("--page-images", dest="page_images", action="store_true", default=True,
                    help="PDF only: renders the pages with vector graphics as PNGs "
                         "(diagram guarantee). On by default.")
    ap.add_argument("--no-page-images", dest="page_images", action="store_false",
                    help="Disables page-image rendering.")
    ap.add_argument("--lean-md", action="store_true",
                    help="'Lean' MD: compact links instead of <details><summary>. "
                         "Reduces boilerplate for LLMs (suggested by Gemini).")
    ap.add_argument("--page-image-dpi", type=int, default=150,
                    help="DPI for the extracted images/pages (default 150).")
    ap.add_argument("--vector-threshold", type=int, default=3,
                    help="Vector drawings threshold for considering a page "
                         "'with diagrams' (default 3).")
    # marker options (--engine marker only)
    ap.add_argument("--force-ocr", action="store_true",
                    help="Marker: forces OCR (for scanned PDFs).")
    ap.add_argument("--block-size", type=int, default=0,
                    help="Marker: splits the PDF into blocks of N pages (15-20 recommended "
                         "with 8 GB of RAM). 0 = whole file.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only show what would be converted.")
    args = ap.parse_args()

    # ---- separate URLs from files ----
    urls = [u for u in args.inputs if u.startswith(("http://", "https://"))]
    local_tokens = [u for u in args.inputs if not u.startswith(("http://", "https://"))]

    files, missing = resolve_inputs(local_tokens, args.root, args.recursive)
    for m in missing:
        print(f"[!] not found: {m}", file=sys.stderr)
    files = [f for f in files if f.suffix.lower() in ALL_SUPPORTED
             or f.name.lower() in ("dockerfile", "makefile", "cmakelists.txt",
                                    "rakefile", "gemfile")]

    # ---- download the URLs as local files and add them to 'files' ----
    if urls:
        args.url_dest.mkdir(parents=True, exist_ok=True)
        for u in urls:
            try:
                print(f"Downloading {u} ...")
                _, _, out_md = convert_with_url(u, args.url_dest, args)
                # the downloaded file has already been converted to .md;
                # mark it as "done" by creating the .md entry
                print(f"  -> {out_md}")
            except Exception as e:
                print(f"  download ERROR: {e}", file=sys.stderr)

    if not files and not urls:
        print("No convertible files found.", file=sys.stderr); sys.exit(1)
    if not files:
        return

    print(f"Files to convert: {len(files)}")
    for f in files:
        print(f"  - {f}  ->  {f.with_suffix('.md').name}   "
              f"[{choose_engine(f, args.engine)}]")
    if args.dry_run:
        return

    results = []
    # Keep-awake: prevents Windows sleep during the whole batch.
    # The screen can still turn off (to save energy).
    with _KeepAwake():
        for f in files:
            print(f"\n>>> {f.name}")
            results.append(convert_one(f, args))

    print("\n=================  SUMMARY  =================")
    ok = sum(1 for r in results if r["ok"])
    for r in results:
        flag = "OK " if r["ok"] else "ERR"
        line = f"[{flag}] {Path(r['src']).name} -> {Path(r['out']).name} ({r['engine']})"
        if r.get("report"): line += f"\n        {r['report']}"
        if r.get("note"):   line += f"   {r['note']}"
        print(line)
    print(f"\nCompletati {ok}/{len(results)}.")


if __name__ == "__main__":
    main()
