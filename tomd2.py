#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tomd2.py — Converte file in Markdown ad alta fedelta', GRATIS e in LOCALE.

Engine PRIMARIO per PDF: pymupdf4llm (legge il livello-testo del PDF, velocissimo
su CPU, niente modelli AI da scaricare). Adatto a PDF "nativi" come slide di
PowerPoint esportate (testo selezionabile, vettoriale).

Routing automatico:
  - PDF (.pdf)                              -> pymupdf4llm  [default]
  - Office/Web (.pptx .docx .xlsx .html ..) -> markitdown
  - Immagini (.png .jpg ...)                -> marker (unico engine sensato per immagini)
  - .md .txt                                -> copiati cosi' come sono

Opzionale:
  --engine marker      forza marker (solo per PDF scansionati o casi limite).
                       Lento su CPU senza GPU; usa --block-size per file grandi.

Output:
  <nomefile>.md nella STESSA cartella del file originale.
  Figure e immagini di pagina in media/<nomefile>/.
  (la cartella 'media/' raccoglie le immagini di TUTTI i file convertiti)

Uso rapido:
    python tomd2.py 2.pdf --overwrite                # pymupdf4llm (rapido)
    python tomd2.py "*.pdf" --overwrite              # batch di PDF
    python tomd2.py slide.pdf --engine marker        # forza marker (lento)
    python tomd2.py scan.pdf --engine marker --force-ocr --block-size 15
    python tomd2.py 2.pdf --no-page-images           # senza render pagina

Dipendenze:
    pip install -U markitdown marker-pdf pymupdf pymupdf4llm
"""

from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys, time
from pathlib import Path

# ----------------------------------------------------------------------------
# Estensioni gestite
# ----------------------------------------------------------------------------
OFFICE_WEB_EXT = {".pptx", ".docx", ".xlsx", ".xls", ".doc", ".ppt",
                  ".html", ".htm", ".csv", ".tsv", ".json", ".xml", ".epub", ".rtf",
                  ".odt", ".ods", ".odp"}
IMAGE_EXT      = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
PDF_EXT        = {".pdf"}
PASSTHROUGH    = {".md", ".markdown", ".txt"}

# Codice sorgente: mappa estensione -> linguaggio per il blocco markdown
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

# File di config / dati testuali (anche senza linguaggio specifico)
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

# Archive (zip, tar, ecc.)
ARCHIVE_EXT = {".zip", ".tar", ".tgz", ".tar.gz", ".tbz", ".tar.bz2"}

# E-book proprietari (richiedono strumenti esterni come calibre)
EBOOK_EXT = {".mobi", ".azw", ".azw3", ".kfx", ".fb2", ".djvu", ".djv"}

# Audio/Video (richiedono whisper per la trascrizione)
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv", ".wmv", ".m4v"}

ALL_SUPPORTED  = (OFFICE_WEB_EXT | IMAGE_EXT | PDF_EXT | PASSTHROUGH |
                  CODE_EXT | TEXT_DATA_EXT | IPYNB_EXT | EMAIL_EXT |
                  ARCHIVE_EXT | EBOOK_EXT | AUDIO_EXT | VIDEO_EXT)


# ----------------------------------------------------------------------------
# Ricerca dei file (nomi, percorsi, glob)
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
# Utility immagini: link relativi corretti
# ----------------------------------------------------------------------------
def _fix_charset_mojibake(text: str) -> str:
    """Sistema i caratteri 'mojibake' (UTF-8 letto come CP1252/CP437).
       Esempi rilevati dalle recensioni LLM:
         ┬«  ->  «     (CP437/cp850 misread di U+00AB)
         ┬»  ->  »
         тАФ  ->  —     (CP1252 misread di U+2014 em-dash)
         тАЪ  ->  ‚
         тАЛ  ->  ‌
         тАЭ  ->  ”
         тАЮ  ->  „
         ┬·  ->  ·
         тДв  ->  ™
    """
    # 1) tentativo "intelligente": prova a re-encodare se vediamo i marker tipici
    if "тАФ" in text or "┬«" in text or "тАЭ" in text or "├¿" in text:
        for src_enc, dst_enc in (("cp1252","utf-8"), ("cp437","utf-8"),
                                  ("latin-1","utf-8")):
            try:
                fixed = text.encode(src_enc, errors="strict").decode(dst_enc, errors="strict")
                # Heuristica: il fix riduce le sequenze di byte sospetti
                if (fixed.count("тА") + fixed.count("┬") + fixed.count("├")
                        < text.count("тА") + text.count("┬") + text.count("├")):
                    text = fixed
                    break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    # 2) sostituzioni dirette di sequenze residue
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
    """Pulizie conservative tipiche dei PDF di slide matematiche.
       Protegge i link Markdown ![alt](path) e [text](url) per non corrompere
       i path delle immagini (es. 'Matlab1_AA2526_assets').
    """
    # 0) proteggi link markdown (sia immagini ![..](..) sia link [..](..))
    stash: list[str] = []
    def _hide(m):
        stash.append(m.group(0))
        return f"\x00LNK{len(stash)-1}\x00"
    md = re.sub(r"!?\[[^\]\n]*\]\([^)\n]+\)", _hide, md)

    # 1) <lettera>[<esponente>] -> <lettera>^{<esponente>}
    md = re.sub(
        r"([A-Za-z])\[([A-Za-z0-9 ]{1,6})\](?!\()",
        lambda m: f"{m.group(1)}^{{{m.group(2)}}}", md)

    # 2) 'xn-1' -> 'x_{n-1}' solo dentro italici _..._ (non multilinea)
    def _pedice_in_ital(m):
        body = m.group(1)
        body = re.sub(r"([A-Za-z])(\d+|n-\d+|n\+\d+)", r"\1_{\2}", body)
        return f"_{body}_"
    md = re.sub(r"_([^_\n]{1,160})_", _pedice_in_ital, md)

    # 3) ripristina i link
    md = re.sub(r"\x00LNK(\d+)\x00", lambda m: stash[int(m.group(1))], md)
    return md


def _rewrite_image_paths(md: str, prefix: str) -> str:
    """Fa puntare i link delle immagini alla sottocartella `prefix`.
       Salta link assoluti, URL e link gia' relativi a `prefix`."""
    def repl(m):
        path = m.group(1).strip()
        if path.startswith(("http://", "https://", "/")) or path.startswith(prefix + "/") or path.startswith(prefix):
            return m.group(0)
        # gia' relativo? estrai solo il basename per metterlo sotto prefix
        return f"]({prefix}/{Path(path).name})"
    return re.sub(r"\]\(([^)]+\.(?:png|jpg|jpeg|webp|gif|tif|tiff|bmp))\)",
                  repl, md, flags=re.IGNORECASE)


# Destinazione di link/immagine: ammette UNA coppia di parentesi bilanciate
# interne (es. 'media/DocScanner (2)/img.png'), come da spec CommonMark.
_MD_LINK_RE = re.compile(
    r"(!?\[[^\]\n]*\])\(([^()\n]*(?:\([^()\n]*\)[^()\n]*)*)\)")


def _encode_md_link_urls(md: str) -> str:
    """Percent-encoding dei percorsi nei link/immagini Markdown.

       Una destinazione con spazi nudi ('media/APPUNTI ASTRONOMIA/f.png')
       NON e' CommonMark valido: VS Code, GitHub, Obsidian ecc. non la
       renderizzano e l'immagine (formula inclusa) "sparisce". Con
       'media/APPUNTI%20ASTRONOMIA/f.png' funziona ovunque.
       Non tocca URL esterni, ancore, destinazioni <...> o gia' pulite;
       '%' resta safe quindi i path gia' encodati non vengono ri-encodati."""
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
# Motore 1: markitdown (DOCX / HTML / EPUB) — gratis, istantaneo
# Usato come fallback per Office/Web NON xlsx/pptx (per quelli c'e' un motore
# dedicato, vedi sotto).
# ----------------------------------------------------------------------------
def convert_with_markitdown(src: Path) -> str:
    from markitdown import MarkItDown
    return MarkItDown(enable_plugins=False).convert(str(src)).text_content


# ----------------------------------------------------------------------------
# Motore Excel: openpyxl dedicato (qualita' ottima, niente NaN, niente Unnamed)
# ----------------------------------------------------------------------------
def convert_with_xlsx(src: Path) -> str:
    """Converte .xlsx in Markdown leggibile da un'IA:
       - 1 sezione H2 per ogni foglio
       - tabelle markdown vere (non NaN, non 'Unnamed: X')
       - rispetta le merged cells (la prima cella tiene il valore)
       - celle numeriche formattate compatte (no decimali infiniti)"""
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
            parts.append("_(foglio vuoto)_\n"); continue
        # taglia colonne completamente vuote a destra
        max_cols = 0
        for r in rows:
            for i in range(len(r) - 1, -1, -1):
                if r[i] != "":
                    max_cols = max(max_cols, i + 1); break
        if max_cols == 0: parts.append("_(foglio vuoto)_\n"); continue
        rows = [r[:max_cols] + [""] * (max_cols - len(r[:max_cols])) for r in rows]
        # taglia righe completamente vuote in fondo
        while rows and all(c == "" for c in rows[-1]):
            rows.pop()
        if not rows:
            parts.append("_(foglio vuoto)_\n"); continue
        # cerca la prima riga "header": prima riga non vuota con almeno 2 celle
        hdr_idx = 0
        for i, r in enumerate(rows):
            if sum(1 for c in r if c) >= 2:
                hdr_idx = i; break
        # se sopra l'header ci sono righe titolo/note, le metto come testo
        for i in range(hdr_idx):
            tx = " ".join(c for c in rows[i] if c)
            if tx: parts.append(f"_{tx}_\n")
        # tabella markdown
        hdr = rows[hdr_idx]
        # se header e' tutto vuoto, usa Col_1, Col_2...
        if all(c == "" for c in hdr):
            hdr = [f"Col_{i+1}" for i in range(len(hdr))]
        parts.append("| " + " | ".join(hdr) + " |")
        parts.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for r in rows[hdr_idx + 1:]:
            # salta righe completamente vuote
            if all(c == "" for c in r): continue
            parts.append("| " + " | ".join(r) + " |")
        parts.append("")
    wb.close()
    return "\n".join(parts).strip() + "\n"


# ----------------------------------------------------------------------------
# Motore PowerPoint: python-pptx dedicato (testo + TUTTE le immagini estratte)
# ----------------------------------------------------------------------------
def convert_with_pptx(src: Path, media_dirname: str = "media") -> tuple[str, Path]:
    """Converte .pptx in Markdown estraendo:
       - una sezione H2 per ogni slide (testo titolo + corpo + note)
       - TUTTE le immagini incorporate (Picture + Shape con fill image)
       - le tabelle, in markdown vero
       - le note del relatore
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
        """Visita ricorsivamente le shape (gestisce anche i group)."""
        nonlocal img_idx
        out: list[str] = []
        for shape in shapes:
            try:
                stype = shape.shape_type
            except Exception:
                stype = None

            # group: ricorsione
            if stype == MSO_SHAPE_TYPE.GROUP:
                out.extend(_walk_shapes(shape.shapes, slide_no, depth + 1))
                continue

            # immagine: salva e linka
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
                    # heuristic: la prima text frame della slide e' spesso il titolo
                    out.append(text)
        return out

    for i, slide in enumerate(pres.slides, start=1):
        parts.append(f"\n## Slide {i}\n")
        # titolo se presente
        title = None
        try:
            if slide.shapes.title and slide.shapes.title.has_text_frame:
                title = slide.shapes.title.text_frame.text.strip()
        except Exception:
            pass
        if title:
            parts.append(f"### {title}\n")
        # contenuto (tutte le shape tranne il titolo)
        content_shapes = [s for s in slide.shapes
                          if not (slide.shapes.title is not None and s == slide.shapes.title)]
        body = _walk_shapes(content_shapes, i)
        if body:
            parts.append("\n".join(body))
        # note del relatore
        try:
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    parts.append(f"\n**Note relatore:** {notes}")
        except Exception:
            pass

    return ("\n".join(parts).strip() + "\n", assets)


# ----------------------------------------------------------------------------
# Motore CSV / TSV: tabella markdown pulita + statistiche di base
# ----------------------------------------------------------------------------
def _is_number(s: str) -> bool:
    if s is None or s == "": return False
    try: float(s.replace(",", "."))
    except ValueError: return False
    return True


def convert_with_csv(src: Path, max_preview_rows: int = 50) -> str:
    """Converte .csv/.tsv in Markdown con buona euristica per CSV "tecnici":
       - se la prima riga e' un TITOLO (cella sola), lo mette come H1
       - se ci sono righe di UNITA' di misura sotto l'header, le riconosce
       - statistica delle colonne numeriche (n, min, media, max)
       - mostra max N righe per non gonfiare il .md
    """
    import csv
    # rileva delimitatore
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
        return f"# {src.name}\n\n_(file vuoto)_\n"

    # max colonne usate in tutto il file
    max_cols = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    parts: list[str] = [f"# {src.name}\n"]

    # --- 1) cerca le righe "titolo" iniziali (una sola cella piena) ---
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
        parts.append(f"_(titolo: **{tr}**)_\n")

    # --- 2) header = prima riga successiva con tutte celle piene e NON numeriche ---
    hdr_idx = i
    while hdr_idx < len(rows):
        r = rows[hdr_idx]
        non_empty = [c for c in r if c]
        if non_empty and not any(_is_number(c) for c in non_empty):
            break
        hdr_idx += 1
    if hdr_idx >= len(rows):
        # nessun header: invento Col_X e parto dalla prima riga "i"
        hdr_idx = i
        headers = [f"Col_{k+1}" for k in range(max_cols)]
        data_start = i
        unit_row = None
    else:
        headers = list(rows[hdr_idx])
        # se la riga successiva e' "tutte stringhe non numeriche" -> UNITA' di misura
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

    # combina header + unita' se presenti
    headers_disp = []
    for k in range(max_cols):
        h = headers[k] if k < len(headers) else ""
        u = unit_row[k] if unit_row and k < len(unit_row) else ""
        if not h: h = f"Col_{k+1}"
        if u: headers_disp.append(f"{h} [{u}]")
        else: headers_disp.append(h)

    data_rows = rows[data_start:]
    # taglia righe completamente vuote in coda
    while data_rows and not any(data_rows[-1]):
        data_rows.pop()

    parts.append(f"**Righe dati**: {len(data_rows)}  "
                 f"**Colonne**: {max_cols}  "
                 f"**Separatore**: `'{dialect.delimiter}'`\n")

    # --- 3) statistica colonne numeriche ---
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
        parts.append("\n## Statistica colonne numeriche\n")
        parts.append("| Colonna | n | min | media | max |")
        parts.append("|---|---|---|---|---|")
        for h, n, mn, av, mx in stats:
            parts.append(f"| {h} | {n} | {mn:.6g} | {av:.6g} | {mx:.6g} |")

    # --- 4) anteprima dati ---
    parts.append("\n## Dati")
    if len(data_rows) > max_preview_rows:
        parts.append(f"_(mostro le prime {max_preview_rows} righe su "
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
# Motore CODE: sorgenti di programmazione (e di config testuale)
# Mette il contenuto in un blocco markdown ``` con linguaggio corretto.
# ----------------------------------------------------------------------------
def convert_with_code(src: Path) -> str:
    """Converte un file sorgente in Markdown:
       - intestazione H1 col nome file e dimensione/righe
       - blocco fenced code (```lang) col contenuto integrale
    """
    ext = src.suffix.lower()
    # nomi speciali senza estensione (Dockerfile, Makefile, ...)
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
    # protezione: se il file contiene ``` lo "scolpisco" con ` adatto
    fence = "```"
    while fence in text:
        fence += "`"
    n_lines = text.count("\n") + 1
    sz = src.stat().st_size
    parts = [
        f"# {src.name}",
        f"",
        f"**Linguaggio**: `{lang or '(testo)'}`  "
        f"**Righe**: {n_lines}  **Dimensione**: {sz/1024:.1f} KB",
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
    parts = [f"# {src.name}", "", f"**Notebook** (linguaggio: {lang})", ""]
    for i, cell in enumerate(nb.get("cells", []), 1):
        ctype = cell.get("cell_type", "")
        src_lines = cell.get("source", [])
        src_text = "".join(src_lines) if isinstance(src_lines, list) else str(src_lines)
        if ctype == "markdown":
            parts.append(f"<!-- Cella {i}: markdown -->\n{src_text}\n")
        elif ctype == "code":
            parts.append(f"<!-- Cella {i}: code -->")
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
                        parts.append(f"\n*(output: immagine PNG embedded — non rinderizzata in Markdown)*")
                elif otype == "error":
                    parts.append(f"\n*Errore:* `{out.get('ename','')}: {out.get('evalue','')}`")
            parts.append("")
        elif ctype == "raw":
            parts.append(f"<!-- Cella {i}: raw -->\n```\n{src_text.rstrip()}\n```\n")
    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------------
# Motore EMAIL .eml -> markdown (header + corpo + allegati estratti)
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
    parts.append("| Campo | Valore |")
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
        parts.append("## Corpo (testo)\n")
        parts.append(body_plain.strip())
        parts.append("")
    elif body_html:
        parts.append("## Corpo (HTML convertito)\n")
        # mini converter HTML -> testo
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
        # nome sicuro
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", fname)
        try:
            payload = part.get_payload(decode=True)
            (assets / safe).write_bytes(payload or b"")
            attachments.append((fname, safe, len(payload or b"")))
        except Exception:
            pass

    if attachments:
        parts.append("\n## Allegati\n")
        parts.append("| Nome | Dimensione | Link |")
        parts.append("|---|---|---|")
        for orig, safe, sz in attachments:
            parts.append(f"| {orig} | {sz/1024:.1f} KB | [{safe}]({asset_prefix}/{safe}) |")

    return ("\n".join(parts) + "\n", assets)


# ----------------------------------------------------------------------------
# Motore ARCHIVE .zip -> markdown (lista file + converte ricorsivamente i contenuti)
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
        parts.append("_(non e' uno ZIP valido)_")
        return ("\n".join(parts), assets)

    with zipfile.ZipFile(src) as zf:
        infolist = zf.infolist()
        parts.append(f"**File contenuti**: {len([i for i in infolist if not i.is_dir()])}")
        parts.append("")
        parts.append("| Path | Dimensione | Compresso |")
        parts.append("|---|---|---|")
        for info in infolist:
            if info.is_dir(): continue
            parts.append(f"| `{info.filename}` | {info.file_size/1024:.1f} KB "
                         f"| {info.compress_size/1024:.1f} KB |")
        # estraggo TUTTO in assets/_extracted/
        extract_dir = assets / "_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_dir)

    if _depth < max_depth and args is not None:
        parts.append("\n## Contenuto dei file (convertito ricorsivamente)\n")
        for sub in sorted(extract_dir.rglob("*")):
            if not sub.is_file(): continue
            if sub.suffix.lower() not in ALL_SUPPORTED: continue
            try:
                sub_eng = choose_engine(sub, "auto")
                parts.append(f"\n### {sub.relative_to(extract_dir).as_posix()}  "
                             f"`[{sub_eng}]`\n")
                # ricicla convert_one (semplice) — rimanderebbe a un .md affianco
                # qui invece estraiamo il testo "inline" senza scrivere .md per ogni sub
                inner_text = _inline_convert(sub, args)
                # rientra di 1 livello di header per non sovrapporsi
                inner_text = re.sub(r"^(#+) ", r"#\1 ", inner_text, flags=re.M)
                parts.append(inner_text.strip())
            except Exception as e:
                parts.append(f"_(non convertito: {type(e).__name__}: {e})_")

    return ("\n".join(parts) + "\n", assets)


def _inline_convert(sub: Path, args) -> str:
    """Converte un sotto-file in stringa Markdown senza scrivere il .md su disco.
       Usato dall'archive engine per inserire i contenuti nel md padre."""
    eng = choose_engine(sub, "auto")
    if eng == "code":     return convert_with_code(sub)
    if eng == "text_data": return convert_with_code(sub)
    if eng == "ipynb":    return convert_with_ipynb(sub)
    if eng == "csv":      return convert_with_csv(sub)
    if eng == "xlsx":     return convert_with_xlsx(sub)
    if eng == "markitdown": return convert_with_markitdown(sub)
    if eng == "passthrough":
        return f"```\n{sub.read_text(encoding='utf-8', errors='replace')}\n```"
    return f"_(estensione {sub.suffix} non gestita in inline)_"


# ----------------------------------------------------------------------------
# Motore URL -> scarica e converti
# ----------------------------------------------------------------------------
def convert_with_url(url: str, dest_dir: Path, args) -> tuple[str, Path, Path]:
    """Scarica una URL e la converte. Ritorna (markdown, assets, output_md_path).
       Il file scaricato viene salvato in dest_dir/<safe_name>.<ext>."""
    import urllib.request, urllib.parse, tempfile, mimetypes
    parsed = urllib.parse.urlparse(url)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", parsed.netloc + parsed.path) or "download"
    safe = safe[:120]
    req = urllib.request.Request(url, headers={"User-Agent": "tomd2/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
        ctype = r.headers.get("Content-Type", "").split(";")[0].strip()
    ext = mimetypes.guess_extension(ctype) or ".bin"
    # se l'URL ha gia' estensione esplicita, preferiscila
    url_ext = Path(parsed.path).suffix
    if url_ext.lower() in ALL_SUPPORTED:
        ext = url_ext
    # evita doppie estensioni se 'safe' termina gia' con 'ext'
    if safe.lower().endswith(ext.lower()):
        out_file = dest_dir / safe
    else:
        out_file = dest_dir / f"{safe}{ext}"
    out_file.write_bytes(data)
    # ora converti il file
    if ext.lower() in ALL_SUPPORTED:
        eng = choose_engine(out_file, "auto")
        # usiamo convert_one ma in modo da NON sovrascrivere il file scaricato
        # (chiamiamo le funzioni dirette)
        if eng == "html" or out_file.suffix.lower() in {".html", ".htm"}:
            text = convert_with_markitdown(out_file)
            return text, dest_dir, out_file.with_suffix(".md")
        # delega al convert_one tramite args fittizio
        from types import SimpleNamespace
        new_args = SimpleNamespace(**vars(args)) if hasattr(args, "__dict__") else args
        new_args.overwrite = True
        info = convert_one(out_file, new_args)
        return (Path(info["out"]).read_text(encoding="utf-8") if info["ok"] else
                f"_(conversione fallita: {info.get('note')})_",
                dest_dir, Path(info["out"]))
    else:
        # contenuto sconosciuto: blocco code
        try: txt = data.decode("utf-8", errors="replace")
        except Exception: txt = f"_(contenuto binario, {len(data)} bytes)_"
        return (f"# Download da {url}\n\n```\n{txt[:5000]}\n```\n",
                dest_dir, out_file.with_suffix(".md"))


# ----------------------------------------------------------------------------
# Motore EBOOK proprietario (.mobi, .azw3, .djvu, .fb2): richiede calibre
# ----------------------------------------------------------------------------
def convert_with_ebook(src: Path) -> str:
    """Converte e-book proprietari via 'ebook-convert' di Calibre, se installato."""
    exe = shutil.which("ebook-convert")
    if not exe:
        return (f"# {src.name}\n\n"
                f"**Tipo**: e-book ({src.suffix})\n\n"
                f"⚠ Non posso convertirlo automaticamente: serve **Calibre** "
                f"(https://calibre-ebook.com/download). Installalo e riprova.\n"
                f"Comando manuale alternativo:\n"
                f"```\n"
                f"ebook-convert \"{src.name}\" \"{src.stem}.epub\"\n"
                f"```\n"
                f"poi converti l'.epub con tomd2.\n")
    import tempfile, subprocess as sp
    tmp = Path(tempfile.mkdtemp(prefix="ebook_"))
    try:
        epub = tmp / f"{src.stem}.epub"
        sp.run([exe, str(src), str(epub)], check=True, capture_output=True)
        return convert_with_markitdown(epub)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------
# Motore AUDIO / VIDEO: trascrizione con openai-whisper (locale, gratis)
# ----------------------------------------------------------------------------
def convert_with_media(src: Path, model: str = "base") -> str:
    """Trascrive audio/video con openai-whisper (locale).
       Richiede: pip install openai-whisper + ffmpeg installato."""
    try:
        import whisper
    except ImportError:
        return (f"# {src.name}\n\n"
                f"**Tipo**: {('audio' if src.suffix.lower() in AUDIO_EXT else 'video')} "
                f"({src.suffix})\n\n"
                f"⚠ Non posso trascrivere: manca **openai-whisper**. Installa:\n"
                f"```\n"
                f"pip install -U openai-whisper\n"
                f"```\n"
                f"e assicurati di avere **ffmpeg** nel PATH (https://ffmpeg.org).\n")
    if not shutil.which("ffmpeg"):
        return (f"# {src.name}\n\n"
                f"⚠ openai-whisper trovato, ma manca **ffmpeg** nel PATH "
                f"(https://ffmpeg.org/download.html).\n")
    print(f"  Whisper: carico modello '{model}' (prima volta scarica ~150MB)...")
    m = whisper.load_model(model)
    print(f"  Whisper: trascrivo {src.name} ...")
    res = m.transcribe(str(src), verbose=False)
    parts = [f"# {src.name}", "",
             f"**Trascrizione automatica** (Whisper, modello `{model}`, "
             f"lingua: {res.get('language','?')})", ""]
    for seg in res.get("segments", []):
        t0 = seg["start"]; t1 = seg["end"]
        def _ts(s): return f"{int(s//60):02d}:{int(s%60):02d}"
        parts.append(f"`[{_ts(t0)}-{_ts(t1)}]` {seg['text'].strip()}")
    parts.append("")
    parts.append("## Testo completo\n")
    parts.append(res.get("text", "").strip())
    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------------
# Motore 2 (PRIMARIO per PDF): pymupdf4llm
#
# Strategia:
#  - to_markdown(page_chunks=True): otteniamo testo+tabelle in markdown e
#    metadati per pagina (immagini estratte, conteggi, ecc.).
#  - write_images=True: pymupdf4llm salva le immagini incorporate e i grafici
#    rilevati come PNG dentro assets/.
#  - In aggiunta renderizziamo come PNG ad alta risoluzione TUTTE le pagine
#    con grafica vettoriale (drawings): cosi' i diagrammi di flusso, anche se
#    pymupdf4llm li approssima male in Markdown, restano disponibili come
#    immagine (un'IA multimodale li legge).
#  - Il render di pagina viene INSERITO nel .md proprio nella sezione di quella
#    pagina, in modo che testo + immagine di pagina vivano vicini.
# ----------------------------------------------------------------------------
def _pages_with_vector_graphics(src: Path, min_drawings: int = 3) -> set[int]:
    """Ritorna l'insieme delle pagine (1-based) con >= min_drawings drawings
       vettoriali (candidate a contenere diagrammi/flowchart)."""
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
    """Renderizza le pagine indicate come PNG dentro assets/_pages/.
       Ritorna mapping pagina_1based -> nomefile (basename) dentro _pages."""
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
            print(f"  [WARN] render pagina {n} fallito: {e}", file=sys.stderr)
    doc.close()
    return result


def convert_with_pymupdf4llm(src: Path, page_images: bool,
                             page_image_dpi: int,
                             vector_threshold: int,
                             media_dirname: str = "media",
                             lean_md: bool = False) -> tuple[str, Path]:
    """Converte un PDF in Markdown con pymupdf4llm.
       Crea media/<stem>/ con immagini estratte e (opz.) immagini di pagina.

       lean_md=True: niente <details><summary>, link immagine compatti.
                     Riduce boilerplate per IA (suggerito da Gemini)."""
    import pymupdf4llm, tempfile, pymupdf

    stem = src.stem
    assets = src.parent / media_dirname / stem
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)

    # --- pymupdf4llm BUG workaround: i nomi delle immagini vengono normalizzati
    # (spazi -> underscore) ma poi cercati col nome originale. Per evitarlo,
    # lavoriamo in una directory temporanea col PDF rinominato senza spazi e
    # senza caratteri "esotici", poi spostiamo i PNG nella cartella assets reale.
    # In aggiunta usiamo un Document aperto, cosi' pymupdf4llm vede il nome
    # canonicale (es. 'doc.pdf') per costruire i path immagine.
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
                table_strategy="lines_strict",  # evita falsi positivi su footer/header
            )
        finally:
            doc.close()

        # Sposta tutte le immagini prodotte in `assets`, rinominandole
        # per usare il vero stem (con spazi reintrodotti come '_').
        for f in tmp_assets.iterdir():
            if f.is_file():
                shutil.move(str(f), str(assets / f.name))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    asset_prefix = f"{media_dirname}/{stem}"

    # --- 2) page-image "garanzia diagrammi" per pagine con grafica vettoriale ---
    page_img_map: dict[int, str] = {}
    if page_images:
        vec_pages = _pages_with_vector_graphics(src, min_drawings=vector_threshold)
        if vec_pages:
            print(f"  Rendering pagine vettoriali ({len(vec_pages)}/{_pdf_pages(src)}) "
                  f"come PNG (dpi={page_image_dpi}) ...")
            page_img_map = _render_pages_to_png(
                src, assets, vec_pages, dpi=page_image_dpi)

    # --- 3) apro il PDF anche con pymupdf per il fallback testo ---
    # (per pagine in cui pymupdf4llm estrae poco testo ma il PDF ne ha)
    import pymupdf as _pdf
    raw_doc = _pdf.open(str(src))

    # lean_md è già parametro della funzione (default False)
    opt_lean_md = lean_md

    # --- 4) ricomponiamo il Markdown pagina per pagina ---
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

        # ===== FALLBACK TESTO (fix Gemini: formule diventate immagini) =====
        # Se la pagina del PDF originale ha molto piu' testo di quello che
        # pymupdf4llm ci ha dato, affianchiamo l'estrazione raw come backup.
        # Soglia: se il chunk md ha meno di 50% del testo raw (di una pagina con
        # almeno 200 caratteri raw), aggiungiamo una sezione "testo originale".
        try:
            if 0 < page_no - 1 < raw_doc.page_count:
                raw_text = raw_doc[page_no - 1].get_text("text").strip()
            else:
                raw_text = ""
        except Exception:
            raw_text = ""
        # rimuovo i blocchi image markdown dal text per misurarne il "vero" contenuto
        text_only = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
        text_only_chars = len(re.sub(r"\s+", "", text_only))
        raw_chars = len(re.sub(r"\s+", "", raw_text))
        needs_fallback = (raw_chars >= 200 and
                          text_only_chars < raw_chars * 0.5)

        parts.append(f"\n\n<!-- ===== Pagina {page_no} di {n_pages} ===== -->\n")
        # link al render della pagina (se presente) PRIMA del testo
        if page_no in page_img_map:
            rel = f"{asset_prefix}/_pages/{page_img_map[page_no]}"
            if opt_lean_md:
                # versione "lean": link compatto, senza <details>
                parts.append(f"\n![pagina {page_no}]({rel})\n")
            else:
                parts.append(
                    f"\n<details><summary>"
                    f"Pagina {page_no} (immagine ad alta risoluzione — garanzia diagrammi)"
                    f"</summary>\n\n"
                    f"![pagina {page_no}]({rel})\n\n"
                    f"</details>\n"
                )
        parts.append("\n" + text.strip() + "\n")
        # Se serve il fallback, aggiungo il testo raw del PDF (per non perdere
        # formule/numeri intrappolati in immagini)
        if needs_fallback:
            parts.append(f"\n<!-- testo originale del PDF (fallback formule/dati): -->\n")
            parts.append(f"\n<details><summary>Testo grezzo pagina {page_no} "
                         f"(per IA: contiene le formule che pymupdf4llm ha catturato come immagini)"
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
# Rilevamento PDF "scansionato + OCR" (es. libri da Anna's Archive, archive.org)
# ----------------------------------------------------------------------------
def _is_scanned_ocr_pdf(src: Path,
                        page_coverage_threshold: float = 0.5,
                        min_pages_scanned_ratio: float = 0.6,
                        min_text_chars_per_page: int = 100) -> bool:
    """Ritorna True se il PDF e' una SCANSIONE con OCR sottostante:
       - la maggior parte delle pagine ha un'immagine raster che copre
         >= page_coverage_threshold dell'area della pagina;
       - ma c'e' anche testo selezionabile (> min_text_chars_per_page).
       In tal caso pymupdf4llm risulta inefficace (vede ogni pagina come
       'una grande figura' e salva la pagina come PNG senza estrarre il testo).
       Conviene allora usare l'estrazione diretta del testo via fitz.get_text.
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
            # area totale delle immagini raster nella pagina
            img_area = 0
            for img in page.get_images(full=True):
                xref = img[0]
                # bbox via get_image_rects (piu' affidabile)
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
# Motore "OCR-scan PDF": preserva il TESTO OCR e mostra ogni pagina come PNG
# ----------------------------------------------------------------------------
def convert_with_ocr_scan(src: Path, page_image_dpi: int = 150,
                          media_dirname: str = "media") -> tuple[str, Path]:
    """Per i PDF scansionati+OCR (libri da Anna's Archive, archive.org, ecc.):
       - testo estratto direttamente con fitz.get_text (preserva l'OCR)
       - ogni pagina renderizzata come PNG (backup visivo, per figure/formule)
       - NON estrae le immagini incorporate (sono solo le pagine stesse,
         occupano centinaia di MB e sono ridondanti rispetto alle page-image).
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
        # 1) testo OCR estratto dal PDF
        text = page.get_text("text").strip()
        # 2) page-image come backup visivo
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_name = f"page_{i+1:04d}.png"
        pix.save(str(pages_dir / png_name))
        parts.append(f"\n\n<!-- ===== Pagina {i+1} di {n} ===== -->\n")
        parts.append(
            f"\n<details><summary>Pagina {i+1} (immagine scansione originale)"
            f"</summary>\n\n"
            f"![pagina {i+1}]({asset_prefix}/_pages/{png_name})\n\n"
            f"</details>\n")
        if text:
            parts.append("\n" + text + "\n")
        else:
            parts.append("\n_(testo non estratto da questa pagina — usa l'immagine sopra)_\n")
    doc.close()

    full_md = "".join(parts).strip() + "\n"
    return full_md, assets


# ----------------------------------------------------------------------------
# Motore Tesseract: OCR VELOCE e locale per immagini e PDF scansionati
# SENZA testo. Secondi invece delle ore di marker (che resta come opzione
# per la massima qualita' su formule/tabelle complesse).
# ----------------------------------------------------------------------------
def _find_tesseract() -> tuple[str | None, str | None]:
    """Ritorna (path tesseract.exe, tessdata_dir) oppure (None, None).
       tessdata_dir punta alla cartella utente se contiene ita.traineddata
       (installata li' quando Program Files non e' scrivibile)."""
    import os
    exe = shutil.which("tesseract")
    if not exe:
        cand = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if cand.exists():
            exe = str(cand)
    if not exe:
        return None, None
    # tessdata: preferisci la cartella utente se ha l'italiano
    user_td = Path(os.environ.get("LOCALAPPDATA", "")) / "tessdata"
    if (user_td / "ita.traineddata").exists():
        return exe, str(user_td)
    return exe, None


def _tesseract_ocr_pil(img, lang: str = "ita+eng") -> str:
    """OCR di un'immagine PIL via pytesseract. Ritorna testo (o '')."""
    import os
    import pytesseract
    exe, tessdata = _find_tesseract()
    if not exe:
        raise RuntimeError(
            "Tesseract non installato. Installa con: "
            "winget install --id UB-Mannheim.TesseractOCR -e")
    pytesseract.pytesseract.tesseract_cmd = exe
    if tessdata:
        # TESSDATA_PREFIX e' piu' affidabile di --tessdata-dir (quoting fragile)
        os.environ["TESSDATA_PREFIX"] = tessdata
    # se manca ita, degrada a eng senza errori
    try:
        return pytesseract.image_to_string(img, lang=lang)
    except Exception:
        return pytesseract.image_to_string(img, lang="eng")


def convert_with_ocr_image(src: Path, media_dirname: str = "media",
                           lang: str = "ita+eng") -> tuple[str, Path]:
    """Immagine -> Markdown: OCR veloce con Tesseract + immagine linkata.
       Se Tesseract manca, produce comunque un .md con l'immagine e un avviso
       (nessun crash)."""
    from PIL import Image

    stem = src.stem
    assets = src.parent / media_dirname / stem
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)
    asset_prefix = f"{media_dirname}/{stem}"

    # copia l'immagine negli assets (cosi' il .md e' autosufficiente)
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
            parts += ["## Testo estratto (OCR Tesseract)", "", text, ""]
        else:
            parts += ["_(nessun testo rilevato nell'immagine)_", ""]
    except RuntimeError as e:
        parts += [f"⚠ OCR non disponibile: {e}", ""]
    except Exception as e:
        parts += [f"⚠ OCR fallito ({type(e).__name__}: {e}) — resta l'immagine sopra.", ""]

    return "\n".join(parts), assets


def _is_scanned_no_text_pdf(src: Path,
                            page_coverage_threshold: float = 0.5,
                            min_pages_scanned_ratio: float = 0.6) -> bool:
    """True se il PDF e' una scansione SENZA testo selezionabile:
       pagine coperte da raster ma quasi nessun testo estraibile.
       (Il caso 'scansione CON OCR' e' gestito da _is_scanned_ocr_pdf.)"""
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
                and text_chars < n * 50)  # quasi zero testo
    finally:
        doc.close()


def convert_with_ocr_tesseract_pdf(src: Path, page_image_dpi: int = 200,
                                   media_dirname: str = "media",
                                   lang: str = "ita+eng") -> tuple[str, Path]:
    """PDF scansionato SENZA OCR -> Markdown, con Tesseract pagina per pagina.
       Molto piu' veloce di marker su CPU (secondi/pagina invece di minuti).
       Ogni pagina: testo OCR + immagine della pagina come backup visivo."""
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
    print(f"  OCR Tesseract su {n} pagine (dpi={page_image_dpi}) ...")
    for i in range(n):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_name = f"page_{i+1:04d}.png"
        pix.save(str(pages_dir / png_name))
        # OCR sul rendering appena fatto (senza ri-leggere da disco)
        img = Image.open(_io.BytesIO(pix.tobytes("png")))
        try:
            text = _tesseract_ocr_pil(img, lang=lang).strip()
        except RuntimeError as e:
            doc.close()
            raise RuntimeError(str(e))
        parts.append(f"\n\n<!-- ===== Pagina {i+1} di {n} ===== -->\n")
        parts.append(
            f"\n<details><summary>Pagina {i+1} (immagine scansione)"
            f"</summary>\n\n![pagina {i+1}]({asset_prefix}/_pages/{png_name})\n\n"
            f"</details>\n")
        parts.append("\n" + (text if text else
                     "_(nessun testo rilevato in questa pagina)_") + "\n")
        # progress leggibile da CLI e GUI
        print(f"  OCR: {int((i+1)*100/n)}% ({i+1}/{n})", flush=True)
    doc.close()

    full_md = "".join(parts).strip() + "\n"
    full_md = _fix_charset_mojibake(full_md)
    return full_md, assets


# ----------------------------------------------------------------------------
# Motore AI-PACK: PDF pieni di immagini -> .md COMPATTO ottimizzato per le IA.
#
# Problema: slide fotografiche (astronomia, medicina, arte...) producono md +
# PNG giganteschi (centinaia di MB) che nessuna chat IA accetta.
# Soluzione: UNA immagine WebP per pagina, qualita' ~70, lato max 1568 px
# (la risoluzione massima che Claude/GPT usano internamente: piu' grande =
# peso sprecato, l'IA la riduce comunque). Il testo della pagina (layer PDF
# + OCR Tesseract opzionale sulle pagine mute) sta accanto all'immagine.
# Riduzione tipica: 10-20x rispetto all'originale, zero perdita di
# informazione utile per l'IA.
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
        f"<!-- AI-PACK: 1 immagine WebP q{quality} max{max_side}px per pagina, "
        f"testo integrale accanto. Ottimizzato per lettura da IA. -->\n"
    ]
    tot_bytes = 0
    print(f"  AI-pack: {n} pagine -> WebP q{quality}, lato max {max_side}px ...")
    for i in range(n):
        page = doc[i]
        # DPI dinamico: rendo la pagina in modo che il lato lungo ~ max_side
        rect = page.rect
        long_side_pt = max(rect.width, rect.height)     # in punti (1/72")
        zoom = (max_side / long_side_pt) if long_side_pt > 0 else 2.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.open(_io.BytesIO(pix.tobytes("png")))
        webp_name = f"page_{i+1:04d}.webp"
        webp_path = pages_dir / webp_name
        img.save(str(webp_path), "WEBP", quality=quality, method=6)
        tot_bytes += webp_path.stat().st_size

        # testo: layer PDF, con OCR di riserva sulle pagine mute
        text = page.get_text("text").strip()
        if not text and ocr_fallback:
            try:
                text = _tesseract_ocr_pil(img, lang=lang).strip()
                if text:
                    text = "*(testo via OCR)*\n\n" + text
            except Exception:
                pass

        parts.append(f"\n\n<!-- ===== Pagina {i+1} di {n} ===== -->\n")
        parts.append(f"\n![pagina {i+1}]({asset_prefix}/_pages/{webp_name})\n")
        if text:
            parts.append("\n" + text + "\n")
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  AI-pack: {int((i+1)*100/n)}% ({i+1}/{n}, "
                  f"{tot_bytes/1024/1024:.1f} MB)", flush=True)
    doc.close()

    full_md = "".join(parts).strip() + "\n"
    full_md = _fix_charset_mojibake(full_md)
    print(f"  AI-pack completato: immagini totali {tot_bytes/1024/1024:.1f} MB")
    return full_md, assets


# ----------------------------------------------------------------------------
# Motore 3 (OPZIONALE): marker — solo per PDF scansionati o casi limite.
# Lento su CPU senza GPU. Supporta --block-size per evitare OOM.
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Keep-awake: impedisce a Windows di sospendersi durante la conversione lunga.
# Usa l'API SetThreadExecutionState (solo Windows). Su altri OS e' no-op.
# ----------------------------------------------------------------------------
class _KeepAwake:
    """Context manager: chiede a Windows di NON sospendersi mentre il blocco
       e' attivo. Lo schermo puo' spegnersi liberamente (per risparmio energia).

       Uso:
           with _KeepAwake():
               ...lavoro lungo...
    """
    # flag dell'API SetThreadExecutionState
    _ES_CONTINUOUS       = 0x80000000
    _ES_SYSTEM_REQUIRED  = 0x00000001
    _ES_AWAYMODE_REQUIRED = 0x00000040
    # (NON usiamo ES_DISPLAY_REQUIRED, cosi' lo schermo puo' spegnersi)

    def __enter__(self):
        if sys.platform != "win32":
            self._active = False
            return self
        try:
            import ctypes
            self._k = ctypes.windll.kernel32
            flags = self._ES_CONTINUOUS | self._ES_SYSTEM_REQUIRED
            # ES_AWAYMODE_REQUIRED: massima resilienza, anche su laptop a batteria
            flags |= self._ES_AWAYMODE_REQUIRED
            self._k.SetThreadExecutionState(flags)
            self._active = True
            print("  [keep-awake] Sospensione di Windows disabilitata "
                  "(lo schermo si puo' spegnere lo stesso)")
        except Exception as e:
            print(f"  [keep-awake] non disponibile: {e}", file=sys.stderr)
            self._active = False
        return self

    def __exit__(self, *a):
        if getattr(self, "_active", False):
            try:
                self._k.SetThreadExecutionState(self._ES_CONTINUOUS)
                print("  [keep-awake] Stato sospensione di Windows ripristinato")
            except Exception:
                pass


def _run_marker_single(src: Path, out_dir: Path, force_ocr: bool):
    if not shutil.which("marker_single"):
        raise RuntimeError("marker non trovato (pip install marker-pdf).")
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
    """Converte un PDF con marker (lento ma OCR di qualita').
       - block_size>0: spezza il PDF in blocchi e li unisce
       - resume=True (default): salta i blocchi gia' completati su disco,
         cosi' si puo' interrompere e riprendere senza perdere il lavoro fatto
    """
    assets = src.parent / media_dirname / src.stem
    is_pdf = src.suffix.lower() in PDF_EXT
    chunked = is_pdf and block_size and block_size > 0

    if chunked:
        # in modalita' a blocchi: NON cancello assets se resume e' attivo
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
                    # RESUME: blocco gia' completato? Lo riuso senza rifare marker.
                    if resume and existing_mds:
                        produced = existing_mds[0]
                        try:
                            # validazione minima: il .md deve essere non vuoto
                            txt = produced.read_text(encoding="utf-8")
                            if len(txt.strip()) > 50:
                                print(f"  [blocco {idx}/{len(chunks)}] pagine {ps}-{pe} "
                                      f"GIA' FATTO, salto (resume)")
                                rel = produced.parent.relative_to(src.parent).as_posix()
                                txt = _rewrite_image_paths(txt, rel)
                                merged.append(
                                    f"\n\n<!-- ===== Blocco {idx} - pagine {ps}-{pe} "
                                    f"(resume) ===== -->\n\n" + txt.strip() + "\n")
                                skipped += 1
                                continue
                        except Exception:
                            pass
                    # blocco da fare ex novo (cancella eventuali residui parziali)
                    if chunk_out.exists():
                        shutil.rmtree(chunk_out)
                    chunk_out.mkdir(parents=True, exist_ok=True)
                    print(f"  [blocco {idx}/{len(chunks)}] pagine {ps}-{pe} ...")
                    _run_marker_single(chunk_pdf, chunk_out, force_ocr)
                    mds = list(chunk_out.rglob("*.md"))
                    if not mds:
                        raise RuntimeError(f"marker non ha prodotto .md per blocco {idx}")
                    produced = mds[0]
                    text = produced.read_text(encoding="utf-8")
                    rel = produced.parent.relative_to(src.parent).as_posix()
                    text = _rewrite_image_paths(text, rel)
                    merged.append(
                        f"\n\n<!-- ===== Blocco {idx} - pagine {ps}-{pe} ===== -->\n\n"
                        + text.strip() + "\n")
                    done += 1
            if skipped:
                print(f"  [resume] saltati {skipped} blocchi gia' completati, "
                      f"rifatti {done}")
            full_md = "".join(merged)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        # modalita' singola (file intero, no blocchi): non c'e' resume sensato
        if assets.exists():
            shutil.rmtree(assets)
        assets.mkdir(parents=True, exist_ok=True)
        with _KeepAwake():
            _run_marker_single(src, assets, force_ocr)
        mds = list(assets.rglob("*.md"))
        if not mds:
            raise RuntimeError("marker non ha prodotto alcun .md")
        produced = mds[0]
        full_md = produced.read_text(encoding="utf-8")
        rel = produced.parent.relative_to(src.parent).as_posix()
        full_md = _rewrite_image_paths(full_md, rel)
    return full_md, assets


# ----------------------------------------------------------------------------
# Report di qualita'
# ----------------------------------------------------------------------------
def quality_report(md_text: str, assets: Path | None, elapsed: float | None = None,
                   n_pages: int | None = None) -> str:
    n_chars   = len(md_text)
    # formule: blocchi $$..$$ + inline $..$
    n_formule = md_text.count("$$") // 2 + len(re.findall(r"(?<!\$)\$[^\$\n]+\$", md_text))
    n_tabelle = len(re.findall(r"^\s*\|.+\|\s*$", md_text, flags=re.M))
    n_figure  = 0
    if assets and assets.exists():
        n_figure = sum(1 for f in assets.rglob("*")
                       if f.suffix.lower() in IMAGE_EXT)
    bits = []
    if n_pages is not None: bits.append(f"{n_pages} pagine")
    bits.append(f"{n_chars} caratteri")
    bits.append(f"~{n_tabelle} righe-tabella")
    bits.append(f"~{n_formule} formule")
    bits.append(f"{n_figure} figure")
    if elapsed is not None: bits.append(f"{elapsed:.1f}s")
    return " | ".join(bits)


# ----------------------------------------------------------------------------
# Routing + conversione di un singolo file
# ----------------------------------------------------------------------------
CSV_EXT = {".csv", ".tsv"}

def choose_engine(src: Path, forced: str | None) -> str:
    ext = src.suffix.lower()
    name_l = src.name.lower()
    if forced and forced != "auto":
        # 'marker' valido solo su PDF/immagini; 'pymupdf4llm' valido solo su PDF.
        if forced == "pymupdf4llm" and ext not in PDF_EXT:
            return "markitdown" if ext in OFFICE_WEB_EXT else "marker"
        # motori OCR-IA (ocr_vlm.py): solo PDF e immagini
        if forced.startswith("vlm_") and ext not in PDF_EXT and ext not in IMAGE_EXT:
            return choose_engine(src, "auto")
        return forced
    # nomi speciali senza estensione: Dockerfile, Makefile, CMakeLists.txt, ...
    if name_l in ("dockerfile", "makefile", "cmakelists.txt", "rakefile", "gemfile"):
        return "code"
    # routing dedicato per ogni formato (qualita' ottima)
    if ext in PDF_EXT:
        # rilevamento PDF scansionato+OCR (libri da Anna's Archive, ecc.)
        try:
            if _is_scanned_ocr_pdf(src):
                return "ocr_scan"
            # scansione SENZA testo -> OCR Tesseract automatico (veloce);
            # marker resta la scelta manuale per qualita' massima
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
    if ext in TEXT_DATA_EXT:  return "code"  # stesso motore con linguaggio specifico
    if ext in OFFICE_WEB_EXT: return "markitdown"  # docx, html, epub, json, odt, ...
    if ext in IMAGE_EXT:      return "ocr_image"   # Tesseract: secondi, non ore
    if ext in PASSTHROUGH:    return "passthrough"
    return "markitdown"


def convert_one(src: Path, args) -> dict:
    out_md = src.with_suffix(".md")
    info = {"src": str(src), "out": str(out_md), "engine": None,
            "ok": False, "note": "", "report": ""}

    if out_md.exists() and not args.overwrite and src.suffix.lower() not in PASSTHROUGH:
        info.update(ok=True, note="gia' presente (usa --overwrite)")
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
                page_image_dpi=max(args.page_image_dpi, 200),  # OCR vuole dpi alti
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
            # OCR con modelli di visione LOCALI via llama.cpp (ocr_vlm.py):
            # PaddleOCR-VL 1.6 (veloce) / Chandra OCR 2 (manoscritti, figure)
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
            raise ValueError(f"motore non valido: {engine}")

        # Spazi/parentesi nei path immagine rompono il rendering Markdown:
        # percent-encoding qui, cosi' vale per TUTTI i motori.
        text = _encode_md_link_urls(text)

        header = (f"<!-- Generato da tomd2.py | sorgente: {src.name} | "
                  f"motore: {engine} -->\n\n")
        out_md.write_text(header + text.strip() + "\n", encoding="utf-8")

        # Descrizioni IA delle immagini (opzionale, --describe-images):
        # modello di visione LOCALE via Ollama, vedi img_describe.py.
        # Non deve MAI far fallire la conversione: il .md e' gia' scritto.
        if getattr(args, "describe_images", False):
            try:
                import img_describe
                # NB: getattr con 'or' — argparse mette None se il flag manca
                _vmodel = (getattr(args, "describe_model", None)
                           or img_describe.DEFAULT_MODEL)
                print(f"  Descrizioni IA delle immagini (modello {_vmodel}) ...")
                st = img_describe.enrich_markdown(out_md, model=_vmodel)
                info["note"] = (f"descrizioni IA: {st['done']} nuove, "
                                f"{st['cached']} da cache, {st['failed']} fallite")
            except Exception as e:
                info["note"] = f"descrizioni IA NON aggiunte: {e}"
                print(f"  [WARN] {info['note']}", file=sys.stderr)

        elapsed = time.time() - t0
        info.update(ok=True,
                    report=quality_report(text, assets, elapsed, n_pages),
                    elapsed=elapsed)
    except subprocess.CalledProcessError as e:
        info["note"] = f"ERRORE marker (codice {e.returncode}): controlla RAM/modelli."
    except Exception as e:
        info["note"] = f"ERRORE: {type(e).__name__}: {e}"
    return info


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Converte file in Markdown ad alta fedelta', gratis e in locale. "
                    "Engine PDF di default: pymupdf4llm (velocissimo, CPU).")
    ap.add_argument("inputs", nargs="+",
                    help="Nomi, percorsi, pattern glob oppure URL http(s)://...")
    ap.add_argument("--url-dest", default=".", type=lambda s: Path(s).expanduser(),
                    help="Cartella di destinazione per i file scaricati da URL (default: .).")
    ap.add_argument("--root", default=".", type=lambda s: Path(s).expanduser(),
                    help="Cartella in cui cercare nomi/glob (default: corrente).")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false",
                    help="Non cercare nelle sottocartelle.")
    ap.add_argument("--engine", choices=["auto", "pymupdf4llm", "ocr_scan",
                                          "ocr_tesseract", "ocr_image", "ai_pack",
                                          "marker", "markitdown",
                                          "vlm_paddle", "vlm_chandra", "vlm_ibrido",
                                          "xlsx", "pptx", "csv",
                                          "code", "ipynb", "eml", "archive",
                                          "ebook", "audio", "video"],
                    default="auto",
                    help="Forza un motore. Default: auto (sceglie da estensione). "
                         "'ai_pack' = PDF fotografici -> md COMPATTO per IA "
                         "(WebP q70, 10-20x piu' leggero).")
    ap.add_argument("--ai-quality", type=int, default=70,
                    help="AI-pack: qualita' WebP 1-100 (default 70).")
    ap.add_argument("--ai-max-side", type=int, default=1568,
                    help="AI-pack: lato massimo immagini in px (default 1568, "
                         "il massimo che le IA usano internamente).")
    ap.add_argument("--whisper-model", default="base",
                    choices=["tiny", "base", "small", "medium", "large"],
                    help="Modello Whisper per audio/video (default 'base'). "
                         "'tiny' = velocissimo poco preciso; 'large' = lento accuratissimo.")
    ap.add_argument("--ocr-lang", default="ita+eng",
                    help="Lingue Tesseract per ocr_image/ocr_tesseract (default 'ita+eng').")
    ap.add_argument("--describe-images", action="store_true",
                    help="Dopo la conversione descrive OGNI immagine del .md con "
                         "un'IA di visione LOCALE (Ollama): formule in LaTeX, "
                         "grafici e diagrammi spiegati. Lento ma riprendibile "
                         "(cache). Richiede Ollama installato.")
    ap.add_argument("--describe-model", default=None,
                    help="Modello visione Ollama per --describe-images "
                         "(default: qwen2.5vl:3b).")
    # ---- OCR con IA di visione locale (ocr_vlm.py) ----
    ap.add_argument("--pages", default=None,
                    help="Motori vlm_*: pagine da convertire, es. '1-10,15,379-383' "
                         "(default: tutte).")
    ap.add_argument("--hq-pages", default=None,
                    help="vlm_ibrido: pagine da fare con Chandra (scrittura a mano, "
                         "figure), es. '379-453'. Le altre vanno a PaddleOCR-VL.")
    ap.add_argument("--vlm-threads", type=int, default=None,
                    help="Thread CPU per llama.cpp (default: numero di core fisici).")
    ap.add_argument("--vlm-gpu", action="store_true",
                    help="Prova la grafica integrata via Vulkan (sperimentale).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Sovrascrive i .md gia' esistenti.")
    ap.add_argument("--media-dir", default="media",
                    help="Nome della cartella che raccoglie le immagini di TUTTI "
                         "i file (default: 'media'). Struttura: <cartella>/<media-dir>/<nome>/.")
    # opzioni page-image (default ON: ci servono per i diagrammi vettoriali)
    ap.add_argument("--page-images", dest="page_images", action="store_true", default=True,
                    help="Solo PDF: renderizza le pagine con grafica vettoriale come PNG "
                         "(garanzia diagrammi). Attivo di default.")
    ap.add_argument("--no-page-images", dest="page_images", action="store_false",
                    help="Disattiva il rendering pagina-immagine.")
    ap.add_argument("--lean-md", action="store_true",
                    help="MD 'snello': link compatti invece di <details><summary>. "
                         "Riduce boilerplate per LLM (suggerito da Gemini).")
    ap.add_argument("--page-image-dpi", type=int, default=150,
                    help="DPI per le immagini estratte/pagine (default 150).")
    ap.add_argument("--vector-threshold", type=int, default=3,
                    help="Soglia drawings vettoriali per considerare una pagina "
                         "'con diagrammi' (default 3).")
    # opzioni marker (solo --engine marker)
    ap.add_argument("--force-ocr", action="store_true",
                    help="Marker: forza l'OCR (per PDF scansionati).")
    ap.add_argument("--block-size", type=int, default=0,
                    help="Marker: spezza il PDF in blocchi di N pagine (consigliato 15-20 "
                         "con 8 GB RAM). 0 = file intero.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Mostra solo cosa verrebbe convertito.")
    args = ap.parse_args()

    # ---- separa URL dai file ----
    urls = [u for u in args.inputs if u.startswith(("http://", "https://"))]
    local_tokens = [u for u in args.inputs if not u.startswith(("http://", "https://"))]

    files, missing = resolve_inputs(local_tokens, args.root, args.recursive)
    for m in missing:
        print(f"[!] non trovato: {m}", file=sys.stderr)
    files = [f for f in files if f.suffix.lower() in ALL_SUPPORTED
             or f.name.lower() in ("dockerfile", "makefile", "cmakelists.txt",
                                    "rakefile", "gemfile")]

    # ---- scarica gli URL come file locali e aggiungili a 'files' ----
    if urls:
        args.url_dest.mkdir(parents=True, exist_ok=True)
        for u in urls:
            try:
                print(f"Scarico {u} ...")
                _, _, out_md = convert_with_url(u, args.url_dest, args)
                # il file scaricato e' gia' stato convertito in .md;
                # lo segnalo come "fatto" creando l'entry .md
                print(f"  -> {out_md}")
            except Exception as e:
                print(f"  ERRORE download: {e}", file=sys.stderr)

    if not files and not urls:
        print("Nessun file convertibile trovato.", file=sys.stderr); sys.exit(1)
    if not files:
        return

    print(f"File da convertire: {len(files)}")
    for f in files:
        print(f"  - {f}  ->  {f.with_suffix('.md').name}   "
              f"[{choose_engine(f, args.engine)}]")
    if args.dry_run:
        return

    results = []
    # Keep-awake: previene la sospensione di Windows durante l'intero batch.
    # Lo schermo puo' spegnersi comunque (per risparmio energia).
    with _KeepAwake():
        for f in files:
            print(f"\n>>> {f.name}")
            results.append(convert_one(f, args))

    print("\n=================  RIEPILOGO  =================")
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
