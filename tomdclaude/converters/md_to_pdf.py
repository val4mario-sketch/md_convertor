"""MarkdownToPdfConverter — Markdown -> HTML (python-markdown) -> PDF (WeasyPrint).

Obiettivo: NON una stampa grezza del codice Markdown, ma un documento formattato
in stile report/articolo: copertina, indice automatico con numeri di pagina,
numerazione dei capitoli, tipografia leggibile, tabelle pulite, syntax
highlighting del codice e **matematica LaTeX renderizzata** (via SVG).

Pipeline interna:
    testo .md
      -> markdown.Markdown(extensions=[extra, toc, codehilite, arithmatex, ...])
      -> body HTML (le formule sono marcate da arithmatex)
      -> sostituzione delle formule con SVG (MathRenderer)
      -> documento HTML completo: copertina + indice + contenuto
      -> WeasyPrint  -> PDF

Metadati esposti al gatekeeper:
    sorgente: headings, n_tables, text_len
    prodotto: html_headings, n_html_tables, pages, pdf_text, pdf_text_len,
              n_math / n_math_failed / math_failures
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import re as _re
import shutil
import tempfile
from pathlib import Path

import markdown
from bs4 import BeautifulSoup

from ..core.base_converter import BaseConverter
from ..utils.gtk import ensure_gtk, gtk_hint
from ..utils.metadata import (
    count_md_tables,
    html_heading_tree,
    markdown_heading_tree,
    md_plain_length,
    pdf_info,
)
from ..validators.math_validator import MathRenderValidator
from ..validators.md_validator import MarkdownStructureValidator
from ..validators.pdf_validator import PdfIntegrityValidator
from .math_render import MathRenderer, MathStats

_ASSETS = Path(__file__).resolve().parent.parent / "assets"

# Marcatori di citazione lasciati dagli export LLM (es. NotebookLM/Gemini):
# "[cite_start]" e "[cite: 1, 2]". Rumore senza valore: rimossi di default.
_CITE_RE = _re.compile(r"\[cite_start\]\s*|\s*\[cite:[^\]]*\]")

# Normalizzazione della matematica display: gli export LLM mettono spesso i
# "$$...$$" indentati dentro un elenco o in mezzo a una frase, dove arithmatex
# li gestisce male (matcha solo il "$...$" interno). Proteggiamo i blocchi di
# codice e poi: $$ isolato su una riga -> blocco display; $$ inline -> $...$.
_FENCE_SPLIT = _re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]*`)", _re.DOTALL)
_DISPLAY_MATH = _re.compile(r"\$\$(.+?)\$\$", _re.DOTALL)


def _normalize_display_math(text: str) -> str:
    parts = _FENCE_SPLIT.split(text)
    for i in range(0, len(parts), 2):  # 0,2,4... = segmenti NON di codice
        parts[i] = _normalize_segment(parts[i])
    return "".join(parts)


def _normalize_segment(seg: str) -> str:
    out, pos = [], 0
    for m in _DISPLAY_MATH.finditer(seg):
        out.append(seg[pos:m.start()])
        content = m.group(1).strip()
        before_line = seg[:m.start()].split("\n")[-1]
        after_line = seg[m.end():].split("\n", 1)[0]
        isolated = before_line.strip() == "" and after_line.strip() == ""
        out.append(f"\n\n$${content}$$\n\n" if isolated else f"${content}$")
        pos = m.end()
    out.append(seg[pos:])
    return "".join(out)

_MD_EXTENSIONS = [
    "extra",                 # tabelle, fenced_code, footnotes, attr_list, def_list, abbr
    "toc",                   # indice automatico (md.toc)
    "codehilite",            # syntax highlighting via Pygments
    "admonition",            # riquadri note/avvisi
    "sane_lists",
    "smarty",                # tipografia: virgolette curve, trattini
    "meta",                  # front-matter: title:, author:
    "pymdownx.arithmatex",   # isola $...$ e $$...$$ senza toccare il codice
]
_MD_CONFIG = {
    "codehilite": {"guess_lang": False, "css_class": "codehilite"},
    "toc": {"permalink": False, "toc_depth": "1-3"},
    "pymdownx.arithmatex": {"generic": True},
}


class MarkdownToPdfConverter(BaseConverter):
    name = "md_to_pdf"
    source_formats = frozenset({"md", "markdown"})
    target_formats = frozenset({"pdf"})

    # ------------------------------------------------------------------ metadati
    def extract_source_metadata(self, src: Path, opts: dict | None = None) -> dict:
        # Stesso pre-processing della conversione, per un confronto coerente.
        text = self._preprocess(Path(src).read_text(encoding="utf-8", errors="replace"),
                                opts or {})
        return {
            "headings": markdown_heading_tree(text),
            "n_tables": count_md_tables(text),
            "text_len": md_plain_length(text),
        }

    @staticmethod
    def _preprocess(text: str, opts: dict) -> str:
        """Trasformazioni sul Markdown grezzo applicate PRIMA di tutto il resto."""
        if opts.get("strip_cite", True):
            text = _CITE_RE.sub("", text)
        if opts.get("math", True):
            text = _normalize_display_math(text)
        return text

    def extract_target_metadata(self, dst: Path) -> dict:
        info = pdf_info(dst)
        return {
            "pages": info["pages"],
            "pdf_text": info["text"],
            "pdf_text_len": info["text_len"],
        }

    def validators(self) -> list:
        return [MarkdownStructureValidator(), PdfIntegrityValidator(), MathRenderValidator()]

    # ----------------------------------------------------------------- conversione
    def convert(self, src: Path, dst: Path, opts: dict) -> dict:
        src = Path(src)
        text = self._preprocess(src.read_text(encoding="utf-8", errors="replace"), opts)

        md = markdown.Markdown(extensions=_MD_EXTENSIONS,
                               extension_configs=_MD_CONFIG,
                               output_format="html5")
        body_html = md.convert(text)
        toc_html = getattr(md, "toc", "") or ""
        meta = getattr(md, "Meta", {}) or {}

        title = (opts.get("title")
                 or _meta_first(meta, "title")
                 or _first_md_heading(text)
                 or src.stem)
        author = opts.get("author") or _meta_first(meta, "author") or ""

        # --- rendering della matematica (puo' essere disattivato con opts['math']=False) ---
        math_stats = MathStats()
        math_workdir = None
        if opts.get("math", True):
            math_workdir = Path(tempfile.mkdtemp(prefix="tomdclaude_math_"))
            body_html, math_stats = self._render_math(body_html, math_workdir)

        # metadati strutturali DOPO la sostituzione delle formule
        html_headings = html_heading_tree(body_html)
        n_html_tables = body_html.count("<table")

        document = self._build_html(title, author, toc_html, body_html)
        try:
            self._render_pdf(document, dst, base_url=src.parent)
        finally:
            if math_workdir:
                shutil.rmtree(math_workdir, ignore_errors=True)

        md_meta = {
            "html_headings": html_headings,
            "n_html_tables": n_html_tables,
            "title": title,
            "n_math": math_stats.total,
            "n_math_ok": math_stats.ok,
            "n_math_failed": math_stats.failed,
            "math_failures": list(math_stats.failures[:20]),
        }
        note = f"PDF in stile report via WeasyPrint (titolo: {title!r})"
        if math_stats.total:
            note += f"; matematica {math_stats.ok}/{math_stats.total}"
        return {"note": note, "metadata": md_meta}

    # ------------------------------------------------------------------- interni
    def _render_math(self, body_html: str, workdir: Path) -> tuple[str, MathStats]:
        soup = BeautifulSoup(body_html, "html.parser")
        nodes = soup.find_all(class_="arithmatex")
        if not nodes:
            return body_html, MathStats()
        renderer = MathRenderer(workdir, fontsize=11.0, color="#1f2328")
        for node in nodes:
            display = node.name == "div"
            tex = _unwrap_math(node.get_text())
            frag = renderer.render(tex, display)
            repl = BeautifulSoup(frag, "html.parser")
            new_el = repl.find(True)
            if new_el is not None:
                node.replace_with(new_el)
            else:
                node.replace_with(repl.get_text())
        return str(soup), renderer.stats

    def _build_html(self, title: str, author: str, toc_html: str, body_html: str) -> str:
        css = (_ASSETS / "report.css").read_text(encoding="utf-8")
        pyg_css = _pygments_css()
        date = _dt.date.today().strftime("%d/%m/%Y")

        toc_section = ""
        if toc_html.strip():
            toc_section = f'<nav class="toc"><h2>Indice</h2>\n{toc_html}\n</nav>'

        author_html = f'<p class="doc-author">{_html.escape(author)}</p>' if author else ""

        return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>{_html.escape(title)}</title>
<style>
{css}
{pyg_css}
</style>
</head>
<body>
<header class="cover">
  <div class="cover-rule"></div>
  <h1 class="doc-title">{_html.escape(title)}</h1>
  {author_html}
  <p class="doc-date">{date}</p>
  <div class="cover-rule"></div>
</header>
{toc_section}
<main class="content">
{body_html}
</main>
</body>
</html>"""

    def _render_pdf(self, document_html: str, dst: Path, base_url: Path) -> None:
        if ensure_gtk() is None:
            import sys
            if sys.platform == "win32":
                raise RuntimeError(gtk_hint())
        from weasyprint import HTML  # import tardivo: dopo ensure_gtk()

        HTML(string=document_html, base_url=str(base_url)).write_pdf(str(dst))


# --------------------------------------------------------------------- helper modulo
def _pygments_css() -> str:
    try:
        from pygments.formatters import HtmlFormatter
        return HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    except Exception:
        return ""


def _unwrap_math(s: str) -> str:
    s = s.strip()
    for op, cl in (("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$"), ("$", "$")):
        if s.startswith(op) and s.endswith(cl) and len(s) >= len(op) + len(cl):
            return s[len(op):-len(cl)].strip()
    return s


def _meta_first(meta: dict, key: str):
    val = meta.get(key)
    if isinstance(val, (list, tuple)) and val:
        return val[0]
    return val or None


def _first_md_heading(text: str):
    tree = markdown_heading_tree(text)
    return tree[0][1] if tree else None
