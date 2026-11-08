"""MarkdownToPdfConverter — Markdown -> HTML (python-markdown) -> PDF (WeasyPrint).

Goal: NOT a raw printout of the Markdown code, but a formatted document
in report/article style: cover, automatic table of contents with page numbers,
chapter numbering, readable typography, clean tables, syntax
highlighting of code and **rendered LaTeX math** (via SVG).

Internal pipeline:
    .md text
      -> markdown.Markdown(extensions=[extra, toc, codehilite, arithmatex, ...])
      -> HTML body (formulas are marked by arithmatex)
      -> replacement of the formulas with SVGs (MathRenderer)
      -> complete HTML document: cover + contents + content
      -> WeasyPrint  -> PDF

Metadata exposed to the gatekeeper:
    source: headings, n_tables, text_len
    product: html_headings, n_html_tables, pages, pdf_text, pdf_text_len,
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

# Citation markers left by LLM exports (e.g. NotebookLM/Gemini):
# "[cite_start]" and "[cite: 1, 2]". Worthless noise: removed by default.
_CITE_RE = _re.compile(r"\[cite_start\]\s*|\s*\[cite:[^\]]*\]")

# Normalizing display math: LLM exports often put the
# "$$...$$" indented inside a list or in the middle of a sentence, where arithmatex
# handles them badly (it matches only the inner "$...$"). We protect the code
# blocks and then: $$ alone on a line -> display block; inline $$ -> $...$.
_FENCE_SPLIT = _re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]*`)", _re.DOTALL)
_DISPLAY_MATH = _re.compile(r"\$\$(.+?)\$\$", _re.DOTALL)


def _normalize_display_math(text: str) -> str:
    parts = _FENCE_SPLIT.split(text)
    for i in range(0, len(parts), 2):  # 0,2,4... = NON-code segments
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
    "toc",                   # automatic table of contents (md.toc)
    "codehilite",            # syntax highlighting via Pygments
    "admonition",            # riquadri note/avvisi
    "sane_lists",
    "smarty",                # typography: curly quotes, dashes
    "meta",                  # front-matter: title:, author:
    "pymdownx.arithmatex",   # isolates $...$ and $$...$$ without touching the code
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
        # Same pre-processing as the conversion, for a consistent comparison.
        text = self._preprocess(Path(src).read_text(encoding="utf-8", errors="replace"),
                                opts or {})
        return {
            "headings": markdown_heading_tree(text),
            "n_tables": count_md_tables(text),
            "text_len": md_plain_length(text),
        }

    @staticmethod
    def _preprocess(text: str, opts: dict) -> str:
        """Transformations on the raw Markdown applied BEFORE everything else."""
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

        # --- math rendering (can be disabled with opts['math']=False) ---
        math_stats = MathStats()
        math_workdir = None
        if opts.get("math", True):
            math_workdir = Path(tempfile.mkdtemp(prefix="tomdclaude_math_"))
            body_html, math_stats = self._render_math(body_html, math_workdir)

        # structural metadata AFTER the formula replacement
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
        note = f"Report-style PDF via WeasyPrint (title: {title!r})"
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
        from weasyprint import HTML  # late import: after ensure_gtk()

        HTML(string=document_html, base_url=str(base_url)).write_pdf(str(dst))


# --------------------------------------------------------------------- module helper
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
