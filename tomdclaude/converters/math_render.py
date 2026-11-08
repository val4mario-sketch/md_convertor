"""Rendering della matematica LaTeX senza una toolchain TeX.

Strategia: ogni espressione ``$...$`` / ``$$...$$`` viene resa con il motore
*mathtext* di matplotlib (puro Python, offline) in un file **SVG vettoriale**,
poi inserita nell'HTML. Con ``mathtext.fontset = 'cm'`` (Computer Modern) la
resa è praticamente identica a LaTeX.

Limiti di mathtext: niente ambienti ``\\begin{matrix|align|cases}`` (non presenti
nei documenti target). Le espressioni non renderizzabili NON fanno fallire tutto:
vengono mostrate come testo grezzo evidenziato e registrate per il validatore.

Allineamento inline: matplotlib fornisce la *depth* (parte sotto la linea di
base); l'immagine viene quindi spostata con ``vertical-align: -depth`` cosicché
la baseline della formula coincida con quella del testo.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"   # look "LaTeX" (Computer Modern)
matplotlib.rcParams["mathtext.rm"] = "serif"

from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.mathtext import MathTextParser, math_to_image  # noqa: E402

_AGG = MathTextParser("agg")

# Citation markers (noise from LLM exports) that ended up inside formulas.
_CITE = re.compile(r"\[cite_start\]|\[cite:[^\]]*\]")
# Commands not supported by mathtext that we can neutralize without loss:
# styles (\displaystyle...) and sized delimiters (\big \Big \bigg \Bigg [l|r|m]).
_STRIP = re.compile(
    r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle|limits|nolimits)\b"
    r"|\\(?:bigg|Bigg|big|Big)[lrm]?\b"
)
# 1:1 substitutions to the syntax mathtext knows.
_REPLACE = (
    (re.compile(r"\\dfrac\b"), r"\\frac"),
    (re.compile(r"\\tfrac\b"), r"\\frac"),
    (re.compile(r"\\mbox\b"), r"\\text"),
    (re.compile(r"\\operatorname\b"), r"\\mathrm"),
    (re.compile(r"\\le(?![a-zA-Z])"), r"\\leq"),   # \le -> \leq (mathtext has no \le)
    (re.compile(r"\\ge(?![a-zA-Z])"), r"\\geq"),
    (re.compile(r"\\ne(?![a-zA-Z])"), r"\\neq"),
    (re.compile(r"\\implies\b"), r"\\Rightarrow"),
    (re.compile(r"\\iff\b"), r"\\Leftrightarrow"),
)


@dataclass
class MathStats:
    total: int = 0
    ok: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)


class MathRenderer:
    """Renders the formulas as SVGs inside ``workdir`` and keeps statistics."""

    def __init__(self, workdir: Path, fontsize: float = 11.0, color: str = "#1f2328",
                 display_scale: float = 1.45, dpi: int = 300):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.fontsize = fontsize
        self.color = color
        self.display_scale = display_scale
        self.dpi = dpi
        self.stats = MathStats()
        self._cache: dict[tuple[str, bool], str] = {}

    # ------------------------------------------------------------------ pubblico
    def render(self, tex: str, display: bool) -> str:
        """Returns the HTML fragment (img/span) for a formula. Never raises."""
        tex = tex.strip()
        if not tex:
            return ""
        self.stats.total += 1
        key = (tex, display)
        if key in self._cache:
            self.stats.ok += 1  # already rendered successfully earlier
            return self._cache[key]

        try:
            html = self._render_one(tex, display)
            self.stats.ok += 1
            self._cache[key] = html
            return html
        except Exception as e:  # noqa: BLE001
            self.stats.failed += 1
            if tex not in self.stats.failures:
                self.stats.failures.append(tex)
            return self._error_html(tex, display, e)

    # -------------------------------------------------------------------- interni
    def _render_one(self, tex: str, display: bool) -> str:
        clean = self._sanitize(tex)
        size = self.fontsize * (self.display_scale if display else 1.0)
        prop = FontProperties(size=size)
        wrapped = f"${clean}$"

        # 1) metrics for the baseline (px at the chosen dpi)
        ox, oy, width, height, depth, _img = _AGG.parse(wrapped, dpi=self.dpi, prop=prop)
        to_pt = 72.0 / self.dpi
        w_pt, h_pt, d_pt = width * to_pt, height * to_pt, depth * to_pt

        # 2) vector SVG to a file (deduplicated by content)
        h = hashlib.md5(f"{size}|{display}|{clean}".encode("utf-8")).hexdigest()[:16]
        svg_path = self.workdir / f"m_{h}.svg"
        if not svg_path.exists():
            math_to_image(wrapped, str(svg_path), prop=prop, dpi=self.dpi,
                          format="svg", color=self.color)
        uri = svg_path.as_uri()

        if display:
            return (f'<span class="math-block-wrap">'
                    f'<img class="math-block" src="{uri}" '
                    f'style="width:{w_pt:.2f}pt;height:{h_pt:.2f}pt" alt="formula"></span>')
        # inline: baseline allineata via vertical-align negativo = depth
        return (f'<img class="math-inline" src="{uri}" '
                f'style="width:{w_pt:.2f}pt;height:{h_pt:.2f}pt;'
                f'vertical-align:-{d_pt:.2f}pt" alt="formula">')

    @staticmethod
    def _sanitize(tex: str) -> str:
        tex = _CITE.sub("", tex)
        tex = _STRIP.sub("", tex)
        for pat, repl in _REPLACE:
            tex = pat.sub(repl, tex)
        return tex.strip()

    @staticmethod
    def _error_html(tex: str, display: bool, err: Exception) -> str:
        import html as _html
        safe = _html.escape(tex)
        title = _html.escape(f"Formula cannot be rendered: {type(err).__name__}")
        cls = "math-error math-error-block" if display else "math-error"
        delim = "$$" if display else "$"
        return f'<span class="{cls}" title="{title}">{delim}{safe}{delim}</span>'
