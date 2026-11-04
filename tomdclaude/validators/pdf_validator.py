"""Validatore di integrita' del PDF prodotto.

Rilievi:
  - not_a_pdf      (ERROR): il file non inizia con l'header %PDF.
  - no_pages       (ERROR): zero pagine.
  - text_too_short (ERROR): il testo estraibile e' sproporzionatamente piu'
    corto del sorgente -> probabile render fallito / pagina bianca.
"""
from __future__ import annotations

from pathlib import Path

from ..core.result import ValidationIssue
from .base import Validator

# Frazione minima del testo sorgente che ci si aspetta di ritrovare nel PDF.
_MIN_TEXT_RATIO = 0.30
_ABS_MIN_CHARS = 20


class PdfIntegrityValidator(Validator):
    name = "pdf_integrity"

    def validate(self, src_meta, dst_meta, ctx) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        tmp = Path(ctx["tmp"])

        try:
            with open(tmp, "rb") as f:
                header = f.read(5)
        except OSError as e:
            return [self.error("pdf_unreadable", f"PDF non leggibile: {e}")]

        if not header.startswith(b"%PDF"):
            issues.append(self.error("not_a_pdf", "Il file prodotto non e' un PDF valido.",
                                     expected="%PDF", actual=header))

        pages = int(dst_meta.get("pages", 0) or 0)
        if pages < 1:
            issues.append(self.error("no_pages", "Il PDF non contiene pagine."))

        src_len = int(src_meta.get("text_len", 0) or 0)
        pdf_len = int(dst_meta.get("pdf_text_len", dst_meta.get("text_len", 0)) or 0)
        if src_len > 0:
            soglia = max(_ABS_MIN_CHARS, int(_MIN_TEXT_RATIO * src_len))
            if pdf_len < soglia:
                issues.append(self.error(
                    "text_too_short",
                    "Il testo estraibile dal PDF e' troppo corto: render probabilmente fallito.",
                    expected=f">= {soglia} caratteri", actual=pdf_len))

        return issues
