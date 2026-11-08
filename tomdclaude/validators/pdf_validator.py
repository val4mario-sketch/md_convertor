"""Integrity validator of the produced PDF.

Findings:
  - not_a_pdf      (ERROR): the file does not begin with the %PDF header.
  - no_pages       (ERROR): zero pages.
  - text_too_short (ERROR): the extractable text is disproportionately shorter
    than the source -> probably a failed render / blank page.
"""
from __future__ import annotations

from pathlib import Path

from ..core.result import ValidationIssue
from .base import Validator

# Minimum fraction of the source text expected to be found in the PDF.
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
            return [self.error("pdf_unreadable", f"PDF not readable: {e}")]

        if not header.startswith(b"%PDF"):
            issues.append(self.error("not_a_pdf", "The produced file is not a valid PDF.",
                                     expected="%PDF", actual=header))

        pages = int(dst_meta.get("pages", 0) or 0)
        if pages < 1:
            issues.append(self.error("no_pages", "The PDF contains no pages."))

        src_len = int(src_meta.get("text_len", 0) or 0)
        pdf_len = int(dst_meta.get("pdf_text_len", dst_meta.get("text_len", 0)) or 0)
        if src_len > 0:
            soglia = max(_ABS_MIN_CHARS, int(_MIN_TEXT_RATIO * src_len))
            if pdf_len < soglia:
                issues.append(self.error(
                    "text_too_short",
                    "The extractable text of the PDF is too short: render probably failed.",
                    expected=f">= {soglia} caratteri", actual=pdf_len))

        return issues
