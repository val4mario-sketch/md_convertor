"""Math rendering validator.

Checks how many formulas were not rendered. By default it is a warning
(not blocking): a single exotic formula must not stop a document of
dozens of pages from being saved; failed formulas stay visible
in the PDF as highlighted raw text and go into the log.

With ``opts['strict_math'] = True`` the failure becomes blocking (useful for
documents in which every formula must be perfect).
"""
from __future__ import annotations

from ..core.result import ValidationIssue
from .base import Validator


class MathRenderValidator(Validator):
    name = "math_render"

    def validate(self, src_meta, dst_meta, ctx) -> list[ValidationIssue]:
        failed = int(dst_meta.get("n_math_failed", 0) or 0)
        if not failed:
            return []
        fails = dst_meta.get("math_failures", [])
        strict = bool((ctx.get("opts") or {}).get("strict_math", False))
        make = self.error if strict else self.warn
        total = dst_meta.get("n_math", "?")
        return [make(
            "math_render_failed",
            f"{failed}/{total} formulas not rendered (shown as raw text).",
            expected=0, actual=fails)]
