"""Validatore del rendering matematico.

Verifica quante formule non sono state renderizzate. Di default e' un avviso
(non bloccante): una singola formula esotica non deve impedire il salvataggio di
un documento di decine di pagine; le formule fallite restano comunque visibili
nel PDF come testo grezzo evidenziato e finiscono nel log.

Con ``opts['strict_math'] = True`` il fallimento diventa bloccante (utile per i
documenti in cui ogni formula deve essere perfetta).
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
            f"{failed}/{total} formule non renderizzate (mostrate come testo grezzo).",
            expected=0, actual=fails)]
