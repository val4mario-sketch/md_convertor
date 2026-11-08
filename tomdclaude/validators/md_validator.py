"""Markdown heading hierarchy validator.

Compares the headings of the source Markdown with those actually rendered
and checks that no heading was lost in the final PDF.

Findings:
  - heading_structure_mismatch (ERROR): the sequence of source heading LEVELS
    does not match the rendered one -> a heading was
    added/lost or the hierarchy changed.
  - heading_missing_in_pdf  (ERROR): the text of a heading does not appear in the
    extractable PDF text -> heading lost during rendering.
  - heading_text_changed    (WARNING): the text of a heading changed (e.g.
    through "smart" typography); the hierarchy is intact.
  - table_count_lower       (WARNING): fewer tables in the rendering than in the source.
"""
from __future__ import annotations

from ..core.result import ValidationIssue
from ..utils.metadata import normalize_text
from .base import Validator


class MarkdownStructureValidator(Validator):
    name = "md_structure"

    def validate(self, src_meta, dst_meta, ctx) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        src_h = src_meta.get("headings", []) or []          # [(level, text)]
        dst_h = dst_meta.get("html_headings", []) or []      # [(level, text)]

        src_levels = [lvl for lvl, _ in src_h]
        dst_levels = [lvl for lvl, _ in dst_h]

        if src_levels != dst_levels:
            issues.append(self.error(
                "heading_structure_mismatch",
                "The heading hierarchy was not preserved.",
                expected=src_levels, actual=dst_levels))
        else:
            # hierarchy ok: check any text variations (not blocking)
            for (_, st), (_, dt) in zip(src_h, dst_h):
                if normalize_text(st) != normalize_text(dt):
                    issues.append(self.warn(
                        "heading_text_changed",
                        "Heading text changed by the rendering.",
                        expected=st, actual=dt))

        # Check on the FINAL PRODUCT: each heading must appear in the PDF text.
        pdf_text = normalize_text(dst_meta.get("pdf_text", "") or dst_meta.get("text", ""))
        if pdf_text:
            missing = [st for _, st in src_h
                       if normalize_text(st) and normalize_text(st) not in pdf_text]
            if missing:
                issues.append(self.error(
                    "heading_missing_in_pdf",
                    "One or more headings do not appear in the produced PDF.",
                    actual=missing))

        # Tables: not blocking (MD recognition is heuristic).
        ns = src_meta.get("n_tables", 0)
        nd = dst_meta.get("n_html_tables", 0)
        if ns and nd < ns:
            issues.append(self.warn(
                "table_count_lower",
                "Fewer tables rendered than present in the source.",
                expected=ns, actual=nd))

        return issues
