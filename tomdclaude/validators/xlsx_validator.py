"""Tabular structure validator (XLSX).

Compares rows x columns of each sheet between source and product. Intended for
conversions involving XLSX (e.g. xlsx->xlsx, csv->xlsx): if a sheet
loses rows or columns, the conversion is rejected.

NB: at this stage it is not yet connected to a converter (the first
converter implemented is MD->PDF), but it is part of the Validator module as per the
"row/column comparison for XLSX" specification. Ready to use as soon as a data
converter is added.

It expects the ``dimensions`` key in the metadata, in the format produced by
``utils.metadata.xlsx_dimensions``::

    {"Sheet1": {"rows": 100, "cols": 8}, ...}
"""
from __future__ import annotations

from ..core.result import ValidationIssue
from .base import Validator


class XlsxStructureValidator(Validator):
    name = "xlsx_structure"

    def validate(self, src_meta, dst_meta, ctx) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        src_dim = src_meta.get("dimensions") or {}
        dst_dim = dst_meta.get("dimensions") or {}

        if not src_dim or not dst_dim:
            return issues  # nothing to compare

        # Sheets missing from the product.
        missing_sheets = [s for s in src_dim if s not in dst_dim]
        if missing_sheets:
            issues.append(self.error(
                "sheets_missing", "Sheets lost in the conversion.",
                expected=list(src_dim), actual=list(dst_dim)))

        # For each common sheet: rows/columns must not decrease.
        for sheet, sd in src_dim.items():
            dd = dst_dim.get(sheet)
            if not dd:
                continue
            if dd.get("rows", 0) < sd.get("rows", 0):
                issues.append(self.error(
                    "rows_lost", f"Sheet '{sheet}': rows lost.",
                    expected=sd.get("rows"), actual=dd.get("rows")))
            if dd.get("cols", 0) < sd.get("cols", 0):
                issues.append(self.error(
                    "cols_lost", f"Sheet '{sheet}': columns lost.",
                    expected=sd.get("cols"), actual=dd.get("cols")))
        return issues
