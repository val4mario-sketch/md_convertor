"""Validatore della struttura tabellare (XLSX).

Confronta righe x colonne di ogni foglio fra sorgente e prodotto. Pensato per
le conversioni che coinvolgono XLSX (es. xlsx->xlsx, csv->xlsx): se un foglio
perde righe o colonne, la conversione viene rifiutata.

NB: nello step 1 non e' ancora collegato a un convertitore (il primo
convertitore implementato e' MD->PDF), ma fa parte del modulo Validator come da
specifica "confronto righe/colonne per XLSX". Pronto all'uso appena verra'
aggiunto un convertitore di dati.

Si aspetta nei metadati la chiave ``dimensions`` nel formato prodotto da
``utils.metadata.xlsx_dimensions``::

    {"Foglio1": {"rows": 100, "cols": 8}, ...}
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
            return issues  # niente da confrontare

        # Fogli mancanti nel prodotto.
        missing_sheets = [s for s in src_dim if s not in dst_dim]
        if missing_sheets:
            issues.append(self.error(
                "sheets_missing", "Fogli persi nella conversione.",
                expected=list(src_dim), actual=list(dst_dim)))

        # Per ogni foglio comune: righe/colonne non devono diminuire.
        for sheet, sd in src_dim.items():
            dd = dst_dim.get(sheet)
            if not dd:
                continue
            if dd.get("rows", 0) < sd.get("rows", 0):
                issues.append(self.error(
                    "rows_lost", f"Foglio '{sheet}': righe perse.",
                    expected=sd.get("rows"), actual=dd.get("rows")))
            if dd.get("cols", 0) < sd.get("cols", 0):
                issues.append(self.error(
                    "cols_lost", f"Foglio '{sheet}': colonne perse.",
                    expected=sd.get("cols"), actual=dd.get("cols")))
        return issues
