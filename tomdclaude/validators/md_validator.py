"""Validatore della struttura gerarchica Markdown.

Confronta i titoli del Markdown sorgente con quelli effettivamente renderizzati
e verifica che nessun titolo sia stato perso nel PDF finale.

Rilievi:
  - heading_structure_mismatch (ERROR): la sequenza dei LIVELLI dei titoli
    sorgente non coincide con quella renderizzata -> un titolo e' stato
    aggiunto/perso o la gerarchia e' cambiata.
  - heading_missing_in_pdf  (ERROR): il testo di un titolo non compare nel
    testo estraibile del PDF -> titolo perso in fase di render.
  - heading_text_changed    (WARNING): il testo di un titolo e' cambiato (es.
    per la tipografia "smart"); la gerarchia e' intatta.
  - table_count_lower       (WARNING): meno tabelle nel rendering che nel sorgente.
"""
from __future__ import annotations

from ..core.result import ValidationIssue
from ..utils.metadata import normalize_text
from .base import Validator


class MarkdownStructureValidator(Validator):
    name = "md_structure"

    def validate(self, src_meta, dst_meta, ctx) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        src_h = src_meta.get("headings", []) or []          # [(livello, testo)]
        dst_h = dst_meta.get("html_headings", []) or []      # [(livello, testo)]

        src_levels = [lvl for lvl, _ in src_h]
        dst_levels = [lvl for lvl, _ in dst_h]

        if src_levels != dst_levels:
            issues.append(self.error(
                "heading_structure_mismatch",
                "La struttura gerarchica dei titoli non e' stata preservata.",
                expected=src_levels, actual=dst_levels))
        else:
            # gerarchia ok: controlla eventuali variazioni di testo (non bloccanti)
            for (_, st), (_, dt) in zip(src_h, dst_h):
                if normalize_text(st) != normalize_text(dt):
                    issues.append(self.warn(
                        "heading_text_changed",
                        "Testo di un titolo modificato dal rendering.",
                        expected=st, actual=dt))

        # Verifica sul PRODOTTO FINALE: ogni titolo deve comparire nel testo PDF.
        pdf_text = normalize_text(dst_meta.get("pdf_text", "") or dst_meta.get("text", ""))
        if pdf_text:
            missing = [st for _, st in src_h
                       if normalize_text(st) and normalize_text(st) not in pdf_text]
            if missing:
                issues.append(self.error(
                    "heading_missing_in_pdf",
                    "Uno o piu' titoli non compaiono nel PDF prodotto.",
                    actual=missing))

        # Tabelle: segnalazione non bloccante (il riconoscimento MD e' euristico).
        ns = src_meta.get("n_tables", 0)
        nd = dst_meta.get("n_html_tables", 0)
        if ns and nd < ns:
            issues.append(self.warn(
                "table_count_lower",
                "Renderizzate meno tabelle di quante presenti nel sorgente.",
                expected=ns, actual=nd))

        return issues
