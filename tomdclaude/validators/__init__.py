"""Validatori = gatekeeper di integrita'.

Un validatore confronta i metadati del sorgente con quelli del prodotto e
ritorna una lista di ``ValidationIssue``. Se anche un solo rilievo e' di livello
"error", la pipeline rifiuta la conversione e non salva il file.
"""
from .base import Validator
from .math_validator import MathRenderValidator
from .md_validator import MarkdownStructureValidator
from .pdf_validator import PdfIntegrityValidator
from .xlsx_validator import XlsxStructureValidator

__all__ = [
    "Validator",
    "MarkdownStructureValidator",
    "PdfIntegrityValidator",
    "XlsxStructureValidator",
    "MathRenderValidator",
]
