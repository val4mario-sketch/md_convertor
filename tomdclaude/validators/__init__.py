"""Validators = integrity gatekeeper.

A validator compares the source metadata with that of the product and
returns a list of ``ValidationIssue``. If even one finding is of level
"error", the pipeline rejects the conversion and does not save the file.
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
