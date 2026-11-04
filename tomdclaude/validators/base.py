"""Contratto astratto dei validatori."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.result import ValidationIssue


class Validator(ABC):
    """Un gatekeeper. ``validate`` confronta metadati sorgente vs prodotto."""

    name: str = "validator"

    @abstractmethod
    def validate(self, src_meta: dict, dst_meta: dict, ctx: dict) -> list[ValidationIssue]:
        """Ritorna la lista di rilievi. Lista vuota = tutto ok.

        ``ctx`` contiene almeno: src (Path), tmp (Path del file prodotto,
        ancora temporaneo), dst (Path destinazione), opts (dict).
        """
        raise NotImplementedError

    # piccoli helper per costruire i rilievi in modo leggibile
    @staticmethod
    def error(code, message, expected=None, actual=None) -> ValidationIssue:
        return ValidationIssue("error", code, message, expected, actual)

    @staticmethod
    def warn(code, message, expected=None, actual=None) -> ValidationIssue:
        return ValidationIssue("warning", code, message, expected, actual)
