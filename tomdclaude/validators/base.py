"""Abstract contract of the validators."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.result import ValidationIssue


class Validator(ABC):
    """A gatekeeper. ``validate`` compares source and product metadata."""

    name: str = "validator"

    @abstractmethod
    def validate(self, src_meta: dict, dst_meta: dict, ctx: dict) -> list[ValidationIssue]:
        """Returns the list of findings. Empty list = all ok.

        ``ctx`` contains at least: src (Path), tmp (Path of the produced file,
        still temporary), dst (destination Path), opts (dict).
        """
        raise NotImplementedError

    # small helpers to build the findings in a readable way
    @staticmethod
    def error(code, message, expected=None, actual=None) -> ValidationIssue:
        return ValidationIssue("error", code, message, expected, actual)

    @staticmethod
    def warn(code, message, expected=None, actual=None) -> ValidationIssue:
        return ValidationIssue("warning", code, message, expected, actual)
