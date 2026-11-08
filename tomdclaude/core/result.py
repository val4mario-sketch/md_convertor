"""Shared return types: ValidationIssue and ConversionResult.

They are simple dataclasses so CLI, GUI and tests can inspect the outcome of
a conversion in a uniform way (no exceptions for predictable cases).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ValidationIssue:
    """A single finding produced by a validator.

    level: "error"  -> BLOCKING: the conversion is rejected, the file is not
                       saved.
           "warning" -> not blocking: the file is saved but the problem
                       is flagged/logged.
    """

    level: str
    code: str
    message: str
    expected: Any = None
    actual: Any = None

    @property
    def blocking(self) -> bool:
        return self.level == "error"

    def __str__(self) -> str:
        base = f"[{self.level.upper()}] {self.code}: {self.message}"
        if self.expected is not None or self.actual is not None:
            base += f"  (atteso={self.expected!r}, ottenuto={self.actual!r})"
        return base


@dataclass
class ConversionResult:
    """Complete outcome of a conversion, returned by the pipeline."""

    ok: bool
    src: Path
    dst: Path
    converter: str = ""
    note: str = ""
    elapsed: float = 0.0
    committed: bool = False  # True only if the final file was actually written
    src_metadata: dict = field(default_factory=dict)
    dst_metadata: dict = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)
    log_path: Optional[Path] = None

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.blocking]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if not i.blocking]
