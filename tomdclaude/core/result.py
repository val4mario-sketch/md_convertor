"""Tipi di ritorno condivisi: ValidationIssue e ConversionResult.

Sono dataclass semplici cosi' che CLI, GUI e test possano ispezionare l'esito di
una conversione in modo uniforme (niente eccezioni per i casi prevedibili).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ValidationIssue:
    """Un singolo rilievo prodotto da un validatore.

    level: "error"  -> BLOCCANTE: la conversione viene rifiutata, il file non
                       viene salvato.
           "warning" -> non bloccante: il file viene salvato ma il problema
                       viene segnalato/loggato.
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
    """Esito completo di una conversione, restituito dalla pipeline."""

    ok: bool
    src: Path
    dst: Path
    converter: str = ""
    note: str = ""
    elapsed: float = 0.0
    committed: bool = False  # True solo se il file finale e' stato effettivamente scritto
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
