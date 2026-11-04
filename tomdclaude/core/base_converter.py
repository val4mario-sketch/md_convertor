"""Contratto astratto di un convertitore.

Ogni convertitore concreto e' una classe indipendente che eredita da
``BaseConverter`` e dichiara i formati di input/output supportati. La pipeline
(``core/pipeline.py``) si occupa di file temporaneo, validazione e commit: il
convertitore deve solo *produrre* l'output e dichiarare come estrarne i metadati.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


def norm_ext(ext: str) -> str:
    """Normalizza un'estensione: minuscolo, senza punto iniziale."""
    return (ext or "").lower().lstrip(".")


class BaseConverter(ABC):
    # --- da sovrascrivere nelle sottoclassi ---
    name: str = "base"
    source_formats: frozenset[str] = frozenset()
    target_formats: frozenset[str] = frozenset()

    # --- introspezione ---
    @classmethod
    def supports(cls, src_ext: str, dst_ext: str) -> bool:
        return norm_ext(src_ext) in cls.source_formats and norm_ext(dst_ext) in cls.target_formats

    # --- metadati per la validazione (gatekeeper) ---
    def extract_source_metadata(self, src: Path, opts: dict | None = None) -> dict:
        """Metadati del file di partenza (es. albero dei titoli, righe/colonne).

        Riceve ``opts`` perche' deve applicare lo STESSO pre-processing della
        conversione (es. pulizia dei marcatori), altrimenti il confronto con il
        prodotto sarebbe incoerente."""
        return {}

    def extract_target_metadata(self, dst: Path) -> dict:
        """Metadati del file prodotto (es. n. pagine PDF, righe/colonne)."""
        return {}

    def validators(self) -> list:
        """Lista di validatori applicati a questa conversione (gatekeeper)."""
        return []

    # --- conversione vera e propria ---
    @abstractmethod
    def convert(self, src: Path, dst: Path, opts: dict) -> dict:
        """Esegue la conversione scrivendo su ``dst``.

        ATTENZIONE: ``dst`` e' un file TEMPORANEO scelto dalla pipeline; non e'
        ancora la destinazione definitiva. Questo metodo puo' sollevare
        eccezioni: la pipeline le cattura, scarta il temporaneo e logga.

        Ritorna un dict opzionale::

            {"note": "<descrizione breve>", "metadata": {<extra per i validatori>}}
        """
        raise NotImplementedError
