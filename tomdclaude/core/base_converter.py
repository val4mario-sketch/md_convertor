"""Abstract contract of a converter.

Each concrete converter is an independent class inheriting from
``BaseConverter`` and declaring the supported input/output formats. The pipeline
(``core/pipeline.py``) takes care of the temporary file, validation and commit: the
converter only has to *produce* the output and declare how to extract its metadata.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


def norm_ext(ext: str) -> str:
    """Normalizes an extension: lowercase, no leading dot."""
    return (ext or "").lower().lstrip(".")


class BaseConverter(ABC):
    # --- to be overridden in subclasses ---
    name: str = "base"
    source_formats: frozenset[str] = frozenset()
    target_formats: frozenset[str] = frozenset()

    # --- introspezione ---
    @classmethod
    def supports(cls, src_ext: str, dst_ext: str) -> bool:
        return norm_ext(src_ext) in cls.source_formats and norm_ext(dst_ext) in cls.target_formats

    # --- metadata for validation (gatekeeper) ---
    def extract_source_metadata(self, src: Path, opts: dict | None = None) -> dict:
        """Metadata of the starting file (e.g. heading tree, rows/columns).

        Receives ``opts`` because it must apply the SAME pre-processing as the
        conversion (e.g. marker clean-up), otherwise the comparison with the
        product would be inconsistent."""
        return {}

    def extract_target_metadata(self, dst: Path) -> dict:
        """Metadata of the produced file (e.g. number of PDF pages, rows/columns)."""
        return {}

    def validators(self) -> list:
        """List of validators applied to this conversion (gatekeeper)."""
        return []

    # --- the actual conversion ---
    @abstractmethod
    def convert(self, src: Path, dst: Path, opts: dict) -> dict:
        """Runs the conversion, writing to ``dst``.

        WARNING: ``dst`` is a TEMPORARY file chosen by the pipeline; it is not
        yet the final destination. This method may raise
        exceptions: the pipeline catches them, discards the temporary file and logs.

        Returns an optional dict::

            {"note": "<short description>", "metadata": {<extra for the validators>}}
        """
        raise NotImplementedError
