"""Central registry of the available converters.

To add a new converter, just import it and put it in ``_CONVERTERS``.
The order only matters when formats overlap (the first wins).
"""
from __future__ import annotations

from .base_converter import BaseConverter, norm_ext
from ..converters.md_to_pdf import MarkdownToPdfConverter

_CONVERTERS: list[type[BaseConverter]] = [
    MarkdownToPdfConverter,
]


def find_converter(src_ext: str, dst_ext: str) -> BaseConverter | None:
    for cls in _CONVERTERS:
        if cls.supports(src_ext, dst_ext):
            return cls()
    return None


def all_converters() -> list[type[BaseConverter]]:
    return list(_CONVERTERS)


def targets_for(src_ext: str) -> list[str]:
    """Target formats reachable from a given source extension."""
    s = norm_ext(src_ext)
    out: set[str] = set()
    for cls in _CONVERTERS:
        if s in cls.source_formats:
            out |= set(cls.target_formats)
    return sorted(out)
