"""
tomdclaude — Modular file conversion system with "gatekeeper" validation.

Architecture:
    core/        abstract contract, result, registry, pipeline (orchestration)
    converters/  one converter = one independent class (e.g. MarkdownToPdfConverter)
    validators/  integrity checks comparing source/target metadata
    utils/       logging, GTK runtime handling, metadata extraction

Key principle: a conversion is written to a TEMPORARY file and becomes
final (commit) ONLY if it passes all the validators. Otherwise the file
is not saved and a detailed error log is produced.
"""

__version__ = "0.1.0"

from .core.pipeline import convert  # noqa: E402  (convenient public API)

__all__ = ["convert", "__version__"]
