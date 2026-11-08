"""Resolves the GTK3 runtime, needed by WeasyPrint on Windows.

WeasyPrint uses Pango/cairo (libgobject-2.0-0.dll, libpango-1.0-0.dll, ...). On
Windows these DLLs are not on the default PATH: they must be added BEFORE
importing weasyprint, otherwise the import fails with
``cannot load library 'libgobject-2.0-0'``.

``ensure_gtk()`` must therefore always be called before ``import weasyprint``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Typical paths of the GTK3 installer (winget id: tschoonj.GTKForWindows).
_CANDIDATES = (
    r"C:\Program Files\GTK3-Runtime Win64\bin",
    r"C:\Program Files (x86)\GTK3-Runtime Win64\bin",
    r"C:\Program Files\GTK3-Runtime Win32\bin",
)

_resolved: str | None = None


def ensure_gtk() -> str | None:
    """Adds the GTK3 runtime to the DLL loader. Idempotent.

    Returns the path used, or None if not on Windows or GTK was
    not found (in that case WeasyPrint gives an explicit error on import).
    Can be overridden with the ``GTK3_RUNTIME`` environment variable.
    """
    global _resolved
    if _resolved is not None:
        return _resolved
    if sys.platform != "win32":
        return None

    override = os.environ.get("GTK3_RUNTIME")
    candidates = ([override] if override else []) + list(_CANDIDATES)
    for c in candidates:
        if c and (Path(c) / "libgobject-2.0-0.dll").exists():
            try:
                os.add_dll_directory(c)  # Python 3.8+: clean scoping of the loader
            except (AttributeError, OSError):
                pass
            if c.lower() not in os.environ.get("PATH", "").lower():
                os.environ["PATH"] = c + os.pathsep + os.environ.get("PATH", "")
            _resolved = c
            return c
    return None


def gtk_hint() -> str:
    return (
        "GTK3 runtime not found. WeasyPrint on Windows requires it.\n"
        "Installation: winget install --id tschoonj.GTKForWindows -e\n"
        "Or set the GTK3_RUNTIME environment variable to the path of 'bin'."
    )
