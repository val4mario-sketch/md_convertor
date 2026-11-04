"""Risoluzione del runtime GTK3, necessario a WeasyPrint su Windows.

WeasyPrint usa Pango/cairo (libgobject-2.0-0.dll, libpango-1.0-0.dll, ...). Su
Windows queste DLL non sono nel PATH di default: vanno aggiunte PRIMA di
importare weasyprint, altrimenti l'import fallisce con
``cannot load library 'libgobject-2.0-0'``.

``ensure_gtk()`` va quindi chiamata sempre prima di ``import weasyprint``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Percorsi tipici dell'installer GTK3 (winget id: tschoonj.GTKForWindows).
_CANDIDATES = (
    r"C:\Program Files\GTK3-Runtime Win64\bin",
    r"C:\Program Files (x86)\GTK3-Runtime Win64\bin",
    r"C:\Program Files\GTK3-Runtime Win32\bin",
)

_resolved: str | None = None


def ensure_gtk() -> str | None:
    """Aggiunge il runtime GTK3 al loader DLL. Idempotente.

    Ritorna il path usato, oppure None se non siamo su Windows o GTK non e'
    stato trovato (in tal caso WeasyPrint dara' un errore esplicito all'import).
    Override possibile con la variabile d'ambiente ``GTK3_RUNTIME``.
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
                os.add_dll_directory(c)  # Python 3.8+: scoping pulito del loader
            except (AttributeError, OSError):
                pass
            if c.lower() not in os.environ.get("PATH", "").lower():
                os.environ["PATH"] = c + os.pathsep + os.environ.get("PATH", "")
            _resolved = c
            return c
    return None


def gtk_hint() -> str:
    return (
        "Runtime GTK3 non trovato. WeasyPrint su Windows lo richiede.\n"
        "Installazione: winget install --id tschoonj.GTKForWindows -e\n"
        "Oppure imposta la variabile d'ambiente GTK3_RUNTIME sul percorso del 'bin'."
    )
