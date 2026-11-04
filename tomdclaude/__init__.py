"""
tomdclaude — Sistema di conversione file modulare con validazione "gatekeeper".

Architettura:
    core/        contratto astratto, risultato, registry, pipeline (orchestrazione)
    converters/  un convertitore = una classe indipendente (es. MarkdownToPdfConverter)
    validators/  controlli di integrità che confrontano i metadati sorgente/destinazione
    utils/       logging, gestione runtime GTK, estrazione metadati

Principio cardine: una conversione viene scritta su file TEMPORANEO e diventa
definitiva (commit) SOLO se supera tutti i validatori. In caso contrario il file
non viene salvato e viene prodotto un log dettagliato dell'errore.
"""

__version__ = "0.1.0"

from .core.pipeline import convert  # noqa: E402  (API pubblica comoda)

__all__ = ["convert", "__version__"]
