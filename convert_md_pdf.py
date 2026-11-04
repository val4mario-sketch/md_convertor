#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_md_pdf.py — Markdown -> PDF LEGGIBILE (stile report) via `tomdclaude`.

Sostituisce il vecchio percorso md->pdf "stile Word" (fallback LibreOffice) con
un PDF tipografico: copertina, indice con numeri di pagina, capitoli numerati,
tabelle, codice evidenziato e MATEMATICA LaTeX ($...$ / $$...$$) resa in SVG.
Con validazione "gatekeeper": se la conversione perde dati, il file non viene
salvato e viene prodotto un log.

Tutto GRATIS e IN LOCALE: python-markdown + WeasyPrint + matplotlib (mathtext).
Nessuna API, nessuna chiave, nessun upload.

Interfaccia uniforme col resto dei moduli del convertitore universale:
    SUPPORTED_INPUTS, SUPPORTED_OUTPUTS, supports(), convert() -> {ok, note, elapsed}
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

# Il package `tomdclaude/` è una sottocartella di questa: assicuriamone l'import.
sys.path.insert(0, str(Path(__file__).parent))

SUPPORTED_INPUTS = {".md", ".markdown"}
SUPPORTED_OUTPUTS = {".pdf"}


def supports(src_ext: str, dst_ext: str) -> bool:
    return src_ext.lower() in SUPPORTED_INPUTS and dst_ext.lower() in SUPPORTED_OUTPUTS


def convert(src: Path, dst: Path, opts: dict | None = None) -> dict:
    """Converte un .md in PDF leggibile.

    `opts` (tutte opzionali):
      - overwrite (bool, default True)
      - title / author (str)         — copertina/intestazione
      - math (bool, default True)    — rendering matematica LaTeX
      - strict_math (bool, default False) — rifiuta il PDF se una formula non si rende
      - strip_cite (bool, default True)   — rimuove i marcatori [cite_start]/[cite: N]
    """
    opts = opts or {}
    t0 = time.time()
    src = Path(src); dst = Path(dst)

    if not src.exists():
        return {"ok": False, "note": f"File non trovato: {src}", "elapsed": 0.0}
    if dst.suffix.lower() not in SUPPORTED_OUTPUTS:
        return {"ok": False,
                "note": f"convert_md_pdf produce solo .pdf (ricevuto {dst.suffix}).",
                "elapsed": 0.0}

    # Mappa le opzioni del router su quelle di tomdclaude (default: math + pulizia ON).
    tc_opts = {"overwrite": opts.get("overwrite", True)}
    for k in ("title", "author", "math", "strict_math", "strip_cite"):
        if k in opts:
            tc_opts[k] = opts[k]

    # ---- RIPARAZIONE LINK IMMAGINE ROTTI ----
    # Caso tipico: il .md e' stato spostato in una sottocartella ma media/ e'
    # rimasta al livello superiore -> tutti i link "media/x/..." sono rotti.
    # Cerco ogni immagine mancante in: cartella del md, cartella padre,
    # nonna; se la trovo riscrivo il link con percorso ASSOLUTO in una copia
    # temporanea del md (l'originale non viene toccato).
    import re as _re, tempfile as _tf
    src_effettivo = src
    tmp_fixed: Path | None = None
    try:
        text = src.read_text(encoding="utf-8")
        modificato = False

        # (a) I blocchi <details>...</details> (page-image "garanzia diagrammi"
        # dei .md generati da tomd2) NON vengono processati da python-markdown:
        # resterebbero testo morto nel PDF. Per un PDF leggibile li RIMUOVIAMO
        # (le figure di contenuto, fuori dai details, restano). Con
        # opts['include_page_images']=True vengono invece INLINEATI.
        if "<details>" in text:
            n_det = text.count("<details>")
            if opts.get("include_page_images"):
                # tieni solo il corpo (l'immagine), togli i tag detail/summary
                text = _re.sub(r"<details><summary>[^<]*</summary>\s*", "", text)
                text = _re.sub(r"\s*</details>", "", text)
                print(f"  [fix] {n_det} blocchi <details> inlineati (page-image incluse)")
            else:
                text = _re.sub(r"<details><summary>.*?</summary>.*?</details>",
                               "", text, flags=_re.S)
                print(f"  [fix] {n_det} blocchi <details> (snapshot pagina) rimossi "
                      f"per un PDF pulito (usa include_page_images per tenerli)")
            modificato = True

        # (b) Riparazione link rotti: cerca l'immagine nelle cartelle superiori
        # e riscrivi come URI file:// (WeasyPrint non accetta 'C:/...' nudo).
        links = _re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
        broken = [l for l in links
                  if not l.startswith(("http://", "https://", "data:", "file:"))
                  and not (src.parent / l).exists()
                  and not Path(l).is_absolute()]
        if broken:
            candidates = [src.parent, src.parent.parent,
                          src.parent.parent.parent]
            fixed = 0
            for l in set(broken):
                for base in candidates:
                    try:
                        cand = (base / l).resolve()
                    except OSError:
                        continue
                    if cand.exists():
                        text = text.replace(f"]({l})", f"]({cand.as_uri()})")
                        fixed += 1
                        break
            if fixed:
                modificato = True
                print(f"  [fix] {fixed}/{len(set(broken))} link immagine rotti "
                      f"riparati (media/ trovata in una cartella superiore)")

        # (c) Marcatori grezzi "----- Start/End of picture text -----" (testo OCR
        # dentro le immagini, prodotto da pymupdf4llm): resi come nota corsiva.
        if "Start of picture text" in text:
            text = _re.sub(r"\*{0,2}-{3,}\s*Start of picture text\s*-{3,}\*{0,2}",
                           "*Testo nel diagramma:*", text)
            text = _re.sub(r"\*{0,2}-{3,}\s*End of picture text\s*-{3,}\*{0,2}",
                           "", text)
            modificato = True

        # (d) Slide-deck senza H1 (tutti '##'): numerazione '0.1' ripetuta.
        # Promuovo i titoli di un livello: ## -> #, ### -> ## ecc.
        h1 = len(_re.findall(r"^# [^#]", text, flags=_re.M))
        h2 = len(_re.findall(r"^## ", text, flags=_re.M))
        if h1 == 0 and h2 >= 3:
            text = _re.sub(r"^(#{2,6}) ", lambda m: "#" * (len(m.group(1)) - 1) + " ",
                           text, flags=_re.M)
            modificato = True
            print(f"  [fix] documento senza H1: {h2} titoli promossi di un livello "
                  f"(numerazione capitoli corretta)")

        if modificato:
            tmp_fixed = Path(_tf.mkdtemp(prefix="mdfix_")) / src.name
            tmp_fixed.write_text(text, encoding="utf-8")
            src_effettivo = tmp_fixed
    except Exception:
        pass  # la riparazione e' best-effort: mai bloccare la conversione

    try:
        # Import TARDIVO: se manca una dipendenza pesante (WeasyPrint/GTK/matplotlib),
        # niente crash all'import del router -> errore chiaro a runtime.
        from tomdclaude.core.pipeline import convert as tc_convert
        res = tc_convert(src_effettivo, dst, tc_opts)
    except Exception as e:
        return {"ok": False,
                "note": f"Motore tomdclaude non disponibile ({type(e).__name__}): {e}",
                "elapsed": time.time() - t0}
    finally:
        if tmp_fixed is not None:
            import shutil as _sh
            _sh.rmtree(tmp_fixed.parent, ignore_errors=True)

    elapsed = res.elapsed or (time.time() - t0)
    note = res.note or ("ok" if res.ok else "errore")
    if res.warnings:
        note += " | avvisi: " + "; ".join(i.code for i in res.warnings)
    out = {"ok": bool(res.ok), "note": note, "elapsed": elapsed}
    if not res.ok:
        if res.errors:
            out["note"] += " | errori: " + "; ".join(i.code for i in res.errors)
        if res.log_path:
            out["report"] = f"log: {res.log_path}"
    return out


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _self_test() -> bool:
    import shutil
    tmp = Path(__file__).parent / "_test_md_pdf"
    tmp.mkdir(exist_ok=True)
    try:
        md = tmp / "demo.md"
        md.write_text(
            "# Titolo\n\nTesto con math inline $E=mc^2$ e display:\n\n"
            "$$\\int_0^1 x\\,dx = \\frac{1}{2}$$\n\n## Sezione\n\nFine.\n",
            encoding="utf-8")
        r = convert(md, tmp / "demo.pdf", {})
        ok = r["ok"] and (tmp / "demo.pdf").exists() and (tmp / "demo.pdf").stat().st_size > 1000
        print(f"  [{'OK' if ok else 'KO'}] demo.md -> demo.pdf  ({r['note']})")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("convert_md_pdf.py self-test:")
    sys.exit(0 if _self_test() else 1)
