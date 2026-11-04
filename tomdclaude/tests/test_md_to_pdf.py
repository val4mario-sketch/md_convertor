"""Primo test case: conversione MD -> PDF leggibile + verifica del gatekeeper.

Esecuzione (dalla cartella che contiene il package, cioe' "CONVERTITORE UNIVERSALE"):

    python -m tomdclaude.tests.test_md_to_pdf

Compatibile anche con pytest:

    pytest tomdclaude/tests/test_md_to_pdf.py

Copre:
  1. POSITIVO  — report_demo.md -> PDF: conversione riuscita, file salvato,
     PDF valido (>=1 pagina), struttura dei titoli preservata, nessun errore.
  2. GATEKEEPER — un convertitore "rotto" che produce un finto PDF deve essere
     RIFIUTATO: il file NON viene salvato e viene prodotto un log dettagliato.
  3. UNIT      — i validatori segnalano correttamente i casi di perdita dati.
"""
from __future__ import annotations

from pathlib import Path

from tomdclaude.core import registry
from tomdclaude.core.base_converter import BaseConverter
from tomdclaude.core.pipeline import convert
from tomdclaude.utils.logging_setup import get_logger
from tomdclaude.validators.md_validator import MarkdownStructureValidator
from tomdclaude.validators.pdf_validator import PdfIntegrityValidator

_PKG = Path(__file__).resolve().parent.parent
_DEMO_MD = _PKG / "examples" / "report_demo.md"

log = get_logger(verbose=False)


# --------------------------------------------------------------------------- 1
def test_md_to_pdf_readable():
    """report_demo.md -> PDF: conversione valida e file salvato."""
    out = _PKG / "examples" / "report_demo.pdf"
    res = convert(_DEMO_MD, out, {"overwrite": True}, logger=log)

    assert res.ok, f"conversione fallita: {res.note} | {[str(i) for i in res.errors]}"
    assert res.committed and out.exists(), "il PDF non e' stato salvato"
    assert out.stat().st_size > 1000, "PDF troppo piccolo"
    assert res.dst_metadata.get("pages", 0) >= 1, "PDF senza pagine"
    assert not res.errors, f"errori inattesi: {[str(i) for i in res.errors]}"

    # struttura dei titoli preservata (sorgente vs HTML renderizzato)
    src_levels = [lvl for lvl, _ in res.src_metadata["headings"]]
    dst_levels = [lvl for lvl, _ in res.dst_metadata["html_headings"]]
    assert src_levels == dst_levels, f"gerarchia titoli alterata: {src_levels} != {dst_levels}"
    return res, out


# --------------------------------------------------------------------------- 2
class _BrokenConverter(BaseConverter):
    """Simula un convertitore che 'perde i dati': scrive un file NON-PDF."""

    name = "broken_md_to_xpdf"
    source_formats = frozenset({"md", "markdown"})
    target_formats = frozenset({"xpdf"})

    def extract_source_metadata(self, src, opts=None):
        from tomdclaude.utils.metadata import markdown_heading_tree, md_plain_length
        text = Path(src).read_text(encoding="utf-8", errors="replace")
        return {"headings": markdown_heading_tree(text), "text_len": md_plain_length(text)}

    def extract_target_metadata(self, dst):
        return {"pages": 0, "pdf_text": "", "pdf_text_len": 0}

    def validators(self):
        return [PdfIntegrityValidator()]

    def convert(self, src, dst, opts):
        Path(dst).write_bytes(b"QUESTO NON E UN PDF")  # output corrotto di proposito
        return {"note": "output volutamente non valido"}


def test_gatekeeper_blocks_bad_output():
    """Il gatekeeper deve rifiutare l'output corrotto e NON salvare nulla."""
    registry._CONVERTERS.insert(0, _BrokenConverter)
    try:
        out = _PKG / "examples" / "_should_not_exist.xpdf"
        out.unlink(missing_ok=True)
        res = convert(_DEMO_MD, out, {"overwrite": True}, logger=log)

        assert not res.ok, "il gatekeeper NON ha bloccato un output corrotto"
        assert not res.committed, "committed=True su output rifiutato"
        assert not out.exists(), "il file corrotto e' stato salvato (NON doveva)"
        assert res.errors, "nessun errore registrato dal validator"
        assert res.log_path and Path(res.log_path).exists(), "log di fallimento non prodotto"
        codes = {i.code for i in res.errors}
        assert "not_a_pdf" in codes, f"atteso 'not_a_pdf', ottenuti {codes}"
    finally:
        registry._CONVERTERS.remove(_BrokenConverter)
    return res


# --------------------------------------------------------------------------- 3
def test_md_validator_detects_missing_heading():
    """Unit: un titolo assente dal PDF viene segnalato come errore bloccante."""
    src_meta = {"headings": [(1, "Introduzione"), (2, "Obiettivi")], "n_tables": 0, "text_len": 200}
    dst_meta = {
        "html_headings": [(1, "Introduzione"), (2, "Obiettivi")],
        "n_html_tables": 0,
        "pdf_text": "Introduzione e basta, manca l'altro titolo.",  # 'Obiettivi' assente
        "pages": 1, "pdf_text_len": 60,
    }
    issues = MarkdownStructureValidator().validate(src_meta, dst_meta, {})
    blocking = [i for i in issues if i.blocking]
    assert any(i.code == "heading_missing_in_pdf" for i in blocking), \
        f"il validator non ha rilevato il titolo mancante: {[str(i) for i in issues]}"


def test_pdf_validator_detects_short_text():
    """Unit: un PDF quasi vuoto rispetto al sorgente e' bloccato."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.7 minimal")
        tmp = Path(f.name)
    try:
        src_meta = {"text_len": 5000}
        dst_meta = {"pages": 1, "pdf_text_len": 3}
        issues = PdfIntegrityValidator().validate(src_meta, dst_meta, {"tmp": tmp})
        assert any(i.code == "text_too_short" and i.blocking for i in issues), \
            f"testo troppo corto non rilevato: {[str(i) for i in issues]}"
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- 4
_MATH_MD = r"""# Matematica

Inline: l'energia e' $E = mc^2$ e la disuguaglianza $f(a) \cdot f(b) < 0$.

Display isolato:

$$\int_0^1 x \, dx = \frac{1}{2}$$

In un elenco (display indentato, caso problematico):

* **Formula**: il punto medio:
  $$x = \frac{a+b}{2}$$
* Altro punto
"""


def test_math_inline_and_display(tmp=None):
    """Le formule $...$ e $$...$$ diventano SVG; nessun '$' resta nel PDF."""
    import tempfile
    d = Path(tempfile.mkdtemp())
    src = d / "math.md"
    src.write_text(_MATH_MD, encoding="utf-8")
    out = d / "math.pdf"
    res = convert(src, out, {"overwrite": True}, logger=log)

    assert res.ok and out.exists(), f"conversione math fallita: {res.note}"
    assert res.dst_metadata.get("n_math", 0) >= 4, "formule non individuate"
    assert res.dst_metadata.get("n_math_failed", 0) == 0, \
        f"formule fallite: {res.dst_metadata.get('math_failures')}"
    # nessun delimitatore '$' deve sopravvivere nel testo del PDF
    assert "$" not in res.dst_metadata.get("pdf_text", ""), "delimitatori $ rimasti nel PDF"
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def test_strict_math_blocks_on_failure():
    """Con strict_math una formula non renderizzabile fa rifiutare il PDF."""
    import tempfile
    d = Path(tempfile.mkdtemp())
    src = d / "bad.md"
    src.write_text("# T\n\nFormula rotta: $\\comandoinesistente{z}$ nel testo.\n", encoding="utf-8")
    out = d / "bad.pdf"
    try:
        res = convert(src, out, {"overwrite": True, "strict_math": True}, logger=log)
        assert not res.ok, "strict_math non ha bloccato la formula rotta"
        assert not out.exists(), "PDF salvato nonostante strict_math"
        assert any(i.code == "math_render_failed" for i in res.errors), \
            f"errore math atteso, ottenuti {[i.code for i in res.errors]}"
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- runner
def _run_all() -> int:
    tests = [
        ("MD->PDF leggibile (positivo)", test_md_to_pdf_readable),
        ("Gatekeeper blocca output corrotto", test_gatekeeper_blocks_bad_output),
        ("Matematica inline + display (no $ residui)", test_math_inline_and_display),
        ("strict_math blocca formula rotta", test_strict_math_blocks_on_failure),
        ("Validator MD: titolo mancante", test_md_validator_detects_missing_heading),
        ("Validator PDF: testo troppo corto", test_pdf_validator_detects_short_text),
    ]
    print("=" * 70)
    print("tomdclaude — suite di test MD->PDF")
    print("=" * 70)
    failed = 0
    pdf_path = None
    for nome, fn in tests:
        try:
            out = fn()
            if isinstance(out, tuple) and len(out) == 2:
                pdf_path = out[1]
            print(f"[PASS] {nome}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {nome}\n       {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[ERR ] {nome}\n       {type(e).__name__}: {e}")
    print("-" * 70)
    print(f"Risultato: {len(tests) - failed}/{len(tests)} test superati.")
    if pdf_path and Path(pdf_path).exists():
        print(f"PDF di esempio generato: {pdf_path}")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
