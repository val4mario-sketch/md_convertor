"""First test case: readable MD -> PDF conversion + gatekeeper check.

Run (from the folder containing the package):

    python -m tomdclaude.tests.test_md_to_pdf

Also compatible with pytest:

    pytest tomdclaude/tests/test_md_to_pdf.py

Covers:
  1. POSITIVE   — report_demo.md -> PDF: successful conversion, file saved,
     valid PDF (>=1 page), heading structure preserved, no errors.
  2. GATEKEEPER — a "broken" converter producing a fake PDF must be
     REJECTED: the file is NOT saved and a detailed log is produced.
  3. UNIT       — the validators correctly flag the data-loss cases.
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
    """report_demo.md -> PDF: valid conversion and file saved."""
    out = _PKG / "examples" / "report_demo.pdf"
    res = convert(_DEMO_MD, out, {"overwrite": True}, logger=log)

    assert res.ok, f"conversion failed: {res.note} | {[str(i) for i in res.errors]}"
    assert res.committed and out.exists(), "the PDF was not saved"
    assert out.stat().st_size > 1000, "PDF too small"
    assert res.dst_metadata.get("pages", 0) >= 1, "PDF with no pages"
    assert not res.errors, f"unexpected errors: {[str(i) for i in res.errors]}"

    # heading structure preserved (source vs rendered HTML)
    src_levels = [lvl for lvl, _ in res.src_metadata["headings"]]
    dst_levels = [lvl for lvl, _ in res.dst_metadata["html_headings"]]
    assert src_levels == dst_levels, f"heading hierarchy altered: {src_levels} != {dst_levels}"
    return res, out


# --------------------------------------------------------------------------- 2
class _BrokenConverter(BaseConverter):
    """Simulates a converter that 'loses data': writes a NON-PDF file."""

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
        Path(dst).write_bytes(b"THIS IS NOT A PDF")  # deliberately corrupted output
        return {"note": "deliberately invalid output"}


def test_gatekeeper_blocks_bad_output():
    """The gatekeeper must reject the corrupted output and save NOTHING."""
    registry._CONVERTERS.insert(0, _BrokenConverter)
    try:
        out = _PKG / "examples" / "_should_not_exist.xpdf"
        out.unlink(missing_ok=True)
        res = convert(_DEMO_MD, out, {"overwrite": True}, logger=log)

        assert not res.ok, "the gatekeeper did NOT block a corrupted output"
        assert not res.committed, "committed=True su output rifiutato"
        assert not out.exists(), "the corrupted file was saved (it should NOT have been)"
        assert res.errors, "no error recorded by the validator"
        assert res.log_path and Path(res.log_path).exists(), "failure log not produced"
        codes = {i.code for i in res.errors}
        assert "not_a_pdf" in codes, f"expected 'not_a_pdf', got {codes}"
    finally:
        registry._CONVERTERS.remove(_BrokenConverter)
    return res


# --------------------------------------------------------------------------- 3
def test_md_validator_detects_missing_heading():
    """Unit: a heading absent from the PDF is flagged as a blocking error."""
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
        f"the validator did not detect the missing heading: {[str(i) for i in issues]}"


def test_pdf_validator_detects_short_text():
    """Unit: a PDF nearly empty compared with the source is blocked."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.7 minimal")
        tmp = Path(f.name)
    try:
        src_meta = {"text_len": 5000}
        dst_meta = {"pages": 1, "pdf_text_len": 3}
        issues = PdfIntegrityValidator().validate(src_meta, dst_meta, {"tmp": tmp})
        assert any(i.code == "text_too_short" and i.blocking for i in issues), \
            f"text too short not detected: {[str(i) for i in issues]}"
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
    """The $...$ and $$...$$ formulas become SVGs; no '$' remains in the PDF."""
    import tempfile
    d = Path(tempfile.mkdtemp())
    src = d / "math.md"
    src.write_text(_MATH_MD, encoding="utf-8")
    out = d / "math.pdf"
    res = convert(src, out, {"overwrite": True}, logger=log)

    assert res.ok and out.exists(), f"math conversion failed: {res.note}"
    assert res.dst_metadata.get("n_math", 0) >= 4, "formulas not found"
    assert res.dst_metadata.get("n_math_failed", 0) == 0, \
        f"failed formulas: {res.dst_metadata.get('math_failures')}"
    # no '$' delimiter must survive in the PDF text
    assert "$" not in res.dst_metadata.get("pdf_text", ""), "$ delimiters left in the PDF"
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def test_strict_math_blocks_on_failure():
    """With strict_math a formula that cannot be rendered makes the PDF rejected."""
    import tempfile
    d = Path(tempfile.mkdtemp())
    src = d / "bad.md"
    src.write_text("# T\n\nFormula rotta: $\\comandoinesistente{z}$ nel testo.\n", encoding="utf-8")
    out = d / "bad.pdf"
    try:
        res = convert(src, out, {"overwrite": True, "strict_math": True}, logger=log)
        assert not res.ok, "strict_math did not block the broken formula"
        assert not out.exists(), "PDF saved despite strict_math"
        assert any(i.code == "math_render_failed" for i in res.errors), \
            f"expected a math error, got {[i.code for i in res.errors]}"
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- runner
def _run_all() -> int:
    tests = [
        ("Readable MD->PDF (positive)", test_md_to_pdf_readable),
        ("Gatekeeper blocks corrupted output", test_gatekeeper_blocks_bad_output),
        ("Inline + display math (no leftover $)", test_math_inline_and_display),
        ("strict_math blocks a broken formula", test_strict_math_blocks_on_failure),
        ("MD validator: missing heading", test_md_validator_detects_missing_heading),
        ("PDF validator: text too short", test_pdf_validator_detects_short_text),
    ]
    print("=" * 70)
    print("tomdclaude — MD->PDF test suite")
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
    print(f"Result: {len(tests) - failed}/{len(tests)} tests passed.")
    if pdf_path and Path(pdf_path).exists():
        print(f"Example PDF generated: {pdf_path}")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
