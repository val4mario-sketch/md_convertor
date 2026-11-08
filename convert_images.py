"""
convert_images.py
=================

Universal converter module for **images** and images -> PDF conversion.

Engines used:
- **Pillow** : main engine for most raster formats.
- **img2pdf** : to produce PDFs preserving quality (no recompression).
- **cairosvg** : to rasterize SVG to PNG.
- **pillow-heif** (optional): to read HEIC.
- **ImageMagick** (`magick` / `convert`): fallback for special cases.

Public API
----------
- ``SUPPORTED_INPUTS``  : set of accepted input extensions.
- ``SUPPORTED_OUTPUTS`` : set of producible output extensions.
- ``supports(src_ext, dst_ext) -> bool`` : tells whether the pair can be handled.
- ``convert(src, dst, opts=None) -> dict`` : runs the conversion.

``convert`` does not raise exceptions for predictable cases: it returns a
dictionary ``{'ok': bool, 'note': str, 'elapsed': float}``.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Iterable

# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------

SUPPORTED_INPUTS: set[str] = {
    ".png",
    ".jpg", ".jpeg",
    ".webp",
    ".tiff", ".tif",
    ".bmp",
    ".gif",
    ".svg",
    ".heic",
    ".pdf",   # PDF -> images: page rendering via pymupdf
}

SUPPORTED_OUTPUTS: set[str] = {
    ".png",
    ".jpg", ".jpeg",
    ".webp",
    ".tiff",
    ".bmp",
    ".gif",
    ".pdf",
    ".ico",
}

# Maps extension -> Pillow format
_PIL_FORMAT: dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".tiff": "TIFF",
    ".tif": "TIFF",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".ico": "ICO",
}

# Output extensions that need RGB mode (no alpha).
_NEEDS_RGB: set[str] = {".jpg", ".jpeg", ".bmp"}

_HEIC_HINT = "For HEIC: pip install pillow-heif"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _norm_ext(ext: str) -> str:
    if not ext:
        return ""
    ext = ext.lower().strip()
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _result(ok: bool, note: str, t0: float) -> dict:
    return {"ok": ok, "note": note, "elapsed": time.perf_counter() - t0}


def _find_imagemagick() -> Optional[str]:
    """Returns the path of magick (or convert) if available, otherwise None."""
    for name in ("magick", "convert"):
        p = shutil.which(name)
        if p:
            return p
    return None


def _has_pillow_heif() -> bool:
    try:
        import pillow_heif  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _register_heif_if_possible() -> bool:
    """Registers the HEIF plugin in Pillow if the library is present."""
    try:
        import pillow_heif  # type: ignore
        try:
            pillow_heif.register_heif_opener()
        except Exception:
            pass
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# API: supports
# ---------------------------------------------------------------------------

def supports(src_ext: str, dst_ext: str) -> bool:
    """Tells whether the pair (src_ext -> dst_ext) can be handled by this module."""
    s = _norm_ext(src_ext)
    d = _norm_ext(dst_ext)
    if s not in SUPPORTED_INPUTS or d not in SUPPORTED_OUTPUTS:
        return False
    if s == ".pdf" and d == ".pdf":
        return False  # identity makes no sense here (office_pdf handles pdf->pdf-like)
    # All theoretically supported; HEIC requires pillow-heif at run time.
    return True


def _possible_targets_from(src_ext: str) -> list[str]:
    s = _norm_ext(src_ext)
    if s not in SUPPORTED_INPUTS:
        return []
    return sorted(SUPPORTED_OUTPUTS)


# ---------------------------------------------------------------------------
# Pipeline: image loading
# ---------------------------------------------------------------------------

def _load_image(src: Path, src_ext: str):
    """Loads a source and returns a PIL.Image object (or raises).

    - SVG  -> rasterized via cairosvg to PNG, then opened with Pillow.
    - HEIC -> requires pillow-heif (registered dynamically).
    - Other formats: opened directly with Pillow.
    """
    from PIL import Image  # delayed import

    if src_ext == ".svg":
        try:
            import cairosvg  # type: ignore
        except Exception as e:
            raise RuntimeError(f"cairosvg not available: {e}")
        png_bytes = cairosvg.svg2png(url=str(src))
        return Image.open(io.BytesIO(png_bytes))

    if src_ext == ".heic":
        if not _register_heif_if_possible():
            raise RuntimeError(_HEIC_HINT)
        return Image.open(str(src))

    return Image.open(str(src))


# ---------------------------------------------------------------------------
# Pipeline: trasformazioni (resize)
# ---------------------------------------------------------------------------

def _apply_resize(img, opts: dict):
    """Applies a resize according to opts['width'] and/or opts['height']."""
    from PIL import Image

    w = opts.get("width")
    h = opts.get("height")
    if not w and not h:
        return img

    orig_w, orig_h = img.size
    if w and not h:
        ratio = float(w) / float(orig_w) if orig_w else 1.0
        new_size = (int(w), max(1, int(round(orig_h * ratio))))
    elif h and not w:
        ratio = float(h) / float(orig_h) if orig_h else 1.0
        new_size = (max(1, int(round(orig_w * ratio))), int(h))
    else:
        new_size = (int(w), int(h))

    resample = getattr(Image, "Resampling", Image).LANCZOS \
        if hasattr(Image, "Resampling") else Image.LANCZOS
    return img.resize(new_size, resample)


def _flatten_for_rgb(img, background=(255, 255, 255)):
    """Flattens an image with alpha onto a white background (for JPG/BMP)."""
    from PIL import Image

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, background)
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


# ---------------------------------------------------------------------------
# Output: PDF via img2pdf
# ---------------------------------------------------------------------------

def _save_pdf_img2pdf(sources: Iterable[Path], dst: Path, opts: dict) -> None:
    """Saves one or more image files as a PDF using img2pdf.

    img2pdf does not recompress JPEG images and keeps maximum quality.
    If the images contain an alpha channel (e.g. RGBA PNG), img2pdf does not
    accept them directly: in that case we convert them to a temporary JPEG
    or flatten them with Pillow on a white background and save a PNG without alpha.
    """
    import img2pdf  # type: ignore
    from PIL import Image

    prepared: list[bytes] = []
    for s in sources:
        s = Path(s)
        try:
            with Image.open(str(s)) as im:
                needs_flatten = (
                    im.mode in ("RGBA", "LA")
                    or (im.mode == "P" and "transparency" in im.info)
                )
                if needs_flatten:
                    flat = _flatten_for_rgb(im)
                    buf = io.BytesIO()
                    flat.save(buf, format="PNG", optimize=True)
                    prepared.append(buf.getvalue())
                else:
                    prepared.append(Path(s).read_bytes())
        except Exception:
            prepared.append(Path(s).read_bytes())

    layout_kwargs: dict = {}
    dpi = opts.get("dpi")
    if dpi:
        # Set the DPI on all the images.
        layout_kwargs["layout_fun"] = img2pdf.get_layout_fun(
            None, None, None, None, None
        )

    pdf_bytes = img2pdf.convert(prepared)
    Path(dst).write_bytes(pdf_bytes)


# ---------------------------------------------------------------------------
# Output: raster saving via Pillow
# ---------------------------------------------------------------------------

def _save_raster(img, dst: Path, dst_ext: str, opts: dict) -> None:
    """Saves a Pillow image in the requested format."""
    fmt = _PIL_FORMAT.get(dst_ext)
    if fmt is None:
        raise RuntimeError(f"Output format not handled by Pillow: {dst_ext}")

    save_kwargs: dict = {}
    q_val = opts.get("quality")
    quality = int(q_val) if q_val is not None else 85
    optimize = bool(opts.get("optimize", True))
    dpi = opts.get("dpi")

    if dst_ext in (".jpg", ".jpeg"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = optimize
        save_kwargs["progressive"] = True
    elif dst_ext == ".webp":
        save_kwargs["quality"] = quality
        save_kwargs["method"] = 6  # maximum quality/size compression
    elif dst_ext == ".png":
        save_kwargs["optimize"] = optimize
    elif dst_ext == ".tiff":
        save_kwargs["compression"] = opts.get("compression", "tiff_lzw")
    elif dst_ext == ".ico":
        # ICO: resize to a series of standard sizes.
        sizes = opts.get("sizes") or [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        save_kwargs["sizes"] = sizes

    if dpi:
        save_kwargs["dpi"] = (int(dpi), int(dpi))

    # JPG/BMP do not support alpha: flatten first.
    if dst_ext in _NEEDS_RGB:
        img = _flatten_for_rgb(img)
    elif dst_ext == ".ico" and img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")

    img.save(str(dst), format=fmt, **save_kwargs)


# ---------------------------------------------------------------------------
# Fallback: ImageMagick
# ---------------------------------------------------------------------------

def _try_imagemagick(src: Path, dst: Path) -> tuple[bool, str]:
    """Attempts the conversion with ImageMagick. Returns (ok, message)."""
    exe = _find_imagemagick()
    if exe is None:
        return False, "ImageMagick not available"

    cmd: list[str]
    name = Path(exe).stem.lower()
    if name == "magick":
        cmd = [exe, str(src), str(dst)]
    else:
        cmd = [exe, str(src), str(dst)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        return False, f"ImageMagick execution error: {e}"

    if proc.returncode != 0:
        return False, f"ImageMagick failed: {(proc.stderr or proc.stdout).strip()}"

    if not dst.exists() or dst.stat().st_size <= 0:
        return False, "ImageMagick: output missing or empty"

    return True, "fallback ImageMagick"


# ---------------------------------------------------------------------------
# API: convert
# ---------------------------------------------------------------------------

def convert(src: Path, dst: Path, opts: Optional[dict] = None) -> dict:
    """Converts ``src`` to ``dst``.

    Options recognized in ``opts``:
      - ``quality``  (int, default 85) for JPG/WEBP.
      - ``dpi``      (int) sets the DPI of the output image.
      - ``width``    (int) desired width (keeps the aspect if height is absent).
      - ``height``   (int) desired height.
      - ``optimize`` (bool, default True).
      - ``pdf_sources`` (list[Path]) for multi-page PDFs:
            each element is an image; ``src`` is ignored/placed as the first page.
    """
    t0 = time.perf_counter()
    opts = dict(opts or {})

    src = Path(src)
    dst = Path(dst)

    src_ext = _norm_ext(src.suffix)
    dst_ext = _norm_ext(dst.suffix)

    # --- Basic validation -----------------------------------------------------
    if src_ext not in SUPPORTED_INPUTS:
        return _result(
            False,
            f"Source extension '{src_ext}' not handled by this module.",
            t0,
        )
    if dst_ext not in SUPPORTED_OUTPUTS:
        return _result(
            False,
            f"Target extension '{dst_ext}' not handled by this module.",
            t0,
        )
    if not supports(src_ext, dst_ext):
        alts = _possible_targets_from(src_ext)
        return _result(
            False,
            (
                f"Conversione {src_ext}->{dst_ext} not supported. "
                f"Alternatives from {src_ext}: {alts}"
            ),
            t0,
        )

    if not src.exists():
        return _result(False, f"Source file not found: {src}", t0)
    if not src.is_file():
        return _result(False, f"The source is not a regular file: {src}", t0)

    # HEIC: early warning if the library is missing.
    if src_ext == ".heic" and not _has_pillow_heif():
        return _result(False, _HEIC_HINT, t0)

    # --- Input PDF: page rendering via pymupdf --------------------------------
    if src_ext == ".pdf":
        try:
            import pymupdf as _pdf
        except ImportError:
            return _result(False, "pymupdf not installed (pip install pymupdf).", t0)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dpi = int(opts.get("dpi") or 200)
            page_sel = opts.get("page")          # 1-based, optional
            doc = _pdf.open(str(src))
            n = doc.page_count
            mat = _pdf.Matrix(dpi / 72, dpi / 72)
            written: list[Path] = []
            if n == 1 or page_sel:
                idx = (int(page_sel) - 1) if page_sel else 0
                idx = max(0, min(n - 1, idx))
                pix = doc[idx].get_pixmap(matrix=mat, alpha=False)
                pix.save(str(dst))
                written.append(dst)
            else:
                # multi-page: <stem>_p001.ext, _p002.ext, ... next to dst
                for i in range(n):
                    out_i = dst.with_name(f"{dst.stem}_p{i+1:03d}{dst.suffix}")
                    pix = doc[i].get_pixmap(matrix=mat, alpha=False)
                    pix.save(str(out_i))
                    written.append(out_i)
                # "main" dst = copy of the first page (so it always exists)
                shutil.copyfile(written[0], dst)
            doc.close()
            note = (f"PDF rendered: {n} page(s) at {dpi} DPI"
                    + (f" -> {len(written)} file ({written[0].name} ...)" if n > 1
                       else f" -> {dst.name}"))
            return _result(True, note, t0)
        except Exception as e:
            return _result(False, f"PDF rendering failed: {type(e).__name__}: {e}", t0)

    # --- Create the destination folder if needed ------------------------------
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return _result(False, f"Cannot create the destination folder: {e}", t0)

    note_extra = ""

    # --- Special case: PDF output ---------------------------------------------
    if dst_ext == ".pdf":
        try:
            pdf_sources = opts.get("pdf_sources")
            if pdf_sources:
                sources = [Path(p) for p in pdf_sources]
            else:
                sources = [src]

            # If the source is SVG or HEIC, it must first be rasterized to a temporary PNG.
            tmp_files: list[Path] = []
            try:
                final_sources: list[Path] = []
                for s in sources:
                    s_ext = _norm_ext(s.suffix)
                    if s_ext in (".svg", ".heic"):
                        img = _load_image(s, s_ext)
                        tmp = s.with_suffix(".__tmp__.png")
                        img.save(str(tmp), format="PNG", optimize=True)
                        tmp_files.append(tmp)
                        final_sources.append(tmp)
                    else:
                        final_sources.append(s)

                _save_pdf_img2pdf(final_sources, dst, opts)
            finally:
                for t in tmp_files:
                    try:
                        t.unlink()
                    except OSError:
                        pass
        except Exception as e:
            # ImageMagick fallback (single image only).
            if not opts.get("pdf_sources"):
                ok, mg_note = _try_imagemagick(src, dst)
                if ok:
                    note_extra = f" ({mg_note})"
                else:
                    return _result(False, f"PDF generation error: {e}. {mg_note}", t0)
            else:
                return _result(False, f"PDF generation error: {e}", t0)

        return _finalize(dst, src_ext, dst_ext, t0, extra=note_extra)

    # --- General case: raster -------------------------------------------------
    img = None
    try:
        img = _load_image(src, src_ext)

        # Resize
        img = _apply_resize(img, opts)

        # Salvataggio
        _save_raster(img, dst, dst_ext, opts)

    except Exception as e:
        # ImageMagick fallback (e.g. multi-page TIFF or missing codecs).
        ok, mg_note = _try_imagemagick(src, dst)
        if ok:
            note_extra = f" (Pillow failed: {e}; usato {mg_note})"
        else:
            return _result(
                False,
                f"Conversion error {src_ext}->{dst_ext}: {e}. {mg_note}",
                t0,
            )
    finally:
        try:
            if img is not None:
                img.close()
        except Exception:
            pass

    return _finalize(dst, src_ext, dst_ext, t0, extra=note_extra)


def _finalize(dst: Path, src_ext: str, dst_ext: str, t0: float, extra: str = "") -> dict:
    if not dst.exists():
        return _result(False, "Conversion completed but the output file does not exist.", t0)
    try:
        size = dst.stat().st_size
    except OSError as e:
        return _result(False, f"Cannot read the output file: {e}", t0)
    if size <= 0:
        return _result(False, "The output file is empty.", t0)
    return _result(True, f"Conversione {src_ext}->{dst_ext} riuscita ({size} byte).{extra}", t0)


# ---------------------------------------------------------------------------
# Direct execution (debug)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python convert_images.py <src> <dst>")
        sys.exit(2)
    res = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(res)
    sys.exit(0 if res["ok"] else 1)
