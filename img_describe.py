# -*- coding: utf-8 -*-
"""
img_describe.py — local AI descriptions of the images in a Markdown file.

For each image referenced in the .md (`![...](path)`) generates a text
description with a LOCAL vision model via Ollama (default: qwen2.5vl:3b,
free, no cloud) and inserts it UNDER the image as a blockquote:

    ![](media/NOTES/formula.png)

    > **[Descrizione IA]** $$a = \\frac{v}{c}\\sin\\theta$$ — formula
    > for the aberration of light...

This way an AI (or a person) reading the .md understands the content of each
image even without seeing it. Formulas are transcribed in LaTeX when possible.

Features:
- CACHE in a JSON file next to the .md: running again does not redo finished work
  (resume after an interruption; the .md is saved every 15 images).
- IDEMPOTENT: images already described are skipped (use --force to
  regenerate).
- Starts `ollama serve` by itself if it is not running and downloads the
  model by itself if missing (one time).

Standalone use:
    python img_describe.py "file.md" [--model qwen2.5vl:3b] [--force]
                           [--limit N] [--dry-run]

Use from tomd2.py:  --describe-images flag (see convert_one).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote

OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("TOMD2_VISION_MODEL", "qwen2.5vl:3b")
# maximum side (px) of the images sent to the model: smaller = much
# faster on CPU (visual tokens grow with the image area)
MAX_SIDE      = int(os.environ.get("TOMD2_VISION_MAXSIDE", "640"))

# ---- llama.cpp backend (PREFERRED if installed: ~4x faster) ----------
# Ollama enforces a MINIMUM of ~1000 visual tokens per image on qwen2.5vl
# (measured: even a 144x68 px image generates 1084 -> ~140 s/image on
# this CPU). llama-server exposes --image-min/max-tokens: with min=100 and
# max=256 the LaTeX quality stays identical but it drops to ~35 s/image.
LLAMACPP_URL    = os.environ.get("TOMD2_LLAMACPP_URL", "http://127.0.0.1:11435")
LLAMACPP_EXE    = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "llamacpp" / "llama-server.exe"
_LLAMACPP_MODELS = Path.home() / ".llamacpp-models"
LLAMACPP_MODEL  = _LLAMACPP_MODELS / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
LLAMACPP_MMPROJ = _LLAMACPP_MODELS / "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
IMG_MIN_TOKENS  = int(os.environ.get("TOMD2_VISION_MINTOK", "100"))
IMG_MAX_TOKENS  = int(os.environ.get("TOMD2_VISION_MAXTOK", "256"))

# Stable marker: used to recognize images already described.
MARKER = "> **[Descrizione IA]**"


def _log_flush(*a, **kw):
    """print with flush: without it, stdout redirected to a file stays in the buffer
       and the progress is not visible (jobs of hours!)."""
    print(*a, **kw, flush=True)


def _keep_awake(enable: bool) -> None:
    """Stops Windows from suspending the PC during long work
       (same mechanism as tomd2's _KeepAwake). The state is per-thread
       and resets by itself when the process ends: no cleanup
       needed in case of a crash."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass

# Same "balanced parentheses" regex as tomd2._encode_md_link_urls
_IMG_RE = re.compile(r"!\[[^\]\n]*\]\(([^()\n]*(?:\([^()\n]*\)[^()\n]*)*)\)")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

PROMPT = (
    "Describe the image (from university lecture notes) for an AI that cannot "
    "see it. Italian, at most 60 words, ONLY the description, without "
    "inventing anything that cannot be seen. "
    "Mathematical formula: transcribe it EXACTLY in LaTeX between $$ and $$ + one sentence "
    "of explanation. Graph: axes, curves, trend. Diagram: elements and "
    "relationships. Table: summary. Photo: subject and details."
)


# ----------------------------------------------------------------------------
# Ollama: server + model
# ----------------------------------------------------------------------------
def _find_ollama_exe() -> str | None:
    import shutil as _sh
    exe = _sh.which("ollama")
    if exe:
        return exe
    for cand in (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
    ):
        if cand.exists():
            return str(cand)
    return None


def _server_ok(timeout: float = 3) -> bool:
    import requests
    try:
        return requests.get(f"{OLLAMA_URL}/api/version", timeout=timeout).ok
    except Exception:
        return False


def ensure_server(log=_log_flush) -> None:
    """Checks that Ollama responds; if not, tries to start it by itself."""
    if _server_ok():
        return
    exe = _find_ollama_exe()
    if not exe:
        raise RuntimeError(
            "Ollama not found. Install it with:  winget install Ollama.Ollama\n"
            "then try again (the model is downloaded automatically).")
    log("  [AI] starting the Ollama server ...")
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    subprocess.Popen([exe, "serve"], creationflags=flags,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        time.sleep(1)
        if _server_ok():
            return
    raise RuntimeError("Ollama does not respond on " + OLLAMA_URL)


def ensure_model(model: str = DEFAULT_MODEL, log=_log_flush) -> None:
    """Downloads the model if not already present (one time, ~GB)."""
    import requests
    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()
    have = {m.get("name", "") for m in tags.get("models", [])}
    if model in have or f"{model}:latest" in have:
        return
    log(f"  [AI] model '{model}' missing: downloading (one time) ...")
    exe = _find_ollama_exe()
    if exe:
        r = subprocess.run([exe, "pull", model], text=True)
        if r.returncode == 0:
            return
    # fallback: pull via API (streaming)
    with requests.post(f"{OLLAMA_URL}/api/pull",
                       json={"name": model}, stream=True, timeout=None) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            st = json.loads(line)
            if st.get("error"):
                raise RuntimeError(f"pull of '{model}' failed: {st['error']}")
            if "completed" in st and "total" in st and st["total"]:
                pct = 100 * st["completed"] // st["total"]
                print(f"\r  [IA] download {model}: {pct}%", end="", flush=True)
    print()


# ----------------------------------------------------------------------------
# llama.cpp backend: server + description
# ----------------------------------------------------------------------------
def _llamacpp_ok(timeout: float = 3) -> bool:
    import requests
    try:
        r = requests.get(f"{LLAMACPP_URL}/health", timeout=timeout)
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False


def ensure_llamacpp(log=_log_flush) -> bool:
    """True if the llama.cpp backend is ready (starting it if needed).
       False (with no exception) if not installed: Ollama will be used."""
    if _llamacpp_ok():
        return True
    if not (LLAMACPP_EXE.exists() and LLAMACPP_MODEL.exists()
            and LLAMACPP_MMPROJ.exists()):
        return False
    # free the Ollama runner RAM (best effort: 2 models = ~8 GB)
    exe = _find_ollama_exe()
    if exe and _server_ok():
        try:
            subprocess.run([exe, "stop", DEFAULT_MODEL], timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    log("  [AI] starting llama-server (fast backend) ...")
    n_threads = max(2, (os.cpu_count() or 8) // 2)   # physical cores
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    subprocess.Popen(
        [str(LLAMACPP_EXE), "-m", str(LLAMACPP_MODEL),
         "--mmproj", str(LLAMACPP_MMPROJ),
         "--host", "127.0.0.1", "--port", LLAMACPP_URL.rsplit(":", 1)[-1],
         "-c", "4096",
         "--image-min-tokens", str(IMG_MIN_TOKENS),
         "--image-max-tokens", str(IMG_MAX_TOKENS),
         "--threads", str(n_threads)],
        creationflags=flags,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):          # loading the model takes ~20-60 s
        time.sleep(1)
        if _llamacpp_ok():
            return True
    log("  [AI] llama-server does not respond: falling back to Ollama")
    return False


def describe_image_llamacpp(img_path: Path, prompt: str = PROMPT,
                            timeout: int = 300) -> str:
    """Description via llama-server (OpenAI-compatible API)."""
    import requests
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    resp = requests.post(
        f"{LLAMACPP_URL}/v1/chat/completions",
        json={
            "model": "qwen2.5vl",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
            "max_tokens": 260, "temperature": 0.1,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"^(descrizione( ia)?|risposta)\s*[:\-]\s*", "", text,
                  flags=re.IGNORECASE)
    return text.strip()


# ----------------------------------------------------------------------------
# Description of ONE image
# ----------------------------------------------------------------------------
def _prep_image_b64(img_path: Path, max_side: int = MAX_SIDE) -> str:
    """Base64 of the image, resized if larger than max_side px.
       On CPU the visual tokens (and so the time) grow with the area: a
       page rendered at 150 dpi goes from ~2500 to ~500 tokens."""
    try:
        from PIL import Image
        import io as _io
        with Image.open(img_path) as im:
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.LANCZOS)
                buf = _io.BytesIO()
                # PNG: no JPEG artefacts on text/formulas
                im.convert("RGB").save(buf, "PNG")
                return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        pass
    return base64.b64encode(img_path.read_bytes()).decode("ascii")


def describe_image(img_path: Path, model: str = DEFAULT_MODEL,
                   prompt: str = PROMPT, timeout: int = 600) -> str:
    """Returns the description (one line, Italian) of the image."""
    import requests
    b64 = _prep_image_b64(img_path)
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "keep_alive": "30m",          # do not reload the model for each image
            "options": {"temperature": 0.1, "num_predict": 260},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    text = (resp.json().get("response") or "").strip()
    # a single line: multi-line blockquotes complicate insertion
    text = re.sub(r"\s*\n\s*", " ", text)
    # remove any preamble such as "Descrizione:" at the start of the answer
    text = re.sub(r"^(descrizione( ia)?|risposta)\s*[:\-]\s*", "", text,
                  flags=re.IGNORECASE)
    return text.strip()


# ----------------------------------------------------------------------------
# Enriching a Markdown file
# ----------------------------------------------------------------------------
def _load_cache(cache_path: Path) -> dict:
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(cache_path)


def enrich_markdown(md_path: Path, model: str = DEFAULT_MODEL,
                    force: bool = False, limit: int = 0,
                    dry_run: bool = False, log=_log_flush) -> dict:
    """Inserts the AI descriptions under each image of the .md.
       Returns statistics: {total, done, cached, skipped, failed}."""
    md_path = Path(md_path)
    base = md_path.parent
    lines = md_path.read_text(encoding="utf-8").splitlines()

    # image census: (line_index, destination, resolved_path)
    jobs: list[tuple[int, str, Path]] = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith(MARKER):
            continue
        for m in _IMG_RE.finditer(line):
            dest = m.group(1).split(' "')[0].strip()
            if not dest or "://" in dest or dest.startswith("data:"):
                continue
            p = base / unquote(dest)
            if p.suffix.lower() not in IMAGE_EXT:
                continue
            jobs.append((i, dest, p))

    stats = {"total": len(jobs), "done": 0, "cached": 0,
             "skipped": 0, "failed": 0}
    if not jobs:
        log("  [AI] no local image found in the .md")
        return stats
    if dry_run:
        for i, dest, p in jobs:
            log(f"  line {i+1}: {dest}  ({'OK' if p.exists() else 'MISSING'})")
        return stats

    # backend: llama.cpp if installed (~4x faster), otherwise Ollama
    use_llamacpp = ensure_llamacpp(log=log)
    if use_llamacpp:
        log("  [IA] backend: llama.cpp "
            f"(visual tokens {IMG_MIN_TOKENS}-{IMG_MAX_TOKENS}, ~35 s/immagine)")
    else:
        ensure_server(log=log)
        ensure_model(model, log=log)
    _keep_awake(True)     # the PC must not go to sleep halfway through the work

    cache_path = md_path.with_name(md_path.stem + ".desc-cache.json")
    cache = _load_cache(cache_path)

    def _has_desc(idx: int) -> int | None:
        """If right after line idx (skipping empty lines) there is already a
           description, returns its line index, otherwise None."""
        j = idx + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and lines[j].lstrip().startswith(MARKER):
            return j
        return None

    # processing FROM THE END: this way insertions do not shift the indices
    # of the lines still to be processed
    n_calls = 0
    processed = 0
    t_start = time.time()
    for k in range(len(jobs) - 1, -1, -1):
        i, dest, p = jobs[k]
        pos = len(jobs) - k          # progression 1..N (for the GUI bar)
        existing = _has_desc(i)
        if existing is not None and not force:
            stats["skipped"] += 1
            continue
        if not p.exists() or p.stat().st_size < 300:
            stats["skipped"] += 1
            continue

        key = dest
        entry = cache.get(key)
        size = p.stat().st_size
        # NB: the cache does NOT compare the model: qwen via Ollama and qwen via
        # llama.cpp produce equivalent descriptions, no need to redo them
        if entry and entry.get("size") == size and entry.get("desc"):
            desc = entry["desc"]
            stats["cached"] += 1
            src = "cache"
        else:
            if limit and n_calls >= limit:
                stats["skipped"] += 1
                continue
            def _call():
                if use_llamacpp:
                    return describe_image_llamacpp(p)
                return describe_image(p, model=model)
            t0 = time.time()
            try:
                desc = _call()
            except Exception as e:
                # a single retry (the first model load can fail
                # with a timeout on slow machines)
                try:
                    desc = _call()
                except Exception:
                    log(f"  [AI] {pos}/{len(jobs)} ERROR on {p.name}: {e}")
                    stats["failed"] += 1
                    continue
            n_calls += 1
            if not desc:
                stats["failed"] += 1
                continue
            used = "llamacpp:qwen2.5vl-3b" if use_llamacpp else model
            cache[key] = {"desc": desc, "size": size, "model": used}
            _save_cache(cache_path, cache)
            stats["done"] += 1
            src = f"{time.time()-t0:.0f}s"

        block = f"{MARKER} {desc}"
        if existing is not None:      # --force: replace the old one
            lines[existing] = block
        else:
            lines[i+1:i+1] = ["", block]
        processed += 1
        log(f"  [AI] {pos}/{len(jobs)} ({src}) {p.name}")

        # periodic save: long job, lose nothing
        if processed % 15 == 0:
            md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _keep_awake(False)
    dt = time.time() - t_start
    log(f"  [AI] finished in {dt/60:.1f} min: {stats['done']} new, "
        f"{stats['cached']} from cache, {stats['skipped']} skipped, "
        f"{stats['failed']} failed (out of {stats['total']})")
    return stats


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    if os.name == "nt":               # UTF-8 on the Windows console (cp1252)
        for s in (sys.stdout, sys.stderr):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    ap = argparse.ArgumentParser(
        description="Adds AI descriptions (local, via Ollama) under each "
                    "image of a Markdown file.")
    ap.add_argument("md", nargs="+", help=".md file to enrich")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Ollama vision model (default {DEFAULT_MODEL})")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate existing descriptions too")
    ap.add_argument("--limit", type=int, default=0,
                    help="At most N new descriptions (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the images without calling the AI")
    args = ap.parse_args()

    for f in args.md:
        p = Path(f)
        if not p.exists():
            print(f"[KO] file not found: {p}")
            continue
        print(f">>> {p.name}")
        enrich_markdown(p, model=args.model, force=args.force,
                        limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
