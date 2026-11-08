# -*- coding: utf-8 -*-
"""
ocr_vlm.py — "Smart" OCR with LOCAL VISION models (free, offline).

Two engines, both run by llama.cpp (llama-server) on the CPU:

  * PADDLE  = PaddleOCR-VL 1.6 (Baidu, ~0.9 billion parameters)
              fast, the best on typed text, formulas and tables.
              Returns text + LaTeX (no structure/figures).
  * CHANDRA = Chandra OCR 2 (Datalab, ~5 billion parameters, 4-bit quant.)
              slower, the best open model on HANDWRITING;
              returns the structured page (headings, lists, tables,
              KaTeX formulas) and DESCRIBES the figures, which are also cropped
              and saved as images.
  * IBRIDO  = Paddle on all pages; Chandra on the pages given with
              --hq-pages and on those where Paddle fails (text too short
              for the ink, repetition loops, unreadable text).
              In addition (version 4) two CHECKERS, still local:
                - Qwen3.5-9B describes the figures (with the page text as
                  context) and rereads handwritten formula blocks;
                - GLM-4.6V-Flash (a different family) acts as REFEREE when
                  readings disagree and CHECKS the figure descriptions
                  (if it finds errors, Qwen rewrites and GLM checks again).
              Whatever stays uncertain goes into <name>_da_controllare.md.

All local: the models are downloaded ONCE into  <this folder>\\modelli
and llama.cpp is in  <this folder>\\motori\\llamacpp .

Use from tomd2:   python tomd2.py file.pdf --engine vlm_paddle
                  python tomd2.py file.pdf --engine vlm_chandra --pages 1-20
                  python tomd2.py file.pdf --engine vlm_ibrido --hq-pages 379-453
Direct use:       python ocr_vlm.py setup tutti          (downloads models + llama.cpp)
                  python ocr_vlm.py prova file.pdf --pages 12,381 --engine tutti
                    -> writes file_prova_paddle.md / file_prova_chandra.md
                       (does NOT touch existing .md files) and prints the times.

Resumable: each finished page is saved in media/<name>/_vlm_cache/, so
if you interrupt and run again it restarts from the first missing page.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths and configuration
# ----------------------------------------------------------------------------
VERSIONE   = "4.7 (22-10-2026, specialized formula readers)"
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("TOMD2_MODELS_DIR", BASE_DIR / "modelli"))
MOTORI_DIR = BASE_DIR / "motori"
LOG_DIR    = BASE_DIR / "logs"

HF = "https://huggingface.co"
IS_WIN = os.name == "nt"

# Chandra quantization: IQ4_NL (3.05 GB, recommended with 16 GB of RAM).
# Alternatives: IQ3_M (2.49 GB, faster, less accurate), Q6_K (4.08 GB).
CHANDRA_QUANT = os.environ.get("TOMD2_CHANDRA_QUANT", "IQ4_NL")
# Paddle quantization: Q8_0 (498 MB). The model is small: better not to
# go lower, accuracy on formulas would suffer.
PADDLE_QUANT = os.environ.get("TOMD2_PADDLE_QUANT", "Q8_0")
# Checkers: 4-bit (Q4_K_M), ~6 GB of RAM each, one at a time.
QWEN_QUANT = os.environ.get("TOMD2_QWEN_QUANT", "Q4_K_M")
GLM_QUANT = os.environ.get("TOMD2_GLM_QUANT", "Q4_K_M")

MODELS = {
    "paddle": {
        "nome": "PaddleOCR-VL 1.6",
        "files": [
            # (HuggingFace repo, remote file, local name)
            ("SanjeevSOLANKI/PaddleOCR-VL-1.6-GGUF",
             f"PaddleOCR-VL-1.6-{PADDLE_QUANT}.gguf", None),
            ("SanjeevSOLANKI/PaddleOCR-VL-1.6-GGUF",
             "PaddleOCR-VL-1.6-mmproj.gguf", None),
            # chat template (some llama.cpp builds crash without it)
            ("EstevaoNaval/PaddleOCR-VL-1.6-GGUF",
             "chat_template.jinja", "PaddleOCR-VL-1.6-chat_template.jinja"),
        ],
        "port": 11436,
        "ctx": 8192,
        "max_tokens": int(os.environ.get("TOMD2_PADDLE_MAXTOK", "3000")),
        "dpi": 200,
        "max_pixels": 2_000_000,      # ~1190x1680 on A4: a good CPU compromise
        "prompt": "OCR:",             # DO NOT change: it is a specialized model
        "sampling": {"temperature": 0.0, "repeat_penalty": 1.3,
                     "dry_multiplier": 0.8},
    },
    "chandra": {
        "nome": "Chandra OCR 2",
        "files": [
            ("SandLogicTechnologies/chandra-ocr-2-GGUF",
             f"chandra-ocr-2_{CHANDRA_QUANT}.gguf", None),
            ("SandLogicTechnologies/chandra-ocr-2-GGUF",
             "chandra-ocr-2.mmproj-f16.gguf", None),
        ],
        "port": 11437,
        "ctx": 16384,
        "max_tokens": int(os.environ.get("TOMD2_CHANDRA_MAXTOK", "5000")),
        "dpi": 192,                   # like the official project
        # the official one uses up to 3072x2048 (6.3 MP) on GPU; too much on CPU:
        # 3.1 MP (~1470x2090 on A4) keeps subscripts and handwriting readable.
        "max_pixels": 3_100_000,
        "prompt": None,               # -> CHANDRA_PROMPT (sotto)
        "sampling": {"temperature": 0.0, "top_p": 0.1},
    },
    # --- CHECKERS (version 4) -------------------------------------------------
    # Qwen3.5-9B (Alibaba): GENERAL-PURPOSE vision model, understands drawings.
    # Describes the figures (knowing what the page is about) and is a second
    # reader of handwritten pages.
    "qwen": {
        "nome": "Qwen3.5-9B",
        "files": [
            ("unsloth/Qwen3.5-9B-GGUF", f"Qwen3.5-9B-{QWEN_QUANT}.gguf", None),
            ("unsloth/Qwen3.5-9B-GGUF", "mmproj-F16.gguf", "Qwen3.5-9B-mmproj-F16.gguf"),
        ],
        "port": 11438,
        "ctx": 12288,
        "max_tokens": 1800,
        "dpi": 200,
        "max_pixels": 1_400_000,
        "prompt": "",
        "sampling": {"temperature": 0.2, "top_p": 0.8, "top_k": 20,
                     "repeat_penalty": 1.05},
    },
    # GLM-4.6V-Flash (Zhipu AI, 9B): a DIFFERENT family from Qwen/Chandra, so
    # it makes mistakes in different places. Acts as REFEREE when readings disagree
    # and CHECKS the figure descriptions written by Qwen.
    "glm": {
        "nome": "GLM-4.6V-Flash",
        "files": [
            ("ggml-org/GLM-4.6V-Flash-GGUF", f"GLM-4.6V-Flash-{GLM_QUANT}.gguf", None),
            ("ggml-org/GLM-4.6V-Flash-GGUF", "mmproj-GLM-4.6V-Flash-Q8_0.gguf", None),
        ],
        "port": 11439,
        "ctx": 12288,
        "max_tokens": 2500,
        "dpi": 200,
        "max_pixels": 1_400_000,
        "prompt": "",
        "sampling": {"temperature": 0.1, "top_p": 0.8},
    },
}

# Official Chandra OCR 2 prompt (datalab-to/chandra, chandra/prompts.py,
# OCR_LAYOUT_PROMPT): page -> block HTML with bboxes and labels.
_ALLOWED_TAGS = ["math", "br", "i", "b", "u", "del", "sup", "sub", "table",
                 "tr", "td", "p", "th", "div", "pre", "h1", "h2", "h3", "h4",
                 "h5", "ul", "ol", "li", "input", "a", "span", "img", "hr",
                 "tbody", "small", "caption", "strong", "thead", "big", "code",
                 "chem"]
_ALLOWED_ATTRS = ["class", "colspan", "rowspan", "display", "checked", "type",
                  "border", "value", "style", "href", "alt", "align",
                  "data-bbox", "data-label"]
_PROMPT_ENDING = f"""
Only use these tags {_ALLOWED_TAGS}, and these attributes {_ALLOWED_ATTRS}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()
CHANDRA_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page

{_PROMPT_ENDING}
""".strip()


CHANDRA_LINGUA = os.environ.get("TOMD2_CHANDRA_LINGUA", "it").lower()
if CHANDRA_LINGUA.startswith("it"):
    CHANDRA_PROMPT += ("\n* Language: the document is in Italian. Write every description of "
                       "images, figures, charts and diagrams (alt text and the text inside the "
                       "div) in Italian, and keep all symbols and labels exactly as written.")


def _log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------
# Download (with resume) of models and llama.cpp
# ----------------------------------------------------------------------------
def _download(url: str, dest: Path, label: str = "") -> Path:
    """Downloads url -> dest with RESUME (.part file) and an on-screen percentage."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "tomd2-ocr-vlm/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:          # already complete
            part.rename(dest)
            return dest
        raise RuntimeError(f"download failed ({e.code}) for {url}") from e
    total = resp.headers.get("Content-Length")
    total = int(total) + have if total else None
    if have and resp.status != 206:         # the server ignores Range: restarting
        have = 0
        mode = "wb"
    else:
        mode = "ab" if have else "wb"
    name = label or dest.name
    done = have
    last = -1
    t0 = time.time()
    with open(part, mode) as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = int(done * 100 / total)
                if pct != last:
                    speed = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                    _log(f"  Download {name}: {pct}% "
                         f"({done/1e6:.0f}/{total/1e6:.0f} MB, {speed:.1f} MB/s)")
                    last = pct
    if total and done < total:
        raise RuntimeError(f"download interrupted ({done}/{total} bytes): run it again, it resumes from there")
    part.rename(dest)
    return dest


def model_paths(key: str) -> dict:
    spec = MODELS[key]
    out = {}
    for repo, remote, local in spec["files"]:
        p = MODELS_DIR / (local or remote)
        if remote.endswith(".jinja"):
            out["template"] = p
        elif "mmproj" in remote:
            out["mmproj"] = p
        else:
            out["model"] = p
    return out


def ensure_model(key: str) -> dict:
    spec = MODELS[key]
    for repo, remote, local in spec["files"]:
        dest = MODELS_DIR / (local or remote)
        if dest.exists() and dest.stat().st_size > 0:
            continue
        _log(f"Downloading {spec['nome']}: {remote} (one time) ...")
        url = f"{HF}/{repo}/resolve/main/{remote}?download=true"
        try:
            _download(url, dest, label=remote)
        except Exception as e:
            if remote.endswith(".jinja"):   # the template is optional
                _log(f"  [warning] template not downloaded ({e}): continuing without it.")
                continue
            raise
    return model_paths(key)


def _server_exe_candidates(gpu: bool) -> list[Path]:
    exe = "llama-server.exe" if IS_WIN else "llama-server"
    c = []
    env = os.environ.get("TOMD2_LLAMA_SERVER")
    if env:
        c.append(Path(env))
    if gpu:                      # the GPU needs the Vulkan build, not the CPU one
        c.append(MOTORI_DIR / "llamacpp-vulkan" / exe)
        return c
    c.append(MOTORI_DIR / "llamacpp" / exe)
    if IS_WIN:   # installation already on the PC (used by img_describe)
        c.append(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "llamacpp" / exe)
    w = shutil.which("llama-server")
    if w:
        c.append(Path(w))
    return c


def ensure_llamacpp(gpu: bool = False) -> Path:
    """Returns the path of llama-server; if missing, downloads the latest
       Windows build from GitHub (CPU, or Vulkan for Intel graphics if gpu=True)."""
    for p in _server_exe_candidates(gpu):
        if p.exists():
            return p
    if not IS_WIN:
        raise RuntimeError("llama-server not found (install llama.cpp or set TOMD2_LLAMA_SERVER).")
    flavor = "vulkan-x64" if gpu else "cpu-x64"
    target = MOTORI_DIR / ("llamacpp-vulkan" if gpu else "llamacpp")
    _log(f"Downloading llama.cpp ({flavor}) from GitHub ...")
    with urllib.request.urlopen(urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
            headers={"User-Agent": "tomd2"}), timeout=60) as r:
        rel = json.load(r)
    asset = next((a for a in rel["assets"]
                  if a["name"].endswith(f"bin-win-{flavor}.zip")), None)
    if not asset:
        raise RuntimeError(f"build llama.cpp '{flavor}' not found in the release {rel.get('tag_name')}")
    zpath = MOTORI_DIR / asset["name"]
    _download(asset["browser_download_url"], zpath)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(target)
    (target / "VERSION.txt").write_text(rel.get("tag_name", "?"), encoding="utf-8")
    exe = target / "llama-server.exe"
    if not exe.exists():
        raise RuntimeError("llama-server.exe not found in the downloaded zip")
    return exe


# ----------------------------------------------------------------------------
# llama.cpp server (automatic start/stop)
# ----------------------------------------------------------------------------
def _http_json(url: str, payload: dict | None = None, timeout: float = 30):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class LlamaServer:
    """with LlamaServer('chandra') as srv: srv.ocr(img) ...
       Starts llama-server with the right model, waits until it is ready, and
       closes it on exit. The log goes to logs/llama_server_<engine>.log."""

    def __init__(self, key: str, threads: int | None = None, gpu: bool = False):
        self.key = key
        self.spec = MODELS[key]
        self.paths = ensure_model(key)
        self.exe = ensure_llamacpp(gpu)
        self.gpu = gpu
        self.threads = threads or max(1, (os.cpu_count() or 4) // 2)  # physical cores
        self.port = int(os.environ.get(f"TOMD2_{key.upper()}_PORT", self.spec["port"]))
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc = None
        self.logf = None

    def _cmd(self, use_template: bool) -> list[str]:
        cmd = [str(self.exe),
               "-m", str(self.paths["model"]),
               "--mmproj", str(self.paths["mmproj"]),
               "--host", "127.0.0.1", "--port", str(self.port),
               "-c", str(self.spec["ctx"]),
               "-t", str(self.threads),
               "-np", "1"]
        if self.gpu:
            cmd += ["-ngl", "99"]
        tpl = self.paths.get("template")
        if use_template and tpl and tpl.exists():
            cmd += ["--jinja", "--chat-template-file", str(tpl)]
        else:
            cmd += ["--jinja"]
        return cmd

    def _healthy(self) -> bool:
        try:
            r = _http_json(self.url + "/health", timeout=5)
            return r.get("status") == "ok"
        except Exception:
            return False

    def start(self):
        if self._healthy():           # already running (e.g. started by hand)
            return self
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        flags = 0x08000000 if IS_WIN else 0   # CREATE_NO_WINDOW
        for use_template in (True, False):
            self.logf = open(LOG_DIR / f"llama_server_{self.key}.log", "w",
                             encoding="utf-8", errors="replace")
            cmd = self._cmd(use_template)
            self.logf.write("CMD: " + " ".join(cmd) + "\n\n")
            self.logf.flush()
            _log(f"  Starting {self.spec['nome']} (llama.cpp, {self.threads} thread"
                 f"{', Vulkan GPU' if self.gpu else ''}) ...")
            self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT,
                                         creationflags=flags)
            t0 = time.time()
            while time.time() - t0 < 600:      # loading from disk: up to 10 min
                if self.proc.poll() is not None:
                    break
                if self._healthy():
                    _log(f"  {self.spec['nome']} ready in {time.time()-t0:.0f} s")
                    return self
                time.sleep(2)
            self.stop()
            if not use_template:
                break
            _log("  [warning] start-up failed with the chat template: retrying without.")
        raise RuntimeError(f"llama-server does not start for {self.spec['nome']}: "
                           f"see {LOG_DIR / f'llama_server_{self.key}.log'} "
                           "(often: llama.cpp too old or not enough RAM)")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()
        self.proc = None
        if self.logf:
            self.logf.close()
            self.logf = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    # -- OCR request -----------------------------------------------------------
    def ocr(self, img, prompt: str | None = None, max_tokens: int | None = None,
            temperature: float | None = None, expected_tokens: float | None = None,
            label: str = "", sampling: dict | None = None) -> tuple[str, dict]:
        """Sends the page to the model in STREAMING mode: shows the progress every
           ~10 s (tokens, speed, estimated %, last line) and stops the generation
           IMMEDIATELY if the model enters a loop."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        limit = max_tokens or self.spec["max_tokens"]
        testo = prompt or self.spec.get("prompt") or CHANDRA_PROMPT
        if self.key == "glm" and "/nothink" not in testo:
            testo += "\n/nothink"          # GLM: no long reasoning
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": testo},
            ]}],
            "max_tokens": limit,
            "stream": True,
            "stream_options": {"include_usage": True},
            # no hidden "reasoning" (Qwen3.5-type models): output only
            "chat_template_kwargs": {"enable_thinking": False},
        }
        payload.update(self.spec["sampling"])
        if sampling:
            payload.update(sampling)
        if temperature is not None:
            payload["temperature"] = temperature
        req = urllib.request.Request(self.url + "/v1/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        parts, think = [], []
        ntok = 0
        usage = {}
        t0 = time.time()
        t_first = None
        last_print = t0
        stopped = None
        with urllib.request.urlopen(req, timeout=4 * 3600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except Exception:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        parts.append(d["content"]); ntok += 1
                    if d.get("reasoning_content"):
                        think.append(d["reasoning_content"]); ntok += 1
                if ntok and t_first is None:
                    t_first = time.time()
                    _log(f"    {label} image read in {t_first - t0:.0f} s, writing the text...")
                now = time.time()
                if now - last_print >= 10 and ntok:
                    last_print = now
                    speed = ntok / max(now - (t_first or t0), 1e-6)
                    pct = ""
                    if expected_tokens:
                        pct = f" (~{min(99, int(ntok * 100 / expected_tokens))}% estimated)"
                    tail = "".join(parts)[-70:].replace("\n", " ").strip()
                    _log(f"    {label} {ntok} token{pct} | {speed:.1f} t/s | ...{tail}")
                # loop check every ~100 tokens: if it repeats, stop right away
                if ntok % 100 == 0 and ntok >= 300 and detect_repeat("".join(parts)):
                    stopped = "loop"
                    _log(f"    {label} [LOOP] the model is repeating itself: stopping at {ntok} token")
                    break
        text = "".join(parts)
        if not text.strip() and think:          # model that wrote only in the "reasoning"
            text = "".join(think)
        usage = dict(usage or {})
        usage.setdefault("completion_tokens", ntok)
        usage["stopped"] = stopped or ("limite" if ntok >= limit else None)
        return text, usage


# ----------------------------------------------------------------------------
# Page images
# ----------------------------------------------------------------------------
def scale_to_fit(img, max_pixels: int, min_pixels: int = 1792 * 28, grid: int = 28):
    """Resizes keeping the proportions, multiples of 28 px (vision encoder
       patch), within max_pixels. Ported from chandra/model/util.py."""
    from PIL import Image
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    ar = w / h
    px = w * h
    scale = 1.0
    if px > max_pixels:
        scale = (max_pixels / px) ** 0.5
    elif px < min_pixels:
        scale = (min_pixels / px) ** 0.5
    wb = max(1, round(w * scale / grid))
    hb = max(1, round(h * scale / grid))
    while wb * hb * grid * grid > max_pixels and (wb > 1 or hb > 1):
        if wb == 1:
            hb -= 1; continue
        if hb == 1:
            wb -= 1; continue
        if abs((wb - 1) / hb - ar) < abs(wb / (hb - 1) - ar):
            wb -= 1
        else:
            hb -= 1
    nw, nh = wb * grid, hb * grid
    if (nw, nh) == (w, h):
        return img
    return img.resize((nw, nh), Image.LANCZOS)


def render_page(doc, i: int, dpi: int):
    import fitz
    from PIL import Image
    page = doc[i]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def ink_ratio(img) -> float:
    """Fraction of 'dark' pixels (text/drawings) in the central part of the
       page: used to tell whether an empty page is really empty."""
    g = img.convert("L")
    w, h = g.size
    g = g.crop((int(w * .05), int(h * .06), int(w * .95), int(h * .94)))
    g.thumbnail((400, 600))
    hist = g.histogram()
    dark = sum(hist[:140])
    return dark / max(1, sum(hist))


# ----------------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------------
def detect_repeat(text: str, window: int = 500, base_max: int = 4,
                  scaling: float = 3.0) -> bool:
    """True if the text ends with a sequence repeated too many times
       (model loop). Simplified port from chandra/model/util.py."""
    t = text[-4000:]
    for L in range(1, window // 2 + 1):
        cand = t[-L:]
        if not cand.strip():
            continue
        max_rep = int(base_max + scaling * (1 / L) ** 0.5 * 10) if L < 10 else base_max
        reps = 0
        pos = len(t)
        while pos - L >= 0 and t[pos - L:pos] == cand:
            reps += 1
            pos -= L
        if reps > max_rep:
            return True
    # consecutive identical lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    run = 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b else 1
        if run >= 8:
            return True
    return False


def trim_repeats(text: str) -> str:
    """Cuts identical lines repeated beyond the 3rd consecutive time."""
    out, prev, run = [], None, 0
    for ln in text.splitlines():
        run = run + 1 if ln.strip() and ln == prev else 1
        prev = ln
        if run <= 3:
            out.append(ln)
    return "\n".join(out)


def normalize_math(md: str) -> str:
    md = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n$$\n" + m.group(1).strip() + "\n$$\n", md, flags=re.S)
    md = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", md, flags=re.S)
    return md


def _n_left(f: str) -> int:      # \left( ...  but NOT \leftarrow
    return len(re.findall(r"\\left(?![A-Za-z])", f))


def _n_right(f: str) -> int:     # \right) ... but NOT \rightarrow
    return len(re.findall(r"\\right(?![A-Za-z])", f))


def math_issues(md: str) -> int:
    """Counts formulas with unbalanced braces or unmatched \\left/\\right
       (a light check, without KaTeX)."""
    n = 0
    formulas = re.findall(r"\$\$(.+?)\$\$", md, flags=re.S)
    rest = re.sub(r"\$\$(.+?)\$\$", " ", md, flags=re.S)
    formulas += re.findall(r"(?<![\\$])\$([^$\n]+?)\$", rest)
    for f in formulas:
        f2 = f.replace(r"\{", "").replace(r"\}", "")
        if f2.count("{") != f2.count("}") or _n_left(f) != _n_right(f):
            n += 1
    if (rest.count("$") - rest.count(r"\$")) % 2:
        n += 1
    return n


def _md_escape_fallback(html: str) -> str:
    """Minimal HTML->Markdown conversion (if bs4/markdownify are missing)."""
    s = html
    s = re.sub(r"<math[^>]*display=\"?block\"?[^>]*>(.*?)</math>", r"\n$$\n\1\n$$\n", s, flags=re.S)
    s = re.sub(r"<math[^>]*>(.*?)</math>", r"$\1$", s, flags=re.S)
    for h in range(1, 6):
        s = re.sub(rf"<h{h}[^>]*>(.*?)</h{h}>", "\n" + "#" * h + r" \1\n", s, flags=re.S)
    s = re.sub(r"<li[^>]*>", "\n- ", s)
    s = re.sub(r"<(b|strong)>(.*?)</\1>", r"**\2**", s, flags=re.S)
    s = re.sub(r"<(i)>(.*?)</\1>", r"*\2*", s, flags=re.S)
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</p>|</div>|</tr>", "\n\n", s)
    s = re.sub(r"<(?!/?(table|tr|td|th|tbody|thead|sub|sup|u)\b)[^>]+>", "", s)
    import html as _h
    return _h.unescape(re.sub(r"\n{3,}", "\n\n", s)).strip()


def _frag_to_md(fragment: str) -> str:
    """HTML fragment (Chandra style) -> Markdown, <math> formulas intact."""
    from bs4 import BeautifulSoup
    from markdownify import MarkdownConverter

    class _MD(MarkdownConverter):
        def convert_table(self, el, text, *a, **k):
            return "\n\n" + str(el) + "\n\n"     # complex tables: HTML intact

    frag = BeautifulSoup(fragment, "html.parser")
    stash = []
    for m in frag.find_all("math"):
        tex = m.get_text().strip()
        if m.get("display") == "block":
            stash.append("\n\n$$\n" + tex + "\n$$\n\n")
        else:
            stash.append("$" + tex + "$")
        m.replace_with(f"XMATHX{len(stash)-1}XMATHX")
    md = _MD(heading_style="ATX", bullets="-", escape_misc=False,
             escape_underscores=False, escape_asterisks=False,
             sub_symbol="<sub>", sup_symbol="<sup>").convert(str(frag))
    md = re.sub(r"XMATHX(\d+)XMATHX", lambda mm: stash[int(mm.group(1))], md)
    md = fix_list_markers(md)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def chandra_describe(raw: str) -> str:
    """Chandra text on a figure CROP -> description in Markdown
       (alt + block content; the text blocks inside the figure,
       e.g. labels, are kept)."""
    raw = _strip_fences(raw)
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        out = []
        for div in soup.find_all("div", recursive=False) or [soup]:
            im = div.find("img")
            if im is not None:
                alt = (im.get("alt") or "").strip()
                im.decompose()
                if alt:
                    out.append(alt)
            body = _frag_to_md(str(div.decode_contents()) if div is not soup else str(div))
            if body.strip():
                out.append(body)
        txt = "\n\n".join(out).strip()
        return txt or _md_escape_fallback(raw)
    except Exception:
        return _md_escape_fallback(raw)


def dedup_frasi(text: str) -> str:
    """Cuts the text at the first sentence the model repeats in a loop."""
    frasi = re.split(r"(?<=[.!?;])\s+", text)
    out, viste = [], set()
    for f in frasi:
        k = f.strip().lower()
        if k and k in viste:
            break
        viste.add(k)
        out.append(f)
    return " ".join(out).strip()


def fix_list_markers(md: str) -> str:
    """'- - text', '- o text', '- • text' -> '- text' (the model also copies
       the original dash/bullet into the list item)."""
    return re.sub(r"(?m)^([ \t]*)[-*][ \t]+(?:[-–•◦▪○o]|\\-)[ \t]+(?=\S)", r"\1- ", md)


def chandra_html_to_md(html: str, img, page_no: int, fig_dir: Path,
                       fig_prefix: str, blocchi: list | None = None) -> str:
    """Chandra block HTML -> Markdown in the tomd2 style:
       - skips headers/footers (Page-Header/Page-Footer)
       - figures: crop saved as PNG + description block
       - <math> formulas -> $...$ / $$...$$ (KaTeX)"""
    try:
        from bs4 import BeautifulSoup
        from markdownify import MarkdownConverter
    except ImportError:
        return _md_escape_fallback(html)

    to_md = _frag_to_md

    soup = BeautifulSoup(html, "html.parser")
    divs = soup.find_all("div", recursive=False)
    if not divs:
        return to_md(html)
    w, h = img.size
    parts, nfig = [], 0
    for div in divs:
        label = (div.get("data-label") or "").strip()
        if label in ("Blank-Page", "Page-Header", "Page-Footer"):
            continue
        if label in ("Image", "Figure", "Diagram", "Chemical-Block"):
            nfig += 1
            alt = ""
            im = div.find("img")
            if im is not None:
                alt = (im.get("alt") or "").strip()
                im.decompose()
            desc = to_md(str(div.decode_contents()))
            link = ""
            try:
                x0, y0, x1, y1 = [int(v) for v in (div.get("data-bbox") or "").split()]
                box = (max(0, x0 * w // 1000), max(0, y0 * h // 1000),
                       min(w, x1 * w // 1000), min(h, y1 * h // 1000))
                if box[2] - box[0] > 10 and box[3] - box[1] > 10:
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    name = f"p{page_no:04d}_fig{nfig}.png"
                    img.crop(box).save(fig_dir / name)
                    link = f"![Figura pag. {page_no}]({fig_prefix}/{name})\n\n"
            except Exception:
                pass
            block = [f"> **[Figura — pag. {page_no}]** {alt}".rstrip()]
            for ln in desc.splitlines():
                block.append("> " + ln if ln.strip() else ">")
            parts.append(link + "\n".join(block))
            continue
        body = to_md(str(div.decode_contents()))
        if blocchi is not None and div.find("math") is not None and body.strip():
            # block with formulas: placeholder, it will be verified (Qwen + referee)
            try:
                bb = [int(v) for v in (div.get("data-bbox") or "").split()][:4]
            except Exception:
                bb = []
            if len(bb) == 4 and bb[2] > bb[0] and bb[3] > bb[1]:
                bid = f"p{page_no:04d}_b{len(blocchi) + 1:02d}"
                blocchi.append({"id": bid, "bbox": bb, "label": label,
                                "chandra": normalize_math(body).strip()})
                parts.append("{{BLOCCO:" + bid + "}}")
                continue
        parts.append(body)
    md = "\n\n".join(p for p in parts if p.strip())
    return re.sub(r"\n{3,}", "\n\n", md).strip()


# ----------------------------------------------------------------------------
# Page selection
# ----------------------------------------------------------------------------
def parse_pages(spec: str | None, n: int) -> list[int]:
    """'1-10,15,379-383' -> [1..10, 15, 379..383] (1-based, within n)."""
    if not spec:
        return list(range(1, n + 1))
    out = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a) if a else 1
            b = int(b) if b else n
            out.update(range(max(1, a), min(n, b) + 1))
        else:
            k = int(part)
            if 1 <= k <= n:
                out.add(k)
    return sorted(out)


# ----------------------------------------------------------------------------
# Converting a PDF / image
# ----------------------------------------------------------------------------
def _cache_file(cache_dir: Path, key: str, page: int) -> Path:
    return cache_dir / key / f"page_{page:04d}.md"


def _strip_fences(raw: str) -> str:
    m = re.match(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", raw, flags=re.S)
    return m.group(1) if m else raw


def _ocr_one(srv: LlamaServer, key: str, page_img, page_no: int,
             fig_dir: Path, fig_prefix: str, expected_tokens: float | None = None,
             raw_file: Path | None = None) -> tuple[str, dict]:
    spec = MODELS[key]
    img = scale_to_fit(page_img, spec["max_pixels"])
    t0 = time.time()
    label = f"[{key} p.{page_no}]"
    raw, usage = srv.ocr(img, expected_tokens=expected_tokens, label=label)
    retried = False
    if usage.get("stopped") == "loop":     # loop: second "shifted" attempt
        retried = True
        _log(f"    {label} second attempt...")
        raw2, usage2 = srv.ocr(img, temperature=0.3, expected_tokens=expected_tokens,
                               label=label)
        if usage2.get("stopped") != "loop" or len(raw2) > len(raw):
            raw, usage = raw2, usage2
    if raw_file is not None:               # always: the raw text of the model
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        raw_file.write_text(raw, encoding="utf-8")
    raw_clean = _strip_fences(raw)
    md = ""
    if key == "chandra":
        try:
            md = chandra_html_to_md(raw_clean, img, page_no, fig_dir, fig_prefix)
        except Exception as e:
            _log(f"    {label} [warning] HTML conversion failed ({e}): using the fallback")
        if not md.strip() and raw_clean.strip():
            md = _md_escape_fallback(raw_clean)      # truncated or abnormal HTML
    else:
        md = raw_clean
    md = normalize_math(trim_repeats(md)).strip()
    if not md and raw_clean.strip():
        md = raw_clean.strip()                       # never lose the text that was read
    meta = {"secondi": round(time.time() - t0, 1),
            "token_out": usage.get("completion_tokens"),
            "token_in": usage.get("prompt_tokens"),
            "fermato": usage.get("stopped"),
            "loop": usage.get("stopped") == "loop", "ritentato": retried}
    return md, meta


# ----------------------------------------------------------------------------
# "Full" PaddleOCR-VL (official Baidu pipeline): a small LAYOUT
# model (PP-DocLayoutV3, runs on CPU) splits the page into blocks (headings,
# paragraphs, formulas, tables, figures, notes) and sends EACH block to the
# PaddleOCR-VL model on our llama-server with the right prompt ("OCR:",
# "Formula Recognition:", "Table Recognition:"). This is how Paddle was
# designed and benchmarked: much better than the whole page.
# Richiede:  pip install paddlepaddle "paddleocr[doc-parser]"   (setup layout)
# If not installed, it falls back to the whole page with "OCR:".
# ----------------------------------------------------------------------------
_LAYOUT_ERR = None
_LAYOUT_OK = None


def _import_paddle_stack():
    """On Windows, if PyTorch is also installed, it must be imported BEFORE paddle:
       otherwise paddle's DLLs 'steal' torch's and the import of
       paddleocr (which loads torch through modelscope) fails with
       'WinError 127 ... shm.dll'."""
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    import paddle  # noqa: F401
    import paddleocr  # noqa: F401


def paddle_layout_disponibile() -> bool:
    global _LAYOUT_ERR, _LAYOUT_OK
    if os.environ.get("TOMD2_PADDLE_LAYOUT", "1") == "0":
        _LAYOUT_ERR = "disabled with TOMD2_PADDLE_LAYOUT=0"
        return False
    if _LAYOUT_OK is None:
        try:
            _import_paddle_stack()
            _LAYOUT_OK = True
        except Exception as e:
            _LAYOUT_OK = False
            _LAYOUT_ERR = f"{type(e).__name__}: {e}"
    return _LAYOUT_OK


class PaddleLayout:
    def __init__(self, server_url: str):
        _import_paddle_stack()
        from paddleocr import PaddleOCRVL
        _log("  Loading the PP-DocLayoutV3 layout model (downloaded the first time) ...")
        self.pipe = PaddleOCRVL(
            pipeline_version="v1.6",
            vl_rec_backend="llama-cpp-server",
            vl_rec_server_url=server_url.rstrip("/") + "/v1",
            vl_rec_max_concurrency=1,          # one block at a time: CPU
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_chart_recognition=False,
            # remove headers/footers/numbers, but KEEP the notes
            markdown_ignore_labels=["number", "header", "header_image",
                                    "footer", "footer_image", "aside_text"],
        )

    def page_md(self, img, page_no: int, fig_dir: Path, fig_prefix: str):
        """Returns (markdown, [cropped figure file names], [formulas]).
           Each display formula becomes a {{FORMULA:id}} placeholder in the
           text: the final version is chosen after verification
           (high-resolution reread + Chandra + vote)."""
        import tempfile
        formule = []
        with tempfile.TemporaryDirectory() as td:
            ip = Path(td) / f"p{page_no:04d}.png"
            img.save(ip)
            parts = []
            for res in self.pipe.predict(str(ip)):
                try:
                    blocks = list(res["parsing_res_list"])
                except Exception:
                    blocks = []
                for bi, b in enumerate(blocks):
                    lab = getattr(b, "label", "") or ""
                    cont = (getattr(b, "content", "") or "").strip()
                    if "formula" not in lab or lab == "formula_number" or not cont:
                        continue
                    num = None
                    if bi + 1 < len(blocks) and getattr(blocks[bi + 1], "label", "") == "formula_number":
                        num = tex_core(getattr(blocks[bi + 1], "content", "") or "").strip() or None
                    fid = f"p{page_no:04d}_f{len(formule) + 1:02d}"
                    try:
                        bbox = [float(v) for v in list(b.bbox)[:4]]
                    except Exception:
                        bbox = None
                    formule.append({"id": fid, "label": lab, "bbox": bbox, "numero": num,
                                    "display": lab != "inline_formula",
                                    "paddle": tex_core(cont)})
                    b.content = "{{FORMULA:" + fid + "}}"
                mdd = res.markdown
                text = mdd.get("markdown_texts", "") or ""
                for rel, pil in (mdd.get("markdown_images") or {}).items():
                    if rel not in text:          # e.g. logo in the footer: discarded
                        continue
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    name = f"p{page_no:04d}_" + Path(rel).name
                    try:
                        pil.save(fig_dir / name)
                    except Exception:
                        continue
                    text = text.replace(rel, f"{fig_prefix}/{name}")
                parts.append(text)
        md = "\n\n".join(parts)
        figs = []

        def _img(m):
            src = m.group(1)
            name = Path(src).name
            figs.append(name)
            return (f"\n\n![Figura pag. {page_no}]({src})\n\n"
                    f"> **[Figura — pag. {page_no}]** {{{{DESCR:{name}}}}}\n\n")
        md = re.sub(r"<div[^>]*>\s*(<img[^>]*>)\s*</div>", r"\1", md)
        md = re.sub(r"<img[^>]*?src=\"([^\"]+)\"[^>]*>", _img, md)
        # formula placeholders: always on a line of their own (any surrounding $ removed)
        md = re.sub(r"\${0,2}\s*\{\{FORMULA:([^}]+)\}\}\s*\${0,2}",
                    lambda m: "\n\n{{FORMULA:" + m.group(1) + "}}\n\n", md)
        inline_ids = {f["id"] for f in formule if not f["display"]}
        if inline_ids:   # inline formulas stay in the paragraph
            md = re.sub(r"\n\n\{\{FORMULA:([^}]+)\}\}\n\n",
                        lambda m: (" {{FORMULA:" + m.group(1) + "}} ") if m.group(1) in inline_ids
                        else m.group(0), md)
        return clean_md(md), figs, formule


_CIRCLED = {i: chr(0x2460 + i - 1) for i in range(1, 21)}


def clean_md(md: str) -> str:
    """General touch-ups: inline formulas with no inner spaces ($ x $ -> $x$),
       \\textcircled{n} (not supported by KaTeX) -> ①, multiple empty lines."""
    md = normalize_math(md)
    md = re.sub(r"\\textcircled\{\s*(\d+)\s*\}",
                lambda m: _CIRCLED.get(int(m.group(1)), f"({m.group(1)})"), md)
    md = re.sub(r"\$\$\s*(.+?)\s*\$\$", lambda m: "\n$$\n" + m.group(1).strip() + "\n$$\n",
                md, flags=re.S)
    # $...$ pairs taken in order from the left (never "between" two formulas)
    md = re.sub(r"(?<!\$)\$(?!\$)([^$\n]*?)(?<!\$)\$(?!\$)",
                lambda m: "$" + m.group(1).strip() + "$", md)
    md = fix_list_markers(md)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


# ----------------------------------------------------------------------------
# Recognizing a page that is "unreadable" for Paddle (typically handwritten):
# words that do not exist in the document's language and characters of
# foreign alphabets (Cyrillic, Chinese...) the model invents when it does not understand.
# Uses 'wordfreq' (Italian, English, French, Spanish, German) if present.
# ----------------------------------------------------------------------------
_LINGUE = ["it", "en", "fr", "es", "de"]


def leggibilita(md: str) -> dict:
    testo = re.sub(r"\$\$.+?\$\$", " ", md, flags=re.S)
    testo = re.sub(r"\$[^$\n]+\$", " ", testo)
    testo = re.sub(r"<[^>]+>|!\[[^\]]*\]\([^)]*\)|\{\{(DESCR|FORMULA):[^}]+\}\}", " ", testo)
    lettere = [c for c in testo if c.isalpha()]
    estranei = [c for c in lettere if not ("a" <= c.lower() <= "z" or "\u00c0" <= c <= "\u024f")]
    frac_estranei = len(estranei) / max(1, len(lettere))
    # long words only (>= 5 letters): short ones nearly always exist
    parole = [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]{5,}", testo)]
    info = {"parole": len(parole), "estranei": round(frac_estranei, 3), "note": None}
    try:
        from wordfreq import zipf_frequency
        if parole:
            best = None
            for lg in _LINGUE:
                ok = sum(1 for w in parole if zipf_frequency(w, lg) >= 1.5) / len(parole)
                if best is None or ok > best[1]:
                    best = (lg, ok)
            info["lingua"], info["parole_note"] = best[0], round(best[1], 2)
    except Exception:
        info["note"] = "wordfreq not installed"
    illeg = (frac_estranei > 0.01
             or (info.get("parole_note") is not None and info["parole"] >= 8
                 and info["parole_note"] < 0.90))
    info["illeggibile"] = bool(illeg)
    return info


# header/footer lines to remove (Paddle transcribes them)
_PAGE_NUM_RE = re.compile(r"^\s*(pag(\.|ina)?\s*)?\d+\s*(a|di|/|of)\s*\d+\s*$", re.I)


def strip_running_lines(pages_md: dict[int, str]) -> dict[int, str]:
    """Removes headers/footers: IDENTICAL lines appearing in
       the first/last 2 lines of at least half the pages (minimum 3), and
       lines such as 'Pag. 12 a 506'. Never touches the rest of the text."""
    from collections import Counter
    def norm(l):
        return re.sub(r"\s+", " ", l.strip().lower())
    def edges(md):
        # never figures (>), formulas ($), images, tables or comments
        idx = [i for i, l in enumerate(md.splitlines())
               if l.strip() and not l.lstrip().startswith((">", "$", "!", "|", "<!--", "\\"))]
        return set(idx[:2] + idx[-2:]) if len(idx) > 4 else set()
    cnt = Counter()
    for md in pages_md.values():
        lines = md.splitlines()
        for l in {norm(lines[i]) for i in edges(md)}:
            cnt[l] += 1
    n = len(pages_md)
    rep_lines = {k for k, v in cnt.items()
                 if v >= max(3, 0.5 * n) and 0 < len(k) < 120}
    out = {}
    for p, md in pages_md.items():
        lines = md.splitlines()
        e = edges(md)
        keep = [l for i, l in enumerate(lines)
                if not (_PAGE_NUM_RE.match(l) or (i in e and norm(l) in rep_lines))]
        out[p] = "\n".join(keep).strip()
    return out


# ----------------------------------------------------------------------------
# FORMULA VERIFICATION (pages done by Paddle)
# Each display formula is read THREE times, by different readers:
#   1) paddle     : the pipeline's reading (page at 200 dpi)
#   2) paddle_hr  : Paddle again, on the crop at 300 dpi (sharper details)
#   3) chandra    : Chandra OCR 2 on the crop at 300 dpi (a different model)
# The readings are compared "by content" (spaces, \left/\right,
# \dfrac/\frac, \text{...} etc. do not count). If Paddle and Chandra
# agree the formula is CONFIRMED; otherwise the majority wins; if there is no
# majority the formula is DOUBTFUL: a comment with the alternative readings
# stays in the .md and it goes into the <name>_da_controllare.md file, with the
# image crop and a space for the correction (which the program
# applies by itself on the next run, without redoing the OCR).
#   TOMD2_VERIFICA_FORMULE = tutte (default) | dubbie (Chandra only where the
#   two Paddle readings differ) | paddle (no Chandra) | 0 (off)
# ----------------------------------------------------------------------------
VERIFICA_FORMULE = os.environ.get("TOMD2_VERIFICA_FORMULE", "tutte").strip().lower()
FORMULA_DPI = int(os.environ.get("TOMD2_FORMULA_DPI", "300"))
LAYOUT_CACHE_V = 2            # version of the cached data for layout mode
CHANDRA_OCR_PROMPT = "OCR this image to HTML.\n\n" + _PROMPT_ENDING


_SINGOLI_RE = re.compile(r"(?<![\\A-Za-z0-9])([A-Za-z0-9](?: [A-Za-z0-9])+)(?![A-Za-z0-9])")


def _graffe_doppie(s: str) -> str:
    """'{ { X } }' -> '{X}' (redundant braces, also nested)."""
    for _ in range(20):
        # brace pairs: opening position -> closing position
        st, match = [], {}
        for i, c in enumerate(s):
            if c == "{" and (i == 0 or s[i - 1] != "\\"):
                st.append(i)
            elif c == "}" and (i == 0 or s[i - 1] != "\\") and st:
                match[st.pop()] = i
        togli = set()
        for o, c in match.items():
            j = o + 1
            while j < len(s) and s[j] == " ":
                j += 1
            if j in match:
                k = match[j] + 1
                while k < len(s) and s[k] == " ":
                    k += 1
                if k == c:            # { {X} } : the outer pair is useless
                    togli.update((o, c))
        if not togli:
            break
        s = "".join(ch for i, ch in enumerate(s) if i not in togli)
    return s


def compatta_tex(s: str) -> str:
    """'Spaced' LaTeX (im2latex style, typical of Qwen):
       '\\frac { 1 } { e _ { 0 } }' -> '\\frac{1}{e_{0}}',
       '\\operatorname* { d e t }' -> '\\det', '{ { { C } } }' -> '{C}'.
       Text inside \\text{...} is not touched."""
    if not s:
        return s
    s = re.sub(r"[ \t]+", " ", s)
    stash = []

    def _st(m):
        stash.append(m.group(0))
        return f"\x00{len(stash) - 1}\x00"
    s = re.sub(r"\\(?:text|textrm|textit|textbf)\s*\{[^{}]*[A-Za-zÀ-ÿ]{2,}[^{}]*\}", _st, s)
    s = _graffe_doppie(s)
    s = re.sub(r"\s*([_^])\s*", r"\1", s)
    s = re.sub(r"(\\[A-Za-z]+\*?)\s+\{", r"\1{", s)
    s = re.sub(r"([{(\[])\s+", r"\1", s)
    s = re.sub(r"\s+([})\]])", r"\1", s)
    s = re.sub(r"\}\s+\{", "}{", s)
    s = _SINGOLI_RE.sub(lambda m: m.group(1).replace(" ", ""), s)   # 'd e t' -> 'det'
    s = re.sub(r"\\(?:operatorname\*?|mathrm)\{([A-Za-z]+)\}",
               lambda m: ("\\" + m.group(1)) if m.group(1) in _TEX_FUNZ else m.group(0), s)
    s = re.sub(r"\\mathrm\{a\}\s*\\(sin|cos|tan)\b", r"\\arc\1", s)
    s = re.sub(r"\\mathrm\{arc\}\s*\\(sin|cos|tan)\b", r"\\arc\1", s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], s)
    return re.sub(r" {2,}", " ", s).strip()


def compatta_md(md: str) -> str:
    """Applies compatta_tex only to the math parts of a Markdown text."""
    md = re.sub(r"\$\$(.+?)\$\$", lambda m: "$$" + compatta_tex(m.group(1)) + "$$", md or "",
                flags=re.S)
    return re.sub(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", lambda m: "$" + compatta_tex(m.group(1)) + "$", md)


def tex_core(s: str) -> str:
    """Removes the delimiters ($$ $ \\[ \\] \\( \\)) around a formula."""
    s = (s or "").strip()
    # "\text" read as TAB+"ext", "\frac" as form-feed+"rac" etc.:
    # restores the LaTeX commands ruined by a double "unescape"
    s = re.sub(r"[\t\x0c\x08\r\x0b\x07](?=[A-Za-z])",
               lambda m: {"\t": "\\t", "\x0c": "\\f", "\x08": "\\b", "\r": "\\r",
                          "\x0b": "\\v", "\x07": "\\a"}[m.group(0)], s)
    for a, b in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if s.startswith(a) and s.endswith(b) and len(s) >= len(a) + len(b):
            s = s[len(a):len(s) - len(b)].strip()
    return s.strip()


_TEX_DROP = {
    "\\displaystyle", "\\textstyle", "\\scriptstyle", "\\left", "\\right",
    "\\big", "\\Big", "\\bigg", "\\Bigg", "\\bigl", "\\bigr", "\\Bigl", "\\Bigr",
    "\\biggl", "\\biggr", "\\Biggl", "\\Biggr", "\\middle",
    "\\limits", "\\nolimits", "\\,", "\\;", "\\:", "\\!", "\\ ", "\\quad", "\\qquad",
    "\\text", "\\textrm", "\\textit", "\\textbf", "\\mathrm", "\\mathit", "\\rm",
    "\\operatorname", "\\boxed", "\\mathord", "\\hspace", "\\enspace", "\\thinspace",
    "~", "&", "\\\\", "\\nonumber", "\\notag", "\\cancel", "\\displaylimits",
}
_TEX_MAP = {
    "\\dfrac": "\\frac", "\\tfrac": "\\frac", "\\cfrac": "\\frac",
    "\\bar": "\\overline", "\\widehat": "\\hat", "\\widetilde": "\\tilde",
    "\\overrightarrow": "\\vec", "\\boldsymbol": "\\mathbf", "\\bm": "\\mathbf",
    "\\iff": "\\Leftrightarrow", "\\Longleftrightarrow": "\\Leftrightarrow",
    "\\implies": "\\Rightarrow", "\\Longrightarrow": "\\Rightarrow",
    "\\to": "\\rightarrow", "\\longrightarrow": "\\rightarrow",
    "\\le": "\\leq", "\\ge": "\\geq", "\\ne": "\\neq", "\\leqslant": "\\leq",
    "\\geqslant": "\\geq", "\\varepsilon": "\\epsilon", "\\varphi": "\\phi",
    "\\vartheta": "\\theta", "\\prime": "'", "\\lbrace": "\\{", "\\rbrace": "\\}",
    "\\vert": "|", "\\lvert": "|", "\\rvert": "|", "\\mid": "|",
    "\\ast": "*", "\\cdotp": "\\cdot", "\\infin": "\\infty",
    "α": "\\alpha", "β": "\\beta", "γ": "\\gamma", "δ": "\\delta", "ε": "\\epsilon",
    "θ": "\\theta", "λ": "\\lambda", "μ": "\\mu", "ν": "\\nu", "π": "\\pi",
    "ρ": "\\rho", "σ": "\\sigma", "τ": "\\tau", "φ": "\\phi", "ψ": "\\psi",
    "ω": "\\omega", "Δ": "\\Delta", "Γ": "\\Gamma", "Ω": "\\Omega", "∞": "\\infty",
    "≤": "\\leq", "≥": "\\geq", "≠": "\\neq", "→": "\\rightarrow", "⇒": "\\Rightarrow",
    "⇔": "\\Leftrightarrow", "·": "\\cdot", "×": "\\times", "∂": "\\partial", "−": "-",
    "′": "'", "\\colon": ":", "\\vcentcolon": ":", "\\dots": "\\ldots", "\\cdots": "\\ldots",
    "\\simeq": "\\cong", "\\approx": "\\cong",
}
# named functions: \arcsin == \operatorname{arcsin} == arcsin
_TEX_FUNZ = {"sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan",
             "sinh", "cosh", "tanh", "coth", "log", "ln", "exp", "det", "lim", "max",
             "min", "sup", "inf", "arg", "deg", "dim", "ker", "sgn", "tg", "arctg",
             "asin", "acos", "atan"}
_TEX_FUNZ_ALIAS = {"asin": "arcsin", "acos": "arccos", "atan": "arctan"}


def tex_tokens(tex: str) -> list[str]:
    """Formula -> sequence of 'content symbols' for the comparison."""
    s = tex_core(tex)
    s = re.sub(r"\\tag\*?\{[^}]*\}", " ", s)
    # equation number written at the end ("\quad (3.2)"): does not count
    s = re.sub(r"(?:\\q?quad|\\hfill|~|\s)*\(\s*\d+(?:\.\d+)+\s*\)\s*$", " ", s)
    s = re.sub(r"(?:\\q?quad|\\hfill)+\s*\(\s*\d+\s*\)\s*$", " ", s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}(\{[^}]*\})?", " ", s)
    s = re.sub(r"\\(left|right|bigl|bigr|Bigl|Bigr|big|Big)\s*\.", " ", s)   # invisible \left.
    s = s.replace("°", "^\\circ")
    s = re.sub(r"\\(hspace|vspace)\*?\{[^}]*\}", " ", s)
    s = re.sub(r"\^\s*\{?\s*\\prime\s*\\prime\s*\}?", "''", s)
    s = re.sub(r"\^\s*\{?\s*\\prime\s*\}?", "'", s)
    s = re.sub(r"\^\s*\{\s*'+\s*\}", lambda m: "'" * m.group(0).count("'"), s)
    s = re.sub(r"\\(?:operatorname\*?|mathrm|rm|text|textrm)\s*\{\s*([A-Za-z]+)\s*\}",
               lambda m: ("\\" + m.group(1)) if m.group(1) in _TEX_FUNZ else m.group(0), s)
    out = []
    for t in re.findall(r"\\[A-Za-z]+|\\.|[^\s{}]", s):
        t = _TEX_MAP.get(t, t)
        if t in _TEX_DROP:
            continue
        if t.startswith("\\") and t[1:] in _TEX_FUNZ:
            # \sin, \operatorname{sin} and "s i n" written in letters are the same thing
            out.extend(_TEX_FUNZ_ALIAS.get(t[1:], t[1:]))
            continue
        out.append(t)
    while out and out[-1] in {".", ",", ";"}:
        out.pop()
    return out


def tex_valida(tex: str) -> bool:
    s = (tex or "").replace("\\{", "").replace("\\}", "")
    return (bool(tex and tex.strip()) and s.count("{") == s.count("}")
            and _n_left(tex) == _n_right(tex)
            and tex.count("\\begin") == tex.count("\\end"))


def tex_differenze(a: str, b: str, max_n: int = 3) -> str:
    """'\\sin -> arcsin' : the pieces in which two readings differ."""
    import difflib
    ta, tb = tex_tokens(a), tex_tokens(b)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        # one symbol of context before and after, to see WHERE the difference is
        a0, b0 = max(0, i1 - 1), max(0, j1 - 1)
        a1, b1 = min(len(ta), i2 + 1), min(len(tb), j2 + 1)
        out.append(f"«{' '.join(ta[a0:a1])}» ↔ «{' '.join(tb[b0:b1])}»")
        if len(out) >= max_n:
            out.append("...")
            break
    return "; ".join(out)


def chandra_tex(raw: str) -> str:
    """LaTeX from Chandra's answers on a formula crop."""
    import html as _h
    raw = _strip_fences(raw or "")
    maths = re.findall(r"<math[^>]*>(.*?)</math>", raw, flags=re.S)
    if maths:
        return " \\quad ".join(_h.unescape(m).strip() for m in maths if m.strip())
    txt = _h.unescape(re.sub(r"<[^>]+>", " ", raw))
    return tex_core(re.sub(r"\s+", " ", txt))


LETTORI_FORMULA = ("paddle", "paddle_hr", "chandra", "glm", "unimernet", "ppformulanet")
_FAMIGLIA_BAIDU = {"paddle", "paddle_hr", "ppformulanet"}
LETTORI_EXTRA = [x.strip().lower() for x in
                 os.environ.get("TOMD2_LETTORI_EXTRA", "unimernet,ppformulanet").split(",")
                 if x.strip() and x.strip() != "0"]
_MODELLI_EXTRA = {"unimernet": "UniMERNet", "ppformulanet": "PP-FormulaNet_plus-L"}


def lettori_extra_formule(formule: list[dict], form_dir: Path, cache_dir: Path):
    """Readers SPECIALIZED in formulas (PaddleOCR, already installed): UniMERNet
       (also trained on handwritten formulas) and PP-FormulaNet. Small and
       fast: seconds per formula instead of minutes. Only on doubtful formulas."""
    for chi in LETTORI_EXTRA:
        nome = _MODELLI_EXTRA.get(chi)
        if not nome:
            continue
        cdir = cache_dir / "extra_formule" / chi
        cdir.mkdir(parents=True, exist_ok=True)
        todo = []
        for f in formule:
            cf = cdir / (f["id"] + ".txt")
            if cf.exists():
                f[chi] = cf.read_text(encoding="utf-8")
            elif f.get("crop") and (form_dir / f["crop"]).exists():
                todo.append(f)
        if not todo:
            continue
        try:
            _import_paddle_stack()
            from paddleocr import FormulaRecognition
            _log(f"  [{nome}] loading the model (downloaded the first time) ...")
            mod = FormulaRecognition(model_name=nome, device="cpu")
        except Exception as e:
            _log(f"  [{nome}] not available ({type(e).__name__}: {e}): skipping")
            continue
        _log(f"  [{nome}] rereads {len(todo)} doubtful formulas ...")
        t_all = time.time()
        for i, f in enumerate(todo, 1):
            try:
                res = next(iter(mod.predict(str(form_dir / f["crop"]))))
                t = tex_core(str(res["rec_formula"] or ""))
            except Exception as e:
                _log(f"    [{nome} {f['id']}] error: {e}")
                continue
            (cdir / (f["id"] + ".txt")).write_text(t, encoding="utf-8")
            f[chi] = t
            if i % 20 == 0 or i == len(todo):
                media = (time.time() - t_all) / i
                _log(f"  [{nome}] {i}/{len(todo)} | time left {media*(len(todo)-i)/60:.0f} min")


def scegli_formula(f: dict) -> tuple[str, str, str]:
    """Returns (chosen formula, status, note). Statuses: corrected (by hand),
       confirmed, majority, consistent (Paddle only), doubtful."""
    if f.get("corretta"):
        return f["corretta"], "corretta", ""
    letture = {k: tex_core(f.get(k)) for k in LETTORI_FORMULA
               if f.get(k) and tex_valida(tex_core(f.get(k)))}
    if not letture:
        return f.get("paddle") or "", "dubbia", "no valid reading"
    sig = {k: tuple(tex_tokens(v)) for k, v in letture.items()}
    same = lambda a, b: a in sig and b in sig and sig[a] == sig[b]

    def pick(keys):        # among equivalent readings: the best written one
        return letture[max(keys, key=lambda k: (_stile_tex(letture[k]), k == "paddle"))]
    if same("paddle", "chandra"):
        return tex_bella(pick([k for k in sig if sig[k] == sig["paddle"]])), "confermata", ""
    if same("paddle_hr", "chandra"):
        nota = ("paddle(200dpi): " + tex_differenze(letture["paddle_hr"], letture["paddle"])
                if "paddle" in letture else "")
        return tex_bella(pick(["paddle_hr", "chandra"])), "maggioranza", nota
    if "glm" in sig:           # the referee (GLM) sides with someone
        alleati = [k for k in sig if k != "glm" and sig[k] == sig["glm"]]
        if alleati:
            altri = [k for k in sig if sig[k] != sig["glm"]]
            nota = " | ".join(f"{k}: {tex_differenze(letture['glm'], letture[k])}" for k in altri)
            return tex_bella(pick(alleati + ["glm"])), "arbitrata", nota
    # general vote (with the specialized readers UniMERNet / PP-FormulaNet):
    # the largest group of identical readings wins, if it is unique and not
    # made only of Baidu models (Paddle, Paddle 300 dpi, PP-FormulaNet), which
    # they tend to make the same mistakes.
    gruppi = {}
    for k in sig:
        gruppi.setdefault(sig[k], []).append(k)
    ordinati = sorted(gruppi.values(), key=len, reverse=True)
    if len(ordinati) > 1 and len(ordinati[0]) >= 2 and len(ordinati[0]) > len(ordinati[1]):
        top = ordinati[0]
        if set(top) - _FAMIGLIA_BAIDU or len(top) >= 3:
            altri = [k for k in sig if k not in top]
            nota = " | ".join(f"{k}: {tex_differenze(letture[top[0]], letture[k])}" for k in altri)
            return tex_bella(pick(top)), "maggioranza", nota
    if "chandra" not in letture:
        if same("paddle", "paddle_hr") or len(letture) == 1:
            k0 = "paddle" if "paddle" in letture else next(iter(letture))
            stato = "coerente" if f.get("chandra_fatto") is None else "dubbia"
            nota = "" if stato == "coerente" else "Chandra gave no valid reading"
            return tex_bella(letture[k0]), stato, nota
    # no agreement. With 3 readings all different, the one "closest"
    # to the other two wins (edit distance in symbols): in practice, on the
    # disputed points the third reading decides. On a tie Chandra wins (the larger
    # model, read on the high-resolution crop).
    # Paddle and Paddle-300dpi equal count as ONE vote only (same model)
    pool = [k for k in sig if not (k == "paddle_hr" and same("paddle", "paddle_hr"))]
    if len(pool) <= 2 and same("paddle", "paddle_hr"):
        best = "paddle"
    else:
        best = min(pool, key=lambda k: (sum(_lev(sig[k], sig[o]) for o in pool if o != k),
                                        k != "chandra", k != "unimernet", k != "paddle"))
    note = [f"{k}: {tex_differenze(letture[best], v)}" for k, v in letture.items()
            if k != best and sig[k] != sig[best]]
    return tex_bella(letture[best]), "dubbia", " | ".join(note)


def _lev(a, b) -> int:
    """Edit distance between two symbol sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _stile_tex(t: str) -> int:
    return t.count("\\text{") - t.count("~") - 2 * t.count("\\mathrm{a")


def tex_bella(t: str) -> str:
    """Form touch-ups: _{angolo~di~Mach} -> _{\\text{angolo di Mach}}."""
    t = re.sub(r"([_^])\{\s*((?:[A-Za-zÀ-ÿ]+\s*~\s*)+[A-Za-zÀ-ÿ]+)\s*\}",
               lambda m: m.group(1) + "{\\text{" + re.sub(r"\s*~\s*", " ", m.group(2)) + "}}", t)
    return t


def paddle_rilettura_formule(srv, formule: list[dict], img, img_hr, form_dir: Path,
                             page_no: int):
    """Crops each formula from the page at FORMULA_DPI, saves it (needed by
       Chandra and the review file) and has Paddle reread it."""
    if not formule:
        return
    form_dir.mkdir(parents=True, exist_ok=True)
    sx = img_hr.width / max(1, img.width)
    sy = img_hr.height / max(1, img.height)
    for f in formule:
        if not f.get("bbox"):
            continue
        x0, y0, x1, y1 = f["bbox"]
        pad = 6
        box = (max(0, int((x0 - pad) * sx)), max(0, int((y0 - pad) * sy)),
               min(img_hr.width, int((x1 + pad) * sx)), min(img_hr.height, int((y1 + pad) * sy)))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        crop = img_hr.crop(box)
        crop.save(form_dir / (f["id"] + ".png"))
        f["crop"] = f["id"] + ".png"
        if srv is None:
            continue
        try:
            im = scale_to_fit(crop, 1_200_000, min_pixels=120_000)
            raw, usage = srv.ocr(im, prompt="Formula Recognition:", max_tokens=700,
                                 label=f"[formula {f['id']}]")
            if usage.get("stopped") != "loop":
                f["paddle_hr"] = tex_core(normalize_math(raw).replace("$$", "").strip())
        except Exception as e:
            _log(f"    [formula {f['id']}] Paddle reread failed: {e}")
    _log(f"    [paddle p.{page_no}] {len(formule)} formulas cropped and reread at {FORMULA_DPI} dpi")


def chandra_verifica_formule(formule: list[dict], form_dir: Path, cache_dir: Path,
                             threads=None, gpu=False):
    """Has Chandra read the formula crops (with cache)."""
    from PIL import Image
    cdir = cache_dir / "chandra_formule"
    cdir.mkdir(parents=True, exist_ok=True)
    todo = []
    for f in formule:
        if not f.get("crop"):
            continue
        cf = cdir / (f["id"] + ".raw.txt")
        if cf.exists():
            f["chandra"] = chandra_tex(cf.read_text(encoding="utf-8"))
            f["chandra_fatto"] = True
            continue
        if VERIFICA_FORMULE == "dubbie" and f.get("paddle_hr") and \
                tex_tokens(f["paddle_hr"]) == tex_tokens(f.get("paddle", "")):
            continue
        todo.append(f)
    if not todo:
        return
    _log(f"  [formula check] Chandra rereads {len(todo)} formulas (on the crops at {FORMULA_DPI} dpi) ...")
    t_all = time.time()
    with LlamaServer("chandra", threads=threads, gpu=gpu) as srv:
        for i, f in enumerate(todo, 1):
            t0 = time.time()
            try:
                im = Image.open(form_dir / f["crop"]).convert("RGB")
                im = scale_to_fit(im, 1_000_000, min_pixels=150_000)
                raw, usage = srv.ocr(im, prompt=CHANDRA_OCR_PROMPT, max_tokens=900,
                                     label=f"[formula {f['id']}]")
            except Exception as e:
                _log(f"    [formula {f['id']}] Chandra failed: {e}")
                continue
            if usage.get("stopped") == "loop":
                raw = ""
            (cdir / (f["id"] + ".raw.txt")).write_text(raw, encoding="utf-8")
            f["chandra"] = chandra_tex(raw)
            f["chandra_fatto"] = True
            media = (time.time() - t_all) / i
            _log(f"  [formula check] {i}/{len(todo)} in {time.time()-t0:.0f} s"
                 f" | time left {media*(len(todo)-i)/60:.0f} min")


def _formula_md(f: dict, tex: str, stato: str, nota: str) -> str:
    if not f.get("display", True):
        return f"${tex}$"
    num = f.get("numero")
    body = tex
    if num and "\\tag" not in tex:
        body = tex + " \\tag*{" + num.replace("}", "") + "}"
    out = "\n\n$$\n" + body + "\n$$\n"
    if stato == "dubbia":
        alt = "; ".join(f"{k}: `{f[k].replace('`', chr(39))}`"
                        for k in LETTORI_FORMULA
                        if f.get(k) and tex_tokens(f[k]) != tex_tokens(tex))
        out += (f"\n> ⚠ **Uncertain formula** ({f['id']}): the automatic readings do not agree "
                f"({nota}). Other readings: {alt}. *Work out the right one from the context.*\n")
    return out + "\n"


_CORR_RE = re.compile(r"^###\s+(p\d{4}_f\d{2})\b(?:(?!\n###).)*?\nCORREZIONE[^\n]*\n\s*\$\$\s*\n(.*?)\n\s*\$\$",
                      re.S | re.M)


def leggi_correzioni(report: Path, cache_dir: Path, scelte_auto: dict) -> dict:
    """Reads the corrections written in the _da_controllare.md file (by hand or by
       Claude) and saves them in the cache: they also apply if the file is regenerated.
       It is a correction ONLY if the text differs from what the
       program wrote in the file last time (report_scritto.json):
       this way a change of automatic choice is not mistaken for a
       manual correction."""
    return _leggi_corr(report, cache_dir, scelte_auto, _CORR_RE,
                       "correzioni_formule_v2.json",
                       lambda a, b: tex_tokens(a) == tex_tokens(b))


def _carica_json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _leggi_corr(report: Path, cache_dir: Path, auto: dict, regex, nome_store: str,
                uguali) -> dict:
    store = cache_dir / nome_store
    corr = _carica_json(store)
    scritto = _carica_json(cache_dir / "report_scritto.json")
    if report.exists():
        for k, t in regex.findall(report.read_text(encoding="utf-8")):
            t = t.strip()
            if not t or k not in auto or k not in scritto:
                continue
            if not uguali(t, scritto[k]):
                corr[k] = t                      # you changed it (or Claude did)
    if corr:
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps(corr, ensure_ascii=False, indent=1), encoding="utf-8")
    return corr


def parole_sospette(md: str, max_n: int = 25, freq_doc=None) -> list[str]:
    """Words that do not exist in any of the main languages (possible
       misreadings, e.g. 'semolno'). Technical terms that recur several
       times in the document (e.g. 'omentropici') are not flagged."""
    try:
        from wordfreq import zipf_frequency
    except Exception:
        return []
    md = "\n".join(l for l in md.splitlines() if not l.lstrip().startswith(">"))
    testo = re.sub(r"\$\$.+?\$\$", " ", md, flags=re.S)
    testo = re.sub(r"\$[^$\n]+\$|<[^>]+>|!\[[^\]]*\]\([^)]*\)|\{\{[^}]+\}\}|\([^)]*\.(jpg|png|webp)\)",
                   " ", testo)
    out, viste = [], set()
    for w in re.findall(r"(?<![\w/\\])[A-Za-zÀ-ÿ]{5,}(?![\w/])", testo):
        k = w.lower()
        if k in viste:
            continue
        viste.add(k)
        if w[0].isupper() and not w.isupper():
            continue            # proper names (Mach, Prandtl, ...)
        if freq_doc is not None and freq_doc.get(k, 0) >= 2:
            continue
        if max(zipf_frequency(k, lg) for lg in _LINGUE) < 0.6:
            out.append(w)
            if len(out) >= max_n:
                break
    return out


def scrivi_da_controllare(report: Path, form_prefix: str, voci: list[dict],
                          parole: dict[int, list[str]], pages_link: dict[int, str],
                          voci_blocchi: list[dict] | None = None,
                          voci_fig: list[dict] | None = None, ctrl_prefix: str = "",
                          cache_dir: Path | None = None):
    voci_blocchi = voci_blocchi or []
    scritto = {}          # what the program writes under CORREZIONE (to tell later
                          # whether you changed it)
    voci_fig = voci_fig or []

    def conta(vv):
        c = {}
        for v in vv:
            c[v["stato"]] = c.get(v["stato"], 0) + 1
        return ", ".join(f"{k} {n}" for k, n in sorted(c.items())) or "nessuna"
    L = [f"# To review — {report.stem.replace('_da_controllare', '')}", "",
         "Generated by ocr_vlm.py.", "",
         f"- Printed formulas: {conta(voci)}",
         f"- Handwritten blocks: {conta(voci_blocchi)}",
         f"- Figures: {conta(voci_fig)}", "",
         "**How to use it:** look at the crop and, if needed, write the right version under "
         "*CORREZIONE* (formulas between the `$$`, text and descriptions inside the ``` box). "
         "Then run the same conversion again: it uses the cache (does not redo the OCR) and applies the "
         "corrections. You can also give this file to Claude and have it correct ONLY these points.", ""]

    def blocco_formula(v):
        f = v["f"]
        out = [f"### {f['id']} — page {v['pagina']} ({v['stato']})", ""]
        if f.get("crop"):
            out += [f"![formula {f['id']}]({form_prefix}/{f['crop']})", ""]
        if v["nota"]:
            out += [f"Differences: {v['nota']}", ""]
        for k in LETTORI_FORMULA:
            if f.get(k):
                out.append(f"- **{k}**: `{f[k]}`")
        out += ["", "CORREZIONE (leave as it is if the formula is right):", "$$", v["scelta"], "$$", ""]
        scritto[f["id"]] = v["scelta"]
        return out

    def blocco_mano(v):
        b = v["b"]
        out = [f"### {b['id']} — page {v['pagina']} ({v['stato']})", ""]
        if b.get("crop"):
            out += [f"![block {b['id']}]({form_prefix}/{b['crop']})", ""]
        if v["nota"]:
            out += [f"Differences in the formulas: {v['nota']}", ""]
        for k in ("chandra", "qwen", "glm"):
            if b.get(k):
                out += [f"**{k}:**", "", "```text", b[k], "```", ""]
        out += ["CORREZIONE (block text in Markdown; leave as it is if right):",
                "```markdown", v["scelta"], "```", ""]
        scritto[b["id"]] = v["scelta"]
        return out

    def blocco_fig(v):
        j = v["j"]
        out = [f"### FIG {j['key']} — page {j['page']} ({v['stato']})", ""]
        if j.get("crop"):
            out += [f"![figure]({ctrl_prefix}/{j['crop']})", ""]
        if v["nota"]:
            out += ["The checker (GLM) reports:", "", v["nota"], ""]
        out += ["CORREZIONE (description; leave as it is if fine):",
                "```text", v["scelta"], "```", ""]
        scritto[j["key"]] = v["scelta"]
        return out

    sezioni = [
        ("Doubtful formulas — the readings do not agree", voci, ("dubbia",), blocco_formula),
        ("Doubtful handwritten blocks", voci_blocchi, ("dubbia",), blocco_mano),
        ("Figures disputed by the checker", voci_fig, ("contestata", "non verificata"), blocco_fig),
        ("Formulas decided by majority or by the referee — quick check", voci,
         ("maggioranza", "arbitrata"), blocco_formula),
        ("Handwritten blocks decided by the referee — quick check", voci_blocchi, ("arbitrata",),
         blocco_mano),
        ("Already corrected (editable)", voci, ("corretta",), blocco_formula),
        ("Blocks already corrected (editable)", voci_blocchi, ("corretta",), blocco_mano),
        ("Figures already corrected (editable)", voci_fig, ("corretta",), blocco_fig),
    ]
    vuoto = True
    for titolo, vv, stati, fn in sezioni:
        sel = [v for v in vv if v["stato"] in stati]
        if sel:
            vuoto = False
            L += [f"## {titolo} ({len(sel)})", ""]
            for v in sel:
                L += fn(v)
    pp = {p: w for p, w in parole.items() if w}
    if pp:
        vuoto = False
        L += ["## Suspicious words (they do not exist in it/en/fr/es/de: possible misreadings)", ""]
        for p, ws in sorted(pp.items()):
            link = f" — [page]({pages_link[p]})" if p in pages_link else ""
            L.append(f"- p. {p}{link}: " + ", ".join(ws))
        L.append("")
    if vuoto:
        L.append("No doubtful points found.")
    report.write_text("\n".join(L) + "\n", encoding="utf-8")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "report_scritto.json").write_text(
            json.dumps(scritto, ensure_ascii=False, indent=1), encoding="utf-8")


FIG_MAX_PER_PAGINA = int(os.environ.get("TOMD2_FIG_MAX_PER_PAGINA", "6"))
_BOX_RE = re.compile(r"box_(\d+)_(\d+)_(\d+)_(\d+)")


def describe_figures(page_figs: dict, page_image, cache_dir: Path, fig_dir: Path,
                     fig_desc: dict, threads=None, gpu=False):
    """Describes with Chandra the figures cropped by Paddle. If a page has
       too many figures (e.g. 12 panels), describes them all together in ONE
       call on the box containing them, so as not to take an hour."""
    from PIL import Image
    cdir = cache_dir / "chandra_fig"
    cdir.mkdir(parents=True, exist_ok=True)
    jobs = []          # (cache_key, image, [figure names], label)
    for p, figs in sorted(page_figs.items()):
        pending = [f for f in figs if not (cdir / (f + ".md")).exists()]
        for f in figs:
            cf = cdir / (f + ".md")
            if cf.exists():
                fig_desc[f] = cf.read_text(encoding="utf-8")
        if not pending:
            continue
        if len(figs) > FIG_MAX_PER_PAGINA:
            boxes = [tuple(map(int, m.groups())) for f in figs
                     if (m := _BOX_RE.search(f))]
            if boxes:
                pg = page_image(p, MODELS["paddle"]["dpi"])
                u = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                     max(b[2] for b in boxes), max(b[3] for b in boxes))
                jobs.append((f"p{p:04d}_insieme", pg.crop(u), figs, f"p.{p} (group of {len(figs)} figures)"))
                continue
        for f in pending:
            try:
                jobs.append((f, Image.open(fig_dir / f).convert("RGB"), [f], f"p.{p} {f[:28]}"))
            except Exception:
                pass
    if not jobs:
        return
    _log(f"  [hybrid] describing {len(jobs)} figures/groups with Chandra (on the crops) ...")
    with LlamaServer("chandra", threads=threads, gpu=gpu) as srv:
        for i, (ck, im, names, label) in enumerate(jobs, 1):
            img = scale_to_fit(im, MODELS["chandra"]["max_pixels"], min_pixels=400_000)
            t0 = time.time()
            raw, usage = srv.ocr(img, max_tokens=2000, label=f"[figure {label}]")
            (cdir / (ck + ".raw.txt")).write_text(raw, encoding="utf-8")
            desc = chandra_describe(raw)
            if usage.get("stopped") == "loop":
                desc = dedup_frasi(desc)
            if len(names) > 1:
                first = names[0]
                fig_desc[first] = desc
                (cdir / (first + ".md")).write_text(desc, encoding="utf-8")
                for f in names[1:]:
                    nota = "*(see the overall description in the first figure on this page)*"
                    fig_desc[f] = nota
                    (cdir / (f + ".md")).write_text(nota, encoding="utf-8")
            else:
                fig_desc[names[0]] = desc
                (cdir / (names[0] + ".md")).write_text(desc, encoding="utf-8")
            _log(f"  [figures] {i}/{len(jobs)} done in {time.time()-t0:.0f} s")


# ============================================================================
# CHECKERS (version 4) — all local and free, one model at a time
#
#   Qwen3.5-9B      describes the FIGURES (with the page text as
#                   context) and rereads the formula blocks of
#                   HANDWRITTEN pages (second reading after Chandra).
#   GLM-4.6V-Flash  a different family: REFEREE when two readings
#                   disagree (printed formulas, handwritten blocks) and
#                   CHECKER of the figure descriptions; if it finds
#                   errors, Qwen rewrites and GLM checks again.
#   Whatever still disagrees ends up in <name>_da_controllare.md
#   (last level: you, or Claude on those points only).
#
#   TOMD2_DESCRIZIONI     qwen (default) | chandra | 0
#   TOMD2_VERIFICA_FIGURE tutte (default) | 0
#   TOMD2_ARBITRO         glm (default) | 0
#   TOMD2_VERIFICA_MANO   1 (default) | 0
#   TOMD2_GLOSSARIO       document symbols, helps read handwriting,
#                         e.g. "p pressure, a speed of sound,
#                         delta=(gamma-1)/2, rho density"
# ============================================================================
DESCRIZIONI = os.environ.get("TOMD2_DESCRIZIONI", "qwen").strip().lower()
VERIFICA_FIGURE = os.environ.get("TOMD2_VERIFICA_FIGURE", "tutte").strip().lower()
ARBITRO = os.environ.get("TOMD2_ARBITRO", "glm").strip().lower()
VERIFICA_MANO = os.environ.get("TOMD2_VERIFICA_MANO", "1").strip() != "0"
GLOSSARIO = os.environ.get("TOMD2_GLOSSARIO", "").strip()

PROMPT_FIGURA = """You are the assistant that makes the figures of a technical-scientific document accessible to an artificial intelligence that CANNOT see images.
Text of the page around the figure (only to understand what it is about):
<<<
{contesto}
>>>
Describe the figure IN ITALIAN with exactly these fields, keeping the field names as written:
Tipo: (hand-drawn diagram, Cartesian graph, photograph, technical drawing, x-t diagram, table, ...)
Etichette: ALL the visible writing, symbols, point letters and numbers, transcribed exactly as written (formulas in LaTeX between $...$), and what each refers to.
Descrizione: shapes, axes (name, unit, direction, values), curves and their trend, points, lines (solid/dashed, tangent, parallel), arrows and their direction, regions and how they are numbered, colours if meaningful; relative positions (above/below/left/right, inside/outside).
Cosa mostra: 1-3 sentences on the meaning of the figure in relation to the text.
Rules: do not invent anything that cannot be seen; if something cannot be read write [illeggibile]; the page text is for understanding, NOT for adding elements that are not in the figure; make no guesses about the meaning of a symbol if the page text does not state it (no "probably", "perhaps", "possible error"): in that case report only the symbol; no preamble, start directly with "Tipo:".{extra}"""

PROMPT_REVISIONE = """

A first description of this figure was:
<<<
{descr}
>>>
An independent checker reported these problems:
{problemi}
Rewrite the COMPLETE description with the same fields, correcting the problems ONLY if they turn out to be true when you look at the image."""

PROMPT_VERIFICA = """Look carefully at the image. For each of the numbered statements below answer:
V = true: clearly visible in the image
F = false: the image shows something different, or that element is not there
N = cannot be verified by looking at the image (it is an interpretation)
Be strict and precise: answer V only if you really see it. Beware: some statements are false on purpose.
STATEMENTS:
{elenco}
Answer ONLY with a JSON object, for example:
{{"risposte": {{"1": "V", "2": "F", "3": "N"}}, "correzioni": {{"2": "what is really visible"}}, "mancanti": ["writing or symbols visible in the image that appear in no statement"]}}"""

# FALSE control statements: if the checker says they are true, its
# check on that figure does not count (it says "yes" to everything)
_TRAPPOLE = [
    "Nella figura e' disegnato un orologio con le lancette.",
    "In alto a destra c'e' la bandiera di uno Stato.",
    "La figura contiene la fotografia di un'automobile.",
    "Al centro della figura c'e' un grafico a torta colorato.",
    "In basso nella figura c'e' un codice a barre.",
    "Nella figura compare il disegno di un albero con le foglie.",
]

PROMPT_TRASCRIVI = """Transcribe EXACTLY the content of this image: a block of handwritten Italian university lecture notes{gloss}.
Output Markdown only. Keep the Italian text word by word. Write every mathematical expression in KaTeX-compatible LaTeX: $...$ inline, $$...$$ for display. Keep the structure (lists, line breaks, arrows as \\Rightarrow or \\rightarrow inside math, boxed results as \\boxed{{}}).
Do not solve, do not correct, do not add comments or explanations. If a symbol is unreadable write [illeggibile]."""

PROMPT_FORMULA = ("Transcribe the mathematical formula in this image into KaTeX-compatible LaTeX. "
                  "Output ONLY the LaTeX code, without $ delimiters and without explanations. "
                  "Copy exactly what is written, symbol by symbol: do not simplify, do not correct.")


def _gloss() -> str:
    return f" (symbols used in these notes: {GLOSSARIO})" if GLOSSARIO else ""


def _pulisci(t: str) -> str:
    """Removes <think> reasoning, GLM 'box' tokens and ``` fences."""
    t = re.sub(r"<think>.*?</think>", "", t or "", flags=re.S)
    t = re.sub(r"^.*?</think>", "", t, flags=re.S) if "</think>" in t else t
    t = t.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    t = re.sub(r"^\s*```[a-zA-Z]*\s*\n", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def _json_morbido(t: str) -> dict | None:
    t = _pulisci(t)
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    s = m.group(0)
    for cand in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return None


class ServerPigro:
    """Starts the model server only if there is really work to do."""

    def __init__(self, key, threads=None, gpu=False):
        self.key, self.threads, self.gpu, self.srv = key, threads, gpu, None

    def __call__(self) -> LlamaServer:
        if self.srv is None:
            self.srv = LlamaServer(self.key, threads=self.threads, gpu=self.gpu).start()
        return self.srv

    def close(self):
        if self.srv is not None:
            self.srv.stop()
            self.srv = None


def _contesto(md: str, name: str, prima: int = 1500, dopo: int = 600) -> str:
    i = md.find(name)
    if i < 0:
        return re.sub(r"\s+", " ", md[:prima]).strip()
    s = md[max(0, i - prima):i] + "\n[FIGURA]\n" + md[i:i + dopo]
    s = re.sub(r"<!--.*?-->|!\[[^\]]*\]\([^)]*\)|\{\{[^}]+\}\}", " ", s, flags=re.S)
    s = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith(">"))
    return re.sub(r"[ \t]+", " ", s).strip()


# ---- figures ----------------------------------------------------------------
_FIG_CH_RE = re.compile(r"!\[Figura pag\. (\d+)\]\(([^)]*?/)?(p\d{4}_fig\d+\.png)\)")


def raccogli_figure(results: dict, page_figs: dict, page_image, fig_dir: Path,
                    ctrl_dir: Path) -> list[dict]:
    """One 'job' per figure (or per group, if the page has too many)."""
    from PIL import Image
    jobs = []
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    sc = FORMULA_DPI / MODELS["paddle"]["dpi"]
    for p in sorted(results):
        motore, md = results[p]
        if motore == "paddle":
            names = list(page_figs.get(p) or [])
        else:
            names = [m.group(3) for m in _FIG_CH_RE.finditer(md)]
        names = [n for n in names if n in md]
        if not names:
            continue
        boxes = {n: tuple(map(int, m.groups())) for n in names
                 if motore == "paddle" and (m := _BOX_RE.search(n))}
        hi = None

        def ritaglio(box):
            nonlocal hi
            if hi is None:
                hi = page_image(p, FORMULA_DPI)
            x0, y0, x1, y1 = box
            return hi.crop((int(x0 * sc), int(y0 * sc), int(x1 * sc), int(y1 * sc)))

        gruppi = [[n] for n in names]
        if len(names) > FIG_MAX_PER_PAGINA and len(boxes) == len(names):
            gruppi = [names]
        for gr in gruppi:
            key = gr[0] if len(gr) == 1 else f"p{p:04d}_insieme"
            try:
                if len(gr) > 1:
                    bb = [boxes[n] for n in gr]
                    im = ritaglio((min(b[0] for b in bb), min(b[1] for b in bb),
                                   max(b[2] for b in bb), max(b[3] for b in bb)))
                elif gr[0] in boxes:
                    im = ritaglio(boxes[gr[0]])
                else:
                    im = Image.open(fig_dir / gr[0]).convert("RGB")
            except Exception as e:
                _log(f"    [figure {key}] crop failed: {e}")
                continue
            crop_name = Path(key).stem + ".png"
            try:
                im.save(ctrl_dir / crop_name)
            except Exception:
                pass
            jobs.append({"key": key, "names": gr, "page": p, "img": im, "crop": crop_name,
                         "contesto": _contesto(md, gr[0]), "pannelli": len(gr) > 1})
    return jobs


def _prompt_figura(job) -> str:
    extra = ("\nThe figure contains several panels: describe them one by one, with their letter "
             "or label." if job["pannelli"] else "")
    if GLOSSARIO:
        extra += f"\nSymbols used in this document: {GLOSSARIO}."
    return PROMPT_FIGURA.format(contesto=job["contesto"][:2200] or "(no text)", extra=extra)


def _taglia_descr(t: str) -> str:
    t = _pulisci(t)
    i = t.find("Tipo:")
    return (t[i:] if i >= 0 else t).strip()


def qwen_descrivi(srv_q: ServerPigro, jobs: list[dict], cache_dir: Path, revisione=False):
    sub = "qwen_fig_rev" if revisione else "qwen_fig"
    cdir = cache_dir / sub
    cdir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (cdir / (j["key"] + ".md")).exists()]
    for j in jobs:
        cf = cdir / (j["key"] + ".md")
        if cf.exists():
            j["rev" if revisione else "descr"] = compatta_md(cf.read_text(encoding="utf-8"))
    if not todo:
        return
    _log(f"  [Qwen] {'rewriting' if revisione else 'describing'} {len(todo)} figures ...")
    t_all = time.time()
    for i, j in enumerate(todo, 1):
        prompt = _prompt_figura(j)
        if revisione:
            prompt += PROMPT_REVISIONE.format(descr=j.get("descr", ""), problemi=j["problemi"])
        img = scale_to_fit(j["img"], MODELS["qwen"]["max_pixels"], min_pixels=300_000)
        try:
            raw, usage = srv_q().ocr(img, prompt=prompt, max_tokens=MODELS["qwen"]["max_tokens"],
                                     label=f"[Qwen figure p.{j['page']}]")
        except Exception as e:
            _log(f"    [Qwen figure {j['key']}] error: {e}")
            continue
        d = _taglia_descr(raw)
        if usage.get("stopped") == "loop":
            d = dedup_frasi(d)
        (cdir / (j["key"] + ".raw.txt")).write_text(raw, encoding="utf-8")
        (cdir / (j["key"] + ".md")).write_text(d, encoding="utf-8")
        j["rev" if revisione else "descr"] = compatta_md(d)
        media = (time.time() - t_all) / i
        _log(f"  [Qwen figures] {i}/{len(todo)} | time left {media*(len(todo)-i)/60:.0f} min")


_SCAMBI = [("sinistra", "destra"), ("sopra", "sotto"), ("alto", "basso"),
           ("dentro", "fuori"), ("orizzontale", "verticale"), ("crescente", "decrescente"),
           ("tratteggiat", "continu"), ("rosso", "blu"), ("rossi", "blu"), ("rosse", "blu")]


def _altera(fr: str, seme: int) -> str | None:
    """A WRONG but plausible version of a statement."""
    # 1) a changed number (e.g. 100 -> 300, 1 -> 3)
    nums = list(re.finditer(r"(?<![A-Za-z_\\])\d+(?:[.,]\d+)?", fr))
    if nums:
        m = nums[seme % len(nums)]
        t = m.group(0)
        try:
            v = float(t.replace(",", "."))
            nv = v * 3 if v not in (0, 1) else v + 2
            nt = (str(int(nv)) if float(nv).is_integer() else f"{nv:g}").replace(".", "," if "," in t else ".")
        except Exception:
            nt = None
        if nt and nt != t:
            return fr[:m.start()] + nt + fr[m.end():]
    # 2) a swapped direction / colour
    low = fr.lower()
    for a_, b_ in _SCAMBI:
        for x, y in ((a_, b_), (b_, a_)):
            i = low.find(x)
            if i >= 0:
                return fr[:i] + y + fr[i + len(x):]
    # 3) a changed subscript ($V_{p1}$ -> $V_{p3}$)
    m = re.search(r"_\{?([0-9])\}?", fr)
    if m:
        d = str((int(m.group(1)) + 2) % 10)
        return fr[:m.start(1)] + d + fr[m.end(1):]
    return None


def trappole_sottili(aff: list[str], seme: int, n: int = 2) -> list[str]:
    out = []
    for k in range(len(aff)):
        fr = aff[(seme + 5 * k) % len(aff)]
        alt = _altera(fr, seme + k)
        if alt and alt != fr and alt not in aff and alt not in out:
            out.append(alt)
        if len(out) >= n:
            break
    return out


def affermazioni(descr: str, max_n: int = 24) -> list[str]:
    """Splits the description into statements that can be checked one by one
       (labels, description and 'what it shows', each with its own space)."""
    sezioni = {"Tipo": [], "Etichette": [], "Descrizione": [], "Cosa mostra": []}
    sez = "Descrizione"
    for riga in (descr or "").splitlines():
        r = riga.strip()
        if not r:
            continue
        m = re.match(r"^(Tipo|Etichette|Descrizione|Cosa mostra)\s*:\s*(.*)$", r)
        if m:
            sez, r = m.group(1), m.group(2).strip()
            if not r:
                continue
        r = re.sub(r"^[-*•]\s*", "", r)
        if sez == "Etichette":
            sezioni[sez].append("Nella figura compare: " + r[:200])
        else:
            for f in re.split(r"(?<=[.;!?])\s+", r):
                if len(f.split()) >= 4:
                    sezioni[sez].append(f.strip())
    out = (sezioni["Etichette"][:10] + sezioni["Descrizione"][:8]
           + sezioni["Cosa mostra"][:4] + sezioni["Tipo"][:1])
    return out[:max_n]


def glm_verifica_figure(srv_g: ServerPigro, jobs: list[dict], cache_dir: Path, giro: int = 1):
    """The checker does NOT judge the description 'as a block' (it says yes to everything):
       it answers true/false to each single statement, mixed with two
       false test statements. If it confirms a trap, its check
       on that figure is considered unreliable."""
    import hashlib
    cdir = cache_dir / ("glm_fig_v3" if giro == 1 else "glm_fig2_v3")
    cdir.mkdir(parents=True, exist_ok=True)
    campo = "descr" if giro == 1 else "rev"
    todo = []
    for j in jobs:
        if not j.get(campo):
            continue
        cf = cdir / (j["key"] + ".json")
        if cf.exists():
            try:
                j[f"verifica{giro}"] = json.loads(cf.read_text(encoding="utf-8"))
                continue
            except Exception:
                pass
        todo.append(j)
    if not todo:
        return
    _log(f"  [GLM] checking {len(todo)} figure descriptions (statement by statement) ...")
    t_all = time.time()
    for i, j in enumerate(todo, 1):
        aff = affermazioni(j[campo])
        if not aff:
            continue
        h = int(hashlib.md5(j["key"].encode("utf-8")).hexdigest(), 16)
        # 1 "gross" trap + up to 2 "subtle" traps: copies of
        # true statements with a number, a symbol or a direction changed.
        # Whoever says "true" to those too is not really looking at the details.
        tr = [_TRAPPOLE[h % len(_TRAPPOLE)]] + trappole_sottili(aff, h)
        lista = list(aff)
        for n_t, t in enumerate(tr):
            k = (h // (3 + 4 * n_t)) % (len(lista) + 1)
            lista.insert(k, t)
        elenco = "\n".join(f"{n}. {a}" for n, a in enumerate(lista, 1))
        img = scale_to_fit(j["img"], MODELS["glm"]["max_pixels"], min_pixels=300_000)
        try:
            raw, usage = srv_g().ocr(img, prompt=PROMPT_VERIFICA.format(elenco=elenco),
                                     max_tokens=MODELS["glm"]["max_tokens"], temperature=0.0,
                                     label=f"[GLM figure p.{j['page']}]")
        except Exception as e:
            _log(f"    [GLM figure {j['key']}] error: {e}")
            continue
        d = _json_morbido(raw) or {}
        risp = {str(k).strip(): str(v).strip().upper()[:1]
                for k, v in (d.get("risposte") or {}).items()}
        corr = {str(k).strip(): str(v) for k, v in (d.get("correzioni") or {}).items()}
        v = {"esito": "ok", "errori": [], "mancanti": [], "non_verificabili": 0}
        idx_tr = [n for n, a in enumerate(lista, 1) if a in tr]
        if not risp or sum(1 for n in range(1, len(lista) + 1) if str(n) in risp) < 0.7 * len(lista):
            v["esito"] = "non verificata"
            v["grezzo"] = _pulisci(raw)[:600]
        elif any(risp.get(str(n)) == "V" for n in idx_tr):
            v["esito"] = "non affidabile"
        else:
            for n, a in enumerate(lista, 1):
                if n in idx_tr:
                    continue
                r = risp.get(str(n), "")
                if r == "F":
                    c = corr.get(str(n), "")
                    v["errori"].append(a + (f" (checker: {c})" if c else ""))
                elif r == "N":
                    v["non_verificabili"] += 1
            v["mancanti"] = [str(x) for x in (d.get("mancanti") or []) if str(x).strip()][:10]
            if v["errori"] or v["mancanti"]:
                v["esito"] = "errori"
        v["affermazioni"] = len(aff)
        (cdir / (j["key"] + ".json")).write_text(json.dumps(v, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
        j[f"verifica{giro}"] = v
        media = (time.time() - t_all) / i
        _log(f"  [GLM figures] {i}/{len(todo)}: {v['esito']} ({len(v['errori'])} errors, "
             f"{len(v['mancanti'])} missing) | time left {media*(len(todo)-i)/60:.0f} min")


def _problemi(v: dict) -> str:
    righe = [f"- false according to the checker: {e}" for e in v.get("errori", [])]
    righe += [f"- visible but not mentioned: {m}" for m in v.get("mancanti", [])]
    return "\n".join(righe[:15])


def esito_figura(j: dict) -> tuple[str, str, str]:
    """(final description, status, checker notes)."""
    if j.get("corretta"):
        return j["corretta"], "corretta", ""
    v1 = j.get("verifica1")
    if not j.get("descr"):
        return "", "assente", ""
    if not v1:
        return j["descr"], "non controllata", ""
    if v1.get("esito") == "ok":
        return j["descr"], "confermata", ""
    if v1.get("esito") == "non affidabile":
        return j["descr"], "non verificata", ("the checker also confirmed false test "
                                              "statements: its check does not count here")
    if v1.get("esito") == "non verificata":
        return j["descr"], "non verificata", v1.get("grezzo", "")
    if not j.get("rev"):
        return j["descr"], "contestata", _problemi(v1)
    v2 = j.get("verifica2")
    if v2 and v2.get("esito") == "ok":
        return j["rev"], "rivista", ""
    if v2 and v2.get("esito") in ("non affidabile", "non verificata"):
        return j["rev"], "contestata", _problemi(v1)
    return j["rev"], "contestata", _problemi(v2 or v1)


# ---- blocks of handwritten pages -------------------------------------------
def prepara_blocchi_mano(results: dict, cache_dir: Path, page_image, fig_dir: Path,
                         fig_prefix: str, form_dir: Path) -> list[dict]:
    """Rebuilds Chandra's pages from the cached raw text, putting a
       placeholder in place of each block with formulas, and saves its crop."""
    tutti = []
    for p in sorted(results):
        motore, md = results[p]
        if motore != "chandra":
            continue
        rawf = cache_dir / "chandra" / f"page_{p:04d}.raw.txt"
        if not rawf.exists():
            continue
        html = _strip_fences(rawf.read_text(encoding="utf-8"))
        try:
            img = scale_to_fit(page_image(p, MODELS["chandra"]["dpi"]), MODELS["chandra"]["max_pixels"])
            bl = []
            md2 = chandra_html_to_md(html, img, p, fig_dir, fig_prefix, blocchi=bl)
        except Exception as e:
            _log(f"    [p.{p}] blocks not rebuilt: {e}")
            continue
        md2 = normalize_math(trim_repeats(md2)).strip()
        if not bl or not md2:
            continue
        hi = page_image(p, FORMULA_DPI)
        form_dir.mkdir(parents=True, exist_ok=True)
        W, H = hi.size
        for b in bl:
            x0, y0, x1, y1 = b["bbox"]
            box = (max(0, int((x0 - 6) * W / 1000)), max(0, int((y0 - 6) * H / 1000)),
                   min(W, int((x1 + 6) * W / 1000)), min(H, int((y1 + 6) * H / 1000)))
            hi.crop(box).save(form_dir / (b["id"] + ".png"))
            b["crop"] = b["id"] + ".png"
            b["pagina"] = p
        results[p] = (motore, md2)
        tutti += bl
    return tutti


def _blocco_tok(md: str):
    maths = []
    for a, b in re.findall(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$", md or "", flags=re.S):
        maths += tex_tokens(a or b) + ["|"]
    testo = re.sub(r"\$\$.+?\$\$|\$[^$\n]+?\$", " ", md or "", flags=re.S)
    parole = re.findall(r"[A-Za-zÀ-ÿ]{2,}|\d+(?:[.,]\d+)?", testo.lower())
    return maths, parole


def blocchi_concordi(a: str, b: str) -> bool:
    ma, wa = _blocco_tok(a)
    mb, wb = _blocco_tok(b)
    return ma == mb and _lev(wa, wb) <= max(2, len(wa) // 10)


def _blocco_dist(a: str, b: str) -> int:
    ma, wa = _blocco_tok(a)
    mb, wb = _blocco_tok(b)
    return 2 * _lev(ma, mb) + _lev(wa, wb)


def _diff_blocchi(a: str, b: str, max_n: int = 3) -> str:
    import difflib
    ta, tb = _blocco_tok(a)[0], _blocco_tok(b)[0]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        a0, b0 = max(0, i1 - 1), max(0, j1 - 1)
        out.append(f"«{' '.join(ta[a0:i2 + 1])}» ↔ «{' '.join(tb[b0:j2 + 1])}»")
        if len(out) >= max_n:
            out.append("...")
            break
    return "; ".join(out) or "differences in the text"


def scegli_blocco(b: dict) -> tuple[str, str, str]:
    if b.get("corretta"):
        return b["corretta"], "corretta", ""
    c, q, g = b.get("chandra", ""), b.get("qwen"), b.get("glm")
    if not q:
        return c, "non verificata", ""
    if blocchi_concordi(c, q):
        return c, "confermata", ""
    if g:
        if blocchi_concordi(g, c):
            return c, "arbitrata", "Qwen differs: " + _diff_blocchi(c, q)
        if blocchi_concordi(g, q):
            return q, "arbitrata", "Chandra differs: " + _diff_blocchi(q, c)
        cands = {"chandra": c, "qwen": q, "glm": g}
        best = min(cands, key=lambda k: (sum(_blocco_dist(cands[k], cands[o])
                                             for o in cands if o != k), k != "chandra"))
        nota = " | ".join(f"{k}: {_diff_blocchi(cands[best], v)}"
                          for k, v in cands.items() if k != best)
        return cands[best], "dubbia", nota
    return c, "dubbia", "qwen: " + _diff_blocchi(c, q)


def _trascrivi(srv: ServerPigro, img, label: str, max_tokens: int = 1500) -> str:
    img = scale_to_fit(img, 1_200_000, min_pixels=200_000)
    raw, usage = srv().ocr(img, prompt=PROMPT_TRASCRIVI.format(gloss=_gloss()),
                           max_tokens=max_tokens, temperature=0.0, label=label,
                           sampling={"repeat_penalty": 1.0})
    if usage.get("stopped") == "loop":
        return ""
    return normalize_math(_pulisci(raw)).strip()


def rileggi_blocchi(srv: ServerPigro, chi: str, blocchi: list[dict], form_dir: Path,
                    cache_dir: Path):
    from PIL import Image
    cdir = cache_dir / f"{chi}_blocchi"
    cdir.mkdir(parents=True, exist_ok=True)
    todo = []
    for b in blocchi:
        cf = cdir / (b["id"] + ".md")
        if cf.exists():
            b[chi] = compatta_md(cf.read_text(encoding="utf-8"))
        else:
            todo.append(b)
    if not todo:
        return
    nome = MODELS[chi]["nome"]
    _log(f"  [{nome}] rilegge {len(todo)} handwritten blocks ...")
    t_all = time.time()
    for i, b in enumerate(todo, 1):
        try:
            im = Image.open(form_dir / b["crop"]).convert("RGB")
            t = _trascrivi(srv, im, f"[{chi} block {b['id']}]")
        except Exception as e:
            _log(f"    [{chi} block {b['id']}] error: {e}")
            continue
        (cdir / (b["id"] + ".md")).write_text(t, encoding="utf-8")
        b[chi] = compatta_md(t)
        media = (time.time() - t_all) / i
        _log(f"  [{chi} blocks] {i}/{len(todo)} | time left {media*(len(todo)-i)/60:.0f} min")


def glm_arbitra_formule(srv: ServerPigro, formule: list[dict], form_dir: Path, cache_dir: Path):
    from PIL import Image
    cdir = cache_dir / "glm_formule"
    cdir.mkdir(parents=True, exist_ok=True)
    todo = []
    for f in formule:
        cf = cdir / (f["id"] + ".txt")
        if cf.exists():
            f["glm"] = compatta_tex(cf.read_text(encoding="utf-8"))
        elif f.get("crop"):
            todo.append(f)
    if not todo:
        return
    _log(f"  [GLM] refereeing {len(todo)} doubtful formulas ...")
    for i, f in enumerate(todo, 1):
        try:
            im = scale_to_fit(Image.open(form_dir / f["crop"]).convert("RGB"), 1_000_000,
                              min_pixels=150_000)
            raw, usage = srv().ocr(im, prompt=PROMPT_FORMULA, max_tokens=1500, temperature=0.0,
                                   label=f"[GLM formula {f['id']}]")
        except Exception as e:
            _log(f"    [GLM formula {f['id']}] error: {e}")
            continue
        t = "" if usage.get("stopped") == "loop" else tex_core(_pulisci(raw))
        (cdir / (f["id"] + ".txt")).write_text(t, encoding="utf-8")
        f["glm"] = compatta_tex(t)
        _log(f"  [GLM formulas] {i}/{len(todo)}")


_CORR_TXT_RE = re.compile(r"^###\s+(?:FIG\s+)?(\S+)\s(?:(?!\n###).)*?\nCORREZIONE[^\n]*\n```[a-z]*\n(.*?)\n```",
                          re.S | re.M)


def leggi_correzioni_testo(report: Path, cache_dir: Path, auto: dict) -> dict:
    """Corrections of handwritten blocks and figure descriptions (``` blocks)."""
    return _leggi_corr(report, cache_dir, auto, _CORR_TXT_RE, "correzioni_testo_v2.json",
                       lambda a, b: re.sub(r"\s+", " ", a.strip()) == re.sub(r"\s+", " ", b.strip()))


def _sostituisci_fig_chandra(md: str, nuove: dict) -> str:
    """Chandra's pages: replaces the description block after the image."""
    def r(m):
        name = m.group(3)
        if name not in nuove:
            return m.group(0)
        return m.group(1) + nuove[name]
    pat = re.compile(r"(!\[Figura pag\. (\d+)\]\([^)]*?(p\d{4}_fig\d+\.png)\)\s*\n\s*\n)"
                     r"> \*\*\[Figura — pag\. \d+\]\*\*[^\n]*(?:\n>[^\n]*)*")
    return pat.sub(r, md)


def ripara_dollari(md: str) -> str:
    r"""Repairs unmatched $ (one $ too many pulls all the following text
       into a formula):
       - footnotes read as '$ ^{11} $Text' -> '[^11]: Text'
       - '$a$$\to$$b$' on one line (joined inline formulas) -> '$a$ $\to$ $b$'
       - if an unmatched $ stays in a paragraph, it is 'neutralized'
         (\$) so it does not ruin the rest of the page."""
    out_lines = []
    for ln in md.split("\n"):
        ln = re.sub(r"^(\s*)\$?\s*\^\{(\d{1,3})\}\s*\$\s*(?=\S)", r"\1[^\2]: ", ln)
        ln = re.sub(r"\s*\$\s*\^\{(\d{1,3})\}\s*\$?\s*$", r" [^\1]", ln)
        st = ln.strip()
        if "$$" in st and st != "$$" and not st.startswith(">"):
            singoli = re.sub(r"\$\$", "", st).count("$")
            if singoli:                       # $$ in the middle of inline formulas
                ln = ln.replace("$$", "$ $")
        out_lines.append(ln)
    md = "\n".join(out_lines)
    # paragraph by paragraph (the $$...$$ blocks and `code` do not count)
    pezzi = re.split(r"(\n\s*\n)", md)
    for i, par in enumerate(pezzi):
        if "$" not in par:
            continue
        mask = re.sub(r"\$\$.*?\$\$|`[^`\n]*`", lambda m: " " * len(m.group(0)), par, flags=re.S)
        pos = [k for k, c in enumerate(mask) if c == "$" and (k == 0 or mask[k - 1] != "\\")]
        if len(pos) % 2 == 0:
            continue
        # the unmatched $: the last one, or the one followed by more "normal" text
        k = pos[-1]
        par = par[:k] + "\\$" + par[k + 1:]
        pezzi[i] = par
    md = "".join(pezzi)

    # inside formulas: HTML entities (&lt; &gt; &amp;) and '·' which KaTeX does not accept
    def _m(m):
        t = m.group(0)
        t = t.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("·", "\\cdot ")
        return t
    md = re.sub(r"\$\$.+?\$\$|(?<![\\$])\$[^$\n]+?\$", _m, md, flags=re.S)
    return md


def guida_per_ia() -> str:
    """Instructions at the top of the file for the AI that will read it."""
    gloss = f"\n> - Symbols used in the document: {GLOSSARIO}." if GLOSSARIO else ""
    return ("\n> **Guide for the AI reading this file.** Text converted automatically "
            "from scans with local models. Reliability:\n"
            "> - Printed pages (*paddle* engine): text and formulas HIGH reliability; the "
            "display formulas were read by several models and put to a vote.\n"
            "> - Handwritten pages (*chandra* engine, marked with ⚠): LOW reliability.\n"
            "> - Figure descriptions: written by a model; the structure is usually "
            "right, the details (handwritten labels, values, directions) may be "
            "wrong.\n"
            "> - ⚠ **Uncertain formula** / **Uncertain reading**: the models do not agree; the "
            "alternative readings are given.\n"
            "> When a marked point conflicts with the rest of the text, with other "
            "formulas or with physics, take the consistent version as right and say so. Typical "
            "confusions in automatic reading of handwriting: p↔k, a↔α↔ω↔ϱ, δ↔S, "
            "ℓ↔c↔e, M↔μ, ρ↔p, u↔μ, 1↔7, 0↔6, piston↔proton." + gloss + "\n"
            "> The images of the original pages are given in the comments "
            "`<!-- immagine pagina: ... -->`.\n\n")


def _tieni_sveglio(si: bool):
    """Windows: prevents automatic sleep during work
       (does not apply to closing the lid)."""
    if not IS_WIN:
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if si else 0))
    except Exception:
        pass


def convert_file(src: Path, mode: str, args=None,
                 out_assets: Path | None = None) -> tuple[str, Path]:
    """mode: 'paddle' | 'chandra' | 'ibrido'. Returns (markdown, media_folder).
       During the conversion the PC does not go to sleep by itself."""
    _log(f"  ocr_vlm version {VERSIONE}")
    _tieni_sveglio(True)
    try:
        return _convert_file(src, mode, args, out_assets)
    finally:
        _tieni_sveglio(False)


def _convert_file(src: Path, mode: str, args=None,
                  out_assets: Path | None = None) -> tuple[str, Path]:
    """mode: 'paddle' | 'chandra' | 'ibrido'. Returns (markdown, media_folder)."""
    import fitz
    from PIL import Image

    g = (lambda k, d=None: getattr(args, k, d) if args is not None else d)
    media_dirname = g("media_dir", "media") or "media"
    stem = src.stem
    assets = out_assets or (src.parent / media_dirname / stem)
    assets.mkdir(parents=True, exist_ok=True)
    fig_dir = assets / "figure"
    try:
        fig_prefix = (assets / "figure").relative_to(src.parent).as_posix().replace(" ", "%20")
    except ValueError:
        fig_prefix = f"{media_dirname}/{stem}/figure".replace(" ", "%20")
    cache_dir = assets / "_vlm_cache"
    threads = g("vlm_threads", None)
    gpu = bool(g("vlm_gpu", False))
    keep_page_images = bool(g("page_images", True))

    is_pdf = src.suffix.lower() == ".pdf"
    if is_pdf:
        doc = fitz.open(str(src))
        n = doc.page_count
    else:
        doc = None
        n = 1
    pages = parse_pages(g("pages", None), n)
    hq = set(parse_pages(g("hq_pages", None), n)) if g("hq_pages", None) else set()

    def page_image(p: int, dpi: int):
        if doc is not None:
            return render_page(doc, p - 1, dpi)
        return Image.open(src).convert("RGB")

    # who does what
    if mode == "paddle":
        plan = {"paddle": list(pages), "chandra": []}
    elif mode == "chandra":
        plan = {"paddle": [], "chandra": list(pages)}
    else:  # ibrido
        plan = {"paddle": [p for p in pages if p not in hq],
                "chandra": [p for p in pages if p in hq]}

    results: dict[int, tuple[str, str]] = {}   # page -> (engine, md)
    stats = {"paddle": [], "chandra": []}
    tok_hist = {"paddle": [], "chandra": []}
    page_figs: dict[int, list[str]] = {}
    page_formule: dict[int, list[dict]] = {}
    form_dir = assets / "formule"
    try:
        form_prefix = form_dir.relative_to(src.parent).as_posix().replace(" ", "%20")
    except ValueError:
        form_prefix = f"{media_dirname}/{stem}/formule".replace(" ", "%20")
    report = src.parent / f"{assets.name}_da_controllare.md"
    fig_desc: dict[str, str] = {}
    total_jobs = len(plan["paddle"]) + len(plan["chandra"])
    done_jobs = 0

    def run_engine(key: str, plist: list[int]):
        nonlocal done_jobs
        # separate cache for "full" Paddle mode (layout)
        ckey = "paddle_layout" if (key == "paddle" and plist
                                   and paddle_layout_disponibile()) else key
        if key == "paddle" and plist:
            if ckey == "paddle_layout":
                _log("  [paddle] FULL mode (layout + blocks)")
            else:
                _log("  [paddle] SIMPLE mode (whole page) - layout not available: "
                     f"{_LAYOUT_ERR or 'paddleocr not installed'}")
        todo = []
        retry_with_chandra = []
        vecchie = 0
        for p in plist:
            cf = _cache_file(cache_dir, ckey, p)
            if cf.exists() and cf.read_text(encoding="utf-8").strip():
                try:
                    mj = json.loads(cf.with_suffix(".json").read_text(encoding="utf-8"))
                except Exception:
                    mj = {}
                if ckey == "paddle_layout" and mj.get("v", 1) < LAYOUT_CACHE_V \
                        and not mj.get("vuota"):
                    vecchie += 1           # done with the previous version: redo it
                    todo.append(p)
                    continue
                results[p] = (key, cf.read_text(encoding="utf-8"))
                done_jobs += 1
                page_figs[p] = mj.get("figure") or []
                page_formule[p] = mj.get("formule") or []
                if mj.get("sospetta"):
                    retry_with_chandra.append(p)
            else:
                todo.append(p)
        if vecchie:
            _log(f"  [{key}] {vecchie} cached pages are from the previous version "
                 "(no formula verification): redoing them")
        if plist and len(todo) < len(plist):
            _log(f"  [{key}] {len(plist)-len(todo)} pages already done (cache), remaining {len(todo)}")
        if not todo:
            return retry_with_chandra
        with LlamaServer(key, threads=threads, gpu=gpu) as srv:
            layout = None
            if key == "paddle" and paddle_layout_disponibile():
                try:
                    layout = PaddleLayout(srv.url)
                    _log("  [paddle] FULL mode: layout + recognition by blocks")
                except Exception as e:
                    _log(f"  [paddle] layout not available ({type(e).__name__}: {e}); using the whole page")
            elif key == "paddle":
                _log("  [paddle] simple mode (whole page). For full mode: "
                     "python ocr_vlm.py setup layout")
            for j, p in enumerate(todo, 1):
                img = page_image(p, MODELS[key]["dpi"])
                ink = ink_ratio(img)
                if ink < 0.0003:
                    md, meta = "*(empty page)*", {"secondi": 0, "vuota": True}
                else:
                    toks = [m for m in tok_hist[key] if m]
                    exp = (sum(toks) / len(toks)) if toks else None
                    md, meta = None, None
                    if layout is not None:
                        t0p = time.time()
                        try:
                            md, figs, formule = layout.page_md(img, p, fig_dir, fig_prefix)
                            # formula reread only if the page will stay
                            # with Paddle (handwritten pages go to Chandra anyway)
                            dubbia_pag = mode == "ibrido" and leggibilita(md)["illeggibile"]
                            if formule and VERIFICA_FORMULE != "0" and not dubbia_pag:
                                paddle_rilettura_formule(srv, formule, img,
                                                         page_image(p, FORMULA_DPI),
                                                         form_dir, p)
                            meta = {"secondi": round(time.time() - t0p, 1),
                                    "modo": "layout", "fermato": None, "figure": figs,
                                    "formule": formule, "v": LAYOUT_CACHE_V}
                            page_formule[p] = formule
                        except Exception as e:
                            _log(f"    [paddle p.{p}] layout failed ({type(e).__name__}: {e}); whole page")
                            md = None
                    if md is None:
                        md, meta = _ocr_one(srv, key, img, p, fig_dir, fig_prefix,
                                            expected_tokens=exp,
                                            raw_file=cache_dir / ckey / f"page_{p:04d}.raw.txt")
                    if meta.get("token_out") and meta.get("fermato") is None:
                        tok_hist[key].append(meta["token_out"])
                    # "suspicious" page: loop, little text, or unreadable text
                    if key == "paddle":
                        meta["leggibilita"] = leggibilita(md)
                    if key == "paddle" and mode == "ibrido" and (
                            meta.get("loop") or (ink > 0.03 and len(md) < 150)
                            or meta["leggibilita"]["illeggibile"]):
                        meta["sospetta"] = True
                        retry_with_chandra.append(p)
                        _log(f"    [paddle p.{p}] unreliable text "
                             f"({meta['leggibilita']}) -> will go to Chandra")
                cf = _cache_file(cache_dir, ckey, p)
                cf.parent.mkdir(parents=True, exist_ok=True)
                if ckey == "paddle_layout":
                    meta.setdefault("v", LAYOUT_CACHE_V)
                cf.write_text(md, encoding="utf-8")
                (cf.with_suffix(".json")).write_text(json.dumps(meta), encoding="utf-8")
                results[p] = (key, md)
                page_figs[p] = meta.get("figure") or []
                stats[key].append(meta.get("secondi", 0))
                done_jobs += 1
                media = sum(stats[key]) / max(1, len(stats[key]))
                eta = media * (len(todo) - j)
                _log(f"  [{key}] page {p} ({j}/{len(todo)}): {meta.get('secondi',0):.0f} s"
                     f"{'  [LOOP]' if meta.get('loop') else ''}"
                     f"{'  [CUT AT THE LIMIT]' if meta.get('fermato') == 'limite' else ''}"
                     f"  | time left {eta/60:.0f} min")
                _log(f"  OCR: {int(done_jobs*100/max(1,total_jobs))}% ({done_jobs}/{total_jobs})")
        return retry_with_chandra

    t_start = time.time()
    retry = run_engine("paddle", plan["paddle"])
    if retry:
        _log(f"  [hybrid] {len(retry)} suspicious pages go to Chandra: {retry}")
        total_jobs += len(retry)
    full_chandra = sorted(set(plan["chandra"]) | set(retry))
    run_engine("chandra", full_chandra)

    ibrido = mode == "ibrido"
    ctrl_dir = assets / "controlli"
    try:
        ctrl_prefix = ctrl_dir.relative_to(src.parent).as_posix().replace(" ", "%20")
    except ValueError:
        ctrl_prefix = f"{media_dirname}/{stem}/controlli".replace(" ", "%20")

    # 1) handwritten pages: blocks with formulas become placeholders
    blocchi: list[dict] = []
    if ibrido and VERIFICA_MANO and any(k == "chandra" for k, _ in results.values()):
        blocchi = prepara_blocchi_mano(results, cache_dir, page_image, fig_dir, fig_prefix,
                                       form_dir)
        if blocchi:
            _log(f"  [handwritten] {len(blocchi)} blocks with formulas to reread")

    # 2) printed formulas: Chandra rereads the crops
    formule_finali = []          # (page, formula)
    for p, fl in sorted(page_formule.items()):
        if results.get(p, ("",))[0] == "paddle":
            formule_finali += [(p, f) for f in fl]
    if formule_finali and ibrido and VERIFICA_FORMULE in ("tutte", "dubbie"):
        chandra_verifica_formule([f for _, f in formule_finali], form_dir, cache_dir,
                                 threads=threads, gpu=gpu)
    if formule_finali and ibrido and LETTORI_EXTRA:
        dub = [f for _, f in formule_finali if scegli_formula(f)[1] == "dubbia"]
        if dub:
            lettori_extra_formule(dub, form_dir, cache_dir)

    # 3) figures to describe
    jobs: list[dict] = []
    if ibrido and DESCRIZIONI == "qwen":
        jobs = raccogli_figure(results, page_figs, page_image, fig_dir, ctrl_dir)
    elif ibrido and DESCRIZIONI == "chandra":
        todo_f = {p: f for p, f in page_figs.items()
                  if f and p not in full_chandra and results.get(p, ("",))[0] == "paddle"}
        if todo_f:
            describe_figures(todo_f, page_image, cache_dir, fig_dir, fig_desc,
                             threads=threads, gpu=gpu)

    # 4) Qwen: figure descriptions + second reading of handwritten blocks
    if jobs or blocchi:
        q = ServerPigro("qwen", threads, gpu)
        try:
            if jobs:
                qwen_descrivi(q, jobs, cache_dir)
            if blocchi:
                rileggi_blocchi(q, "qwen", blocchi, form_dir, cache_dir)
        finally:
            q.close()

    # 5) GLM: referee on disagreements + check of the descriptions
    if ibrido and ARBITRO == "glm":
        gs = ServerPigro("glm", threads, gpu)
        try:
            dub_f = [f for _, f in formule_finali
                     if f.get("crop") and scegli_formula(f)[1] == "dubbia"]
            if dub_f:
                glm_arbitra_formule(gs, dub_f, form_dir, cache_dir)
            dub_b = [b for b in blocchi if b.get("qwen") and scegli_blocco(b)[1] == "dubbia"]
            if dub_b:
                rileggi_blocchi(gs, "glm", dub_b, form_dir, cache_dir)
            if jobs and VERIFICA_FIGURE != "0":
                glm_verifica_figure(gs, jobs, cache_dir, giro=1)
        finally:
            gs.close()
        # 6) Qwen rewrites the disputed figures, 7) GLM checks them again
        contest = [j for j in jobs if (j.get("verifica1") or {}).get("esito") == "errori"]
        if contest:
            for j in contest:
                j["problemi"] = _problemi(j["verifica1"])
            q = ServerPigro("qwen", threads, gpu)
            try:
                qwen_descrivi(q, contest, cache_dir, revisione=True)
            finally:
                q.close()
            gs = ServerPigro("glm", threads, gpu)
            try:
                glm_verifica_figure(gs, contest, cache_dir, giro=2)
            finally:
                gs.close()

    # final choices + corrections written by hand in the review file
    scelte = {}
    for p, f in formule_finali:
        scelte[f["id"]] = scegli_formula(f)[0]
    corr = leggi_correzioni(report, cache_dir, scelte) if formule_finali else {}
    voci, formula_md = [], {}
    for p, f in formule_finali:
        if f["id"] in corr:
            f["corretta"] = corr[f["id"]]
        tex, stato, nota = scegli_formula(f)
        voci.append({"f": f, "pagina": p, "scelta": tex, "stato": stato, "nota": nota})
        formula_md[f["id"]] = _formula_md(f, tex, stato, nota)

    auto_txt = {b["id"]: scegli_blocco(b)[0] for b in blocchi}
    auto_txt.update({j["key"]: esito_figura(j)[0] for j in jobs})
    corr_t = leggi_correzioni_testo(report, cache_dir, auto_txt) if auto_txt else {}
    voci_blocchi, blocco_md = [], {}
    for b in blocchi:
        if b["id"] in corr_t:
            b["corretta"] = corr_t[b["id"]]
        t, stato, nota = scegli_blocco(b)
        voci_blocchi.append({"b": b, "pagina": b["pagina"], "scelta": t, "stato": stato, "nota": nota})
        if stato == "dubbia":
            alt = [b[k] for k in ("chandra", "qwen", "glm")
                   if b.get(k) and b[k].strip() != t.strip()]
            # the alternative readings go in `code`: the AI reads them, but they are not
            # interpreted as formulas (a cut piece would break the $...$)
            alt = " ⟂ ".join("`" + re.sub(r"\s+", " ", a_).strip()[:400].replace("`", "'") + "`"
                             for a_ in alt)
            t += (f"\n\n> ⚠ **Uncertain reading** (block {b['id']}, handwritten): the automatic "
                  f"readings do not agree ({nota}). Other readings: {alt}")
        blocco_md[b["id"]] = t
    voci_fig = []
    fig_pag = {}
    for j in jobs:
        if j["key"] in corr_t:
            j["corretta"] = corr_t[j["key"]]
        d, stato, nota = esito_figura(j)
        voci_fig.append({"j": j, "scelta": d, "stato": stato, "nota": nota})
        if not d:
            continue
        testo = d
        if stato == "contestata" and nota:
            testo += ("\n\n⚠ *DISPUTED automatic description: the second model flags these "
                      "points as false or missing (check against the page text):*\n" + nota)
        elif stato == "non verificata":
            testo += ("\n\n⚠ *UNVERIFIED automatic description (the checker was not "
                      "reliable on this figure): symbols, values and directions may be "
                      "wrong; in case of conflict the page text prevails.*")
        elif stato == "non controllata":
            testo += ("\n\n*(Automatic description, not checked by a second model: "
                      "details may contain errors.)*")
        elif stato in ("confermata", "rivista"):
            testo += ("\n\n*(Automatic description, checked by a second model; "
                      "fine details may still contain errors.)*")
        fig_desc[j["names"][0]] = testo
        fig_pag[j["names"][0]] = j["page"]
        for nm in j["names"][1:]:
            fig_desc[nm] = "*(see the overall description in the first figure on this page)*"
            fig_pag[nm] = j["page"]
    for nome, vv in (("printed formulas", voci), ("handwritten blocks", voci_blocchi),
                     ("figure", voci_fig)):
        if vv:
            cont = {}
            for v in vv:
                cont[v["stato"]] = cont.get(v["stato"], 0) + 1
            _log(f"  [check {nome}] " + ", ".join(f"{k}: {n}" for k, n in sorted(cont.items())))

    # page images (visual backup, light WebP)
    page_links = {}
    if keep_page_images and doc is not None:
        pdir = assets / "_pages"
        pdir.mkdir(parents=True, exist_ok=True)
        for p in pages:
            name = f"page_{p:04d}.webp"
            if not (pdir / name).exists():
                im = render_page(doc, p - 1, 110)
                im.save(pdir / name, "WEBP", quality=70, method=4)
            page_links[p] = f"{media_dirname}/{stem}/_pages/{name}".replace(" ", "%20")
    if doc is not None:
        doc.close()

    # figure descriptions in place of the placeholders
    def _fill(md: str) -> str:
        def r(m):
            d = fig_desc.get(m.group(1))
            if not d:
                return ("*(cropped image; description available with --engine vlm_ibrido)*"
                        if mode != "ibrido" else "*(no description)*")
            return d.strip().replace("\n", "\n> ")
        md = re.sub(r"\{\{DESCR:([^}]+)\}\}", r, md)

        def rf(m):              # formulas: verified version (or Paddle's)
            fid = m.group(1)
            if fid in formula_md:
                return formula_md[fid]
            return "\n\n*(formula not available)*\n\n"
        md = re.sub(r"\{\{FORMULA:([^}]+)\}\}", rf, md)
        bmap = {b["id"]: b.get("chandra", "") for b in blocchi}
        md = re.sub(r"\{\{BLOCCO:([^}]+)\}\}",
                    lambda m: blocco_md.get(m.group(1), bmap.get(m.group(1), "")), md)
        return re.sub(r"\n{3,}", "\n\n", md)
    results = {p: (k, ripara_dollari(_fill(md))) for p, (k, md) in results.items()}
    # Chandra's pages: figures get the verified description (Qwen/GLM)
    nuove_ch = {nm: (f"> **[Figura — pag. {fig_pag[nm]}]** "
                     + fig_desc[nm].strip().replace("\n", "\n> "))
                for nm in fig_desc if re.match(r"p\d{4}_fig\d+\.png$", nm) and nm in fig_pag}
    if nuove_ch:
        results = {p: (k, _sostituisci_fig_chandra(md, nuove_ch) if k == "chandra" else md)
                   for p, (k, md) in results.items()}

    # "to review" file: doubtful formulas + suspicious words
    from collections import Counter
    freq_doc = Counter(w.lower() for k, md in results.values()
                       for w in re.findall(r"[A-Za-zÀ-ÿ]{5,}", md))
    parole = {p: parole_sospette(md, freq_doc=freq_doc)
              for p, (k, md) in results.items() if k == "paddle"}
    if voci or voci_blocchi or voci_fig or any(parole.values()):
        try:
            scrivi_da_controllare(report, form_prefix, voci, parole, page_links,
                                  voci_blocchi, voci_fig, ctrl_prefix, cache_dir)
            n_dub = sum(1 for v in voci if v["stato"] == "dubbia")
            n_bd = sum(1 for v in voci_blocchi if v["stato"] == "dubbia")
            n_fc = sum(1 for v in voci_fig if v["stato"] in ("contestata", "non verificata"))
            _log(f"  To review: {n_dub} doubtful formulas, {n_bd} doubtful handwritten blocks, "
                 f"{n_fc} disputed figures, "
                 f"{sum(len(w) for w in parole.values())} suspicious words -> {report.name}")
        except Exception as e:
            _log(f"  [warning] cannot write {report.name}: {e}")

    # assembly (without repeated headers/footers)
    if len(results) >= 2:
        cleaned = strip_running_lines({p: md for p, (k, md) in results.items()})
        results = {p: (results[p][0], cleaned[p]) for p in results}
    else:
        results = {p: (k, "\n".join(l for l in md.splitlines()
                                     if not _PAGE_NUM_RE.match(l)).strip())
                   for p, (k, md) in results.items()}
    parts = []
    issues = 0
    for p in pages:
        key, md = results.get(p, ("?", "*(page not converted)*"))
        issues += math_issues(md)
        parts.append(f"\n\n<!-- ===== Page {p} of {n} | engine: {key} ===== -->\n")
        if p in page_links:
            parts.append(f"<!-- page image: {page_links[p]} -->\n")
        if key == "chandra" and mode == "ibrido":
            parts.append("\n> ⚠ **Handwritten page** — automatic transcription with LOW "
                         "reliability: symbols and numbers may be misread (see the "
                         "guide at the top of the file). If something does not add up, rebuild it from the "
                         "context, nearby formulas and physics.\n")
        parts.append("\n" + md + "\n")
    tot = time.time() - t_start
    sp = stats["paddle"]; sc = stats["chandra"]
    summary = (f"<!-- OCR VLM locale: {len(pages)} pages | "
               f"Paddle {len(sp)} pages (average {sum(sp)/max(1,len(sp)):.0f} s) | "
               f"Chandra {len(sc)} pages (average {sum(sc)/max(1,len(sc)):.0f} s) | "
               f"verified display formulas: {len(voci)} (doubtful: "
               f"{sum(1 for v in voci if v['stato'] == 'dubbia')}) | "
               f"verified handwritten blocks: {len(voci_blocchi)} (doubtful: "
               f"{sum(1 for v in voci_blocchi if v['stato'] == 'dubbia')}) | "
               f"figures: {len(voci_fig)} (disputed: "
               f"{sum(1 for v in voci_fig if v['stato'] == 'contestata')}) | see {report.name} | "
               f"formulas with suspicious syntax: {issues} | total {tot/60:.1f} min -->\n")
    _log("  " + summary.strip("<!-> \n"))
    guida = guida_per_ia() if mode == "ibrido" else ""
    return summary + guida + "".join(parts).strip() + "\n", assets


def prova_modello(key: str) -> bool:
    """Starts the model and has it read a small test image."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (640, 200), "white")
    d = ImageDraw.Draw(im)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=48)
    except Exception:
        font = None
    d.text((30, 60), "PROVA 2026 ok", fill="black", font=font)
    _log(f"Test of {MODELS[key]['nome']} ...")
    try:
        with LlamaServer(key) as srv:
            t0 = time.time()
            raw, _ = srv.ocr(im, prompt="What text is written in the image? Answer with the text only.",
                             max_tokens=300, label=f"[test {key}]")
        ok = "2026" in raw
        _log(f"  answer: {_pulisci(raw)[:80]!r} ({time.time()-t0:.0f} s) -> "
             f"{'WORKS' if ok else 'odd answer, check it'}")
        return ok
    except Exception as e:
        _log(f"  [ERROR] {MODELS[key]['nome']} does not start: {e}\n"
             "  Probably llama.cpp is too old for this model: run\n"
             "      python ocr_vlm.py setup aggiorna-llamacpp\n  and try again.")
        return False


# ----------------------------------------------------------------------------
# CLI: setup / test
# ----------------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(description="Local OCR with PaddleOCR-VL and Chandra OCR 2 (llama.cpp).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="Downloads llama.cpp and the models (one time).")
    s.add_argument("quali", nargs="?", default="tutti",
                   choices=["tutti", "paddle", "chandra", "layout", "qwen", "glm",
                            "controllori", "aggiorna-llamacpp"])
    s.add_argument("--gpu", action="store_true", help="Also download llama.cpp Vulkan (Intel graphics).")
    p = sub.add_parser("prova", help="Test on some pages: writes <name>_prova_<engine>.md")
    p.add_argument("file")
    p.add_argument("--pages", default="1-3")
    p.add_argument("--engine", default="tutti", choices=["tutti", "paddle", "chandra"])
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    if a.cmd == "setup" and a.quali == "layout":
        _log("Installing PaddlePaddle (CPU) + PaddleOCR (full pipeline with layout) ...")
        r = subprocess.call([sys.executable, "-m", "pip", "install", "-U",
                             "--timeout", "120", "--retries", "10",
                             "paddlepaddle", "paddleocr[doc-parser]", "wordfreq"])
        if r != 0:
            _log("Installation failed: run the command again (did the network drop?)")
            return
        _log("Done. The layout model downloads by itself on first use.")
        return
    if a.cmd == "setup" and a.quali == "aggiorna-llamacpp":
        old = MOTORI_DIR / "llamacpp"
        if old.exists():
            bak = MOTORI_DIR / f"llamacpp_vecchio_{time.strftime('%Y%m%d_%H%M%S')}"
            old.rename(bak)
            _log(f"Previous version moved to {bak.name}")
        exe = ensure_llamacpp(False)
        _log(f"llama.cpp updated: {exe} ({(exe.parent / 'VERSION.txt').read_text() if (exe.parent / 'VERSION.txt').exists() else '?'})")
        return
    if a.cmd == "setup":
        ensure_llamacpp(False)
        if a.gpu:
            ensure_llamacpp(True)
        keys = {"tutti": ["paddle", "chandra", "qwen", "glm"],
                "controllori": ["qwen", "glm"]}.get(a.quali, [a.quali])
        for k in keys:
            ensure_model(k)
        _log("Setup completed. Models in: " + str(MODELS_DIR))
        for k in keys:
            if k in ("qwen", "glm"):
                prova_modello(k)
        return

    src = Path(a.file)
    keys = ["paddle", "chandra"] if a.engine == "tutti" else [a.engine]
    for k in keys:
        class A: pass
        args = A()
        args.pages, args.vlm_threads, args.vlm_gpu = a.pages, a.threads, a.gpu
        args.media_dir, args.page_images = "media", False
        assets = src.parent / "media" / f"{src.stem}_prova_{k}"
        t0 = time.time()
        md, _ = convert_file(src, k, args, out_assets=assets)
        out = src.with_name(f"{src.stem}_prova_{k}.md")
        out.write_text(md, encoding="utf-8")
        _log(f"==> {out.name}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    _cli()
