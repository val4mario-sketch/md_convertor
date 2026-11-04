# -*- coding: utf-8 -*-
"""
ocr_vlm.py — OCR "intelligente" con modelli di VISIONE LOCALI (gratis, offline).

Due motori, entrambi eseguiti da llama.cpp (llama-server) sulla CPU:

  * PADDLE  = PaddleOCR-VL 1.6 (Baidu, ~0.9 miliardi di parametri)
              veloce, il migliore su testo dattiloscritto, formule e tabelle.
              Restituisce testo + LaTeX (niente struttura/figure).
  * CHANDRA = Chandra OCR 2 (Datalab, ~5 miliardi di parametri, quant. 4 bit)
              piu' lento, il migliore tra i modelli aperti su SCRITTURA A MANO;
              restituisce la pagina strutturata (titoli, liste, tabelle,
              formule KaTeX) e DESCRIVE le figure, che vengono anche ritagliate
              e salvate come immagini.
  * IBRIDO  = Paddle su tutte le pagine; Chandra sulle pagine indicate con
              --hq-pages e su quelle dove Paddle fallisce (testo troppo corto
              rispetto all'inchiostro, ripetizioni in loop).

Tutto in locale: i modelli si scaricano UNA volta in  <questa cartella>\\modelli
e llama.cpp sta in  <questa cartella>\\motori\\llamacpp .

Uso da tomd2:   python tomd2.py file.pdf --engine vlm_paddle
                python tomd2.py file.pdf --engine vlm_chandra --pages 1-20
                python tomd2.py file.pdf --engine vlm_ibrido --hq-pages 379-453
Uso diretto:    python ocr_vlm.py setup tutti          (scarica modelli + llama.cpp)
                python ocr_vlm.py prova file.pdf --pages 12,381 --engine tutti
                    -> scrive file_prova_paddle.md / file_prova_chandra.md
                       (NON tocca file .md esistenti) e stampa i tempi.

Riprendibile: ogni pagina finita e' salvata in media/<nome>/_vlm_cache/, quindi
se interrompi e rilanci riparte dalla prima pagina mancante.
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
# Percorsi e configurazione
# ----------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("TOMD2_MODELS_DIR", BASE_DIR / "modelli"))
MOTORI_DIR = BASE_DIR / "motori"
LOG_DIR    = BASE_DIR / "logs"

HF = "https://huggingface.co"
IS_WIN = os.name == "nt"

# Quantizzazione Chandra: IQ4_NL (3,05 GB, consigliata con 16 GB di RAM).
# Alternative: IQ3_M (2,49 GB, piu' veloce, meno preciso), Q6_K (4,08 GB).
CHANDRA_QUANT = os.environ.get("TOMD2_CHANDRA_QUANT", "IQ4_NL")
# Quantizzazione Paddle: Q8_0 (498 MB). Il modello e' piccolo: meglio non
# scendere di piu', la precisione su formule ne risentirebbe.
PADDLE_QUANT = os.environ.get("TOMD2_PADDLE_QUANT", "Q8_0")

MODELS = {
    "paddle": {
        "nome": "PaddleOCR-VL 1.6",
        "files": [
            # (repo HuggingFace, file remoto, nome locale)
            ("SanjeevSOLANKI/PaddleOCR-VL-1.6-GGUF",
             f"PaddleOCR-VL-1.6-{PADDLE_QUANT}.gguf", None),
            ("SanjeevSOLANKI/PaddleOCR-VL-1.6-GGUF",
             "PaddleOCR-VL-1.6-mmproj.gguf", None),
            # template di chat (alcune build di llama.cpp crashano senza)
            ("EstevaoNaval/PaddleOCR-VL-1.6-GGUF",
             "chat_template.jinja", "PaddleOCR-VL-1.6-chat_template.jinja"),
        ],
        "port": 11436,
        "ctx": 8192,
        "max_tokens": int(os.environ.get("TOMD2_PADDLE_MAXTOK", "3000")),
        "dpi": 200,
        "max_pixels": 2_000_000,      # ~1190x1680 su A4: buon compromesso CPU
        "prompt": "OCR:",             # NON cambiare: e' un modello specializzato
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
        "dpi": 192,                   # come il progetto ufficiale
        # l'ufficiale usa fino a 3072x2048 (6,3 MP) su GPU; su CPU e' troppo:
        # 3,1 MP (~1470x2090 su A4) tiene leggibili pedici e scrittura a mano.
        "max_pixels": 3_100_000,
        "prompt": None,               # -> CHANDRA_PROMPT (sotto)
        "sampling": {"temperature": 0.0, "top_p": 0.1},
    },
}

# Prompt ufficiale di Chandra OCR 2 (datalab-to/chandra, chandra/prompts.py,
# OCR_LAYOUT_PROMPT): pagina -> HTML a blocchi con bbox ed etichette.
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
# Download (con ripresa) di modelli e llama.cpp
# ----------------------------------------------------------------------------
def _download(url: str, dest: Path, label: str = "") -> Path:
    """Scarica url -> dest con RIPRESA (file .part) e percentuale a video."""
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
        if e.code == 416 and have:          # gia' completo
            part.rename(dest)
            return dest
        raise RuntimeError(f"download fallito ({e.code}) per {url}") from e
    total = resp.headers.get("Content-Length")
    total = int(total) + have if total else None
    if have and resp.status != 206:         # il server ignora il Range: riparto
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
        raise RuntimeError(f"download interrotto ({done}/{total} byte): rilancia, riprende da li'")
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
        _log(f"Scarico {spec['nome']}: {remote} (una tantum) ...")
        url = f"{HF}/{repo}/resolve/main/{remote}?download=true"
        try:
            _download(url, dest, label=remote)
        except Exception as e:
            if remote.endswith(".jinja"):   # il template e' facoltativo
                _log(f"  [avviso] template non scaricato ({e}): proseguo senza.")
                continue
            raise
    return model_paths(key)


def _server_exe_candidates(gpu: bool) -> list[Path]:
    exe = "llama-server.exe" if IS_WIN else "llama-server"
    c = []
    env = os.environ.get("TOMD2_LLAMA_SERVER")
    if env:
        c.append(Path(env))
    if gpu:                      # con la GPU serve la build Vulkan, non la CPU
        c.append(MOTORI_DIR / "llamacpp-vulkan" / exe)
        return c
    c.append(MOTORI_DIR / "llamacpp" / exe)
    if IS_WIN:   # installazione gia' presente sul PC (usata da img_describe)
        c.append(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "llamacpp" / exe)
    w = shutil.which("llama-server")
    if w:
        c.append(Path(w))
    return c


def ensure_llamacpp(gpu: bool = False) -> Path:
    """Ritorna il percorso di llama-server; se manca scarica l'ultima build
       Windows da GitHub (CPU, oppure Vulkan per la grafica Intel se gpu=True)."""
    for p in _server_exe_candidates(gpu):
        if p.exists():
            return p
    if not IS_WIN:
        raise RuntimeError("llama-server non trovato (installa llama.cpp o imposta TOMD2_LLAMA_SERVER).")
    flavor = "vulkan-x64" if gpu else "cpu-x64"
    target = MOTORI_DIR / ("llamacpp-vulkan" if gpu else "llamacpp")
    _log(f"Scarico llama.cpp ({flavor}) da GitHub ...")
    with urllib.request.urlopen(urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
            headers={"User-Agent": "tomd2"}), timeout=60) as r:
        rel = json.load(r)
    asset = next((a for a in rel["assets"]
                  if a["name"].endswith(f"bin-win-{flavor}.zip")), None)
    if not asset:
        raise RuntimeError(f"build llama.cpp '{flavor}' non trovata nella release {rel.get('tag_name')}")
    zpath = MOTORI_DIR / asset["name"]
    _download(asset["browser_download_url"], zpath)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(target)
    (target / "VERSION.txt").write_text(rel.get("tag_name", "?"), encoding="utf-8")
    exe = target / "llama-server.exe"
    if not exe.exists():
        raise RuntimeError("llama-server.exe non trovato nello zip scaricato")
    return exe


# ----------------------------------------------------------------------------
# Server llama.cpp (avvio/arresto automatico)
# ----------------------------------------------------------------------------
def _http_json(url: str, payload: dict | None = None, timeout: float = 30):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class LlamaServer:
    """with LlamaServer('chandra') as srv: srv.ocr(img) ...
       Avvia llama-server col modello giusto, aspetta che sia pronto, lo
       chiude all'uscita. Il log va in logs/llama_server_<motore>.log."""

    def __init__(self, key: str, threads: int | None = None, gpu: bool = False):
        self.key = key
        self.spec = MODELS[key]
        self.paths = ensure_model(key)
        self.exe = ensure_llamacpp(gpu)
        self.gpu = gpu
        self.threads = threads or max(1, (os.cpu_count() or 4) // 2)  # core fisici
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
        if self._healthy():           # gia' in esecuzione (es. lanciato a mano)
            return self
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        flags = 0x08000000 if IS_WIN else 0   # CREATE_NO_WINDOW
        for use_template in (True, False):
            self.logf = open(LOG_DIR / f"llama_server_{self.key}.log", "w",
                             encoding="utf-8", errors="replace")
            cmd = self._cmd(use_template)
            self.logf.write("CMD: " + " ".join(cmd) + "\n\n")
            self.logf.flush()
            _log(f"  Avvio {self.spec['nome']} (llama.cpp, {self.threads} thread"
                 f"{', GPU Vulkan' if self.gpu else ''}) ...")
            self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT,
                                         creationflags=flags)
            t0 = time.time()
            while time.time() - t0 < 600:      # caricamento da disco: fino a 10 min
                if self.proc.poll() is not None:
                    break
                if self._healthy():
                    _log(f"  {self.spec['nome']} pronto in {time.time()-t0:.0f} s")
                    return self
                time.sleep(2)
            self.stop()
            if not use_template:
                break
            _log("  [avviso] avvio fallito col template di chat: riprovo senza.")
        raise RuntimeError(f"llama-server non parte per {self.spec['nome']}: "
                           f"guarda {LOG_DIR / f'llama_server_{self.key}.log'} "
                           "(spesso: llama.cpp troppo vecchio o RAM insufficiente)")

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

    # -- richiesta OCR -------------------------------------------------------
    def ocr(self, img, prompt: str | None = None, max_tokens: int | None = None,
            temperature: float | None = None, expected_tokens: float | None = None,
            label: str = "") -> tuple[str, dict]:
        """Invia la pagina al modello in STREAMING: mostra l'avanzamento ogni
           ~10 s (token, velocita', % stimata, ultima riga) e interrompe
           SUBITO la generazione se il modello entra in un loop."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        limit = max_tokens or self.spec["max_tokens"]
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt or self.spec["prompt"] or CHANDRA_PROMPT},
            ]}],
            "max_tokens": limit,
            "stream": True,
            "stream_options": {"include_usage": True},
            # niente "ragionamento" nascosto (modelli tipo Qwen3.5): solo l'output
            "chat_template_kwargs": {"enable_thinking": False},
        }
        payload.update(self.spec["sampling"])
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
                    _log(f"    {label} immagine letta in {t_first - t0:.0f} s, scrivo il testo...")
                now = time.time()
                if now - last_print >= 10 and ntok:
                    last_print = now
                    speed = ntok / max(now - (t_first or t0), 1e-6)
                    pct = ""
                    if expected_tokens:
                        pct = f" (~{min(99, int(ntok * 100 / expected_tokens))}% stimato)"
                    tail = "".join(parts)[-70:].replace("\n", " ").strip()
                    _log(f"    {label} {ntok} token{pct} | {speed:.1f} t/s | ...{tail}")
                # controllo loop ogni ~100 token: se si ripete, stop subito
                if ntok % 100 == 0 and ntok >= 300 and detect_repeat("".join(parts)):
                    stopped = "loop"
                    _log(f"    {label} [LOOP] il modello si ripete: interrompo a {ntok} token")
                    break
        text = "".join(parts)
        if not text.strip() and think:          # modello che ha scritto solo nel "ragionamento"
            text = "".join(think)
        usage = dict(usage or {})
        usage.setdefault("completion_tokens", ntok)
        usage["stopped"] = stopped or ("limite" if ntok >= limit else None)
        return text, usage


# ----------------------------------------------------------------------------
# Immagini di pagina
# ----------------------------------------------------------------------------
def scale_to_fit(img, max_pixels: int, min_pixels: int = 1792 * 28, grid: int = 28):
    """Ridimensiona mantenendo le proporzioni, multipli di 28 px (patch
       del vision encoder), entro max_pixels. Porting da chandra/model/util.py."""
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
    """Frazione di pixel 'scuri' (testo/disegni) nella parte centrale della
       pagina: serve a capire se una pagina vuota e' davvero vuota."""
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
    """True se il testo finisce con una sequenza ripetuta troppe volte
       (loop del modello). Porting semplificato da chandra/model/util.py."""
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
    # righe identiche consecutive
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    run = 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b else 1
        if run >= 8:
            return True
    return False


def trim_repeats(text: str) -> str:
    """Taglia le righe identiche ripetute oltre la 3a volta consecutiva."""
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


def math_issues(md: str) -> int:
    """Conta le formule con parentesi graffe sbilanciate o \\left/\\right
       spaiati (controllo leggero, senza KaTeX)."""
    n = 0
    formulas = re.findall(r"\$\$(.+?)\$\$", md, flags=re.S)
    rest = re.sub(r"\$\$(.+?)\$\$", " ", md, flags=re.S)
    formulas += re.findall(r"(?<![\\$])\$([^$\n]+?)\$", rest)
    for f in formulas:
        f2 = f.replace(r"\{", "").replace(r"\}", "")
        if f2.count("{") != f2.count("}") or f.count(r"\left") != f.count(r"\right"):
            n += 1
    if (rest.count("$") - rest.count(r"\$")) % 2:
        n += 1
    return n


def _md_escape_fallback(html: str) -> str:
    """Conversione HTML->Markdown minimale (se mancano bs4/markdownify)."""
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


def fix_list_markers(md: str) -> str:
    """'- - testo', '- o testo', '- • testo' -> '- testo' (il modello copia
       anche il trattino/cerchietto originale dentro la voce di elenco)."""
    return re.sub(r"(?m)^([ \t]*)[-*][ \t]+(?:[-–•◦▪○o]|\\-)[ \t]+(?=\S)", r"\1- ", md)


def chandra_html_to_md(html: str, img, page_no: int, fig_dir: Path,
                       fig_prefix: str) -> str:
    """HTML a blocchi di Chandra -> Markdown nello stile tomd2:
       - salta intestazioni/piè di pagina (Page-Header/Page-Footer)
       - figure: ritaglio salvato come PNG + blocco descrizione
       - formule <math> -> $...$ / $$...$$ (KaTeX)"""
    try:
        from bs4 import BeautifulSoup
        from markdownify import MarkdownConverter
    except ImportError:
        return _md_escape_fallback(html)

    class _MD(MarkdownConverter):
        def convert_table(self, el, text, *a, **k):
            return "\n\n" + str(el) + "\n\n"     # tabelle complesse: HTML intatto

    def to_md(fragment: str) -> str:
        # le formule vengono tolte PRIMA di markdownify (che altrimenti
        # "escaperebbe" _ * \\ ...) e rimesse dopo, intatte
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
        parts.append(to_md(str(div.decode_contents())))
    md = "\n\n".join(p for p in parts if p.strip())
    return re.sub(r"\n{3,}", "\n\n", md).strip()


# ----------------------------------------------------------------------------
# Selezione pagine
# ----------------------------------------------------------------------------
def parse_pages(spec: str | None, n: int) -> list[int]:
    """'1-10,15,379-383' -> [1..10, 15, 379..383] (1-based, entro n)."""
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
# Conversione di un PDF / immagine
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
    if usage.get("stopped") == "loop":     # loop: secondo tentativo "mosso"
        retried = True
        _log(f"    {label} secondo tentativo...")
        raw2, usage2 = srv.ocr(img, temperature=0.3, expected_tokens=expected_tokens,
                               label=label)
        if usage2.get("stopped") != "loop" or len(raw2) > len(raw):
            raw, usage = raw2, usage2
    if raw_file is not None:               # sempre: il testo grezzo del modello
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        raw_file.write_text(raw, encoding="utf-8")
    raw_clean = _strip_fences(raw)
    md = ""
    if key == "chandra":
        try:
            md = chandra_html_to_md(raw_clean, img, page_no, fig_dir, fig_prefix)
        except Exception as e:
            _log(f"    {label} [avviso] conversione HTML fallita ({e}): uso il ripiego")
        if not md.strip() and raw_clean.strip():
            md = _md_escape_fallback(raw_clean)      # HTML troncato o anomalo
    else:
        md = raw_clean
    md = normalize_math(trim_repeats(md)).strip()
    if not md and raw_clean.strip():
        md = raw_clean.strip()                       # mai perdere il testo letto
    meta = {"secondi": round(time.time() - t0, 1),
            "token_out": usage.get("completion_tokens"),
            "token_in": usage.get("prompt_tokens"),
            "fermato": usage.get("stopped"),
            "loop": usage.get("stopped") == "loop", "ritentato": retried}
    return md, meta


# ----------------------------------------------------------------------------
# PaddleOCR-VL "completo" (pipeline ufficiale Baidu): un piccolo modello di
# LAYOUT (PP-DocLayoutV3, gira su CPU) divide la pagina in blocchi (titoli,
# paragrafi, formule, tabelle, figure, note) e manda OGNI blocco al modello
# PaddleOCR-VL sul nostro llama-server con il prompt giusto ("OCR:",
# "Formula Recognition:", "Table Recognition:"). E' cosi' che Paddle e' stato
# pensato e misurato nei benchmark: molto meglio della pagina intera.
# Richiede:  pip install paddlepaddle "paddleocr[doc-parser]"   (setup layout)
# Se non e' installato, si ripiega sulla pagina intera con "OCR:".
# ----------------------------------------------------------------------------
def paddle_layout_disponibile() -> bool:
    if os.environ.get("TOMD2_PADDLE_LAYOUT", "1") == "0":
        return False
    try:
        import paddle  # noqa: F401
        import paddleocr  # noqa: F401
        return True
    except Exception:
        return False


class PaddleLayout:
    def __init__(self, server_url: str):
        from paddleocr import PaddleOCRVL
        _log("  Carico il modello di layout PP-DocLayoutV3 (la prima volta lo scarica) ...")
        self.pipe = PaddleOCRVL(
            pipeline_version="v1.6",
            vl_rec_backend="llama-cpp-server",
            vl_rec_server_url=server_url.rstrip("/") + "/v1",
            vl_rec_max_concurrency=1,          # un blocco alla volta: CPU
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_chart_recognition=False,
            # togliamo intestazioni/pie' di pagina/numeri, ma TENIAMO le note
            markdown_ignore_labels=["number", "header", "header_image",
                                    "footer", "footer_image", "aside_text"],
        )

    def page_md(self, img, page_no: int, fig_dir: Path, fig_prefix: str) -> str:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ip = Path(td) / f"p{page_no:04d}.png"
            img.save(ip)
            parts = []
            for res in self.pipe.predict(str(ip)):
                mdd = res.markdown
                text = mdd.get("markdown_texts", "") or ""
                for rel, pil in (mdd.get("markdown_images") or {}).items():
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    name = f"p{page_no:04d}_" + Path(rel).name
                    try:
                        pil.save(fig_dir / name)
                    except Exception:
                        continue
                    text = text.replace(rel, f"{fig_prefix}/{name}")
                parts.append(text)
        md = "\n\n".join(parts)
        # segnaposto descrittivo accanto a ogni figura ritagliata
        md = re.sub(r"(<img[^>]*>|!\[[^\]]*\]\([^)]*\))",
                    lambda m: m.group(1) + f"\n\n> **[Figura — pag. {page_no}]** "
                    "*(immagine ritagliata; nessuna descrizione con questo motore)*", md)
        return normalize_math(md).strip()


# righe di intestazione / pie' di pagina da eliminare (Paddle le trascrive)
_PAGE_NUM_RE = re.compile(r"^\s*(pag(\.|ina)?\s*)?\d+\s*(a|di|/|of)\s*\d+\s*$", re.I)


def strip_running_lines(pages_md: dict[int, str]) -> dict[int, str]:
    """Toglie intestazioni / pie' di pagina: righe IDENTICHE che compaiono
       nelle prime/ultime 2 righe di almeno meta' delle pagine (minimo 3), e
       le righe tipo 'Pag. 12 a 506'. Non tocca mai il resto del testo."""
    from collections import Counter
    def norm(l):
        return re.sub(r"\s+", " ", l.strip().lower())
    def edges(md):
        idx = [i for i, l in enumerate(md.splitlines()) if l.strip()]
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


def convert_file(src: Path, mode: str, args=None,
                 out_assets: Path | None = None) -> tuple[str, Path]:
    """mode: 'paddle' | 'chandra' | 'ibrido'. Ritorna (markdown, cartella_media)."""
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

    # chi fa cosa
    if mode == "paddle":
        plan = {"paddle": list(pages), "chandra": []}
    elif mode == "chandra":
        plan = {"paddle": [], "chandra": list(pages)}
    else:  # ibrido
        plan = {"paddle": [p for p in pages if p not in hq],
                "chandra": [p for p in pages if p in hq]}

    results: dict[int, tuple[str, str]] = {}   # pagina -> (motore, md)
    stats = {"paddle": [], "chandra": []}
    tok_hist = {"paddle": [], "chandra": []}
    total_jobs = len(plan["paddle"]) + len(plan["chandra"])
    done_jobs = 0

    def run_engine(key: str, plist: list[int]):
        nonlocal done_jobs
        # cache separata per la modalita' Paddle "completa" (layout)
        ckey = "paddle_layout" if (key == "paddle" and plist
                                   and paddle_layout_disponibile()) else key
        todo = []
        retry_with_chandra = []
        for p in plist:
            cf = _cache_file(cache_dir, ckey, p)
            if cf.exists() and cf.read_text(encoding="utf-8").strip():
                results[p] = (key, cf.read_text(encoding="utf-8"))
                done_jobs += 1
                try:
                    if json.loads(cf.with_suffix(".json").read_text(encoding="utf-8")).get("sospetta"):
                        retry_with_chandra.append(p)
                except Exception:
                    pass
            else:
                todo.append(p)
        if plist and len(todo) < len(plist):
            _log(f"  [{key}] {len(plist)-len(todo)} pagine gia' fatte (cache), ne restano {len(todo)}")
        if not todo:
            return retry_with_chandra
        with LlamaServer(key, threads=threads, gpu=gpu) as srv:
            layout = None
            if key == "paddle" and paddle_layout_disponibile():
                try:
                    layout = PaddleLayout(srv.url)
                    _log("  [paddle] modalita' COMPLETA: layout + riconoscimento per blocchi")
                except Exception as e:
                    _log(f"  [paddle] layout non disponibile ({type(e).__name__}: {e}); uso la pagina intera")
            elif key == "paddle":
                _log("  [paddle] modalita' semplice (pagina intera). Per la modalita' completa: "
                     "python ocr_vlm.py setup layout")
            for j, p in enumerate(todo, 1):
                img = page_image(p, MODELS[key]["dpi"])
                ink = ink_ratio(img)
                if ink < 0.0003:
                    md, meta = "*(pagina vuota)*", {"secondi": 0, "vuota": True}
                else:
                    toks = [m for m in tok_hist[key] if m]
                    exp = (sum(toks) / len(toks)) if toks else None
                    md, meta = None, None
                    if layout is not None:
                        t0p = time.time()
                        try:
                            md = layout.page_md(img, p, fig_dir, fig_prefix)
                            meta = {"secondi": round(time.time() - t0p, 1),
                                    "modo": "layout", "fermato": None}
                        except Exception as e:
                            _log(f"    [paddle p.{p}] layout fallito ({type(e).__name__}: {e}); pagina intera")
                            md = None
                    if md is None:
                        md, meta = _ocr_one(srv, key, img, p, fig_dir, fig_prefix,
                                            expected_tokens=exp,
                                            raw_file=cache_dir / ckey / f"page_{p:04d}.raw.txt")
                    if meta.get("token_out") and meta.get("fermato") is None:
                        tok_hist[key].append(meta["token_out"])
                    # pagina "sospetta": tanto inchiostro, poco testo, o loop
                    if key == "paddle" and mode == "ibrido" and (
                            meta.get("loop") or (ink > 0.03 and len(md) < 150)):
                        meta["sospetta"] = True
                        retry_with_chandra.append(p)
                cf = _cache_file(cache_dir, ckey, p)
                cf.parent.mkdir(parents=True, exist_ok=True)
                cf.write_text(md, encoding="utf-8")
                (cf.with_suffix(".json")).write_text(json.dumps(meta), encoding="utf-8")
                results[p] = (key, md)
                stats[key].append(meta.get("secondi", 0))
                done_jobs += 1
                media = sum(stats[key]) / max(1, len(stats[key]))
                eta = media * (len(todo) - j)
                _log(f"  [{key}] pagina {p} ({j}/{len(todo)}): {meta.get('secondi',0):.0f} s"
                     f"{'  [LOOP]' if meta.get('loop') else ''}"
                     f"{'  [TAGLIATA AL LIMITE]' if meta.get('fermato') == 'limite' else ''}"
                     f"  | stima residua {eta/60:.0f} min")
                _log(f"  OCR: {int(done_jobs*100/max(1,total_jobs))}% ({done_jobs}/{total_jobs})")
        return retry_with_chandra

    t_start = time.time()
    retry = run_engine("paddle", plan["paddle"])
    if retry:
        _log(f"  [ibrido] {len(retry)} pagine sospette passano a Chandra: {retry}")
        total_jobs += len(retry)
    run_engine("chandra", sorted(set(plan["chandra"]) | set(retry)))

    # immagini di pagina (backup visivo, WebP leggero)
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

    # assemblaggio (senza intestazioni / pie' di pagina ripetuti)
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
        key, md = results.get(p, ("?", "*(pagina non convertita)*"))
        issues += math_issues(md)
        parts.append(f"\n\n<!-- ===== Pagina {p} di {n} | motore: {key} ===== -->\n")
        if p in page_links:
            parts.append(f"<!-- immagine pagina: {page_links[p]} -->\n")
        parts.append("\n" + md + "\n")
    tot = time.time() - t_start
    sp = stats["paddle"]; sc = stats["chandra"]
    summary = (f"<!-- OCR VLM locale: {len(pages)} pagine | "
               f"Paddle {len(sp)} pag. (media {sum(sp)/max(1,len(sp)):.0f} s) | "
               f"Chandra {len(sc)} pag. (media {sum(sc)/max(1,len(sc)):.0f} s) | "
               f"formule sospette: {issues} | totale {tot/60:.1f} min -->\n")
    _log("  " + summary.strip("<!-> \n"))
    return summary + "".join(parts).strip() + "\n", assets


# ----------------------------------------------------------------------------
# CLI: setup / prova
# ----------------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(description="OCR locale con PaddleOCR-VL e Chandra OCR 2 (llama.cpp).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="Scarica llama.cpp e i modelli (una tantum).")
    s.add_argument("quali", nargs="?", default="tutti",
                   choices=["tutti", "paddle", "chandra", "layout"])
    s.add_argument("--gpu", action="store_true", help="Scarica anche llama.cpp Vulkan (grafica Intel).")
    p = sub.add_parser("prova", help="Prova su alcune pagine: scrive <nome>_prova_<motore>.md")
    p.add_argument("file")
    p.add_argument("--pages", default="1-3")
    p.add_argument("--engine", default="tutti", choices=["tutti", "paddle", "chandra"])
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    if a.cmd == "setup" and a.quali == "layout":
        _log("Installo PaddlePaddle (CPU) + PaddleOCR (pipeline completa con layout) ...")
        r = subprocess.call([sys.executable, "-m", "pip", "install", "-U",
                             "--timeout", "120", "--retries", "10",
                             "paddlepaddle", "paddleocr[doc-parser]"])
        if r != 0:
            _log("Installazione non riuscita: rilancia il comando (la rete ha interrotto?)")
            return
        _log("Fatto. Il modello di layout si scarica da solo al primo uso.")
        return
    if a.cmd == "setup":
        ensure_llamacpp(False)
        if a.gpu:
            ensure_llamacpp(True)
        keys = ["paddle", "chandra"] if a.quali == "tutti" else [a.quali]
        for k in keys:
            ensure_model(k)
        _log("Setup completato. Modelli in: " + str(MODELS_DIR))
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
