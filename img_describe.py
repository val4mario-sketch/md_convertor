# -*- coding: utf-8 -*-
"""
img_describe.py — descrizioni IA (locali) delle immagini di un file Markdown.

Per ogni immagine referenziata nel .md (`![...](path)`) genera una descrizione
testuale con un modello di visione LOCALE via Ollama (default: qwen2.5vl:3b,
gratis, niente cloud) e la inserisce SOTTO l'immagine come blockquote:

    ![](media/APPUNTI%20ASTRONOMIA/formula.png)

    > **[Descrizione IA]** $$a = \\frac{v}{c}\\sin\\theta$$ — formula
    > dell'aberrazione della luce...

Cosi' un'IA (o una persona) che legge il .md capisce il contenuto di ogni
immagine anche senza vederla. Formule trascritte in LaTeX quando possibile.

Caratteristiche:
- CACHE su file JSON accanto al .md: rilanciare non rifa' il lavoro gia' fatto
  (resume dopo interruzione; il .md viene salvato ogni 15 immagini).
- IDEMPOTENTE: le immagini gia' descritte vengono saltate (usa --force per
  rigenerare).
- Avvia da solo `ollama serve` se non e' in esecuzione e scarica da solo il
  modello se manca (una tantum).

Uso standalone:
    python img_describe.py "file.md" [--model qwen2.5vl:3b] [--force]
                           [--limit N] [--dry-run]

Uso da tomd2.py:  flag --describe-images (vedi convert_one).
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
# lato massimo (px) delle immagini inviate al modello: piu' piccolo = molto
# piu' veloce su CPU (i token visivi crescono con l'area dell'immagine)
MAX_SIDE      = int(os.environ.get("TOMD2_VISION_MAXSIDE", "640"))

# ---- backend llama.cpp (PREFERITO se installato: ~4x piu' veloce) ----------
# Ollama impone un MINIMO di ~1000 token visivi per immagine su qwen2.5vl
# (misurato: anche un'immagine 144x68 px ne genera 1084 -> ~140 s/immagine su
# questa CPU). llama-server espone --image-min/max-tokens: con min=100 e
# max=256 la qualita' LaTeX resta identica ma si scende a ~35 s/immagine.
LLAMACPP_URL    = os.environ.get("TOMD2_LLAMACPP_URL", "http://127.0.0.1:11435")
LLAMACPP_EXE    = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "llamacpp" / "llama-server.exe"
_LLAMACPP_MODELS = Path.home() / ".llamacpp-models"
LLAMACPP_MODEL  = _LLAMACPP_MODELS / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
LLAMACPP_MMPROJ = _LLAMACPP_MODELS / "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
IMG_MIN_TOKENS  = int(os.environ.get("TOMD2_VISION_MINTOK", "100"))
IMG_MAX_TOKENS  = int(os.environ.get("TOMD2_VISION_MAXTOK", "256"))

# Marcatore stabile: serve per riconoscere le immagini gia' descritte.
MARKER = "> **[Descrizione IA]**"


def _log_flush(*a, **kw):
    """print con flush: senza, stdout rediretto su file resta nel buffer
       e il progresso non si vede (lavori da ore!)."""
    print(*a, **kw, flush=True)


def _keep_awake(enable: bool) -> None:
    """Impedisce a Windows di sospendere il PC durante il lavoro lungo
       (stesso meccanismo del _KeepAwake di tomd2). Lo stato e' per-thread
       e si azzera da solo quando il processo termina: niente cleanup
       necessario in caso di crash."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass

# Stessa regex "parentesi bilanciate" di tomd2._encode_md_link_urls
_IMG_RE = re.compile(r"!\[[^\]\n]*\]\(([^()\n]*(?:\([^()\n]*\)[^()\n]*)*)\)")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

PROMPT = (
    "Descrivi l'immagine (da appunti universitari) per un'IA che non puo' "
    "vederla. Italiano, massimo 60 parole, SOLO la descrizione, senza "
    "inventare nulla che non si veda. "
    "Formula matematica: trascrivila ESATTA in LaTeX tra $$ e $$ + una frase "
    "di spiegazione. Grafico: assi, curve, andamento. Schema: elementi e "
    "relazioni. Tabella: riassunto. Foto: soggetto e dettagli."
)


# ----------------------------------------------------------------------------
# Ollama: server + modello
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
    """Verifica che Ollama risponda; se no prova ad avviarlo da solo."""
    if _server_ok():
        return
    exe = _find_ollama_exe()
    if not exe:
        raise RuntimeError(
            "Ollama non trovato. Installalo con:  winget install Ollama.Ollama\n"
            "poi riprova (il modello viene scaricato automaticamente).")
    log("  [IA] avvio del server Ollama ...")
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    subprocess.Popen([exe, "serve"], creationflags=flags,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        time.sleep(1)
        if _server_ok():
            return
    raise RuntimeError("Ollama non risponde su " + OLLAMA_URL)


def ensure_model(model: str = DEFAULT_MODEL, log=_log_flush) -> None:
    """Scarica il modello se non e' gia' presente (una tantum, ~GB)."""
    import requests
    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()
    have = {m.get("name", "") for m in tags.get("models", [])}
    if model in have or f"{model}:latest" in have:
        return
    log(f"  [IA] modello '{model}' assente: download in corso (una tantum) ...")
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
                raise RuntimeError(f"pull di '{model}' fallito: {st['error']}")
            if "completed" in st and "total" in st and st["total"]:
                pct = 100 * st["completed"] // st["total"]
                print(f"\r  [IA] download {model}: {pct}%", end="", flush=True)
    print()


# ----------------------------------------------------------------------------
# Backend llama.cpp: server + descrizione
# ----------------------------------------------------------------------------
def _llamacpp_ok(timeout: float = 3) -> bool:
    import requests
    try:
        r = requests.get(f"{LLAMACPP_URL}/health", timeout=timeout)
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False


def ensure_llamacpp(log=_log_flush) -> bool:
    """True se il backend llama.cpp e' pronto (avviandolo se serve).
       False (senza eccezioni) se non installato: si usera' Ollama."""
    if _llamacpp_ok():
        return True
    if not (LLAMACPP_EXE.exists() and LLAMACPP_MODEL.exists()
            and LLAMACPP_MMPROJ.exists()):
        return False
    # libera la RAM del runner Ollama (best effort: 2 modelli = ~8 GB)
    exe = _find_ollama_exe()
    if exe and _server_ok():
        try:
            subprocess.run([exe, "stop", DEFAULT_MODEL], timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    log("  [IA] avvio llama-server (backend veloce) ...")
    n_threads = max(2, (os.cpu_count() or 8) // 2)   # core fisici
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
    for _ in range(120):          # il caricamento del modello impiega ~20-60 s
        time.sleep(1)
        if _llamacpp_ok():
            return True
    log("  [IA] llama-server non risponde: ripiego su Ollama")
    return False


def describe_image_llamacpp(img_path: Path, prompt: str = PROMPT,
                            timeout: int = 300) -> str:
    """Descrizione via llama-server (API OpenAI-compatibile)."""
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
# Descrizione di UNA immagine
# ----------------------------------------------------------------------------
def _prep_image_b64(img_path: Path, max_side: int = MAX_SIDE) -> str:
    """Base64 dell'immagine, ridimensionata se piu' grande di max_side px.
       Su CPU i token visivi (e quindi il tempo) crescono con l'area: una
       pagina renderizzata a 150 dpi passa da ~2500 a ~500 token."""
    try:
        from PIL import Image
        import io as _io
        with Image.open(img_path) as im:
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.LANCZOS)
                buf = _io.BytesIO()
                # PNG: niente artefatti JPEG su testo/formule
                im.convert("RGB").save(buf, "PNG")
                return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        pass
    return base64.b64encode(img_path.read_bytes()).decode("ascii")


def describe_image(img_path: Path, model: str = DEFAULT_MODEL,
                   prompt: str = PROMPT, timeout: int = 600) -> str:
    """Ritorna la descrizione (una riga, italiano) dell'immagine."""
    import requests
    b64 = _prep_image_b64(img_path)
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "keep_alive": "30m",          # non ricaricare il modello a ogni immagine
            "options": {"temperature": 0.1, "num_predict": 260},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    text = (resp.json().get("response") or "").strip()
    # una sola riga: i blockquote multilinea complicano l'inserimento
    text = re.sub(r"\s*\n\s*", " ", text)
    # via eventuali premesse tipo "Descrizione:" a inizio risposta
    text = re.sub(r"^(descrizione( ia)?|risposta)\s*[:\-]\s*", "", text,
                  flags=re.IGNORECASE)
    return text.strip()


# ----------------------------------------------------------------------------
# Arricchimento di un file Markdown
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
    """Inserisce le descrizioni IA sotto ogni immagine del .md.
       Ritorna statistiche: {total, done, cached, skipped, failed}."""
    md_path = Path(md_path)
    base = md_path.parent
    lines = md_path.read_text(encoding="utf-8").splitlines()

    # censimento immagini: (indice_riga, destinazione, path_risolto)
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
        log("  [IA] nessuna immagine locale trovata nel .md")
        return stats
    if dry_run:
        for i, dest, p in jobs:
            log(f"  riga {i+1}: {dest}  ({'OK' if p.exists() else 'MANCANTE'})")
        return stats

    # backend: llama.cpp se installato (~4x piu' veloce), altrimenti Ollama
    use_llamacpp = ensure_llamacpp(log=log)
    if use_llamacpp:
        log("  [IA] backend: llama.cpp "
            f"(token visivi {IMG_MIN_TOKENS}-{IMG_MAX_TOKENS}, ~35 s/immagine)")
    else:
        ensure_server(log=log)
        ensure_model(model, log=log)
    _keep_awake(True)     # il PC non deve addormentarsi a meta' lavoro

    cache_path = md_path.with_name(md_path.stem + ".desc-cache.json")
    cache = _load_cache(cache_path)

    def _has_desc(idx: int) -> int | None:
        """Se subito dopo la riga idx (saltando righe vuote) c'e' gia' una
           descrizione, ritorna il suo indice riga, altrimenti None."""
        j = idx + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and lines[j].lstrip().startswith(MARKER):
            return j
        return None

    # elaboro DAL FONDO: cosi' gli inserimenti non spostano gli indici
    # delle righe ancora da elaborare
    n_calls = 0
    processed = 0
    t_start = time.time()
    for k in range(len(jobs) - 1, -1, -1):
        i, dest, p = jobs[k]
        pos = len(jobs) - k          # progressione 1..N (per la barra GUI)
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
        # NB: la cache NON confronta il modello: qwen via Ollama e qwen via
        # llama.cpp producono descrizioni equivalenti, inutile rifarle
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
                # un solo retry (il primo caricamento del modello puo' fallire
                # per timeout su macchine lente)
                try:
                    desc = _call()
                except Exception:
                    log(f"  [IA] {pos}/{len(jobs)} ERRORE su {p.name}: {e}")
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
        if existing is not None:      # --force: sostituisco la vecchia
            lines[existing] = block
        else:
            lines[i+1:i+1] = ["", block]
        processed += 1
        log(f"  [IA] {pos}/{len(jobs)} ({src}) {p.name}")

        # salvataggio periodico: lavoro lungo, non perdere nulla
        if processed % 15 == 0:
            md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _keep_awake(False)
    dt = time.time() - t_start
    log(f"  [IA] finito in {dt/60:.1f} min: {stats['done']} nuove, "
        f"{stats['cached']} da cache, {stats['skipped']} saltate, "
        f"{stats['failed']} fallite (su {stats['total']})")
    return stats


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    if os.name == "nt":               # UTF-8 su console Windows (cp1252)
        for s in (sys.stdout, sys.stderr):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    ap = argparse.ArgumentParser(
        description="Aggiunge descrizioni IA (locali, via Ollama) sotto ogni "
                    "immagine di un file Markdown.")
    ap.add_argument("md", nargs="+", help="File .md da arricchire")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Modello visione Ollama (default {DEFAULT_MODEL})")
    ap.add_argument("--force", action="store_true",
                    help="Rigenera anche le descrizioni gia' presenti")
    ap.add_argument("--limit", type=int, default=0,
                    help="Massimo N nuove descrizioni (0 = tutte)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Elenca le immagini senza chiamare l'IA")
    args = ap.parse_args()

    for f in args.md:
        p = Path(f)
        if not p.exists():
            print(f"[KO] file non trovato: {p}")
            continue
        print(f">>> {p.name}")
        enrich_markdown(p, model=args.model, force=args.force,
                        limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
