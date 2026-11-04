"""
convert_av.py
--------------
Modulo di conversione per audio e video tramite FFmpeg.

Estensioni supportate in input :
    .mp3 .wav .flac .aac .ogg .m4a .opus .wma
    .mp4 .mov .avi .mkv .webm .flv .wmv .m4v
    .gif

Estensioni supportate in output:
    .mp3 .wav .flac .aac .ogg .m4a .opus
    .mp4 .mov .avi .mkv .webm
    .gif

Funzione principale:
    convert(src: Path, dst: Path, opts: dict | None = None) -> dict

Ritorna sempre un dict con chiavi:
    ok      -> bool
    note    -> str (messaggio informativo o di errore)
    elapsed -> float (secondi)

opts puo' contenere:
    'bitrate'   -> str, es. '192k' (audio)
    'fps'       -> int, frame rate video di output
    'image'     -> str/Path, immagine da usare per audio->video
    'duration'  -> float, durata in secondi (audio->video se serve troncare)
    'gif_fps'   -> int, default 10
    'gif_scale' -> int, larghezza in pixel, default 480

Dipende da FFmpeg gia' installato sul sistema (winget Gyan.FFmpeg oppure
KeyShot Studio). Il percorso viene risolto a import-time e memorizzato in
FFMPEG_PATH / FFPROBE_PATH.
"""

from __future__ import annotations

import os
import glob
import time
import shutil
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Localizzazione di ffmpeg/ffprobe
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str | None:
    """Trova ffmpeg.exe e ritorna il path assoluto, oppure None."""
    # 1) Prova il PATH normale
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 2) Cerca tra i pacchetti winget Gyan.FFmpeg
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        pattern = os.path.join(
            local_appdata,
            "Microsoft", "WinGet", "Packages",
            "Gyan.FFmpeg_*", "ffmpeg-*-full_build", "bin", "ffmpeg.exe",
        )
        matches = glob.glob(pattern)
        if matches:
            # Prendi il piu' recente in ordine alfabetico (la versione piu' alta).
            matches.sort()
            return matches[-1]

    # 3) Fallback: KeyShot Studio porta una copia di ffmpeg.
    keyshot = r"C:\Program Files\KeyShot Studio\bin\ffmpeg.exe"
    if os.path.isfile(keyshot):
        return keyshot

    return None


def _derive_ffprobe(ffmpeg_path: str | None) -> str | None:
    """Restituisce il path di ffprobe nella stessa cartella di ffmpeg, se esiste."""
    if not ffmpeg_path:
        return shutil.which("ffprobe")
    candidate = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe.exe")
    if os.path.isfile(candidate):
        return candidate
    return shutil.which("ffprobe")


FFMPEG_PATH: str | None = _find_ffmpeg()
FFPROBE_PATH: str | None = _derive_ffprobe(FFMPEG_PATH)


# ---------------------------------------------------------------------------
# Capacita' dichiarate
# ---------------------------------------------------------------------------

_AUDIO_EXTS: tuple[str, ...] = (
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".opus", ".wma",
)
_VIDEO_EXTS: tuple[str, ...] = (
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
)

SUPPORTED_INPUTS: tuple[str, ...] = _AUDIO_EXTS + _VIDEO_EXTS + (".gif",)

_AUDIO_OUT: tuple[str, ...] = (
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".opus",
)
_VIDEO_OUT: tuple[str, ...] = (
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
)

# .srt/.vtt = sottotitoli generati con Whisper (trascrizione locale)
_SUB_OUT: tuple[str, ...] = (".srt", ".vtt")

SUPPORTED_OUTPUTS: tuple[str, ...] = _AUDIO_OUT + _VIDEO_OUT + (".gif",) + _SUB_OUT


# Codec di default per audio in container "esotici"
_AUDIO_CODEC_BY_EXT: dict[str, str] = {
    ".mp3": "libmp3lame",
    ".wav": "pcm_s16le",
    ".flac": "flac",
    ".aac": "aac",
    ".ogg": "libvorbis",
    ".m4a": "aac",
    ".opus": "libopus",
}


def _norm_ext(ext: str) -> str:
    if not ext:
        return ""
    ext = ext.strip().lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _is_audio(ext: str) -> bool:
    return _norm_ext(ext) in _AUDIO_EXTS


def _is_video(ext: str) -> bool:
    return _norm_ext(ext) in _VIDEO_EXTS


def _is_gif(ext: str) -> bool:
    return _norm_ext(ext) == ".gif"


def supports(src_ext: str, dst_ext: str) -> bool:
    """Ritorna True se la coppia (src_ext, dst_ext) e' gestita.

    Le combinazioni gestite sono:
        audio -> audio
        video -> audio (estrazione)
        video -> video
        video -> gif
        audio -> video (servira' un'immagine; ne viene generata una se manca)
        gif   -> video
        gif   -> gif (riencode)
    """
    s = _norm_ext(src_ext)
    d = _norm_ext(dst_ext)

    if s not in SUPPORTED_INPUTS or d not in SUPPORTED_OUTPUTS:
        return False

    if _is_audio(s) and d in _AUDIO_OUT:
        return True
    if _is_video(s) and d in _AUDIO_OUT:
        return True
    if _is_video(s) and d in _VIDEO_OUT:
        return True
    if _is_video(s) and d == ".gif":
        return True
    if _is_audio(s) and d in _VIDEO_OUT:
        return True
    if _is_gif(s) and d in _VIDEO_OUT:
        return True
    if _is_gif(s) and d == ".gif":
        return True
    # sottotitoli: da qualsiasi audio o video (trascrizione Whisper)
    if (_is_audio(s) or _is_video(s)) and d in _SUB_OUT:
        return True
    # audio -> gif non e' significativo
    return False


# ---------------------------------------------------------------------------
# Helper esecuzione
# ---------------------------------------------------------------------------

import re as _re

def _probe_duration(src: Path) -> float | None:
    """Durata totale in secondi del file media (via ffprobe). None se ignota."""
    if not FFPROBE_PATH:
        return None
    try:
        p = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
            capture_output=True, text=True, check=False)
        val = (p.stdout or "").strip()
        return float(val) if val and val != "N/A" else None
    except Exception:
        return None


_TIME_RE = _re.compile(r"time=(\d+):(\d+):(\d+\.?\d*)")

def _run_ffmpeg(args: list[str], total_duration: float | None = None) -> tuple[int, str, str]:
    """Esegue ffmpeg. Se total_duration è nota, stampa il progresso in %
       (su stdout, riga 'PROGRESS_AV: NN%') leggendo lo stderr in tempo reale.
       La GUI cattura quelle righe e muove la barra; in CLI si vede comunque."""
    cmd = [FFMPEG_PATH] + args

    if not total_duration or total_duration <= 0:
        # Modalità semplice (conversione breve): cattura tutto alla fine.
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", check=False)
        except FileNotFoundError as e:
            return 127, "", f"ffmpeg non trovato: {e}"
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    # Modalità con progresso: Popen + lettura stderr riga per riga.
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace")
    except FileNotFoundError as e:
        return 127, "", f"ffmpeg non trovato: {e}"

    err_tail: list[str] = []
    last_pct = -1
    try:
        for line in proc.stderr:
            err_tail.append(line)
            if len(err_tail) > 40:
                err_tail.pop(0)
            m = _TIME_RE.search(line)
            if m:
                h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                done = h * 3600 + mn * 60 + s
                pct = int(min(100, done * 100 / total_duration))
                if pct != last_pct:
                    last_pct = pct
                    print(f"PROGRESS_AV: {pct}%", flush=True)
    except Exception:
        pass
    proc.wait()
    if last_pct < 100:
        print("PROGRESS_AV: 100%", flush=True)
    return proc.returncode, "", "".join(err_tail)


def _short_err(stderr: str, max_lines: int = 6) -> str:
    """Estrae le ultime righe non vuote di stderr per un messaggio sintetico."""
    if not stderr:
        return ""
    lines = [ln.rstrip() for ln in stderr.splitlines() if ln.strip()]
    return " | ".join(lines[-max_lines:])


# ---------------------------------------------------------------------------
# Generazione immagine placeholder per audio -> video
# ---------------------------------------------------------------------------

def _make_black_image(path: Path, w: int = 1280, h: int = 720) -> bool:
    """Crea un'immagine PNG nera w x h. Prova prima Pillow, altrimenti ffmpeg."""
    try:
        from PIL import Image  # type: ignore

        img = Image.new("RGB", (w, h), (0, 0, 0))
        img.save(path, format="PNG")
        return True
    except Exception:
        pass

    # Fallback: usa ffmpeg con il filtro color
    args = [
        "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s={w}x{h}:d=1",
        "-frames:v", "1",
        str(path),
    ]
    code, _out, _err = _run_ffmpeg(args)
    return code == 0 and path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Singole strategie di conversione
# ---------------------------------------------------------------------------

def _convert_audio_to_audio(src: Path, dst: Path, opts: dict) -> tuple[bool, str]:
    bitrate = str(opts.get("bitrate") or "192k")
    codec = _AUDIO_CODEC_BY_EXT.get(dst.suffix.lower())
    args = ["-y", "-i", str(src), "-vn"]
    if codec:
        args += ["-c:a", codec]
    if dst.suffix.lower() in (".flac", ".wav"):
        # I formati lossless ignorano il bitrate; lo omettiamo.
        pass
    else:
        args += ["-b:a", bitrate]
    args.append(str(dst))

    code, _out, err = _run_ffmpeg(args, total_duration=_probe_duration(src))
    if code != 0:
        return False, f"ffmpeg fallito (audio->audio): {_short_err(err)}"
    return True, f"Audio convertito ({src.suffix} -> {dst.suffix})."


def _convert_video_to_audio(src: Path, dst: Path, opts: dict) -> tuple[bool, str]:
    bitrate = str(opts.get("bitrate") or "192k")
    codec = _AUDIO_CODEC_BY_EXT.get(dst.suffix.lower())
    args = ["-y", "-i", str(src), "-vn"]
    if codec:
        args += ["-c:a", codec]
    if dst.suffix.lower() not in (".flac", ".wav"):
        args += ["-b:a", bitrate]
    args.append(str(dst))

    code, _out, err = _run_ffmpeg(args, total_duration=_probe_duration(src))
    if code != 0:
        return False, f"ffmpeg fallito (estrazione audio): {_short_err(err)}"
    return True, f"Audio estratto dal video ({dst.suffix})."


def _convert_video_to_video(src: Path, dst: Path, opts: dict) -> tuple[bool, str]:
    fps = opts.get("fps")
    dst_ext = dst.suffix.lower()

    args = ["-y", "-i", str(src)]

    # Codec video sensati per container
    if dst_ext == ".webm":
        args += ["-c:v", "libvpx-vp9", "-c:a", "libopus"]
    elif dst_ext == ".mkv":
        args += ["-c:v", "libx264", "-c:a", "aac"]
    elif dst_ext == ".avi":
        args += ["-c:v", "mpeg4", "-c:a", "libmp3lame"]
    elif dst_ext == ".mov":
        args += ["-c:v", "libx264", "-c:a", "aac"]
    else:  # .mp4
        args += ["-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart"]
        args += ["-pix_fmt", "yuv420p"]

    if fps:
        args += ["-r", str(int(fps))]

    args.append(str(dst))

    code, _out, err = _run_ffmpeg(args, total_duration=_probe_duration(src))
    if code != 0:
        return False, f"ffmpeg fallito (video->video): {_short_err(err)}"
    return True, f"Video transcodificato ({src.suffix} -> {dst.suffix})."


def _convert_video_to_gif(src: Path, dst: Path, opts: dict) -> tuple[bool, str]:
    fps = int(opts.get("gif_fps", 10))
    scale = int(opts.get("gif_scale", 480))
    vf = f"fps={fps},scale={scale}:-1:flags=lanczos"
    args = ["-y", "-i", str(src), "-vf", vf, str(dst)]

    code, _out, err = _run_ffmpeg(args, total_duration=_probe_duration(src))
    if code != 0:
        return False, f"ffmpeg fallito (video->gif): {_short_err(err)}"
    return True, f"GIF generata (fps={fps}, scale={scale})."


def _convert_audio_to_video(src: Path, dst: Path, opts: dict) -> tuple[bool, str]:
    """Combina un'immagine statica con l'audio per generare un video."""
    image_opt = opts.get("image")
    generated_image: Path | None = None
    note_extra = ""

    if image_opt:
        image_path = Path(image_opt)
        if not image_path.exists():
            return False, f"Immagine indicata non trovata: {image_path}"
    else:
        # Generiamo un'immagine nera al volo
        image_path = dst.parent / f"_av_placeholder_{os.getpid()}.png"
        ok = _make_black_image(image_path)
        if not ok:
            return False, "Impossibile generare un'immagine placeholder."
        generated_image = image_path
        note_extra = " (immagine nera 1280x720 generata automaticamente)"

    args = [
        "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-i", str(src),
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", str(opts.get("bitrate") or "192k"),
        "-pix_fmt", "yuv420p",
        "-shortest",
    ]
    if "duration" in opts:
        args += ["-t", str(opts["duration"])]
    # mp4 vuole moov all'inizio
    if dst.suffix.lower() == ".mp4":
        args += ["-movflags", "+faststart"]
    args.append(str(dst))

    code, _out, err = _run_ffmpeg(args)

    # Pulizia immagine generata
    if generated_image is not None:
        try:
            generated_image.unlink()
        except OSError:
            pass

    if code != 0:
        return False, f"ffmpeg fallito (audio->video): {_short_err(err)}"
    return True, f"Video creato da audio + immagine{note_extra}."


def _convert_gif_to_video(src: Path, dst: Path, opts: dict) -> tuple[bool, str]:
    dst_ext = dst.suffix.lower()
    args = [
        "-y",
        "-i", str(src),
        "-movflags", "faststart",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    ]
    if dst_ext == ".webm":
        args += ["-c:v", "libvpx-vp9"]
    else:
        args += ["-c:v", "libx264"]
    args.append(str(dst))

    code, _out, err = _run_ffmpeg(args)
    if code != 0:
        return False, f"ffmpeg fallito (gif->video): {_short_err(err)}"
    return True, f"GIF convertita in video ({dst_ext})."


def _convert_gif_to_gif(src: Path, dst: Path, opts: dict) -> tuple[bool, str]:
    fps = int(opts.get("gif_fps", 10))
    scale = int(opts.get("gif_scale", 480))
    vf = f"fps={fps},scale={scale}:-1:flags=lanczos"
    args = ["-y", "-i", str(src), "-vf", vf, str(dst)]
    code, _out, err = _run_ffmpeg(args)
    if code != 0:
        return False, f"ffmpeg fallito (gif->gif): {_short_err(err)}"
    return True, "GIF re-encoded."


def _fmt_ts(sec: float, vtt: bool) -> str:
    """Formatta un timestamp per SRT (virgola) o VTT (punto)."""
    h = int(sec // 3600); m = int(sec % 3600 // 60); s = sec % 60
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{int(s):02d}{sep}{int((s % 1) * 1000):03d}"


def _convert_to_subtitles(src: Path, dst: Path, opts: dict) -> tuple[bool, str]:
    """Audio/video -> sottotitoli .srt/.vtt con Whisper (trascrizione locale).
       Lento su CPU ma completamente offline e gratuito."""
    try:
        import whisper
    except ImportError:
        return False, ("Per i sottotitoli serve openai-whisper: "
                       "pip install -U openai-whisper (e ffmpeg nel PATH).")
    model_name = str(opts.get("whisper_model") or "base")
    print(f"  Whisper '{model_name}': trascrivo {src.name} (lento su CPU)...",
          flush=True)
    model = whisper.load_model(model_name)
    res = model.transcribe(str(src), verbose=False)
    segs = res.get("segments", [])
    if not segs:
        return False, "Whisper non ha prodotto segmenti (audio vuoto?)."
    vtt = dst.suffix.lower() == ".vtt"
    lines: list[str] = ["WEBVTT", ""] if vtt else []
    for i, seg in enumerate(segs, 1):
        start = _fmt_ts(float(seg["start"]), vtt)
        end = _fmt_ts(float(seg["end"]), vtt)
        if not vtt:
            lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    dst.write_text("\n".join(lines), encoding="utf-8")
    return True, (f"Sottotitoli {dst.suffix} generati: {len(segs)} segmenti, "
                  f"lingua {res.get('language', '?')} (Whisper {model_name}).")


# ---------------------------------------------------------------------------
# API pubblica
# ---------------------------------------------------------------------------

def convert(src: Path, dst: Path, opts: dict | None = None) -> dict:
    """Converte un file audio/video tramite ffmpeg.

    Vedi modulo docstring per dettagli su estensioni ed opzioni.
    """
    t0 = time.perf_counter()
    opts = dict(opts or {})

    try:
        src = Path(src)
        dst = Path(dst)

        if FFMPEG_PATH is None:
            return {
                "ok": False,
                "note": (
                    "ffmpeg.exe non trovato. Installa FFmpeg "
                    "(winget install Gyan.FFmpeg) o assicurati che sia nel PATH."
                ),
                "elapsed": time.perf_counter() - t0,
            }

        if not src.exists():
            return {
                "ok": False,
                "note": f"File sorgente non trovato: {src}",
                "elapsed": time.perf_counter() - t0,
            }

        src_ext = _norm_ext(src.suffix)
        dst_ext = _norm_ext(dst.suffix)

        if src_ext not in SUPPORTED_INPUTS:
            return {
                "ok": False,
                "note": (
                    f"Estensione di input '{src_ext}' non supportata. "
                    f"Supportate: {', '.join(SUPPORTED_INPUTS)}."
                ),
                "elapsed": time.perf_counter() - t0,
            }

        if dst_ext not in SUPPORTED_OUTPUTS:
            return {
                "ok": False,
                "note": (
                    f"Estensione di output '{dst_ext}' non supportata. "
                    f"Supportate: {', '.join(SUPPORTED_OUTPUTS)}."
                ),
                "elapsed": time.perf_counter() - t0,
            }

        if not supports(src_ext, dst_ext):
            # Suggerimenti
            alt: list[str] = []
            if _is_audio(src_ext):
                alt = list(_AUDIO_OUT) + list(_VIDEO_OUT)
            elif _is_video(src_ext):
                alt = list(_AUDIO_OUT) + list(_VIDEO_OUT) + [".gif"]
            elif _is_gif(src_ext):
                alt = list(_VIDEO_OUT) + [".gif"]
            return {
                "ok": False,
                "note": (
                    f"Coppia {src_ext} -> {dst_ext} non supportata. "
                    f"Alternative: {', '.join(alt)}."
                ),
                "elapsed": time.perf_counter() - t0,
            }

        dst.parent.mkdir(parents=True, exist_ok=True)

        # Dispatch
        if _is_audio(src_ext) and dst_ext in _AUDIO_OUT:
            ok, note = _convert_audio_to_audio(src, dst, opts)
        elif _is_video(src_ext) and dst_ext in _AUDIO_OUT:
            ok, note = _convert_video_to_audio(src, dst, opts)
        elif _is_video(src_ext) and dst_ext in _VIDEO_OUT:
            ok, note = _convert_video_to_video(src, dst, opts)
        elif _is_video(src_ext) and dst_ext == ".gif":
            ok, note = _convert_video_to_gif(src, dst, opts)
        elif _is_audio(src_ext) and dst_ext in _VIDEO_OUT:
            ok, note = _convert_audio_to_video(src, dst, opts)
        elif _is_gif(src_ext) and dst_ext in _VIDEO_OUT:
            ok, note = _convert_gif_to_video(src, dst, opts)
        elif _is_gif(src_ext) and dst_ext == ".gif":
            ok, note = _convert_gif_to_gif(src, dst, opts)
        elif dst_ext in _SUB_OUT:
            ok, note = _convert_to_subtitles(src, dst, opts)
        else:
            ok, note = False, "Coppia gestita logicamente non implementata."

        return {
            "ok": ok,
            "note": note,
            "elapsed": time.perf_counter() - t0,
        }

    except Exception as e:  # rete di sicurezza
        return {
            "ok": False,
            "note": f"Errore imprevisto: {e}",
            "elapsed": time.perf_counter() - t0,
        }
