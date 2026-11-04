#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tomd2_gui.py — Interfaccia grafica per tomd2.py.

Caratteristiche:
  - Drag-and-drop di file e cartelle (richiede tkinterdnd2)
  - Aggiungi file singoli o intere cartelle
  - Cattura lo stdout di pymupdf4llm nel log della GUI
  - Progress bar per il file corrente + barra totale
  - Bottone "Annulla" funzionante
  - Salva le tue impostazioni in .tomd2_gui_config.json
  - Mostra solo le opzioni rilevanti per il motore scelto
  - Thread-safe (uso di queue + root.after, no chiamate tk da thread)
  - Pulsante "Apri cartella output" per Esplora risorse
"""
from __future__ import annotations
import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Drag-and-drop opzionale: se manca, la GUI parte lo stesso ma senza DnD.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except Exception:
    HAS_DND = False

# Importa le funzioni dello script principale
sys.path.insert(0, str(Path(__file__).parent))
from tomd2 import ALL_SUPPORTED, PDF_EXT, choose_engine, convert_one, _KeepAwake

# Il router (convertitore universale ogni->ogni) trascina dipendenze pesanti
# (pandas, cairosvg, img2pdf, ...). Se UNA fallisce l'import, la GUI deve
# comunque aprirsi in modalità "solo Markdown" invece di non partire.
try:
    import router as _router
    _ROUTER_OK = True
    _ROUTER_ERR = ""
except Exception as _e:  # noqa: BLE001
    _router = None
    _ROUTER_OK = False
    _ROUTER_ERR = f"{type(_e).__name__}: {_e}"

CONFIG_FILE = Path(__file__).parent / ".tomd2_gui_config.json"


# ----------------------------------------------------------------------------
# Helper: redirige stdout/stderr verso una queue (per il log)
# ----------------------------------------------------------------------------
class QueueWriter:
    def __init__(self, q: queue.Queue, prefix: str = ""):
        self.q, self.prefix = q, prefix
        self._buf = ""
    def write(self, s):
        if not s:
            return
        # accodo per righe per evitare flicker; le righe con \r di tqdm le
        # gestisco come "linee live" sostituendo l'ultima
        self._buf += s
        while "\n" in self._buf or "\r" in self._buf:
            # cerca il primo terminatore
            i_nl = self._buf.find("\n")
            i_cr = self._buf.find("\r")
            if i_nl == -1: i = i_cr
            elif i_cr == -1: i = i_nl
            else: i = min(i_nl, i_cr)
            line = self._buf[:i]
            term = self._buf[i]
            self._buf = self._buf[i+1:]
            if line.strip():
                self.q.put(("LOG_LIVE" if term == "\r" else "LOG", self.prefix + line))
    def flush(self):
        if self._buf.strip():
            self.q.put(("LOG", self.prefix + self._buf))
            self._buf = ""


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
class TomdGui:
    def __init__(self, root):
        self.root = root
        self.root.title("tomd2 — Convertitore PDF/Office → Markdown")
        self.root.geometry("1000x780")
        self.root.minsize(820, 560)

        # Stato
        self.files: list[Path] = []
        self.queue: queue.Queue = queue.Queue()
        self.cancel_flag = threading.Event()
        self.worker: threading.Thread | None = None

        # Variabili opzioni (default + carica da config)
        cfg = self._load_config()
        self.var_overwrite       = tk.BooleanVar(value=cfg.get("overwrite", True))
        self.var_page_images     = tk.BooleanVar(value=cfg.get("page_images", True))
        self.var_force_ocr       = tk.BooleanVar(value=cfg.get("force_ocr", False))
        self.var_whisper_model   = tk.StringVar (value=cfg.get("whisper_model", "base"))
        self.var_engine          = tk.StringVar (value=cfg.get("engine", "auto"))
        self.var_dpi             = tk.IntVar    (value=cfg.get("page_image_dpi", 150))
        self.var_vector_threshold= tk.IntVar    (value=cfg.get("vector_threshold", 3))
        self.var_block_size      = tk.IntVar    (value=cfg.get("block_size", 0))
        self.var_media_dir       = tk.StringVar (value=cfg.get("media_dir", "media"))
        self.var_target_ext      = tk.StringVar (value=cfg.get("target_ext", ".md"))
        self.var_lean_md         = tk.BooleanVar(value=cfg.get("lean_md", False))
        self.var_describe_images = tk.BooleanVar(value=cfg.get("describe_images", False))
        self.var_vlm_pages       = tk.StringVar (value=cfg.get("vlm_pages", ""))
        self.var_vlm_hq_pages    = tk.StringVar (value=cfg.get("vlm_hq_pages", ""))

        self._build_ui()
        self._update_engine_visibility()
        self.root.after(80, self._pump_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=8)
        style.configure("Danger.TButton", foreground="#a00")

        root = ttk.Frame(self.root, padding=10)
        root.pack(fill="both", expand=True)

        # ===== Sezione 1: File =====
        hdr1 = ttk.Frame(root)
        hdr1.pack(fill="x")
        ttk.Label(hdr1, text="1) File da convertire",
                  font=("Segoe UI", 11, "bold")).pack(side="left")
        dnd_hint = "  (trascina file qui)" if HAS_DND else \
                   "  (drag-and-drop disabilitato: pip install tkinterdnd2)"
        ttk.Label(hdr1, text=dnd_hint, foreground="#666").pack(side="left")

        list_frame = ttk.Frame(root)
        list_frame.pack(fill="both", expand=False, pady=(4, 8))
        self.lst = tk.Listbox(list_frame, height=8, activestyle="dotbox",
                              selectmode="extended")
        self.lst.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.lst.yview)
        sb.pack(side="right", fill="y")
        self.lst.config(yscrollcommand=sb.set)

        if HAS_DND:
            self.lst.drop_target_register(DND_FILES)
            self.lst.dnd_bind("<<Drop>>", self._on_drop)

        btns = ttk.Frame(root)
        btns.pack(fill="x", pady=(0, 8))
        ttk.Button(btns, text="+ Aggiungi file", command=self._add_files).pack(side="left", padx=2)
        ttk.Button(btns, text="+ Aggiungi cartella", command=self._add_folder).pack(side="left", padx=2)
        ttk.Button(btns, text="− Rimuovi selezionati", command=self._remove_sel).pack(side="left", padx=2)
        ttk.Button(btns, text="✕ Pulisci tutto", command=self._clear).pack(side="left", padx=2)
        self.lbl_count = ttk.Label(btns, text="0 file")
        self.lbl_count.pack(side="right")

        # ===== Sezione 2: Opzioni =====
        ttk.Label(root, text="2) Opzioni",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        opts = ttk.LabelFrame(root, text="Impostazioni", padding=10)
        opts.pack(fill="x", pady=(4, 8))

        # ---- Riga 0: Converti in (formato di destinazione) ----
        r0 = ttk.Frame(opts); r0.pack(fill="x", pady=2)
        ttk.Label(r0, text="Converti in:", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.cb_target = ttk.Combobox(r0, textvariable=self.var_target_ext,
                                       state="readonly", width=14,
                                       values=[".md"])  # popolata dinamicamente
        self.cb_target.pack(side="left", padx=4)
        self.cb_target.bind("<<ComboboxSelected>>",
                             lambda e: self._update_engine_visibility())
        ttk.Label(r0, text="  (selezione dinamica in base ai file caricati)",
                  foreground="#666").pack(side="left")

        # ---- Riga 1: motore (solo per .md) + opzioni comuni ----
        r1 = ttk.Frame(opts); r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Motore:").pack(side="left")
        cb = ttk.Combobox(r1, textvariable=self.var_engine, state="readonly", width=16,
                          values=("auto", "pymupdf4llm", "ocr_scan",
                                  "ocr_tesseract", "ocr_image", "ai_pack",
                                  "xlsx", "pptx", "csv",
                                  "code", "ipynb", "eml", "archive",
                                  "ebook", "audio", "video",
                                  "markitdown", "marker",
                                  "vlm_paddle", "vlm_chandra", "vlm_ibrido"))
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._update_engine_visibility())

        ttk.Checkbutton(r1, text="Sovrascrivi .md esistenti",
                        variable=self.var_overwrite).pack(side="left", padx=10)

        ttk.Checkbutton(r1, text="Descrizioni IA immagini (Ollama, lento)",
                        variable=self.var_describe_images).pack(side="left", padx=10)

        ttk.Label(r1, text="Cartella media:").pack(side="left", padx=(20, 2))
        ttk.Entry(r1, textvariable=self.var_media_dir, width=10).pack(side="left")

        # Riga 2: opzioni pymupdf4llm (page-images, dpi, soglia)
        self.frame_pym = ttk.Frame(opts); self.frame_pym.pack(fill="x", pady=2)
        ttk.Label(self.frame_pym, text="(opzioni PDF)",
                  foreground="#666").pack(side="left", padx=(0, 4))
        ttk.Checkbutton(self.frame_pym, text="Genera immagini di pagina (garanzia diagrammi)",
                        variable=self.var_page_images).pack(side="left", padx=2)
        ttk.Label(self.frame_pym, text="  DPI:").pack(side="left", padx=(15, 2))
        ttk.Spinbox(self.frame_pym, from_=72, to=600, increment=25,
                    textvariable=self.var_dpi, width=6).pack(side="left")
        ttk.Label(self.frame_pym, text="  Soglia disegni:").pack(side="left", padx=(15, 2))
        ttk.Spinbox(self.frame_pym, from_=0, to=50, textvariable=self.var_vector_threshold,
                    width=5).pack(side="left")
        ttk.Checkbutton(self.frame_pym, text="MD snello (no <details>)",
                        variable=self.var_lean_md).pack(side="left", padx=15)

        # Riga 3: opzioni marker (force-ocr, block-size)
        self.frame_mk = ttk.Frame(opts); self.frame_mk.pack(fill="x", pady=2)
        ttk.Label(self.frame_mk, text="(opzioni marker)",
                  foreground="#666").pack(side="left", padx=(0, 4))
        ttk.Checkbutton(self.frame_mk, text="Forza OCR (PDF scansionati)",
                        variable=self.var_force_ocr).pack(side="left", padx=2)
        ttk.Label(self.frame_mk, text="  Blocco pagine (0 = file intero):").pack(side="left", padx=(15, 2))
        ttk.Spinbox(self.frame_mk, from_=0, to=200, textvariable=self.var_block_size,
                    width=5).pack(side="left")
        ttk.Label(self.frame_mk, text=" ⚠ Marker e' molto lento senza GPU",
                  foreground="#a00").pack(side="left", padx=15)

        # Riga 3b: OCR con IA locale (PaddleOCR-VL / Chandra OCR 2)
        self.frame_vlm = ttk.Frame(opts); self.frame_vlm.pack(fill="x", pady=2)
        ttk.Label(self.frame_vlm, text="(OCR IA locale)",
                  foreground="#666").pack(side="left", padx=(0, 4))
        ttk.Label(self.frame_vlm, text="Pagine:").pack(side="left")
        ttk.Entry(self.frame_vlm, textvariable=self.var_vlm_pages,
                  width=14).pack(side="left", padx=2)
        ttk.Label(self.frame_vlm, text="  Pagine alta qualita' (Chandra):").pack(side="left")
        ttk.Entry(self.frame_vlm, textvariable=self.var_vlm_hq_pages,
                  width=14).pack(side="left", padx=2)
        ttk.Label(self.frame_vlm, text=" vuoto = tutte  ·  es. 1-10,379",
                  foreground="#666").pack(side="left", padx=8)

        # Riga 4: opzioni audio/video (whisper model)
        self.frame_wh = ttk.Frame(opts); self.frame_wh.pack(fill="x", pady=2)
        ttk.Label(self.frame_wh, text="(opzioni audio/video e sottotitoli .srt/.vtt)",
                  foreground="#666").pack(side="left", padx=(0, 4))
        ttk.Label(self.frame_wh, text="Modello Whisper:").pack(side="left", padx=(0, 4))
        ttk.Combobox(self.frame_wh, textvariable=self.var_whisper_model,
                     state="readonly", width=8,
                     values=("tiny", "base", "small", "medium", "large")).pack(side="left")
        ttk.Label(self.frame_wh,
                  text="(tiny=veloce ma approssimativo; base/small=consigliati; "
                       "large=massima qualita' ma MOLTO lento su questo PC)",
                  foreground="#666").pack(side="left", padx=10)

        # ===== Sezione 3: Log + Progress =====
        # NB: impacchettamento "dal basso" per garantire che i PULSANTI e le
        # BARRE siano SEMPRE visibili, anche rimpicciolendo la finestra.
        # tkinter riserva prima lo spazio degli elementi side="bottom",
        # poi il log (expand) riempie solo ciò che resta.
        ttk.Label(root, text="3) Log",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")

        # --- Pulsanti azione (ancorati IN FONDO, packati per primi) ---
        actions = ttk.Frame(root); actions.pack(side="bottom", fill="x", pady=(8, 0))
        self.btn_convert = ttk.Button(actions, text="▶ AVVIA CONVERSIONE",
                                      command=self._start, style="Accent.TButton")
        self.btn_convert.pack(side="left")
        self.btn_cancel = ttk.Button(actions, text="■ Annulla", command=self._cancel,
                                     state="disabled", style="Danger.TButton")
        self.btn_cancel.pack(side="left", padx=8)
        ttk.Button(actions, text="📂 Apri cartella output",
                   command=self._open_output).pack(side="left", padx=8)
        self.status = ttk.Label(actions, text="Pronto", foreground="#06a")
        self.status.pack(side="right")

        # --- Barra del FILE CORRENTE (sopra i pulsanti) ---
        prog_frame2 = ttk.Frame(root); prog_frame2.pack(side="bottom", fill="x", pady=(2, 0))
        ttk.Label(prog_frame2, text="In corso:", width=8).pack(side="left")
        self.pb_cur = ttk.Progressbar(prog_frame2, mode="determinate", maximum=100)
        self.pb_cur.pack(side="left", fill="x", expand=True, padx=4)
        self.lbl_cur_progress = ttk.Label(prog_frame2, text="--", width=8)
        self.lbl_cur_progress.pack(side="left", padx=4)
        self._cur_indeterminate = False  # stato barra animata

        # --- Barra TOTALE (sopra la barra corrente) ---
        prog_frame = ttk.Frame(root); prog_frame.pack(side="bottom", fill="x")
        ttk.Label(prog_frame, text="Totale:", width=8).pack(side="left")
        self.pb_file = ttk.Progressbar(prog_frame, mode="determinate")
        self.pb_file.pack(side="left", fill="x", expand=True, padx=4)
        self.lbl_file_progress = ttk.Label(prog_frame, text="0 / 0", width=8)
        self.lbl_file_progress.pack(side="left", padx=4)

        # --- Log (riempie lo spazio rimanente nel mezzo) ---
        self.log = scrolledtext.ScrolledText(root, height=10, wrap="word",
                                             font=("Consolas", 9))
        self.log.pack(side="top", fill="both", expand=True, pady=(4, 6))
        self.log.tag_config("live", foreground="#666")
        self.log.tag_config("ok",   foreground="#0a6")
        self.log.tag_config("err",  foreground="#a00")
        self.log.tag_config("info", foreground="#06a")

    # ------------------------------------------------------------------------
    # Eventi UI
    # ------------------------------------------------------------------------
    def _update_engine_visibility(self):
        eng = self.var_engine.get()
        # pymupdf4llm opzioni: page-images, DPI, soglia
        if eng in ("auto", "pymupdf4llm"):
            self._show_frame(self.frame_pym)
        else:
            self._hide_frame(self.frame_pym)
        # marker opzioni: force-OCR, block-size
        if eng in ("marker",):
            self._show_frame(self.frame_mk)
        else:
            self._hide_frame(self.frame_mk)
        # OCR IA locale: pagine / pagine alta qualita'
        if eng.startswith("vlm_"):
            self._show_frame(self.frame_vlm)
        else:
            self._hide_frame(self.frame_vlm)
        # whisper opzioni: audio / video, oppure destinazione sottotitoli
        target = (self.var_target_ext.get() or "").lower()
        if eng in ("auto", "audio", "video") or target in (".srt", ".vtt"):
            self._show_frame(self.frame_wh)
        else:
            self._hide_frame(self.frame_wh)

    def _is_packed(self, fr):
        # pack_info() solleva TclError se non impacchettato; ritorna dict altrimenti
        try:
            return bool(fr.pack_info())
        except tk.TclError:
            return False

    def _show_frame(self, fr):
        if not self._is_packed(fr):
            fr.pack(fill="x", pady=2)

    def _hide_frame(self, fr):
        if self._is_packed(fr):
            fr.pack_forget()

    def _add_files(self):
        pats = [
            ("Tutti i supportati",
             "*.pdf *.pptx *.ppt *.docx *.doc *.xlsx *.xls *.odt *.ods *.odp "
             "*.csv *.tsv *.html *.htm *.json *.xml *.epub *.rtf "
             "*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp *.gif "
             "*.md *.txt *.ipynb *.eml *.zip *.tar *.tgz "
             "*.mobi *.azw *.azw3 *.djvu *.fb2 "
             "*.mp3 *.wav *.m4a *.flac *.ogg *.aac "
             "*.mp4 *.mov *.avi *.webm *.mkv "
             "*.py *.js *.ts *.tsx *.jsx *.c *.h *.cpp *.cs *.java *.go *.rs "
             "*.rb *.php *.sh *.ps1 *.sql *.lua *.r *.swift *.kt *.scala "
             "*.css *.scss *.vue *.svelte *.tex *.bib "
             "*.yaml *.yml *.toml *.ini *.cfg *.conf *.env "
             "*.log *.diff *.patch *.dockerfile *.makefile"),
            ("PDF", "*.pdf"),
            ("Excel", "*.xlsx *.xls *.ods"),
            ("PowerPoint", "*.pptx *.ppt *.odp"),
            ("Word", "*.docx *.doc *.odt *.rtf"),
            ("CSV / TSV", "*.csv *.tsv"),
            ("Web / HTML", "*.html *.htm *.epub *.json *.xml"),
            ("Notebook Jupyter", "*.ipynb"),
            ("Email", "*.eml"),
            ("Archive", "*.zip *.tar *.tgz"),
            ("E-book proprietari (servono tool esterni)",
             "*.mobi *.azw *.azw3 *.djvu *.fb2"),
            ("Audio (Whisper)", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.opus"),
            ("Video (Whisper)", "*.mp4 *.mov *.avi *.webm *.mkv"),
            ("Codice sorgente",
             "*.py *.js *.ts *.tsx *.jsx *.c *.h *.cpp *.cs *.java *.go *.rs "
             "*.rb *.php *.sh *.ps1 *.sql *.lua *.r *.swift *.kt *.scala "
             "*.css *.scss *.vue *.svelte *.tex *.bib"),
            ("Config / dati testuali",
             "*.yaml *.yml *.toml *.ini *.cfg *.conf *.env *.log "
             "*.diff *.patch *.dockerfile *.makefile"),
            ("Immagini (lento, marker)", "*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp *.gif"),
            ("Tutti", "*.*"),
        ]
        paths = filedialog.askopenfilenames(title="Seleziona file", filetypes=pats)
        self._extend_files([Path(p) for p in paths])

    def _add_folder(self):
        d = filedialog.askdirectory(title="Seleziona cartella")
        if not d:
            return
        root = Path(d)
        files = [f for f in root.rglob("*")
                 if f.is_file() and f.suffix.lower() in ALL_SUPPORTED]
        self._extend_files(files)

    def _on_drop(self, event):
        # Tk passa i path tra '{ }' se hanno spazi
        raw = self.root.tk.splitlist(event.data)
        added = []
        for r in raw:
            p = Path(r)
            if p.is_dir():
                added.extend([f for f in p.rglob("*")
                              if f.is_file() and f.suffix.lower() in ALL_SUPPORTED])
            elif p.is_file() and p.suffix.lower() in ALL_SUPPORTED:
                added.append(p)
        self._extend_files(added)

    def _extend_files(self, paths: list[Path]):
        seen = {f.resolve() for f in self.files}
        n0 = len(self.files)
        for p in paths:
            pr = p.resolve()
            if pr in seen: continue
            if pr.suffix.lower() not in ALL_SUPPORTED: continue
            seen.add(pr)
            self.files.append(pr)
            self.lst.insert("end", str(pr))
        if len(self.files) > n0:
            self._update_count()

    def _remove_sel(self):
        for i in reversed(self.lst.curselection()):
            self.lst.delete(i)
            del self.files[i]
        self._update_count()

    def _clear(self):
        self.lst.delete(0, "end")
        self.files.clear()
        self._update_count()

    def _update_count(self):
        n = len(self.files)
        pdfs = sum(1 for f in self.files if f.suffix.lower() in PDF_EXT)
        self.lbl_count.config(text=f"{n} file ({pdfs} PDF)")
        self._refresh_target_options()

    def _refresh_target_options(self):
        """Aggiorna la combobox 'Converti in' con le destinazioni compatibili
           per i file attualmente caricati. Sempre .md disponibile."""
        if not self.files or not _ROUTER_OK:
            # Senza router resta solo la conversione verso Markdown
            self.cb_target["values"] = [".md"]
            return
        # Intersezione delle destinazioni supportate per OGNI file caricato
        common: set[str] | None = None
        for f in self.files:
            outs_per_modulo = _router.supported_outputs_for(f.suffix)
            outs = {o for vals in outs_per_modulo.values() for o in vals}
            if common is None:
                common = outs
            else:
                common &= outs
        common = sorted(common or {".md"})
        # .md sempre disponibile e in cima
        if ".md" not in common:
            common = [".md"] + common
        else:
            common = [".md"] + [x for x in common if x != ".md"]
        self.cb_target["values"] = common
        if self.var_target_ext.get() not in common:
            self.var_target_ext.set(".md")

    def _open_output(self):
        # apre la cartella del primo file selezionato (o della GUI)
        target = self.files[0].parent if self.files else Path(__file__).parent
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire la cartella:\n{e}")

    # ------------------------------------------------------------------------
    # Conversione: thread worker -> queue -> _pump_queue (main thread)
    # ------------------------------------------------------------------------
    def _start(self):
        if not self.files:
            messagebox.showwarning("Attenzione", "Aggiungi almeno un file.")
            return
        self.btn_convert.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.status.config(text="Conversione in corso...", foreground="#a60")
        self.log.delete("1.0", "end")
        self.pb_file.config(value=0, maximum=len(self.files))
        self.lbl_file_progress.config(text=f"0 / {len(self.files)}")
        self.cancel_flag.clear()
        self._save_config()

        self.worker = threading.Thread(target=self._worker_run, daemon=True)
        self.worker.start()

    def _cancel(self):
        if self.cancel_flag.is_set(): return
        self.cancel_flag.set()
        self.status.config(text="Annullamento... (finisce il file corrente)",
                           foreground="#a00")
        self.queue.put(("LOG", "⚠ Richiesto annullamento, finisco il file corrente..."))

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Chiudere?", "Una conversione e' in corso.\n"
                                                   "Vuoi davvero chiudere la finestra?"):
                return
            self.cancel_flag.set()
        self._save_config()
        self.root.destroy()

    def _worker_run(self):
        """Eseguito in un thread separato. NON tocca tkinter direttamente."""
        # Reindirizza stdout/stderr verso la queue: prendiamo cosi' anche i
        # progress di pymupdf4llm e i log di marker.
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = QueueWriter(self.queue)
        sys.stderr = QueueWriter(self.queue, prefix="[err] ")

        class Args: pass
        args = Args()
        args.overwrite        = self.var_overwrite.get()
        args.page_images      = self.var_page_images.get()
        args.page_image_dpi   = self.var_dpi.get()
        args.vector_threshold = self.var_vector_threshold.get()
        args.engine           = self.var_engine.get()
        args.media_dir        = self.var_media_dir.get()
        args.force_ocr        = self.var_force_ocr.get()
        args.block_size       = self.var_block_size.get()
        args.whisper_model    = self.var_whisper_model.get()
        args.lean_md          = self.var_lean_md.get()
        args.describe_images  = self.var_describe_images.get()
        args.describe_model   = None    # default di img_describe (qwen2.5vl:3b)
        args.pages            = self.var_vlm_pages.get().strip() or None
        args.hq_pages         = self.var_vlm_hq_pages.get().strip() or None
        args.vlm_threads      = None
        args.vlm_gpu          = False

        target_ext = (self.var_target_ext.get() or ".md").lower()
        opts_router = {
            "overwrite":       args.overwrite,
            "page_image_dpi":  args.page_image_dpi,
            "page_images":     args.page_images,
            "vector_threshold": args.vector_threshold,
            "media_dir":       args.media_dir,
            "engine":          args.engine,
            "whisper_model":   args.whisper_model,
            "force_ocr":       args.force_ocr,
            "block_size":      args.block_size,
            "describe_images": args.describe_images,
            "pages":           args.pages,
            "hq_pages":        args.hq_pages,
        }

        results = []
        try:
            self.queue.put(("LOG", f"File da convertire: {len(self.files)} -> {target_ext}"))
            # Keep-awake: il PC non si sospende durante la conversione
            with _KeepAwake():
                for i, f in enumerate(self.files, 1):
                    if self.cancel_flag.is_set():
                        self.queue.put(("LOG", f"⏹ Annullato dopo {i-1} file."))
                        break
                    self.queue.put(("PROG", (i - 1, len(self.files), f.name)))
                    self.queue.put(("CUR_START", f.name))  # avvia barra file corrente
                    dst = f.with_suffix(target_ext)
                    # PROTEZIONE ORIGINALI: se la destinazione esiste ed e' piu'
                    # vecchia del sorgente, e' con ogni probabilita' il file
                    # ORIGINALE da cui il sorgente era stato generato (es.
                    # 2.md -> 2.pdf dove 2.pdf e' l'originale). Non lo tocco.
                    if (dst.exists() and dst.resolve() != f.resolve()
                            and dst.stat().st_mtime < f.stat().st_mtime):
                        safe = f.with_name(f"{f.stem}_convertito{target_ext}")
                        self.queue.put(("LOG",
                            f"⚠ {dst.name} esiste ed e' piu' vecchio del sorgente: "
                            f"probabile ORIGINALE. Scrivo {safe.name}."))
                        dst = safe

                    if target_ext in (".md", ".markdown"):
                        # Modalità ad alta fedeltà via tomd2.convert_one
                        self.queue.put(("LOG", f"\n▶ [{i}/{len(self.files)}] "
                                               f"{f.name}  -> .md  "
                                               f"[{choose_engine(f, args.engine)}]"))
                        r = convert_one(f, args)
                        ok_i = r.get("ok", False)
                        rep  = r.get("report") or r.get("note","")
                        results.append({"src": str(f), "out": r.get("out", str(dst)),
                                         "ok": ok_i, "note": rep,
                                         "engine": r.get("engine","auto"),
                                         "module": "markdown"})
                    elif not _ROUTER_OK:
                        # Router non disponibile: conversione universale disattivata
                        ok_i = False
                        results.append({"src": str(f), "out": str(dst),
                                         "ok": False,
                                         "note": ("Conversione universale non disponibile "
                                                  f"(router non caricato: {_ROUTER_ERR}). "
                                                  "Usa il formato .md, oppure reinstalla le "
                                                  "dipendenze."),
                                         "engine": "-", "module": "-"})
                    else:
                        # Conversione universale via router.convert
                        self.queue.put(("LOG", f"\n▶ [{i}/{len(self.files)}] "
                                               f"{f.name}  -> {target_ext}"))
                        r = _router.convert(f, dst, opts_router)
                        ok_i = r.get("ok", False)
                        results.append({"src": str(f), "out": str(dst),
                                         "ok": ok_i, "note": r.get("note",""),
                                         "engine": r.get("module") or "-",
                                         "module": r.get("module") or "-"})
                    self.queue.put(("CUR_DONE", None))  # file corrente completato
                    if ok_i:
                        self.queue.put(("OK",
                            f"   ✓ OK  [{results[-1]['module']}]  "
                            f"{results[-1]['note'][:120]}"))
                    else:
                        self.queue.put(("ERR",
                            f"   ✗ ERR  {results[-1]['note'].splitlines()[0]}"))
                        for ln in results[-1]['note'].splitlines()[1:6]:
                            self.queue.put(("ERR", f"        {ln}"))
                    self.queue.put(("PROG", (i, len(self.files), f.name)))

            ok = sum(1 for r in results if r["ok"])
            self.queue.put(("LOG", "\n===== RIEPILOGO ====="))
            for r in results:
                tag = "OK" if r["ok"] else "ERR"
                self.queue.put((tag,
                    f"[{tag:>3}] {Path(r['src']).name} -> "
                    f"{Path(r['out']).name} ({r.get('module','-')})"))
            self.queue.put(("DONE", (ok, len(results))))
        except Exception as e:
            self.queue.put(("ERR", f"ERRORE WORKER: {type(e).__name__}: {e}"))
            self.queue.put(("DONE", (0, 0)))
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    # Pompa eventi dal thread worker al main thread.
    def _pump_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        self.root.after(80, self._pump_queue)

    # --- gestione barra "file corrente" -------------------------------------
    _PCT_RE = re.compile(r"(\d{1,3})\s*%")
    _FRAC_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")

    def _cur_start(self, name: str):
        """Inizio conversione di un file: barra animata (non sappiamo ancora la %)."""
        self.lbl_cur_progress.config(text="...")
        try:
            self.pb_cur.config(mode="indeterminate")
            self.pb_cur.start(60)
            self._cur_indeterminate = True
        except tk.TclError:
            pass

    def _cur_set_pct(self, pct: int):
        """Percentuale reale rilevata dallo stdout: passa a barra determinata."""
        pct = max(0, min(100, pct))
        if self._cur_indeterminate:
            try: self.pb_cur.stop()
            except tk.TclError: pass
            self.pb_cur.config(mode="determinate", maximum=100)
            self._cur_indeterminate = False
        self.pb_cur["value"] = pct
        self.lbl_cur_progress.config(text=f"{pct}%")

    def _cur_done(self):
        """Fine conversione del file: barra a 100% e ferma."""
        if self._cur_indeterminate:
            try: self.pb_cur.stop()
            except tk.TclError: pass
            self.pb_cur.config(mode="determinate", maximum=100)
            self._cur_indeterminate = False
        self.pb_cur["value"] = 100
        self.lbl_cur_progress.config(text="100%")

    def _maybe_update_cur_from_text(self, text: str):
        """Estrae una percentuale dalle righe di log (tqdm di pymupdf4llm/marker,
           oppure 'PROGRESS_AV: NN%' emesso da convert_av per ffmpeg)."""
        m = self._PCT_RE.search(text)
        if m:
            self._cur_set_pct(int(m.group(1)))
            return
        m2 = self._FRAC_RE.search(text)
        if m2:
            cur, tot = int(m2.group(1)), int(m2.group(2))
            if tot > 0:
                self._cur_set_pct(round(cur * 100 / tot))

    def _handle_event(self, kind: str, payload):
        if kind == "LOG":
            self.log.insert("end", payload + "\n")
            self.log.see("end")
            self._maybe_update_cur_from_text(payload)
        elif kind == "LOG_LIVE":
            # sostituisce l'ultima riga (tipico di tqdm \r)
            self.log.delete("end-2l", "end-1l")
            self.log.insert("end-1l", payload + "\n", "live")
            self.log.see("end")
            self._maybe_update_cur_from_text(payload)
        elif kind == "OK":
            self.log.insert("end", payload + "\n", "ok"); self.log.see("end")
        elif kind == "ERR":
            self.log.insert("end", payload + "\n", "err"); self.log.see("end")
        elif kind == "CUR_START":
            self._cur_start(payload)
        elif kind == "CUR_DONE":
            self._cur_done()
        elif kind == "PROG":
            done, total, name = payload
            self.pb_file["value"] = done
            self.pb_file["maximum"] = total
            self.lbl_file_progress.config(text=f"{done} / {total}")
        elif kind == "DONE":
            ok, tot = payload
            self.btn_convert.config(state="normal")
            self.btn_cancel.config(state="disabled")
            self.lbl_cur_progress.config(text="--")
            self.pb_cur["value"] = 0
            if self.cancel_flag.is_set():
                self.status.config(text=f"Annullato ({ok}/{tot} fatti)", foreground="#a00")
            else:
                self.status.config(text=f"Fatto: {ok}/{tot} OK", foreground="#0a6")
                if tot:
                    messagebox.showinfo("Conversione completata",
                                        f"Completati: {ok} / {tot}")

    # ------------------------------------------------------------------------
    # Persistenza impostazioni
    # ------------------------------------------------------------------------
    def _load_config(self) -> dict:
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_config(self):
        cfg = {
            "overwrite":        bool(self.var_overwrite.get()),
            "page_images":      bool(self.var_page_images.get()),
            "force_ocr":        bool(self.var_force_ocr.get()),
            "whisper_model":    str (self.var_whisper_model.get()),
            "target_ext":       str (self.var_target_ext.get()),
            "lean_md":          bool(self.var_lean_md.get()),
            "describe_images":  bool(self.var_describe_images.get()),
            "engine":           str (self.var_engine.get()),
            "page_image_dpi":   int (self.var_dpi.get()),
            "vector_threshold": int (self.var_vector_threshold.get()),
            "block_size":       int (self.var_block_size.get()),
            "media_dir":        str (self.var_media_dir.get()),
            "vlm_pages":        str (self.var_vlm_pages.get()),
            "vlm_hq_pages":     str (self.var_vlm_hq_pages.get()),
        }
        try:
            CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        except Exception:
            pass  # niente di drammatico se non riesce


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    TomdGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
