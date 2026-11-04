@echo off
REM ============================================================
REM  Installa l'OCR con IA LOCALE (gratis, offline dopo il download):
REM   - librerie Python (beautifulsoup4, markdownify)
REM   - llama.cpp (gia' presente in motori\llamacpp)
REM   - PaddleOCR-VL 1.6 (~1,4 GB), Chandra OCR 2 (~3,7 GB),
REM     Qwen3.5-9B (~6,6 GB) e GLM-4.6V-Flash (~7 GB): controllori
REM  Il download riprende da dove si era fermato se lo rilanci.
REM ============================================================
cd /d "%~dp0"
title Installazione OCR IA locale
set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

echo [1/3] Librerie Python...
"%PYEXE%" -m pip install -U beautifulsoup4 markdownify pymupdf pillow
echo.
echo [2/3] Applico le modifiche a tomd2 / GUI (con backup)...
"%PYEXE%" patch_vlm.py
echo.
echo [3/3] Scarico llama.cpp e i modelli (circa 19 GB, una tantum)...
"%PYEXE%" ocr_vlm.py setup tutti
echo.
echo ============================================================
echo  Finito. Per una prova veloce trascina un PDF su
echo  "Prova_OCR_locale.bat".
echo ============================================================
pause
