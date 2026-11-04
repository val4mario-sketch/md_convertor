@echo off
REM ============================================================
REM  PROVA dell'OCR IA locale: trascina un PDF su questo file.
REM  Converte solo alcune pagine con ENTRAMBI i motori e scrive
REM     <nome>_prova_paddle.md   e   <nome>_prova_chandra.md
REM  accanto al PDF (NON tocca altri .md). Alla fine stampa i tempi.
REM ============================================================
cd /d "%~dp0"
title Prova OCR IA locale
set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"
if "%~1"=="" (
  echo Trascina un PDF sopra questo file .bat
  pause
  exit /b 1
)
set "PAGINE=1-3"
set /p PAGINE=Quali pagine provo? (invio = 1-3, es. 12,115,381,400): 
"%PYEXE%" ocr_vlm.py prova "%~1" --pages %PAGINE% --engine tutti
echo.
echo Log del motore in: %~dp0logs
pause
