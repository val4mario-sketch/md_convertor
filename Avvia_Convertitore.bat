@echo off
REM ============================================================
REM  Launcher del Convertitore Universale.
REM  FAI DOPPIO CLIC SU QUESTO FILE per aprire l'app.
REM  Usa il Python corretto (quello con le librerie installate).
REM  Se c'e' un errore, la finestra resta aperta col messaggio.
REM ============================================================

cd /d "%~dp0"
title Convertitore Universale

REM 1) Python specifico dove sono installate le librerie
set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"

REM 2) Se non c'e', prova 'python' nel PATH
if not exist "%PYEXE%" (
  set "PYEXE=python"
)

echo Avvio del Convertitore...
echo (Python in uso: %PYEXE%)
echo.

"%PYEXE%" tomd2_gui.py
set ERR=%ERRORLEVEL%

if %ERR% NEQ 0 (
  echo.
  echo ============================================================
  echo   L'app si e' chiusa con ERRORE (codice %ERR%).
  echo   Leggi sopra il messaggio.
  echo.
  echo   Se dice "ModuleNotFoundError", installa le librerie:
  echo   "%PYEXE%" -m pip install -U pymupdf pymupdf4llm markitdown openpyxl python-pptx pandas pillow img2pdf cairosvg pypandoc tkinterdnd2
  echo ============================================================
  echo.
  pause
)
