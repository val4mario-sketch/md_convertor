@echo off
REM ============================================================
REM  Universal Converter launcher.
REM  DOUBLE-CLICK THIS FILE to open the app.
REM  Uses the right Python (the one with the libraries installed).
REM  If there is an error, the window stays open with the message.
REM ============================================================

cd /d "%~dp0"
title Universal Converter

REM 1) Specific Python where the libraries are installed
set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"

REM 2) If it is not there, try 'python' on the PATH
if not exist "%PYEXE%" (
  set "PYEXE=python"
)

echo Starting the Converter...
echo (Python in use: %PYEXE%)
echo.

"%PYEXE%" tomd2_gui.py
set ERR=%ERRORLEVEL%

if %ERR% NEQ 0 (
  echo.
  echo ============================================================
  echo   The app closed with an ERROR (code %ERR%).
  echo   Read the message above.
  echo.
  echo   If it says "ModuleNotFoundError", install the libraries:
  echo   "%PYEXE%" -m pip install -U pymupdf pymupdf4llm markitdown openpyxl python-pptx pandas pillow img2pdf cairosvg pypandoc tkinterdnd2
  echo ============================================================
  echo.
  pause
)
