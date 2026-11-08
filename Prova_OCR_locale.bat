@echo off
REM ============================================================
REM  TEST of the local AI OCR: drag a PDF onto this file.
REM  Converts only some pages with BOTH engines and writes
REM     <name>_prova_paddle.md   and   <name>_prova_chandra.md
REM  next to the PDF (does NOT touch other .md files). Prints the times at the end.
REM ============================================================
cd /d "%~dp0"
title Local AI OCR test
set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"
if "%~1"=="" (
  echo Drag a PDF onto this .bat file
  pause
  exit /b 1
)
set "PAGINE=1-3"
set /p PAGINE=Quali pagine provo? (invio = 1-3, es. 12,115,381,400): 
"%PYEXE%" ocr_vlm.py prova "%~1" --pages %PAGINE% --engine tutti
echo.
echo Engine log in: %~dp0logs
pause
