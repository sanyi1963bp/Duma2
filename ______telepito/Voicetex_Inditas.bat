@echo off
chcp 65001 >nul
title Voicetex v3
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [HIBA] Nincs telepitve! Futtasd elobb a Telepito.bat fajlt.
    pause
    exit /b 1
)

REM A ctranslate2 (faster-whisper) GPU-n a cuDNN/cuBLAS DLL-eket keresi.
REM A torch sajat lib mappaja tartalmazza oket, ezert azt a PATH elejere tesszuk.
set "PATH=%~dp0venv\Lib\site-packages\torch\lib;%PATH%"

"venv\Scripts\python.exe" voicetex_v3.py

if errorlevel 1 (
    echo.
    echo [HIBA] A Voicetex hibaval allt le - a fenti uzenetek segitenek.
    pause
)
