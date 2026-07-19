@echo off
chcp 65001 >nul
setlocal EnableExtensions
title Voicetex v3 - Telepito
cd /d "%~dp0"

echo ============================================================
echo   VOICETEX v3 - TELEPITO
echo ============================================================
echo.

REM ── 1. Python keresese (3.10 - 3.12 ajanlott) ────────────────
set "PYCMD="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYCMD (
        py -%%V -c "print()" >nul 2>&1 && set "PYCMD=py -%%V"
    )
)
if not defined PYCMD (
    python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
    echo [HIBA] Nem talaltam megfelelo Pythont ^(3.10 - 3.12 kell^).
    echo.
    echo Telepitsd innen: https://www.python.org/downloads/
    echo Telepiteskor pipald be: "Add python.exe to PATH"
    echo Utana futtasd ujra ezt a telepitot.
    pause
    exit /b 1
)
for /f "delims=" %%P in ('%PYCMD% -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%P"
echo [OK] Python megtalalva: %PYVER%
echo.

REM ── 2. Virtualis kornyezet ───────────────────────────────────
if exist "venv\Scripts\python.exe" (
    echo [OK] Meglevo venv hasznalata.
) else (
    echo [..] Virtualis kornyezet letrehozasa...
    %PYCMD% -m venv venv
    if errorlevel 1 (
        echo [HIBA] A venv letrehozasa nem sikerult.
        pause
        exit /b 1
    )
)
set "VPY=%~dp0venv\Scripts\python.exe"

echo [..] pip frissitese...
"%VPY%" -m pip install --upgrade pip --quiet

REM ── 3. GPU felismerese es torch telepitese ───────────────────
echo.
echo [..] Videokartya azonositasa...
set "GPUNEV="
for /f "delims=" %%G in ('powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ' + '" 2^>nul') do set "GPUNEV=%%G"
if defined GPUNEV echo      Talalt GPU: %GPUNEV%

where nvidia-smi >nul 2>&1
if not errorlevel 1 goto :torch_cuda

REM Nincs aktiv NVIDIA meghajto. Megnezzuk, van-e egyaltalan NVIDIA kartya.
REM FONTOS JAVITAS: a GPU nev zarojelet tartalmazhat - pl. "Intel(R)" -,
REM ezert a %GPUNEV% valtozot TILOS zarojeles if/else blokkon belul
REM kiertekeltetni (a nevben levo zarojel lezarna a blokkot). Ezert megy
REM ez a resz goto-elagazassal, es a nev-vizsgalat sima sorban tortenik.
set "NVKARTYA=0"
if defined GPUNEV echo %GPUNEV%| findstr /i "nvidia geforce rtx gtx quadro" >nul 2>&1 && set "NVKARTYA=1"
if "%NVKARTYA%"=="1" goto :torch_cpu_nvdriver

echo [!] Nincs NVIDIA GPU - CPU-s PyTorch telepitese.
echo.
echo     FONTOS: a faster-whisper GPU-gyorsitasa csak NVIDIA/CUDA
echo     kartyaval mukodik. AMD vagy Intel GPU-n az app CPU modban
echo     fut - mukodik, de a large modellek lassabbak lesznek.
echo     Ilyen gepen a "small" vagy "medium" modell ajanlott
echo     ^(a programban a Whisper modell listabol valaszthato^).
goto :torch_cpu_install

:torch_cpu_nvdriver
echo.
echo [!] NVIDIA kartyat talaltam, de a meghajtoja NINCS telepitve
echo     ^(az nvidia-smi nem elerheto^).
echo     Telepitsd a drivert: https://www.nvidia.com/drivers
echo     majd futtasd ujra ezt a telepitot a GPU-gyorsitasert.
echo.
echo     Most CPU-s valtozat telepul - az app mukodni fog, csak lassabban.

:torch_cpu_install
"%VPY%" -m pip install torch
goto :torch_kesz

:torch_cuda
echo [OK] NVIDIA meghajto aktiv - CUDA-s PyTorch telepitese ^(cu128^).
echo      ^(RTX 50xx / Blackwell kartyakhoz a cu128 valtozat kell.^)
echo      Ez ~3 GB letoltes, eltarthat par percig...
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128

:torch_kesz
if errorlevel 1 (
    echo [HIBA] A torch telepitese nem sikerult. Ellenorizd az internetkapcsolatot.
    pause
    exit /b 1
)

REM ── 4. Tobbi fuggoseg ────────────────────────────────────────
echo.
echo [..] Tovabbi csomagok telepitese ^(faster-whisper, transformers, stb.^)...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [HIBA] A csomagok telepitese nem sikerult teljesen.
    echo        Nezd meg a fenti hibauzenetet, majd futtasd ujra a telepitot.
    pause
    exit /b 1
)

REM ── 5. Gyors onteszt ─────────────────────────────────────────
echo.
echo [..] Telepites ellenorzese...
"%VPY%" -c "import torch, faster_whisper, transformers, sounddevice; print('  torch:', torch.__version__); print('  CUDA elerheto:', torch.cuda.is_available())"
if errorlevel 1 (
    echo [HIBA] Az onteszt nem futott le hibatlanul.
    pause
    exit /b 1
)

REM ── 6. Asztali parancsikon ───────────────────────────────────
echo.
echo [..] Asztali parancsikon letrehozasa...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Voicetex v3.lnk');" ^
  "$s.TargetPath='%~dp0Voicetex_Inditas.bat';" ^
  "$s.WorkingDirectory='%~dp0';" ^
  "$s.Description='Voicetex v3 - magyar diktalo';" ^
  "$s.Save()" >nul 2>&1
if exist "%USERPROFILE%\Desktop\Voicetex v3.lnk" (
    echo [OK] Parancsikon az asztalon: "Voicetex v3"
) else (
    echo [!] A parancsikont nem sikerult letrehozni - inditsd a Voicetex_Inditas.bat fajllal.
)

echo.
echo ============================================================
echo   TELEPITES KESZ!
echo ============================================================
echo.
echo Inditas: asztali "Voicetex v3" ikon vagy Voicetex_Inditas.bat
echo.
echo FONTOS: az elso inditaskor a Whisper modell ^(~3 GB^) automatikusan
echo letoltodik - ez egyszeri, utana helyi cache-bol tolt.
echo.
pause
