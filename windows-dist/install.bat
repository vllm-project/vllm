@echo off
title vLLM Installer
cd /d "%~dp0"
setlocal enabledelayedexpansion

echo =============================================
echo    vLLM for AMD ROCm - Interactive Installer
echo =============================================
echo.

:: ===== WHERE TO INSTALL =====
set "DEF_DIR=E:\VLLM"
:ask_path
set /p "INSTALL_DIR=Install folder [%DEF_DIR%]: "
if "!INSTALL_DIR!"=="" set "INSTALL_DIR=%DEF_DIR%"
:: Validate full path like E:\...
echo !INSTALL_DIR! | findstr /r "^[A-Za-z]:" >nul 2>nul
if not ERRORLEVEL 1 if EXIST "!INSTALL_DIR!" goto :path_ok
if not ERRORLEVEL 1 (
    rem Has drive letter - check parent exists
    for %%I in ("!INSTALL_DIR!") do set "PARENT=%%~dpI"
    if EXIST "!PARENT!" goto :path_ok
)
echo   Enter a valid path like E:\VLLM or C:\vllm
goto :ask_path
:path_ok
if not EXIST "!INSTALL_DIR!" mkdir "!INSTALL_DIR!"
cd /d "!INSTALL_DIR!"
echo Installed to: !INSTALL_DIR!
echo.

:: ===== PYTHON VENV =====
set /p "DO_VENV=Create Python venv? (Y/N) [Y]: "
if "!DO_VENV!"=="" set "DO_VENV=Y"
if /i "!DO_VENV!"=="Y" (
    python -m venv "!INSTALL_DIR!\.venv" 2>nul
    if ERRORLEVEL 1 (
        echo Python 3.12 not found. Install from python.org first.
        pause
        exit /b 1
    )
    call "!INSTALL_DIR!\.venv\Scripts\activate.bat"
    echo Venv: !INSTALL_DIR!\.venv
)
echo.

:: ===== DETECT ROCm VERSION =====
:: Save system env vars before clearing local ones
set "ENV_ROCM_HOME=%ROCM_HOME%"
set "ENV_ROCM_PATH=%ROCM_PATH%"
set "ENV_HIP_PATH=%HIP_PATH%"
set "HIP_PATH="
:: Check environment variables (from saved system values)
if NOT "!ENV_ROCM_HOME!"=="" if EXIST "!ENV_ROCM_HOME!\bin\hipcc.exe" set "HIP_PATH=!ENV_ROCM_HOME!"
if "!HIP_PATH!"=="" if NOT "!ENV_ROCM_PATH!"=="" if EXIST "!ENV_ROCM_PATH!\bin\hipcc.exe" set "HIP_PATH=!ENV_ROCM_PATH!"
if "!HIP_PATH!"=="" if NOT "!ENV_HIP_PATH!"=="" if EXIST "!ENV_HIP_PATH!\bin\hipcc.exe" set "HIP_PATH=!ENV_HIP_PATH!"
:: Check common paths
if "!HIP_PATH!"=="" if EXIST "C:\Program Files\AMD\ROCm\7.13\bin\hipcc.exe" set "HIP_PATH=C:\Program Files\AMD\ROCm\7.13"
if "!HIP_PATH!"=="" if EXIST "C:\Program Files\AMD\ROCm\7.12\bin\hipcc.exe" set "HIP_PATH=C:\Program Files\AMD\ROCm\7.12"
:: Scan C:\Program Files\AMD\ROCm\* for any version
if "!HIP_PATH!"=="" (
    for /f "delims=" %%D in ('dir /b "C:\Program Files\AMD\ROCm\*" 2^>nul') do (
        if EXIST "C:\Program Files\AMD\ROCm\%%D\bin\hipcc.exe" set "HIP_PATH=C:\Program Files\AMD\ROCm\%%D"
    )
)
:: Scan E:\ROCM-* and other drive roots
if "!HIP_PATH!"=="" (
    for %%D in (C D E F G) do (
        for /f "delims=" %%R in ('dir /b "%%D:\ROCM-*" 2^>nul') do (
            if EXIST "%%D:\%%R\bin\hipcc.exe" set "HIP_PATH=%%D:\%%R"
        )
    )
)
:: Extract version number from path (e.g. "7.13" from "ROCM-7.13.0-Windows")
if NOT "!HIP_PATH!"=="" (
    for %%I in ("!HIP_PATH!") do set "ROCVER=%%~nxI"
    :: Extract just X.Y where X and Y are the first two numbers found
    for /f "tokens=1,2,3 delims=-." %%a in ("!ROCVER!") do set "ROCVER=%%a.%%b"
    :: Remove any remaining non-numeric characters after the version
    set "ROCVER=!ROCVER:ROCM_=!"
    set "ROCVER=!ROCVER:rocm=!"
    set "ROCVER=!ROCVER:Rocm=!"
    :: Ensure it's just X.Y format
    for /f "tokens=1,2 delims=." %%a in ("!ROCVER!") do set "ROCVER=%%a.%%b"
    echo [OK] ROCm at !HIP_PATH! (version !ROCVER!)
) else (
    echo [WARN] ROCm not detected. Will attempt to install anyway.
    set "ROCVER=7.13"
)

:: Check Python - prefer 3.12, but any version works for the venv
python --version >nul 2>&1
if ERRORLEVEL 1 (
    echo Python not found. Download Python 3.12 from python.org first.
    pause
    exit /b 1
)

:: Check Python version - warn if not 3.12
for /f "tokens=2 delims= " %%V in ('python --version 2^>nul') do set "PY_VER=%%V"
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do set "PY_MAJ=%%a" & set "PY_MIN=%%b"
echo Python: !PY_VER!

if NOT "!PY_MAJ!"=="3" (
    echo Python 3 required. You have !PY_VER!. Install Python 3.12.
    pause
    exit /b 1
)
if !PY_MIN! GTR 13 (
    echo WARNING: Python !PY_VER! may not have ROCm torch wheels.
    echo ROCm wheels support Python 3.10-3.13. Try Python 3.12 instead.
    echo Continuing anyway...
)
if !PY_MIN! LSS 10 (
    echo Python !PY_VER! is too old. Use Python 3.12.
    pause
    exit /b 1
)
echo.

:: Choose matching torch version based on ROCm version + Python version
set "TORCH_VER=2.11.0+rocm7.13.0"
set "TV_VER=0.22.0+rocm7.13.0"
if "!ROCVER!"=="7.13" set "TORCH_VER=2.11.0+rocm7.13.0" & set "TV_VER=0.22.0+rocm7.13.0"
if "!ROCVER!"=="7.12" set "TORCH_VER=2.10.0+rocm7.12.0" & set "TV_VER=0.21.0+rocm7.12.0"
if "!ROCVER!"=="7.11" set "TORCH_VER=2.9.1+rocm7.11.0" & set "TV_VER=0.20.1+rocm7.11.0"
if "!ROCVER!"=="7.10" set "TORCH_VER=2.9.1+rocm7.10.0" & set "TV_VER=0.20.1+rocm7.10.0"

echo Detected: ROCm !ROCVER! + Python !PY_TAG! -> torch !TORCH_VER!
echo.

:: ===== PYTORCH WITH ROCM =====
set /p "DO_TORCH=Install PyTorch !TORCH_VER! with ROCm? (Y/N) [Y]: "
if "!DO_TORCH!"=="" set "DO_TORCH=Y"
if /i "!DO_TORCH!"=="Y" (
    :: Build Python tag for direct wheel URL (e.g. cp312-cp312-win_amd64)
    for /f "tokens=2 delims= " %%V in ('python --version 2^>nul') do set "PY_FULL=%%V"
    for /f "tokens=1,2 delims=." %%a in ("!PY_FULL!") do set "PY_TAG=cp%%a%%b"
    set "WHEEL_TAG=!PY_TAG!-!PY_TAG!-win_amd64"

    echo Installing from AMD repo...
    pip install torch==!TORCH_VER! torchvision==!TV_VER! --extra-index-url https://repo.amd.com/rocm/whl/gfx120X-all/ --timeout 120
    if ERRORLEVEL 1 (
        echo AMD index failed. Trying direct wheel download...
        pip install https://repo.amd.com/rocm/whl/gfx120X-all/torch/torch-!TORCH_VER!-!WHEEL_TAG!.whl --timeout 120
        if ERRORLEVEL 1 (
            echo Direct download failed. Trying PyTorch official ROCm repo...
            pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm!ROCVER! --timeout 120
            if ERRORLEVEL 1 (
                echo PyTorch install failed.
                pause
                exit /b 1
            )
        )
    )
    echo PyTorch + ROCm installed.
)
echo.

:: ===== CLONE vLLM =====
if EXIST "!INSTALL_DIR!\vllm-windows" (
    echo vLLM already cloned.
) else (
    git clone https://github.com/Maxritz/vllm-windows.git "!INSTALL_DIR!\vllm-windows"
    cd "!INSTALL_DIR!\vllm-windows"
    git checkout WINDOWS-PORT 2>nul
)
set "SRC=!INSTALL_DIR!\vllm-windows"
echo Source: !SRC!
echo.

:: ===== INSTALL vLLM =====
cd /d "!SRC!"
pip install -e . 2>nul
echo vLLM package installed.

:: ===== COPY _C.pyd + vllm.exe =====
if EXIST "%~dp0_C.pyd" copy /Y "%~dp0_C.pyd" "!SRC!\vllm\_C.pyd" >nul
if EXIST "%~dp0vllm.exe" copy /Y "%~dp0vllm.exe" "!SRC!\" >nul
echo Binaries installed.

:: ===== CREATE sitecustomize.py =====
:: Use forward slashes to avoid Python backslash escape issues (e.g. \7 → bell)
set "HIP_SAFE=!HIP_PATH:\=/!"
python -c "import sys; p=[x for x in sys.path if 'site-packages' in x][0]" 2>nul >"%TEMP%\sp.txt"
set /p "SP=" <"%TEMP%\sp.txt"
if not "!SP!"=="" (
    if EXIST "!SP!\sitecustomize.py" del "!SP!\sitecustomize.py"
    echo import os > "!SP!\sitecustomize.py"
    echo os.environ.setdefault('HIP_PATH', '!HIP_SAFE!') >> "!SP!\sitecustomize.py"
    echo os.environ.setdefault('VLLM_NO_USAGE_STATS', 'true') >> "!SP!\sitecustomize.py"
    echo [OK] sitecustomize.py created
)

:: ===== DONE =====
echo.
echo ===== INSTALLATION COMPLETE =====
echo.
echo vLLM is ready in: !SRC!
echo.
echo Run:  "!SRC!\vllm.exe"
echo.
pause
