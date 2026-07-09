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
set "HIP_PATH="
if EXIST "C:\Program Files\AMD\ROCm\7.13\bin\hipcc.exe" set "HIP_PATH=C:\Program Files\AMD\ROCm\7.13" & set "ROCVER=7.13"
if EXIST "C:\Program Files\AMD\ROCm\7.12\bin\hipcc.exe" set "HIP_PATH=C:\Program Files\AMD\ROCm\7.12" & set "ROCVER=7.12"
if EXIST "C:\Program Files\AMD\ROCm\7.11\bin\hipcc.exe" set "HIP_PATH=C:\Program Files\AMD\ROCm\7.11" & set "ROCVER=7.11"
if "!HIP_PATH!"=="" (
    for /f "delims=" %%D in ('dir /b "C:\Program Files\AMD\ROCm\*" 2^>nul') do (
        if EXIST "C:\Program Files\AMD\ROCm\%%D\bin\hipcc.exe" set "HIP_PATH=C:\Program Files\AMD\ROCm\%%D" & set "ROCVER=%%D"
    )
)
if "!HIP_PATH!"=="" (
    if NOT "!ROCM_HOME!"=="" set "HIP_PATH=!ROCM_HOME!"
    if NOT "!HIP_PATH!"=="" if EXIST "!HIP_PATH!\bin\hipcc.exe" for %%I in ("!HIP_PATH!") do set "ROCVER=%%~nxI"
)
if "!HIP_PATH!"=="" (
    echo [WARN] ROCm not detected. Will attempt to install anyway.
    set "ROCVER=7.13"
) else (
    echo [OK] ROCm !ROCVER! at !HIP_PATH!
)

:: Choose matching torch version based on ROCm version
if "!ROCVER!"=="7.13" set "TORCH_VER=2.11.0+rocm7.13.0" & set "TV_VER=0.22.0+rocm7.13.0"
if "!ROCVER!"=="7.12" set "TORCH_VER=2.10.0+rocm7.12.0" & set "TV_VER=0.21.0+rocm7.12.0"
if "!ROCVER!"=="7.11" set "TORCH_VER=2.9.1+rocm7.11.0" & set "TV_VER=0.20.1+rocm7.11.0"
if "!ROCVER!"=="7.10" set "TORCH_VER=2.9.1+rocm7.10.0" & set "TV_VER=0.20.1+rocm7.10.0"
if "!TORCH_VER!"=="" set "TORCH_VER=2.11.0+rocm7.13.0" & set "TV_VER=0.22.0+rocm7.13.0"

echo Detected: ROCm !ROCVER! -> torch !TORCH_VER!
echo.

:: ===== PYTORCH WITH ROCM =====
set /p "DO_TORCH=Install PyTorch !TORCH_VER! with ROCm? (Y/N) [Y]: "
if "!DO_TORCH!"=="" set "DO_TORCH=Y"
if /i "!DO_TORCH!"=="Y" (
    echo Installing from https://repo.amd.com/rocm/whl/gfx120X-all/ ...
    pip install torch==!TORCH_VER! torchvision==!TV_VER! --find-links https://repo.amd.com/rocm/whl/gfx120X-all/ --timeout 120
    if ERRORLEVEL 1 (
        echo AMD repo failed. Trying PyTorch official repo...
        pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm!ROCVER! --timeout 120
        if ERRORLEVEL 1 (
            echo PyTorch install failed.
            pause
            exit /b 1
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
