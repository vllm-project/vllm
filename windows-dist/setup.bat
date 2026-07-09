@echo off
title vLLM Windows ROCm Setup
setlocal enabledelayedexpansion

echo ========================================
echo   vLLM Windows + AMD ROCm Setup
echo ========================================
echo.

:: --- Find vLLM source directory ---
set "VLLM_DIR="
if EXIST "%~dp0..\vllm\__init__.py" set "VLLM_DIR=%~dp0.."
if EXIST "%~dp0..\..\vllm\__init__.py" set "VLLM_DIR=%~dp0..\.."
if EXIST "%CD%\vllm\__init__.py" set "VLLM_DIR=%CD%"
if EXIST "vllm\__init__.py" set "VLLM_DIR=%CD%"

if "%VLLM_DIR%"=="" (
    echo [ERROR] vLLM source not found.
    echo.
    echo Clone it first:
    echo   git clone https://github.com/Maxritz/vllm-windows.git
    echo   cd vllm-windows
    echo   git checkout WINDOWS-PORT
    echo.
    set /p "VLLM_DIR=Enter path to vllm source: "
    if "!VLLM_DIR!"=="" exit /b 1
)

echo [OK] vLLM source at %VLLM_DIR%
echo.

:: --- Install _C.pyd ---
set "PYD_SRC=%~dp0_C.pyd"
set "PYD_DST=%VLLM_DIR%\vllm\_C.pyd"
if EXIST "%PYD_SRC%" (
    copy /Y "%PYD_SRC%" "%PYD_DST%" >nul
    for %%F in ("%PYD_SRC%") do set PYD_SIZE=%%~zF
    set /a PYD_MB=!PYD_SIZE! / 1048576
    echo [OK] Installed _C.pyd (!PYD_MB! MB)
) else (
    echo [WARN] _C.pyd not found alongside setup.bat
)

:: --- Copy build harness ---
set "HARNESS_SRC=%~dp0build-harness"
set "HARNESS_DST=%VLLM_DIR%\windows-dist\build-harness"
if EXIST "%HARNESS_SRC%" (
    if not EXIST "%HARNESS_DST%" mkdir "%HARNESS_DST%" >nul 2>&1
    xcopy /E /I /Y "%HARNESS_SRC%" "%HARNESS_DST%" >nul 2>&1
    echo [OK] Build harness copied
)

:: --- Find Python site-packages and create sitecustomize.py ---
echo.
echo [..] Creating sitecustomize.py...
for /f "delims=" %%P in ('python -c "import sys; [print(p) for p in sys.path if p.endswith('site-packages')]" 2^>nul') do (
    set "SITE_PKG=%%P"
)
if not "!SITE_PKG!"=="" (
    if not EXIST "!SITE_PKG!\sitecustomize.py" (
        echo import os > "!SITE_PKG!\sitecustomize.py"
        echo os.environ.setdefault('HIP_PATH', 'C:\Program Files\AMD\ROCm\7.13') >> "!SITE_PKG!\sitecustomize.py"
        echo os.environ.setdefault('VLLM_NO_USAGE_STATS', 'true') >> "!SITE_PKG!\sitecustomize.py"
        echo [OK] Created sitecustomize.py in !SITE_PKG!
    ) else (
        echo [OK] sitecustomize.py already exists
    )
) else (
    echo [WARN] Could not find Python site-packages
)

:: --- Verify PyTorch + ROCm ---
echo.
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm:', torch.cuda.is_available())" 2>nul
if ERRORLEVEL 1 (
    echo.
    echo [HINT] Install PyTorch with ROCm:
    echo   pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/
    echo.
    pause
)

:: --- Done ---
echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Run a model:
echo   python -m vllm.entrypoints.openai.api_server --model ^<model-path^> --enforce-eager
echo.
echo Rebuild _C.pyd:
echo   cd windows-dist\build-harness
echo   set HIP_PATH=C:\Program Files\AMD\ROCm\7.13
echo   python build_c_win.py
echo.
echo For help: https://github.com/Maxritz/vllm-windows
echo.
pause
