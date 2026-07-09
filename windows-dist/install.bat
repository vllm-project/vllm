@echo off
title vLLM Installer
cd /d "%~dp0"
setlocal enabledelayedexpansion

echo =============================================
echo    vLLM for AMD ROCm - Interactive Installer
echo =============================================
echo.
echo This will install vLLM on your system.
echo.
echo.

:: ===== WHERE TO INSTALL =====
set "DEF_DIR=E:\VLLM"
:ask_path
set /p "INSTALL_DIR=Install folder [%DEF_DIR%]: "
if "!INSTALL_DIR!"=="" set "INSTALL_DIR=%DEF_DIR%"
:: Validate it's a proper path (has a colon for drive letter or starts with a backslash)
echo !INSTALL_DIR! | findstr /r "^[A-Za-z]:\\" >nul
if ERRORLEVEL 1 (
    if "!INSTALL_DIR!"=="%DEF_DIR%" goto :path_ok
    echo   Please enter a full path like E:\VLLM or C:\vllm
    goto :ask_path
)
:path_ok
echo.

:: Check if directory exists
if not EXIST "!INSTALL_DIR!" (
    echo [..] Creating directory...
    mkdir "!INSTALL_DIR!" >nul 2>&1
)
cd /d "!INSTALL_DIR!"
echo [OK] Installing to: !INSTALL_DIR!
echo.

:: ===== PYTHON VENV =====
set /p "CREATE_VENV=Create Python virtual environment? (Y/N) [Y]: "
if "!CREATE_VENV!"=="" set "CREATE_VENV=Y"
if /i "!CREATE_VENV!"=="Y" (
    echo [..] Creating virtual environment...
    python -m venv "!INSTALL_DIR!\.venv"
    if ERRORLEVEL 1 (
        echo [ERROR] Failed to create venv. Install Python 3.12 first.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created: !INSTALL_DIR!\.venv
    call "!INSTALL_DIR!\.venv\Scripts\activate.bat"
    echo [OK] Virtual environment activated.
) else (
    echo [SKIP] Using system Python.
)
echo.

:: ===== PYTORCH WITH ROCM =====
set /p "INSTALL_TORCH=Install PyTorch with ROCm? (Y/N) [Y]: "
if "!INSTALL_TORCH!"=="" set "INSTALL_TORCH=Y"
if /i "!INSTALL_TORCH!"=="Y" (
    echo [..] Installing PyTorch with ROCm...
    echo     This downloads ~3 GB from repo.amd.com
    pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ --timeout 120
    if ERRORLEVEL 1 (
        echo [ERROR] PyTorch install failed.
        echo         Try: pip install torch --index-url https://download.pytorch.org/whl/rocm7.13
        pause
        exit /b 1
    )
    echo [OK] PyTorch with ROCm installed.
) else (
    echo [SKIP] PyTorch install skipped.
)
echo.

:: ===== CLONE vLLM SOURCE =====
echo [..] Cloning vLLM Windows port...
if EXIST "!INSTALL_DIR!\vllm-windows" (
    echo [OK] Already cloned.
) else (
    git clone https://github.com/Maxritz/vllm-windows.git "!INSTALL_DIR!\vllm-windows"
    cd "!INSTALL_DIR!\vllm-windows"
    git checkout WINDOWS-PORT
    if ERRORLEVEL 1 (
        echo [WARN] Branch checkout failed, continuing...
    )
)
set "VLLM_SRC=!INSTALL_DIR!\vllm-windows"
echo [OK] vLLM source: !VLLM_SRC!
echo.

:: ===== INSTALL vLLM PACKAGE =====
cd /d "!VLLM_SRC!"
echo [..] Installing vLLM Python package...
pip install -e .
if ERRORLEVEL 1 (
    echo [WARN] pip install had issues, continuing...
)
echo [OK] vLLM package installed.
echo.

:: ===== COPY _C.pyd + vllm.exe =====
echo [..] Installing binaries...
set "PYD_SRC=%~dp0_C.pyd"
if EXIST "!PYD_SRC!" (
    copy /Y "!PYD_SRC!" "!VLLM_SRC!\vllm\_C.pyd" >nul
    for %%F in ("!PYD_SRC!") do set /a PYD_MB=%%~zF/1048576
    echo [OK] _C.pyd installed (!PYD_MB! MB)
) else (
    echo [WARN] _C.pyd not found in zip.
)
if EXIST "%~dp0vllm.exe" (
    copy /Y "%~dp0vllm.exe" "!VLLM_SRC!\" >nul
    echo [OK] vllm.exe copied.
)
echo.

:: ===== CONFIGURE ROCm =====
echo [..] Detecting ROCm...
set "HIP_PATH="
if EXIST "C:\Program Files\AMD\ROCm\7.13\bin\hipcc.exe" set "HIP_PATH=C:\Program Files\AMD\ROCm\7.13"
if EXIST "C:\Program Files\AMD\ROCm\7.12\bin\hipcc.exe" set "HIP_PATH=C:\Program Files\AMD\ROCm\7.12"
if "!HIP_PATH!"=="" (
    for /f "delims=" %%D in ('dir /b "C:\Program Files\AMD\ROCm\*" 2^>nul') do (
        if EXIST "C:\Program Files\AMD\ROCm\%%D\bin\hipcc.exe" set "HIP_PATH=C:\Program Files\AMD\ROCm\%%D"
    )
)
if "!HIP_PATH!"=="" (
    if NOT "!ROCM_HOME!"=="" set "HIP_PATH=!ROCM_HOME!"
    if NOT "!HIP_PATH!"=="" if EXIST "!HIP_PATH!\bin\hipcc.exe" set "HIP_PATH=!HIP_PATH!"
)
if "!HIP_PATH!"=="" (
    echo [WARN] ROCm not found. Set HIP_PATH manually in sitecustomize.py.
) else (
    echo [OK] ROCm found: !HIP_PATH!
)

:: ===== FIX sitecustomize.py (remove stale one with errors) =====
for /f "delims=" %%P in ('python -c "import sys; [print(p) for p in sys.path if p.endswith('site-packages')]" 2^>nul') do set "SITE_PKG=%%P"
if not "!SITE_PKG!"=="" (
    if EXIST "!SITE_PKG!\sitecustomize.py" del "!SITE_PKG!\sitecustomize.py"
    echo import os > "!SITE_PKG!\sitecustomize.py"
    echo os.environ.setdefault('HIP_PATH', '!HIP_PATH!') >> "!SITE_PKG!\sitecustomize.py"
    echo os.environ.setdefault('VLLM_NO_USAGE_STATS', 'true') >> "!SITE_PKG!\sitecustomize.py"
    echo [OK] sitecustomize.py written
)
echo.

:: ===== VERIFY =====
echo [..] Verifying vLLM...
python -c "import vllm; print('vLLM:', vllm.__version__)" 2>nul
if ERRORLEVEL 1 (
    echo [WARN] vLLM import failed. Check the installation.
) else (
    python -c "import vllm._C; print('_C.pyd: OK')" 2>nul
)
echo.

:: ===== DONE =====
echo =============================================
echo   INSTALLATION COMPLETE
echo =============================================
echo.
echo   Installed to: !INSTALL_DIR!
echo   vLLM source:  !VLLM_SRC!
echo.
echo   To start the server from terminal:
echo     "!VLLM_SRC!.venv\Scripts\activate.bat"
echo     python -m vllm.entrypoints.openai.api_server --model ^<model-path^> --enforce-eager
echo.
echo   Or use the GUI:
echo     "!VLLM_SRC!\vllm.exe"
echo.
echo   Press any key to exit...
pause >nul
