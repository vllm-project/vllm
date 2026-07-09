@echo off
title vLLM for AMD ROCm - One Click Setup
cd /d "%~dp0"
echo ==============================================
echo    vLLM for AMD ROCm - One Click Setup
echo ==============================================
echo.
echo This will install everything needed to run vLLM
echo on your AMD GPU. Internet connection required.
echo.

:: ===== 1. CHECK/SETUP FOLDER =====
set "ROOT=%~dp0"
if EXIST "%ROOT%..\vllm\__init__.py" set "ROOT=%~dp0.."
set "VENV_DIR=%ROOT%.venv"

:: ===== 2. CHECK PYTHON =====
echo [1/6] Checking Python...
python --version >nul 2>&1
if ERRORLEVEL 1 (
    echo   Python not found. Downloading Python 3.12...
    curl -sL -o "%TEMP%\python-installer.exe" https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe
    if ERRORLEVEL 1 (
        echo   Download failed. Install Python 3.12 manually from python.org
        pause
        exit /b 1
    )
    start /wait "" "%TEMP%\python-installer.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    echo   Python 3.12 installed.
) else (
    echo   Python found.
)

:: ===== 3. CHECK GIT =====
echo [2/6] Checking Git...
git --version >nul 2>&1
if ERRORLEVEL 1 (
    echo   Git not found. Downloading Git...
    curl -sL -o "%TEMP%\git-installer.exe" https://github.com/git-for-windows/git/releases/download/v2.48.1.windows.1/Git-2.48.1-64-bit.exe
    if ERRORLEVEL 1 (
        echo   Download failed. Install Git manually from git-scm.com
        pause
        exit /b 1
    )
    start /wait "" "%TEMP%\git-installer.exe" /SILENT
    echo   Git installed.
) else (
    echo   Git found.
)

:: ===== 4. SETUP VENV =====
echo [3/6] Creating virtual environment...
if not EXIST "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
    echo   Virtual environment created.
) else (
    echo   Virtual environment already exists.
)

:: ===== 5. INSTALL PYTORCH WITH ROCM =====
echo [4/6] Installing PyTorch with ROCm (this downloads ~3 GB)...
echo   This will take a while...
call "%VENV_DIR%\Scripts\activate.bat"
pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ --timeout 120
if ERRORLEVEL 1 (
    echo.
    echo   PyTorch install failed. Try:
    echo   pip install torch --index-url https://download.pytorch.org/whl/rocm7.13
    pause
)
echo   PyTorch installed.

:: ===== 6. CLONE vLLM (if needed) =====
echo [5/6] Setting up vLLM...
if not EXIST "%ROOT%vllm\__init__.py" (
    if EXIST "%ROOT%..\vllm\__init__.py" (
        set "ROOT=%~dp0.."
    ) else (
        echo   Cloning vLLM Windows port...
        cd "%ROOT%"
        git clone https://github.com/Maxritz/vllm-windows.git vllm-src
        cd vllm-src
        git checkout WINDOWS-PORT
        set "ROOT=%ROOT%vllm-src"
    )
)

:: Install vLLM package
cd /d "%ROOT%"
pip install -e .
echo   vLLM package installed.

:: Copy _C.pyd
if EXIST "%~dp0_C.pyd" (
    copy /Y "%~dp0_C.pyd" "%ROOT%\vllm\_C.pyd" >nul
    echo   _C.pyd installed.
) else (
    echo   WARNING: _C.pyd not found in zip.
)

:: ===== 7. CREATE sitecustomize.py =====
for /f "delims=" %%P in ('python -c "import sys; [print(p) for p in sys.path if p.endswith('site-packages')]" 2^>nul') do set "SITE_PKG=%%P"
if not "!SITE_PKG!"=="" (
    if not EXIST "!SITE_PKG!\sitecustomize.py" (
        echo import os > "!SITE_PKG!\sitecustomize.py"
        echo os.environ.setdefault('HIP_PATH', 'C:\Program Files\AMD\ROCm\7.13') >> "!SITE_PKG!\sitecustomize.py"
        echo os.environ.setdefault('VLLM_NO_USAGE_STATS', 'true') >> "!SITE_PKG!\sitecustomize.py"
        echo [OK] sitecustomize.py created
    )
)

:: ===== DONE =====
echo.
echo ==============================================
echo   INSTALL COMPLETE!
echo ==============================================
echo.
echo   vLLM is ready to use.
echo.
echo   To start the server:
echo     "%ROOT%.venv\Scripts\activate.bat"
echo     python -m vllm.entrypoints.openai.api_server --model ^<model-path^> --enforce-eager
echo.
echo   Or double-click vllm.exe in this folder for the GUI.
echo.
echo   Press any key to exit...
pause >nul
