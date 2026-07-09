@echo off
title vLLM Server
setlocal enabledelayedexpansion

:: --- Auto-detect ROCm ---
if EXIST "C:\Program Files\AMD\ROCm\7.13\bin\hipcc.exe" set "HIP_PATH=C:\Program Files\AMD\ROCm\7.13"
if EXIST "C:\Program Files\AMD\ROCm\*\bin\hipcc.exe" for /f "delims=" %%A in ('dir /b "C:\Program Files\AMD\ROCm\*"') do set "HIP_PATH=C:\Program Files\AMD\ROCm\%%A"
if EXIST "E:\ROCM-7.13.0-Windows\bin\hipcc.exe" set "HIP_PATH=E:\ROCM-7.13.0-Windows"

:: --- Find model ---
set "MODEL=%1"
if "%MODEL%"=="" (
    if EXIST "F:\VLLM-Models\Qwen2.5-3B-Instruct" set "MODEL=F:\VLLM-Models\Qwen2.5-3B-Instruct"
    if EXIST "F:\VLLM-Models\Qwen2.5-7B-Instruct" set "MODEL=F:\VLLM-Models\Qwen2.5-7B-Instruct"
)

if "%MODEL%"=="" (
    echo Usage: run.bat ^<path-to-model^>
    echo.
    echo Or download a model first:
    echo   pip install huggingface_hub
    echo   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-3B-Instruct', local_dir=r'F:\VLLM-Models\Qwen2.5-3B-Instruct')"
    echo.
    echo Then: run.bat F:\VLLM-Models\Qwen2.5-3B-Instruct
    pause
    exit /b 1
)

echo [vLLM] Starting server...
echo [vLLM] Model: %MODEL%
echo [vLLM] Open http://localhost:8001/ in your browser
echo.

python -m vllm.entrypoints.openai.api_server --model "%MODEL%" --enforce-eager --dtype float16 --port 8001

if ERRORLEVEL 1 (
    echo.
    echo [ERROR] vLLM failed to start.
    echo Make sure you've activated your venv and run setup.bat first.
    pause
)
