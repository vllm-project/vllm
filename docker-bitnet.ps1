<#
.SYNOPSIS
    Launch vLLM in Docker for BitNet development.

.DESCRIPTION
    Runs the vLLM Docker container with:
    - GPU passthrough
    - HuggingFace cache mounted (avoids re-downloading models)
    - bitnet_vllm plugin directory mounted

.EXAMPLE
    # Start vLLM OpenAI-compatible API server (port 8000)
    .\docker-bitnet.ps1 -Serve

    # Chat interactively (requires -Serve running in another terminal)
    .\docker-bitnet.ps1 -Chat

    # Quick generation test
    .\docker-bitnet.ps1 -Test

    # Run the parity test against HuggingFace
    .\docker-bitnet.ps1 -Parity

    # Interactive shell inside the container
    .\docker-bitnet.ps1

    # Run a specific script
    .\docker-bitnet.ps1 -Script "bitnet_vllm/scripts/test_parity.py"
#>
param(
    [string]$Script = "",
    [switch]$Serve,
    [switch]$Chat,
    [switch]$Test,
    [switch]$Pytest,
    [switch]$Parity,
    [switch]$Build
)

$HF_CACHE = "$env:USERPROFILE\.cache\huggingface"
$PROJECT_DIR = $PSScriptRoot
$CONTAINER_NAME = "vllm-bitnet-dev"

# Convert Windows paths to Docker-compatible paths
$HF_CACHE_DOCKER = $HF_CACHE -replace '\\', '/' -replace '^C:', '/c'
$PROJECT_DIR_DOCKER = $PROJECT_DIR -replace '\\', '/' -replace '^C:', '/c'

if ($Serve) {
    Write-Host "Starting vLLM OpenAI-compatible API server..." -ForegroundColor Cyan
    Write-Host "  Model: microsoft/bitnet-b1.58-2B-4T-bf16" -ForegroundColor DarkGray
    Write-Host "  Port:  8000" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "Once running, you can:" -ForegroundColor Yellow
    Write-Host "  - Chat interactively:  .\docker-bitnet.ps1 -Chat" -ForegroundColor Yellow
    Write-Host "  - Use curl:" -ForegroundColor Yellow
    Write-Host '    curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d "{\"model\": \"microsoft/bitnet-b1.58-2B-4T-bf16\", \"prompt\": \"Hello world\", \"max_tokens\": 64}"' -ForegroundColor DarkGray
    Write-Host ""

    docker run --rm `
        --gpus all `
        --name $CONTAINER_NAME `
        -v "${HF_CACHE_DOCKER}:/root/.cache/huggingface" `
        -v "${PROJECT_DIR_DOCKER}/bitnet_vllm:/app/bitnet_vllm" `
        -p 8000:8000 `
        -e PYTHONPATH=/app `
        --entrypoint python3 `
        vllm/vllm-openai:latest `
        /app/bitnet_vllm/scripts/serve.py
}
elseif ($Chat) {
    Write-Host "Starting interactive chat client..." -ForegroundColor Cyan
    Write-Host "(Make sure the server is running with: .\docker-bitnet.ps1 -Serve)" -ForegroundColor DarkGray
    Write-Host ""
    python "$PROJECT_DIR\bitnet_vllm\scripts\chat.py"
}
elseif ($Test) {
    Write-Host "Running quick BitNet generation test..." -ForegroundColor Cyan
    docker run --rm `
        --gpus all `
        --name $CONTAINER_NAME `
        -v "${HF_CACHE_DOCKER}:/root/.cache/huggingface" `
        -v "${PROJECT_DIR_DOCKER}/bitnet_vllm:/app/bitnet_vllm" `
        -e PYTHONPATH=/app `
        --entrypoint python3 `
        vllm/vllm-openai:latest `
        /app/bitnet_vllm/scripts/test_multi.py
}
elseif ($Pytest) {
    Write-Host "Running pytest: test_bitnet.py inside Docker..." -ForegroundColor Cyan
    Write-Host "  This runs the HF vs vLLM logprob comparison test." -ForegroundColor DarkGray
    Write-Host ""

    # Mount tests/ into a workspace dir, and overlay bitnet.py + registry.py
    # into the installed vLLM package. We do NOT mount the full source tree
    # because that would shadow the installed vllm package (with compiled C extensions).
    docker run --rm `
        --gpus all `
        --name "${CONTAINER_NAME}-pytest" `
        -v "${HF_CACHE_DOCKER}:/root/.cache/huggingface" `
        -v "${PROJECT_DIR_DOCKER}/tests:/workspace/vllm/tests" `
        -v "${PROJECT_DIR_DOCKER}/vllm/model_executor/models/bitnet.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/bitnet.py" `
        -v "${PROJECT_DIR_DOCKER}/vllm/model_executor/models/registry.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py" `
        --workdir /workspace/vllm `
        --entrypoint bash `
        vllm/vllm-openai:latest `
        -c "pip install pytest tblib -q && python3 -m pytest tests/models/language/generation/test_bitnet.py -v -x -s"
}
elseif ($Parity) {
    Write-Host "Running parity test (HF vs vLLM)..." -ForegroundColor Cyan
    Write-Host "Step 1: Generate HF reference..." -ForegroundColor Yellow

    docker run --rm `
        --gpus all `
        --name "${CONTAINER_NAME}-hf" `
        -v "${HF_CACHE_DOCKER}:/root/.cache/huggingface" `
        -v "${PROJECT_DIR_DOCKER}/bitnet_vllm:/app/bitnet_vllm" `
        -e PYTHONPATH=/app `
        --entrypoint python3 `
        vllm/vllm-openai:latest `
        /app/bitnet_vllm/scripts/hf_reference.py

    Write-Host "`nStep 2: Run vLLM and compare..." -ForegroundColor Yellow

    docker run --rm `
        --gpus all `
        --name "${CONTAINER_NAME}-vllm" `
        -v "${HF_CACHE_DOCKER}:/root/.cache/huggingface" `
        -v "${PROJECT_DIR_DOCKER}/bitnet_vllm:/app/bitnet_vllm" `
        -e PYTHONPATH=/app `
        --entrypoint python3 `
        vllm/vllm-openai:latest `
        /app/bitnet_vllm/scripts/test_parity.py
}
elseif ($Script -ne "") {
    Write-Host "Running script: $Script" -ForegroundColor Cyan
    docker run --rm `
        --gpus all `
        --name $CONTAINER_NAME `
        -v "${HF_CACHE_DOCKER}:/root/.cache/huggingface" `
        -v "${PROJECT_DIR_DOCKER}/bitnet_vllm:/app/bitnet_vllm" `
        -e PYTHONPATH=/app `
        --entrypoint python3 `
        vllm/vllm-openai:latest `
        "/app/$Script"
}
else {
    Write-Host "Starting interactive vLLM container..." -ForegroundColor Cyan
    Write-Host "  HF cache: $HF_CACHE" -ForegroundColor DarkGray
    Write-Host "  Project:  $PROJECT_DIR" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "Inside the container, you can run:" -ForegroundColor Yellow
    Write-Host "  python3 -c 'import vllm; print(vllm.__version__)'" -ForegroundColor Yellow
    Write-Host "  python3 /app/bitnet_vllm/scripts/test_load.py" -ForegroundColor Yellow
    Write-Host ""
    docker run --rm -it `
        --gpus all `
        --name $CONTAINER_NAME `
        -v "${HF_CACHE_DOCKER}:/root/.cache/huggingface" `
        -v "${PROJECT_DIR_DOCKER}/bitnet_vllm:/app/bitnet_vllm" `
        -e PYTHONPATH=/app `
        -p 8000:8000 `
        --entrypoint /bin/bash `
        vllm/vllm-openai:latest
}
