param(
    [string]$VllmDir = ".",
    [string]$HipPath = "E:\ROCM-7.13.0-Windows"
)

Write-Host "=== vLLM Windows ROCm Installer ===" -ForegroundColor Cyan

# 1. Verify ROCm
if (-not (Test-Path "$HipPath\bin\hipcc.exe")) {
    Write-Host "ERROR: HIP_PATH not found at $HipPath" -ForegroundColor Red
    Write-Host "Set -HipPath to your ROCm installation directory"
    exit 1
}
Write-Host "[OK] ROCm found at $HipPath" -ForegroundColor Green

# 2. Copy _C.pyd
$src = Join-Path $PSScriptRoot "_C.pyd"
$dst = Join-Path $VllmDir "vllm\_C.pyd"
if (Test-Path $src) {
    Copy-Item -LiteralPath $src -Destination $dst -Force
    Write-Host "[OK] Installed _C.pyd (13.8 MB)" -ForegroundColor Green
} else {
    Write-Host "[WARN] _C.pyd not found alongside installer" -ForegroundColor Yellow
}

# 3. Build harness
$harnessDir = Join-Path $VllmDir "dist\build-harness"
if (Test-Path (Join-Path $PSScriptRoot "build-harness")) {
    if (-not (Test-Path $harnessDir)) {
        New-Item -ItemType Directory -Path $harnessDir -Force | Out-Null
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "build-harness\*") -Destination $harnessDir -Recurse -Force
    Write-Host "[OK] Build harness copied" -ForegroundColor Green
}

# 4. Set up environment
$siteCust = Join-Path (Split-Path (python -c "import sys; print(sys.path[-1])" 2>$null)) "sitecustomize.py"
if ($siteCust -and -not (Test-Path $siteCust)) {
    Set-Content -Path $siteCust -Value @"
import os
os.environ.setdefault("HIP_PATH", r"$HipPath")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "true")
"@
    Write-Host "[OK] Created sitecustomize.py" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Install Complete ===" -ForegroundColor Cyan
Write-Host "To run a model:" -ForegroundColor Yellow
Write-Host "  `$env:HIP_PATH = '$HipPath'"
Write-Host "  python -m vllm.entrypoints.openai.api_server --model F:\VLLM-Models\Qwen2.5-3B-Instruct --enforce-eager"
Write-Host ""
Write-Host "To rebuild _C.pyd from source:" -ForegroundColor Yellow
Write-Host "  cd dist\build-harness"
Write-Host "  `$env:HIP_PATH = '$HipPath'"
Write-Host "  python build_c_win.py"
