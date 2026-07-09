param(
    [string]$VllmDir = "",
    [string]$HipPath = "",
    [string]$VenvDir = "",
    [switch]$SkipVenv
)

function Find-ROCm {
    $candidates = @(
        "$env:ProgramFiles\AMD\ROCm\*\bin\hipcc.exe",
        "${env:ProgramFiles(x86)}\AMD\ROCm\*\bin\hipcc.exe",
        "C:\ROCm\*\bin\hipcc.exe",
        "E:\ROCM-*\bin\hipcc.exe",
        "D:\ROCM-*\bin\hipcc.exe"
    )
    # Also check environment
    if ($env:ROCM_HOME -and (Test-Path "$env:ROCM_HOME\bin\hipcc.exe")) {
        return $env:ROCM_HOME
    }
    if ($env:ROCM_PATH -and (Test-Path "$env:ROCM_PATH\bin\hipcc.exe")) {
        return $env:ROCM_PATH
    }
    if ($env:HIP_PATH -and (Test-Path "$env:HIP_PATH\bin\hipcc.exe")) {
        return $env:HIP_PATH
    }
    foreach ($pattern in $candidates) {
        $matches = Resolve-Path $pattern -ErrorAction SilentlyContinue
        if ($matches) {
            return Split-Path (Split-Path $matches[-1] -Parent) -Parent
        }
    }
    return $null
}

function Find-VllmDir {
    # Try common locations relative to script
    $scriptDir = $PSScriptRoot
    # Script is in windows-dist/ or dist/ — look for vllm/ alongside
    $checks = @(
        (Join-Path $scriptDir "..\vllm"),
        (Join-Path $scriptDir "..\..\vllm"),
        (Join-Path (Get-Location) "vllm"),
        (Get-Location)
    )
    foreach ($d in $checks) {
        $resolved = Resolve-Path (Join-Path $d "__init__.py") -ErrorAction SilentlyContinue
        if ($resolved) {
            return Split-Path $resolved -Parent
        }
        $resolved = Resolve-Path (Join-Path $d "vllm\__init__.py") -ErrorAction SilentlyContinue
        if ($resolved) {
            return Split-Path (Split-Path $resolved -Parent) -Parent
        }
    }
    return (Get-Location)
}

Write-Host "=== vLLM Windows ROCm Installer ===" -ForegroundColor Cyan
Write-Host ""

# --- Detect ROCm ---
if (-not $HipPath) { $HipPath = Find-ROCm }
if (-not $HipPath -or -not (Test-Path "$HipPath\bin\hipcc.exe")) {
    Write-Host "ERROR: ROCm not found. Provide -HipPath or install ROCm from:" -ForegroundColor Red
    Write-Host "  https://rocm.docs.amd.com"
    Write-Host "  pip index: https://repo.amd.com/rocm/whl/gfx120X-all/"
    exit 1
}
Write-Host "[OK] ROCm $((Get-Item "$HipPath\bin\hipcc.exe").VersionInfo.ProductVersion)" -ForegroundColor Green
Write-Host "     $HipPath"

# --- Detect vLLM source ---
if (-not $VllmDir) { $VllmDir = Find-VllmDir }
$vllmPkg = Join-Path $VllmDir "vllm\__init__.py"
if (-not (Test-Path $vllmPkg)) {
    Write-Host "ERROR: vLLM source not found at $VllmDir. Provide -VllmDir" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] vLLM source at $VllmDir"

# --- Install _C.pyd ---
$srcPyd = Join-Path $PSScriptRoot "_C.pyd"
$dstPyd = Join-Path $VllmDir "vllm\_C.pyd"
if (Test-Path $srcPyd) {
    Copy-Item -LiteralPath $srcPyd -Destination $dstPyd -Force
    Write-Host "[OK] Installed _C.pyd ($((Get-Item $srcPyd).Length / 1MB -as [int]) MB)" -ForegroundColor Green
} else {
    Write-Host "[WARN] _C.pyd not found — build from source or download" -ForegroundColor Yellow
}

# --- Copy build harness ---
$harnessSrc = Join-Path $PSScriptRoot "build-harness"
$harnessDst = Join-Path $VllmDir "windows-dist\build-harness"
if (Test-Path $harnessSrc) {
    if (-not (Test-Path $harnessDst)) { New-Item -ItemType Directory -Path $harnessDst -Force | Out-Null }
    Copy-Item "$harnessSrc\*" -Destination $harnessDst -Recurse -Force
    Write-Host "[OK] Build harness at windows-dist\build-harness" -ForegroundColor Green
}

# --- Create sitecustomize.py in active venv ---
if (-not $SkipVenv) {
    try {
        # Use Python file to avoid PowerShell parsing conflicts with brackets
        $pyCode = @'
import sys
for p in sys.path:
    if p.endswith("site-packages"):
        print(p)
        break
'@
        $pythonLib = python -c $pyCode 2>$null
        if ($pythonLib) {
            $siteCust = Join-Path $pythonLib "sitecustomize.py"
            if (-not (Test-Path $siteCust)) {
                $scContent = 'import os' + [Environment]::NewLine + `
                    'os.environ.setdefault("HIP_PATH", r"' + $HipPath + '")' + [Environment]::NewLine + `
                    'os.environ.setdefault("VLLM_NO_USAGE_STATS", "true")'
                Set-Content -Path $siteCust -Value $scContent
                Write-Host "[OK] Created sitecustomize.py in $pythonLib" -ForegroundColor Green
            } else {
                Write-Host "[OK] sitecustomize.py already exists" -ForegroundColor Green
            }
        }
    } catch {
        Write-Host "[WARN] Could not auto-configure sitecustomize.py" -ForegroundColor Yellow
    }
}

# --- PyTorch ROCm install hint ---
try {
    $torchOk = python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($torchOk -ne "True") {
        Write-Host ""
        Write-Host "[HINT] Install PyTorch with ROCm:" -ForegroundColor Yellow
        Write-Host "  pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/" -ForegroundColor Yellow
        Write-Host "  Or (stable): https://download.pytorch.org/whl/rocm7.13" -ForegroundColor Yellow
    } else {
        Write-Host "[OK] PyTorch + ROCm CUDA detected" -ForegroundColor Green
    }
} catch {}

# --- Security notes ---
Write-Host ""
Write-Host "=== Security Notes ===" -ForegroundColor Cyan
Write-Host "* Firewall: vLLM listens on ports 8000-8001 (API) and 29500 (distributed)."
Write-Host "  On Windows, use: New-NetFirewallRule -DisplayName 'vLLM' -Direction Inbound -Protocol TCP -LocalPort 8000,8001,29500 -Action Block"
Write-Host "* Cache dir: %LOCALAPPDATA%\vllm\cache — restrict with icacls if multi-user system."
Write-Host "* Do NOT bind to 0.0.0.0 on untrusted networks (VPN, public WiFi)."

# --- Summary ---
Write-Host ""
Write-Host "=== Install Complete ===" -ForegroundColor Cyan
Write-Host "ROCm:      $HipPath"
Write-Host "vLLM:      $VllmDir"
Write-Host ""
Write-Host "Run:  `$env:HIP_PATH = '$HipPath'"
Write-Host "      `$env:PYTHONPATH = '$VllmDir'"
Write-Host "      python -m vllm.entrypoints.openai.api_server --model <path-to-model> --enforce-eager --dtype float16"
Write-Host ""
Write-Host "Rebuild _C.pyd:  cd windows-dist\build-harness && `$env:HIP_PATH='$HipPath' && python build_c_win.py"
Write-Host "Distribution zip packed in windows-dist\vllm-windows-rocm-dist.zip"
