param(
    [string]$VllmDir = "",
    [string]$HipPath = "",
    [string]$VenvDir = "",
    [switch]$SkipVenv
)

function Find-ROCm {
    $envCandidates = @("ROCM_HOME", "ROCM_PATH", "HIP_PATH")
    foreach ($e in $envCandidates) {
        $v = [Environment]::GetEnvironmentVariable($e)
        if ($v -and (Test-Path "$v/bin/hipcc.exe")) { return $v }
    }
    $paths = @(
        "$env:ProgramFiles/AMD/ROCm/*/bin/hipcc.exe",
        "${env:ProgramFiles(x86)}/AMD/ROCm/*/bin/hipcc.exe",
        "C:/ROCm/*/bin/hipcc.exe",
        "E:/ROCM-*/bin/hipcc.exe"
    )
    foreach ($p in $paths) {
        $m = Resolve-Path $p -ErrorAction SilentlyContinue
        if ($m) { return Split-Path (Split-Path $m[-1] -Parent) -Parent }
    }
    return $null
}

function Find-VllmDir {
    $scriptDir = $PSScriptRoot
    $checks = @(
        (Join-Path $scriptDir "../vllm"),
        (Join-Path $scriptDir "../../vllm"),
        (Join-Path (Get-Location) "vllm"),
        (Get-Location)
    )
    foreach ($d in $checks) {
        $resolved = Resolve-Path (Join-Path $d "__init__.py") -ErrorAction SilentlyContinue
        if ($resolved) { return Split-Path $resolved -Parent }
        $resolved = Resolve-Path (Join-Path $d "vllm/__init__.py") -ErrorAction SilentlyContinue
        if ($resolved) { return Split-Path (Split-Path $resolved -Parent) -Parent }
    }
    return (Get-Location)
}

Write-Host "=== vLLM Windows ROCm Installer ===" -ForegroundColor Cyan
Write-Host ""

# --- Detect ROCm ---
if (-not $HipPath) { $HipPath = Find-ROCm }
if (-not $HipPath -or -not (Test-Path "$HipPath/bin/hipcc.exe")) {
    Write-Host "ERROR: ROCm not found." -ForegroundColor Red
    Write-Host "Install from: https://rocm.docs.amd.com" -ForegroundColor Yellow
    Write-Host "Pip wheels: https://repo.amd.com/rocm/whl/gfx120X-all/" -ForegroundColor Yellow
    exit 1
}
$rocVer = (Get-Item "$HipPath/bin/hipcc.exe").VersionInfo.ProductVersion
Write-Host "[OK] ROCm $rocVer" -ForegroundColor Green
Write-Host "     $HipPath"

# --- Detect vLLM source ---
if (-not $VllmDir) { $VllmDir = Find-VllmDir }
$vllmPkg = Join-Path $VllmDir "vllm/__init__.py"
if (-not (Test-Path $vllmPkg)) {
    Write-Host "ERROR: vLLM source not found at $VllmDir" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] vLLM source at $VllmDir"

# --- Install _C.pyd ---
$srcPyd = Join-Path $PSScriptRoot "_C.pyd"
$dstPyd = Join-Path $VllmDir "vllm/_C.pyd"
if (Test-Path $srcPyd) {
    Copy-Item -LiteralPath $srcPyd -Destination $dstPyd -Force
    $size = [math]::Round((Get-Item $srcPyd).Length / 1MB, 0)
    Write-Host "[OK] Installed _C.pyd ($size MB)" -ForegroundColor Green
} else {
    Write-Host "[WARN] _C.pyd not found" -ForegroundColor Yellow
}

# --- Copy build harness ---
$harnessSrc = Join-Path $PSScriptRoot "build-harness"
$harnessDst = Join-Path $VllmDir "windows-dist/build-harness"
if (Test-Path $harnessSrc -and (Resolve-Path $harnessSrc).Path -ne (Resolve-Path $harnessDst).Path) {
    if (-not (Test-Path $harnessDst)) { New-Item -ItemType Directory -Path $harnessDst -Force | Out-Null }
    Copy-Item "$harnessSrc/*" -Destination $harnessDst -Recurse -Force
    Write-Host "[OK] Build harness copied" -ForegroundColor Green
}

# --- Create sitecustomize.py ---
if (-not $SkipVenv) {
    try {
        $tmpf = [System.IO.Path]::GetTempFileName() + ".py"
        Set-Content -Path $tmpf -Value "import sys`nfor p in sys.path:`n if p.endswith('site-packages'):`n  print(p); break"
        $pythonLib = python $tmpf 2>$null
        Remove-Item $tmpf -Force -ErrorAction SilentlyContinue
        if ($pythonLib) {
            $siteCust = Join-Path $pythonLib "sitecustomize.py"
            if (-not (Test-Path $siteCust)) {
                $scLines = @(
                    "import os",
                    "os.environ.setdefault('HIP_PATH', '$HipPath')",
                    "os.environ.setdefault('VLLM_NO_USAGE_STATS', 'true')"
                )
                Set-Content -Path $siteCust -Value ($scLines -join "`n")
                Write-Host "[OK] Created sitecustomize.py" -ForegroundColor Green
            } else {
                Write-Host "[OK] sitecustomize.py exists" -ForegroundColor Green
            }
        }
    } catch {
        Write-Host "[WARN] Could not create sitecustomize.py" -ForegroundColor Yellow
    }
}

# --- PyTorch check ---
try {
    $tmpf = [System.IO.Path]::GetTempFileName() + ".py"
    Set-Content -Path $tmpf -Value "import torch; print(torch.cuda.is_available())"
    $torchOk = python $tmpf 2>$null
    Remove-Item $tmpf -Force -ErrorAction SilentlyContinue
    if ($torchOk -ne "True") {
        Write-Host ""
        Write-Host "[HINT] Install PyTorch with ROCm:" -ForegroundColor Yellow
        Write-Host "  pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/" -ForegroundColor Yellow
        Write-Host "  Or: pip install torch --index-url https://download.pytorch.org/whl/rocm7.13" -ForegroundColor Yellow
    } else {
        Write-Host "[OK] PyTorch + ROCm detected" -ForegroundColor Green
    }
} catch {}

# --- Security notes ---
Write-Host ""
Write-Host "=== Security Notes ===" -ForegroundColor Cyan
Write-Host "* Firewall block on public networks:"
Write-Host "  New-NetFirewallRule -DisplayName 'vLLM' -Direction Inbound -Protocol TCP -LocalPort 8000,8001,29500 -Action Block"
Write-Host "* Cache directory: %LOCALAPPDATA%/vllm/cache"
Write-Host "* API has no auth -- use a reverse proxy in production"

# --- Summary ---
Write-Host ""
Write-Host "=== Install Complete ===" -ForegroundColor Cyan
Write-Host "ROCm: $HipPath"
Write-Host "vLLM: $VllmDir"
Write-Host ""
Write-Host "Run: python -m vllm.entrypoints.openai.api_server --model <model-path> --enforce-eager"
Write-Host "Rebuild _C.pyd: cd windows-dist/build-harness && python build_c_win.py"
