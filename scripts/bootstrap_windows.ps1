# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
<#
.SYNOPSIS
Build and prove the Windows CUDA vLLM fork from scratch.

.DESCRIPTION
This script creates or reuses a Python virtual environment, clones or updates
the Windows compatibility branch, installs CUDA runtime dependencies, builds
vLLM from source, installs it editable, and runs the Windows CUDA smoke test.

Run from a normal PowerShell session. If Visual Studio Build Tools are
installed, the script imports the x64 developer environment automatically.
#>

[CmdletBinding()]
param(
    [string]$InstallRoot = "C:\tmp\vllm-windows-bootstrap",
    [string]$RepoUrl = "https://github.com/ericleigh007/vllm-windows.git",
    [string]$Branch = "windows-compat",
    [string]$RepoPath = "",
    [string]$VenvPath = "C:\tmp\vllmvenv",
    [string]$CudaPath = "C:\tmp\cuda13",
    [string]$CudaArch = "120",
    [string]$FetchContentBaseDir = "C:\tmp\vllm_deps",
    [int]$MaxJobs = 4,
    [int]$NvccThreads = 1,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130",
    [string[]]$TorchPackages = @("torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0"),
    [string[]]$ExtraPackages = @("triton-windows==3.6.0.post26", "flashinfer-python==0.6.8.post1", "huggingface_hub>=1.0.0"),
    [switch]$SkipDependencyInstall,
    [switch]$SkipBuild,
    [switch]$SkipSmoke,
    [switch]$AllowDirtyRepo
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Require-Command {
    param([string]$Name, [string]$InstallHint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found on PATH. $InstallHint"
    }
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Import-VisualStudioDevShell {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        Write-Host "MSVC compiler already available on PATH."
        return
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "cl.exe was not found and vswhere.exe is missing. Install Visual Studio 2022 Build Tools with the C++ workload."
    }

    $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $installPath) {
        throw "Visual Studio C++ Build Tools were not found. Install Visual Studio 2022 Build Tools with the C++ workload."
    }

    $devCmd = Join-Path $installPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $devCmd)) {
        throw "VsDevCmd.bat was not found at $devCmd."
    }

    Write-Host "Importing Visual Studio developer environment from $devCmd"
    $envLines = cmd.exe /c "`"$devCmd`" -arch=x64 -host_arch=x64 >nul && set"
    foreach ($line in $envLines) {
        if ($line -match "^(.*?)=(.*)$") {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
}

function Ensure-Venv {
    param([string]$Path)
    $python = Join-Path $Path "Scripts\python.exe"
    if (Test-Path $python) {
        return $python
    }

    Write-Step "Creating Python virtual environment at $Path"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        Invoke-Native $pyLauncher.Source "-3.12" "-m" "venv" $Path
    }
    else {
        Require-Command "python.exe" "Install Python 3.12, or install the Windows Python launcher."
        Invoke-Native "python.exe" "-m" "venv" $Path
    }
    return $python
}

function Ensure-Repo {
    param([string]$Path, [string]$Url, [string]$Ref)
    if (-not (Test-Path $Path)) {
        Write-Step "Cloning $Url ($Ref) to $Path"
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
        Invoke-Native "git.exe" "clone" "-b" $Ref $Url $Path
        return
    }

    if (-not (Test-Path (Join-Path $Path ".git"))) {
        throw "$Path exists but is not a Git repository."
    }

    Push-Location $Path
    try {
        $dirty = git status --porcelain
        if ($dirty -and -not $AllowDirtyRepo) {
            throw "$Path has uncommitted changes. Re-run with -AllowDirtyRepo to skip the safety check."
        }
        Write-Step "Updating existing repository at $Path"
        Invoke-Native "git.exe" "fetch" "origin" $Ref
        Invoke-Native "git.exe" "checkout" $Ref
        Invoke-Native "git.exe" "pull" "--ff-only" "origin" $Ref
    }
    finally {
        Pop-Location
    }
}

function Set-BootstrapEnvironment {
    param([string]$PythonExe)
    if (-not (Test-Path $CudaPath)) {
        throw "CUDA path $CudaPath was not found. Install CUDA or pass -CudaPath. Use a space-free path or junction if possible."
    }
    $nvcc = Join-Path $CudaPath "bin\nvcc.exe"
    if (-not (Test-Path $nvcc)) {
        throw "nvcc.exe was not found at $nvcc."
    }

    $torchLib = Join-Path (Split-Path -Parent (Split-Path -Parent $PythonExe)) "Lib\site-packages\torch\lib"
    $env:PATH = "$CudaPath\bin;$torchLib;$env:PATH"
    $env:CUDA_HOME = $CudaPath
    $env:CUDA_PATH = $CudaPath
    $env:CUDACXX = $nvcc
    $env:VLLM_TARGET_DEVICE = "cuda"
    $env:MAX_JOBS = [string]$MaxJobs
    $env:NVCC_THREADS = [string]$NvccThreads
    $env:FETCHCONTENT_BASE_DIR = $FetchContentBaseDir
    $env:CMAKE_ARGS = "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler"
    $env:HF_HUB_DISABLE_SYMLINKS = "1"
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
    $env:VLLM_WORKER_MULTIPROC_METHOD = "spawn"
}

if (-not $RepoPath) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    $candidateRepo = Resolve-Path (Join-Path $scriptRoot "..") -ErrorAction SilentlyContinue
    if ($candidateRepo -and (Test-Path (Join-Path $candidateRepo ".git"))) {
        $RepoPath = $candidateRepo.Path
    }
    else {
        $RepoPath = Join-Path $InstallRoot "vllm-windows"
    }
}

Write-Step "Checking host tools"
Require-Command "git.exe" "Install Git for Windows."
Import-VisualStudioDevShell

Ensure-Repo -Path $RepoPath -Url $RepoUrl -Ref $Branch
$pythonExe = Ensure-Venv -Path $VenvPath
Set-BootstrapEnvironment -PythonExe $pythonExe

Write-Step "Upgrading pip build tools"
Invoke-Native $pythonExe "-m" "pip" "install" "-U" "pip" "setuptools" "wheel" "ninja" "cmake"

if (-not $SkipDependencyInstall) {
    Write-Step "Installing PyTorch CUDA packages"
    $torchArgs = @("-m", "pip", "install")
    if ($TorchIndexUrl) {
        $torchArgs += @("--index-url", $TorchIndexUrl)
    }
    $torchArgs += $TorchPackages
    Invoke-Native $pythonExe @torchArgs

    if ($ExtraPackages.Count -gt 0) {
        Write-Step "Installing Windows runtime helper packages"
        Invoke-Native $pythonExe "-m" "pip" "install" @ExtraPackages
    }
}

Push-Location $RepoPath
try {
    if (-not $SkipBuild) {
        Write-Step "Building vLLM CUDA extensions in-place"
        Invoke-Native $pythonExe "setup.py" "build_ext" "--inplace"
    }

    Write-Step "Installing vLLM editable"
    Invoke-Native $pythonExe "-m" "pip" "install" "-e" "." "--no-build-isolation"

    if (-not $SkipSmoke) {
        Write-Step "Running Windows CUDA smoke test"
        Invoke-Native $pythonExe "scripts\windows_cuda_smoke.py"
    }
}
finally {
    Pop-Location
}

Write-Step "vLLM Windows bootstrap complete"
Write-Host "Repository: $RepoPath"
Write-Host "Virtualenv:  $VenvPath"
Write-Host "CUDA:        $CudaPath"
