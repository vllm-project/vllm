# vLLM with CPU Offloading - Windows Installation Script
# Requires PowerShell 5.1+ and Administrator privileges

#Requires -Version 5.1

param(
    [switch]$Unattended = $false
)

$VERSION = "1.0.0"
$ErrorActionPreference = "Stop"

# Default values
$script:InstallDir = ""
$script:ModelPath = ""
$script:ModelURL = ""
$script:InstallManager = $false
$script:VLLMPort = 8000
$script:ManagerPort = 7999
$script:MaxModelLen = 262144
$script:GPUMemoryUtil = 0.88
$script:NumCPUBlocks = 16710
$script:ToolParser = "qwen3_coder"

# Color output functions
function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error-Message {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Warning-Message {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Cyan
}

function Write-Header {
    param([string]$Message)
    Write-Host "`n================================" -ForegroundColor Blue
    Write-Host $Message -ForegroundColor Blue
    Write-Host "================================`n" -ForegroundColor Blue
}

# Check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check prerequisites
function Test-Prerequisites {
    Write-Header "Checking Prerequisites"

    $missingDeps = @()

    # Check Python 3.8+
    try {
        $pythonVersion = & python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Python found: $pythonVersion"
        } else {
            $missingDeps += "Python 3.8+"
        }
    } catch {
        $missingDeps += "Python 3.8+"
    }

    # Check pip
    try {
        $pipVersion = & pip --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "pip found"
        } else {
            $missingDeps += "pip"
        }
    } catch {
        $missingDeps += "pip"
    }

    # Check git
    try {
        $gitVersion = & git --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "git found"
        } else {
            $missingDeps += "git"
        }
    } catch {
        $missingDeps += "git"
    }

    # Check NVIDIA GPU
    try {
        $gpuInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "NVIDIA GPU detected"
            Write-Host $gpuInfo
        } else {
            Write-Warning-Message "nvidia-smi not found. GPU support may not be available."
        }
    } catch {
        Write-Warning-Message "nvidia-smi not found. GPU support may not be available."
    }

    # Check CUDA
    if ($env:CUDA_PATH) {
        Write-Success "CUDA found: $env:CUDA_PATH"
    } else {
        Write-Warning-Message "CUDA_PATH not set. GPU support may not work."
    }

    if ($missingDeps.Count -gt 0) {
        Write-Error-Message "Missing dependencies: $($missingDeps -join ', ')"
        Write-Host "`nInstall Python from: https://www.python.org/downloads/"
        Write-Host "Install Git from: https://git-scm.com/download/win"
        exit 1
    }
}

# Interactive configuration
function Get-Configuration {
    Write-Header "Installation Configuration"

    # Choose installation directory
    Write-Host "`nWhere would you like to install vLLM?"
    Write-Host "1) User directory ($env:USERPROFILE\.vllm)"
    Write-Host "2) Program Files (C:\Program Files\vLLM)"
    Write-Host "3) Custom path"
    $installChoice = Read-Host "Choice [1]"
    if ([string]::IsNullOrWhiteSpace($installChoice)) { $installChoice = "1" }

    switch ($installChoice) {
        "1" {
            $script:InstallDir = "$env:USERPROFILE\.vllm"
        }
        "2" {
            $script:InstallDir = "C:\Program Files\vLLM"
        }
        "3" {
            $script:InstallDir = Read-Host "Enter custom path"
        }
        default {
            Write-Error-Message "Invalid choice"
            exit 1
        }
    }

    Write-Info "Installation directory: $script:InstallDir"

    # Model configuration
    Write-Host "`nModel Configuration:"
    Write-Host "1) Use existing model (provide path)"
    Write-Host "2) Download model from HuggingFace"
    Write-Host "3) Download model from custom URL"
    $modelChoice = Read-Host "Choice [1]"
    if ([string]::IsNullOrWhiteSpace($modelChoice)) { $modelChoice = "1" }

    switch ($modelChoice) {
        "1" {
            $script:ModelPath = Read-Host "Enter model path"
            if (-not (Test-Path $script:ModelPath)) {
                Write-Error-Message "Model path does not exist: $script:ModelPath"
                exit 1
            }
        }
        "2" {
            $hfModelId = Read-Host "Enter HuggingFace model ID (e.g., Qwen/Qwen3-Coder-30B)"
            $modelName = Split-Path $hfModelId -Leaf
            $script:ModelPath = "$script:InstallDir\models\$modelName"
            $script:ModelURL = "hf://$hfModelId"
        }
        "3" {
            $script:ModelURL = Read-Host "Enter model download URL"
            $modelName = Read-Host "Enter local model name"
            $script:ModelPath = "$script:InstallDir\models\$modelName"
        }
    }

    # Server configuration
    Write-Host ""
    $portInput = Read-Host "vLLM server port [8000]"
    if (-not [string]::IsNullOrWhiteSpace($portInput)) { $script:VLLMPort = [int]$portInput }

    $maxLenInput = Read-Host "Maximum model length [262144]"
    if (-not [string]::IsNullOrWhiteSpace($maxLenInput)) { $script:MaxModelLen = [int]$maxLenInput }

    $gpuMemInput = Read-Host "GPU memory utilization (0.0-1.0) [0.88]"
    if (-not [string]::IsNullOrWhiteSpace($gpuMemInput)) { $script:GPUMemoryUtil = [double]$gpuMemInput }

    $cpuBlocksInput = Read-Host "Number of CPU blocks for offloading [16710]"
    if (-not [string]::IsNullOrWhiteSpace($cpuBlocksInput)) { $script:NumCPUBlocks = [int]$cpuBlocksInput }

    $toolParserInput = Read-Host "Tool parser (leave empty for none) [qwen3_coder]"
    if (-not [string]::IsNullOrWhiteSpace($toolParserInput)) { $script:ToolParser = $toolParserInput }

    # Manager installation
    Write-Host ""
    $installMgr = Read-Host "Install vLLM Manager web interface? (yes/no) [no]"
    $script:InstallManager = ($installMgr -eq "yes")

    if ($script:InstallManager) {
        $mgrPortInput = Read-Host "Manager port [7999]"
        if (-not [string]::IsNullOrWhiteSpace($mgrPortInput)) { $script:ManagerPort = [int]$mgrPortInput }
    }

    # Summary
    Write-Header "Installation Summary"
    Write-Host "Installation directory: $script:InstallDir"
    Write-Host "Model path: $script:ModelPath"
    if ($script:ModelURL) { Write-Host "Model URL: $script:ModelURL" }
    Write-Host "vLLM port: $script:VLLMPort"
    Write-Host "Max model length: $script:MaxModelLen"
    Write-Host "GPU memory utilization: $script:GPUMemoryUtil"
    Write-Host "CPU blocks: $script:NumCPUBlocks"
    Write-Host "Tool parser: $(if ($script:ToolParser) { $script:ToolParser } else { 'none' })"
    Write-Host "Install manager: $script:InstallManager"
    if ($script:InstallManager) { Write-Host "Manager port: $script:ManagerPort" }

    Write-Host ""
    $confirm = Read-Host "Continue with installation? (yes/no) [yes]"
    if ([string]::IsNullOrWhiteSpace($confirm)) { $confirm = "yes" }

    if ($confirm -ne "yes") {
        Write-Info "Installation cancelled"
        exit 0
    }
}

# Create directory structure
function New-DirectoryStructure {
    Write-Header "Creating Directory Structure"

    $dirs = @(
        $script:InstallDir,
        "$script:InstallDir\bin",
        "$script:InstallDir\models",
        "$script:InstallDir\logs",
        "$script:InstallDir\config"
    )

    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }

    Write-Success "Directories created"
}

# Install Python dependencies
function Install-Dependencies {
    Write-Header "Installing Python Dependencies"

    # Create virtual environment
    if (-not (Test-Path "$script:InstallDir\venv")) {
        Write-Info "Creating virtual environment..."
        & python -m venv "$script:InstallDir\venv"
    }

    # Activate venv and upgrade pip
    Write-Info "Upgrading pip..."
    & "$script:InstallDir\venv\Scripts\pip.exe" install --upgrade pip wheel setuptools

    # Install vLLM
    Write-Info "Installing vLLM (this may take a while)..."
    & "$script:InstallDir\venv\Scripts\pip.exe" install vllm==0.11.0

    # Install additional dependencies
    Write-Info "Installing additional dependencies..."
    & "$script:InstallDir\venv\Scripts\pip.exe" install nvidia-ml-py requests aiohttp fastapi uvicorn

    Write-Success "Dependencies installed"
}

# Download model if needed
function Get-Model {
    if (-not $script:ModelURL) {
        return
    }

    Write-Header "Downloading Model"

    New-Item -ItemType Directory -Path $script:ModelPath -Force | Out-Null

    if ($script:ModelURL -like "hf://*") {
        Write-Info "Downloading from HuggingFace..."
        $repoId = $script:ModelURL -replace "hf://", ""
        & "$script:InstallDir\venv\Scripts\pip.exe" install huggingface_hub
        & "$script:InstallDir\venv\Scripts\python.exe" -c @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$repoId',
    local_dir='$($script:ModelPath)',
    resume_download=True
)
"@
    } else {
        Write-Info "Downloading from URL..."
        Invoke-WebRequest -Uri $script:ModelURL -OutFile "$script:ModelPath\model.zip"
        Expand-Archive -Path "$script:ModelPath\model.zip" -DestinationPath $script:ModelPath
        Remove-Item "$script:ModelPath\model.zip"
    }

    Write-Success "Model downloaded"
}

# Create server script
function New-ServerScript {
    Write-Header "Creating Server Script"

    $serverScript = @"
#!/usr/bin/env python3
"""
vLLM OpenAI API Server with CPU KV Cache Offloading
Generated by vLLM Installer v$VERSION
"""

import sys
import runpy
from vllm.config import KVTransferConfig

# Configure CPU offloading for KV cache
kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "num_cpu_blocks": $($script:NumCPUBlocks),
        "block_size": 16,
    },
)

# Build command line arguments
sys.argv = [
    "vllm.entrypoints.openai.api_server",
    "--model", r"$($script:ModelPath)",
    "--dtype", "auto",
    "--max-model-len", "$($script:MaxModelLen)",
    "--gpu-memory-utilization", "$($script:GPUMemoryUtil)",
    "--enforce-eager",
    "--max-num-seqs", "8",
    "--tensor-parallel-size", "1",
    "--enable-prefix-caching",
    "--host", "0.0.0.0",
    "--port", "$($script:VLLMPort)",
"@

    if ($script:ToolParser) {
        $serverScript += @"

    "--tool-call-parser", "$($script:ToolParser)",
    "--enable-auto-tool-choice",
"@
    }

    $serverScript += @'

]

# Monkey-patch the KVTransferConfig
import vllm.engine.arg_utils
original_create_engine_config = vllm.engine.arg_utils.EngineArgs.create_engine_config

def patched_create_engine_config(self, *args, **kwargs):
    if self.kv_transfer_config is None:
        self.kv_transfer_config = kv_transfer_config
    return original_create_engine_config(self, *args, **kwargs)

vllm.engine.arg_utils.EngineArgs.create_engine_config = patched_create_engine_config

# Run the API server
if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
'@

    Set-Content -Path "$script:InstallDir\bin\vllm_server.py" -Value $serverScript -Encoding UTF8
    Write-Success "Server script created"
}

# Create Windows service using NSSM
function New-WindowsService {
    Write-Header "Creating Windows Service"

    # Download NSSM if not present
    $nssmPath = "$script:InstallDir\bin\nssm.exe"
    if (-not (Test-Path $nssmPath)) {
        Write-Info "Downloading NSSM (Non-Sucking Service Manager)..."
        $nssmUrl = "https://nssm.cc/release/nssm-2.24.zip"
        $nssmZip = "$env:TEMP\nssm.zip"
        Invoke-WebRequest -Uri $nssmUrl -OutFile $nssmZip
        Expand-Archive -Path $nssmZip -DestinationPath $env:TEMP -Force
        Copy-Item "$env:TEMP\nssm-2.24\win64\nssm.exe" $nssmPath
        Remove-Item $nssmZip
        Remove-Item "$env:TEMP\nssm-2.24" -Recurse
    }

    # Install vLLM service
    Write-Info "Installing vLLM Windows service..."
    & $nssmPath install vLLM "$script:InstallDir\venv\Scripts\python.exe" "$script:InstallDir\bin\vllm_server.py"
    & $nssmPath set vLLM AppDirectory $script:InstallDir
    & $nssmPath set vLLM AppStdout "$script:InstallDir\logs\vllm.log"
    & $nssmPath set vLLM AppStderr "$script:InstallDir\logs\vllm.err"
    & $nssmPath set vLLM DisplayName "vLLM Server"
    & $nssmPath set vLLM Description "vLLM Large Language Model Server with CPU Offloading"
    & $nssmPath set vLLM Start SERVICE_AUTO_START

    Write-Success "Windows service created"
}

# Start services
function Start-Services {
    Write-Header "Starting Services"

    Write-Info "Starting vLLM service..."
    Start-Service -Name vLLM

    Start-Sleep -Seconds 5

    $service = Get-Service -Name vLLM
    if ($service.Status -eq "Running") {
        Write-Success "vLLM service started"
    } else {
        Write-Error-Message "Failed to start vLLM service"
        Write-Info "Check logs: $script:InstallDir\logs\vllm.err"
        exit 1
    }
}

# Print final instructions
function Show-Instructions {
    Write-Header "Installation Complete!"

    Write-Host ""
    Write-Success "vLLM is now installed and running!"
    Write-Host ""
    Write-Host "Service Management:"
    Write-Host "  Start:   Start-Service vLLM"
    Write-Host "  Stop:    Stop-Service vLLM"
    Write-Host "  Restart: Restart-Service vLLM"
    Write-Host "  Status:  Get-Service vLLM"
    Write-Host "  Logs:    Get-Content $script:InstallDir\logs\vllm.log -Tail 50 -Wait"
    Write-Host ""
    Write-Host "API Endpoint:"
    Write-Host "  http://localhost:$($script:VLLMPort)/v1/chat/completions"
    Write-Host ""
    Write-Host "Test the server:"
    Write-Host "  Invoke-WebRequest http://localhost:$($script:VLLMPort)/v1/models"
    Write-Host ""
    Write-Host "Configuration:"
    Write-Host "  Installation: $script:InstallDir"
    Write-Host "  Model: $script:ModelPath"
    Write-Host "  Logs: $script:InstallDir\logs\"
    Write-Host ""
    Write-Info "For more information, visit: https://github.com/datagram1/vllm"
}

# Main installation flow
function Main {
    Write-Header "vLLM with CPU Offloading Installer v$VERSION (Windows)"

    if (-not (Test-Administrator)) {
        Write-Error-Message "This script must be run as Administrator"
        Write-Info "Right-click PowerShell and select 'Run as Administrator'"
        exit 1
    }

    Test-Prerequisites

    if (-not $Unattended) {
        Get-Configuration
    }

    New-DirectoryStructure
    Install-Dependencies
    Get-Model
    New-ServerScript
    New-WindowsService
    Start-Services
    Show-Instructions
}

# Run main
Main
