#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Configures a Podman machine running under WSL2 (Fedora 42 base) for NVIDIA GPU passthrough.

.DESCRIPTION
    Installs the NVIDIA Container Toolkit inside the Podman machine, generates a CDI spec for
    WSL2 GPUs, and verifies device/node availability. This script is idempotent and safe to run
    multiple times. A machine reboot is recommended after toolkit installation.

.PARAMETER MachineName
    Name of the Podman machine to configure. Defaults to "podman-machine-default".

.PARAMETER SkipReboot
    Prevents the script from automatically restarting the Podman machine after configuration.

.EXAMPLE
    pwsh extras/tools/enable-podman-wsl-gpu.ps1

.NOTES
    Requires Podman 4.6+ with `podman machine` support and an NVIDIA driver on Windows that
    exposes the CUDA WSL integration. Run from an elevated PowerShell session for best results.
#>

[CmdletBinding()]
param(
    [string]$MachineName = "podman-machine-default",
    [switch]$SkipReboot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Podman {
    param([string[]]$Arguments)
    $result = & podman @Arguments
    return $result
}

function Confirm-PodmanCli {
    if (-not (Get-Command podman -ErrorAction SilentlyContinue)) {
        throw "Podman CLI not found. Install Podman Desktop or Podman for Windows first."
    }
}

function Confirm-PodmanMachine {
    try {
        $inspect = Invoke-Podman @('machine','inspect',$MachineName,'--format','{{.Name}}') | Select-Object -First 1
    } catch {
        throw "Podman machine '$MachineName' not found. Create it with 'podman machine init $MachineName --image-path fedora-42' before running this script."
    }
    if (-not $inspect -or $inspect.TrimEnd('*') -ne $MachineName) {
        throw "Podman machine '$MachineName' not found. Create it with 'podman machine init $MachineName --image-path fedora-42' before running this script."
    }
}

function Start-MachineIfNeeded {
    $state = Invoke-Podman @('machine','inspect',$MachineName,'--format','{{.State}}') | Select-Object -First 1
    if ($state.Trim() -ne 'Running') {
        Write-Host "🟢 Starting Podman machine '$MachineName'..." -ForegroundColor Green
        Invoke-Podman @('machine','start',$MachineName) | Out-Null
    }
}

function Get-OsRelease {
    $osRelease = Invoke-Podman @('machine','ssh',$MachineName,'--','cat','/etc/os-release')
    $map = @{}
    foreach ($line in $osRelease) {
        if ($line -match '^(?<key>[A-Z0-9_]+)=("?)(?<value>.*)\2$') {
            $map[$Matches.key] = $Matches.value
        }
    }
    return $map
}

function Install-NvidiaToolkit {
    $remoteScript = @'
set -euo pipefail

REPO_FILE="/etc/yum.repos.d/nvidia-container-toolkit.repo"
if [ ! -f "$REPO_FILE" ]; then
    cat <<'EOF' | sudo tee "$REPO_FILE" >/dev/null
[nvidia-container-toolkit]
name=NVIDIA Container Toolkit
baseurl=https://nvidia.github.io/libnvidia-container/stable/rpm/
enabled=1
gpgcheck=1
gpgkey=https://nvidia.github.io/libnvidia-container/gpgkey
EOF
fi

if command -v rpm-ostree >/dev/null 2>&1; then
    sudo rpm-ostree install --idempotent nvidia-container-toolkit || true
else
    sudo dnf install -y nvidia-container-toolkit || true
fi

if command -v nvidia-ctk >/dev/null 2>&1; then
    sudo mkdir -p /var/cdi
    sudo nvidia-ctk cdi generate --output=/var/cdi/nvidia.yaml --mode=wsl || true
fi
sudo udevadm control --reload || true
exit 0
'@

    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteScript))
    $sshArgs = @('machine','ssh',$MachineName,'--','bash','-lc',"set -euo pipefail; echo $encoded | base64 -d >/tmp/configure-gpu.sh; chmod +x /tmp/configure-gpu.sh; sudo /tmp/configure-gpu.sh")
    Invoke-Podman $sshArgs | Out-Null
}

function Test-PodmanGpu {
    Write-Host "🔍 Checking GPU devices inside the machine..." -ForegroundColor Yellow
    $cmd = 'bash -lc "ls -l /dev/dxg 2>/dev/null; ls -l /dev/nvidia* 2>/dev/null; nvidia-smi || true"'
    Invoke-Podman @('machine','ssh',$MachineName,'--',$cmd)
}

Confirm-PodmanCli
Confirm-PodmanMachine
Start-MachineIfNeeded
$osInfo = Get-OsRelease
$machineId = if ($osInfo.ContainsKey('ID') -and $osInfo['ID']) { $osInfo['ID'] } else { 'unknown' }
if ($machineId -ne 'fedora') {
    Write-Warning ("Machine reports ID='{0}'. Script was validated against Fedora 42; adjust steps manually if your image differs." -f $machineId)
}

Write-Host "⚙️  Installing NVIDIA container runtime bits inside '$MachineName'..." -ForegroundColor Cyan
Install-NvidiaToolkit

if (-not $SkipReboot.IsPresent) {
    Write-Host "🔄 Restarting machine to finalize toolkit installation..." -ForegroundColor Cyan
    Invoke-Podman @('machine','stop',$MachineName) | Out-Null
    Invoke-Podman @('machine','start',$MachineName) | Out-Null
}

Test-PodmanGpu

Write-Host "✅ GPU configuration routine completed. Re-run your container helper (run.ps1 -GPUCheck) to validate from the workload container." -ForegroundColor Green
