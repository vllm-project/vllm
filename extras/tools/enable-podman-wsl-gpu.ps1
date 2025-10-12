#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Provisions a Rocky Linux 10 WSL2 distribution and configures Podman with NVIDIA GPU passthrough.

.DESCRIPTION
        Installs and configures inside the imported WSL distro:
            - systemd (via /etc/wsl.conf)
            - OpenSSH server (for optional podman-remote over SSH)
            - Podman (+ rootless user socket)
            - NVIDIA Container Toolkit and CDI spec for WSL2 GPUs
        Then verifies device/node availability. This script is idempotent and safe to run multiple times.

.PARAMETER MachineName
    Name of the WSL distribution to create/import. Defaults to "podman-machine-default" for continuity.

.PARAMETER SkipReboot
    Prevents the script from automatically restarting the Podman machine after configuration.

.PARAMETER ImagePath
    Optional override for the Rocky Linux 10 container rootfs. Defaults to the Rocky UBI x86_64 image.

.EXAMPLE
    pwsh extras/tools/enable-podman-wsl-gpu.ps1

.NOTES
    Requires WSL2 and an NVIDIA driver on Windows that exposes the CUDA WSL integration.
    Podman for Windows is optional; if present, a podman-remote context can be created to use
    the WSL instance via SSH.
#>

[CmdletBinding()]
param(
    [string]$MachineName = "podman-machine-default",
    [switch]$SkipReboot,
    [switch]$Reset,
    [string]$ImagePath = "https://dl.rockylinux.org/pub/rocky/10/images/x86_64/Rocky-10-Container-UBI.latest.x86_64.tar.xz",
    [switch]$ConvertImage,
    [string]$CacheRoot,

    # Optional: create a Podman remote context on Windows via SSH to the WSL distro
    [switch]$CreatePodmanContext,
    [string]$PodmanSshUser = "podman",
    [string]$PodmanContextName,

    # Deprecated/ignored (kept for compatibility)
    [switch]$Rootful,
    [switch]$AllowSparseUnsafe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$script:PodmanImageCacheRoot = $null

function Wait-FileUnlocked {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Path,
        [int]$TimeoutSeconds = 30
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,[System.IO.FileShare]::None)
            try { return } finally { $stream.Dispose() }
        } catch [System.IO.IOException] {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "Timed out waiting for exclusive access to '$Path'."
}

function Assert-Administrator {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($currentIdentity)
    $isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        throw "This option requires an elevated PowerShell session. Re-run in an 'Administrator: PowerShell' window."
    }
}

function Test-IsAdministrator {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($currentIdentity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-Podman {
    param([string[]]$Arguments)
    $result = & podman @Arguments
    return $result
}

function Invoke-WSL {
    param(
        [Parameter(Mandatory=$true)][string]$Distro,
        [Parameter(Mandatory=$true)][string]$Command
    )
    & wsl.exe -d $Distro -- bash -lc $Command
}

function Test-PodmanMachine {
    try {
        $name = Invoke-Podman @('machine','inspect',$MachineName,'--format','{{.Name}}') | Select-Object -First 1
        if ($name -and $name.TrimEnd('*') -eq $MachineName) {
            return $true
        }
    } catch {
    }
    return $false
}

function Confirm-PodmanCli {
    if (-not (Get-Command podman -ErrorAction SilentlyContinue)) {
        Write-Warning "Podman CLI not found. Skipping Windows context configuration."
    }
}

function Test-WslDistroExists {
    param([string]$Name)
    $distros = & wsl.exe -l -q 2>$null | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    return $distros -contains $Name
}

function Import-RockyDistro {
    if (Test-WslDistroExists -Name $MachineName) {
        Write-Host "ℹ️  WSL distro '$MachineName' already exists." -ForegroundColor DarkGray
        return
    }

    $resolvedImage = Resolve-ImagePath -ImageSpec $ImagePath
    if ($resolvedImage.EndsWith('.tar.xz',[StringComparison]::OrdinalIgnoreCase)) {
        # Use as-is; newer WSL can import .tar.xz directly
        $importImage = $resolvedImage
    } else {
        $importImage = $resolvedImage
    }

    if (-not $script:PodmanImageCacheRoot) {
        $script:PodmanImageCacheRoot = Get-PodmanImageCacheRoot -OverrideRoot $CacheRoot
    }
    $distroDataRoot = Join-Path $script:PodmanImageCacheRoot "wsl\$MachineName"
    if (-not (Test-Path $distroDataRoot)) { New-Item -ItemType Directory -Path $distroDataRoot -Force | Out-Null }

    Write-Host "🆕 Importing Rocky Linux 10 into WSL2 as '$MachineName'..." -ForegroundColor Cyan
    $wslArgs = @('--import', $MachineName, $distroDataRoot, $importImage, '--version', '2')
    $out = & wsl.exe @wslArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "WSL import failed: $($out | Out-String)"
    }

    # Bootstrap systemd and default user
    Write-Host "🧩 Configuring /etc/wsl.conf (systemd=true)" -ForegroundColor Cyan
    Invoke-WSL -Distro $MachineName -Command "sudo bash -lc 'cat >/etc/wsl.conf <<\"EOF\"\n[boot]\nsystemd=true\n[network]\nhostname=$MachineName\nEOF'" | Out-Null

    # Ensure PATH and basic tools
    Invoke-WSL -Distro $MachineName -Command "sudo dnf -y makecache --refresh || true" | Out-Null
}

function Install-BasePackages {
    Write-Host "📦 Installing base packages (openssh-server, podman, overlayfs, slirp4netns)..." -ForegroundColor Cyan
    $installCmd = @(
        'set -euo pipefail',
        'sudo dnf -y install openssh-server podman fuse-overlayfs slirp4netns iptables xz tar procps-ng',
        # Create default user for rootless podman if missing
        "id -u $PodmanSshUser >/dev/null 2>&1 || sudo useradd -m -G wheel $PodmanSshUser",
        # Enable systemd services
        'sudo systemctl enable --now sshd || true'
    ) -join '; '
    Invoke-WSL -Distro $MachineName -Command $installCmd | Out-Null

    # Enable lingering so user services can run without login
    Invoke-WSL -Distro $MachineName -Command "sudo loginctl enable-linger $PodmanSshUser || true" | Out-Null

    # Enable user socket for podman
    $userSockCmd = @(
        "sudo -u $PodmanSshUser bash -lc 'systemctl --user enable --now podman.socket || true'"
    ) -join '; '
    Invoke-WSL -Distro $MachineName -Command $userSockCmd | Out-Null
}

function Initialize-SSHKeyAndContext {
    if (-not (Get-Command podman -ErrorAction SilentlyContinue)) { return }
    # Generate a dedicated SSH key for this context
    $sshDir = Join-Path $HOME ".ssh"
    if (-not (Test-Path $sshDir)) { New-Item -ItemType Directory -Path $sshDir -Force | Out-Null }
    $keyPath = Join-Path $sshDir ("id_ed25519_podman_" + $MachineName)
    if (-not (Test-Path $keyPath)) {
        & ssh-keygen -t ed25519 -N "" -f $keyPath -C "podman-wsl-$MachineName" | Out-Null
    }
    $pub = Get-Content ($keyPath + ".pub") -Raw
    # Install the public key into the WSL user's authorized_keys
    $installKeyCmd = @(
        "set -euo pipefail",
        "sudo -u $PodmanSshUser mkdir -p /home/$PodmanSshUser/.ssh",
        "echo '$($pub.Replace("'","'\''"))' | sudo tee -a /home/$PodmanSshUser/.ssh/authorized_keys >/dev/null",
    "sudo chown -R ${PodmanSshUser}:${PodmanSshUser} /home/${PodmanSshUser}/.ssh",
        "sudo chmod 700 /home/$PodmanSshUser/.ssh",
        "sudo chmod 600 /home/$PodmanSshUser/.ssh/authorized_keys"
    ) -join '; '
    Invoke-WSL -Distro $MachineName -Command $installKeyCmd | Out-Null

    # Get UID to compose the podman socket path
    $uid = (& wsl.exe -d $MachineName -- bash -lc "id -u $PodmanSshUser" | Select-Object -First 1).Trim()
    if (-not $uid) { $uid = '1000' }
    $socketPath = "/run/user/$uid/podman/podman.sock"

    $ctxName = if ($PodmanContextName) { $PodmanContextName } else { "wsl-$MachineName" }
    $uri = "ssh://$PodmanSshUser@localhost$socketPath"

    # Add podman remote connection
    & podman system connection remove $ctxName -f 2>$null | Out-Null
    $addOut = & podman system connection add $ctxName $uri --identity $keyPath --default 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning ("Failed to create podman connection: {0}" -f ($addOut | Out-String))
    } else {
        Write-Host "🔗 Configured podman remote context '$ctxName' -> $uri" -ForegroundColor Green
    }
}

function Get-PodmanImageCacheRoot {
    param([string]$OverrideRoot)

    $candidates = @()
    if ($OverrideRoot) { $candidates += $OverrideRoot }
    if ($env:VLLM_PODMAN_IMAGE_CACHE) { $candidates += $env:VLLM_PODMAN_IMAGE_CACHE }

    $commonData = [Environment]::GetFolderPath([Environment+SpecialFolder]::CommonApplicationData)
    if ($commonData) {
        $candidates += (Join-Path $commonData 'vllm\podman-images')
    }

    $localData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
    if ($localData) {
        $candidates += (Join-Path $localData 'vllm-podman-images')
    }

    $candidates += (Join-Path ([IO.Path]::GetTempPath()) 'vllm-podman-images')

    foreach ($candidate in $candidates) {
        if (-not $candidate) { continue }
        try {
            $full = [IO.Path]::GetFullPath($candidate)
            if (-not (Test-Path $full)) {
                New-Item -ItemType Directory -Path $full -Force | Out-Null
            }
            return $full
        } catch {
            # try next candidate
        }
    }

    throw "Unable to determine a writable cache location. Provide -CacheRoot or set VLLM_PODMAN_IMAGE_CACHE."
}

function Convert-RockyImage {
    param([string]$ImageSpec)

    $resolved = Resolve-ImagePath -ImageSpec $ImageSpec
    if (-not $resolved) {
        throw "Unable to resolve image reference '$ImageSpec'."
    }

    # If the image is a tar.xz, decompress it to tar
    if ($resolved.EndsWith('.tar.xz',[StringComparison]::OrdinalIgnoreCase)) {
        $decompressed = [IO.Path]::ChangeExtension($resolved, '.tar')
        if (-not (Test-Path $decompressed)) {
            Write-Host "🗜️  Decompressing '$resolved' to '$decompressed'..." -ForegroundColor Yellow
            # Use built-in tar if available
            tar -xf $resolved -C (Split-Path $decompressed) --force-local
        }
        $resolved = $decompressed
    }

    if ($resolved.EndsWith('.tar',[StringComparison]::OrdinalIgnoreCase)) {
        Write-Host "ℹ️  Resolved image is a tar archive: $resolved" -ForegroundColor DarkGray
        return $resolved
    }

    throw "Unsupported image format: $resolved. Please provide a .tar.xz or .tar container rootfs."
}

function Resolve-ImagePath {
    param([string]$ImageSpec)

    if (-not $ImageSpec) {
        return $null
    }

    if ($ImageSpec -match '^[a-zA-Z][a-zA-Z0-9+.-]*://') {
        if (-not $script:PodmanImageCacheRoot) {
            $script:PodmanImageCacheRoot = Get-PodmanImageCacheRoot -OverrideRoot $CacheRoot
        }
        $cacheRoot = $script:PodmanImageCacheRoot
        $leaf = Split-Path $ImageSpec -Leaf
        if (-not $leaf) {
            $leaf = 'podman-machine-image.qcow2'
        }
        $localPath = Join-Path $cacheRoot $leaf
        if (-not (Test-Path $localPath)) {
            Write-Host "⬇️  Downloading Podman machine image from '$ImageSpec'..." -ForegroundColor Cyan
            try {
                Invoke-WebRequest -Uri $ImageSpec -OutFile $localPath -UseBasicParsing | Out-Null
            } catch {
                if (Test-Path $localPath) { Remove-Item $localPath -Force }
                throw "Failed to download image from '$ImageSpec': $($_.Exception.Message)"
            }
        } else {
            Write-Host "ℹ️  Reusing cached machine image '$localPath'." -ForegroundColor DarkGray
        }
        $preparedCandidate = [IO.Path]::ChangeExtension($localPath,'prepared.tar')
        if (Test-Path $preparedCandidate) {
            Write-Host "ℹ️  Using prepared machine image '$preparedCandidate'." -ForegroundColor DarkGray
            return $preparedCandidate
        }
        $legacyPrepared = [IO.Path]::ChangeExtension($localPath,'prepared.vhdx')
        if (Test-Path $legacyPrepared) {
            Write-Host "ℹ️  Using legacy prepared VHDX '$legacyPrepared'." -ForegroundColor DarkGray
            return $legacyPrepared
        }
        $fixedCandidate = [IO.Path]::ChangeExtension($localPath,'fixed.vhdx')
        if (Test-Path $fixedCandidate) {
            Write-Host "ℹ️  Using previously converted VHDX '$fixedCandidate'." -ForegroundColor DarkGray
            return $fixedCandidate
        }
        return $localPath
    }

    if (-not (Test-Path $ImageSpec)) {
        throw "Image path '$ImageSpec' does not exist."
    }
    $resolved = (Resolve-Path $ImageSpec).Path
    if ($resolved.EndsWith('.wsl',[StringComparison]::OrdinalIgnoreCase) -or $resolved.EndsWith('.tar',[StringComparison]::OrdinalIgnoreCase)) {
        $preparedCandidate = [IO.Path]::ChangeExtension($resolved,'prepared.tar')
        if (Test-Path $preparedCandidate) {
            Write-Host "ℹ️  Using prepared machine image '$preparedCandidate'." -ForegroundColor DarkGray
            return $preparedCandidate
        }
        $legacyPrepared = [IO.Path]::ChangeExtension($resolved,'prepared.vhdx')
        if (Test-Path $legacyPrepared) {
            Write-Host "ℹ️  Using legacy prepared VHDX '$legacyPrepared'." -ForegroundColor DarkGray
            return $legacyPrepared
        }
        $fixedCandidate = [IO.Path]::ChangeExtension($resolved,'fixed.vhdx')
        if (Test-Path $fixedCandidate) {
            Write-Host "ℹ️  Using previously converted VHDX '$fixedCandidate'." -ForegroundColor DarkGray
            return $fixedCandidate
        }
    }
    return $resolved
}

function Initialize-PodmanMachine { Import-RockyDistro }

function Enable-SparseVhdSupport {
    param([string]$Distribution)
    Write-Host "🪫 Enabling sparse VHD import for '$Distribution' (allows WSL to attach pre-sparse Rocky image)..." -ForegroundColor Yellow
    $attempts = @(
        @('--manage',$Distribution,'--set-sparse','--allow-unsafe'),
        @('--manage',$Distribution,'--set-sparse','--allow-unsafe','true'),
        @('--manage',$Distribution,'--set-sparse','true','--allow-unsafe'),
        @('--manage',$Distribution,'--set-sparse','true','--allow-unsafe','true'),
        @('--manage',$Distribution,'--set-sparse','true','--allow-unsafe=true'),
        @('--manage',$Distribution,'--set-sparse=true','--allow-unsafe'),
        @('--manage',$Distribution,"--set-sparse=true","--allow-unsafe=true")
    )
    $failMessages = @()
    for ($i = 0; $i -lt $attempts.Count; $i++) {
        $cmdArgs = $attempts[$i]
        $output = & wsl.exe @cmdArgs 2>&1
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            if ($i -gt 0) {
                Write-Host "ℹ️  Sparse support enabled using syntax variant: wsl.exe $($cmdArgs -join ' ')" -ForegroundColor DarkGray
            }
            return
        }
        $msg = ($output | Out-String).Trim()
        if (-not $msg) { $msg = "wsl.exe exited with code $exitCode" }
        $failMessages += "- wsl.exe $($cmdArgs -join ' '): $msg"
    }
    $joined = [string]::Join("`n",$failMessages)
    if ($joined -match 'allow-unsafe is not a valid boolean') {
        $guidance = @(
            "WSL on this host does not recognise '--allow-unsafe'.",
            "Update WSL via 'wsl.exe --update --pre-release' or convert the Rocky .wsl archive to a fixed .vhdx and rerun with -ImagePath pointing to that file.",
            "See https://docs.rockylinux.org/10/guides/interoperability/import_rocky_to_wsl/ for manual conversion steps."
        )
        $joined = "$joined`n$([string]::Join(' ', $guidance))"
    }
    throw "Failed to enable sparse VHD support for '$Distribution':`n$joined"
}

function Start-MachineIfNeeded { return }

function Reset-PodmanMachine {
    if (-not $Reset.IsPresent) {
        return
    }
    Write-Host "♻️ Resetting WSL distro '$MachineName'..." -ForegroundColor Yellow
    if (Test-WslDistroExists -Name $MachineName) {
        & wsl.exe --terminate $MachineName 2>$null | Out-Null
        & wsl.exe --unregister $MachineName 2>$null | Out-Null
    } else {
        Write-Host "ℹ️  Distro '$MachineName' already absent." -ForegroundColor DarkGray
    }
}

function Set-PodmanRootfulMode {
    if (-not $Rootful.IsPresent) {
        return
    }
    if (-not (Test-PodmanMachine)) {
        return
    }
    $rootfulState = Invoke-Podman @('machine','inspect',$MachineName,'--format','{{.Rootful}}') | Select-Object -First 1
    if ($rootfulState -and $rootfulState.Trim().ToLower() -eq 'true') {
        return
    }
    Write-Host "🔑 Enabling rootful mode for '$MachineName'..." -ForegroundColor Yellow
    Invoke-Podman @('machine','set','--rootful',$MachineName) | Out-Null
    try {
        Invoke-Podman @('machine','stop',$MachineName) | Out-Null
    } catch {}
}

function Get-OsRelease {
    $osRelease = Invoke-WSL -Distro $MachineName -Command 'cat /etc/os-release'
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
#!/usr/bin/env bash
set -euo pipefail

REPO_FILE="/etc/yum.repos.d/nvidia-container-toolkit.repo"
ARCH="$(uname -m)"
. /etc/os-release
MAJOR="${VERSION_ID%%.*}"
ID_LIKE_LOWER=$(echo "${ID_LIKE:-}" | tr '[:upper:]' '[:lower:]')
ID_LOWER=$(echo "$ID" | tr '[:upper:]' '[:lower:]')

if [ ! -f "$REPO_FILE" ]; then
    if [[ "$ID_LOWER" == "rocky" || "$ID_LOWER" == "rhel" || "$ID_LIKE_LOWER" == *"rhel"* ]]; then
        if [[ "$MAJOR" =~ ^1[0-9]$ ]]; then
            CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/rhel${MAJOR}/${ARCH}/cuda-rhel${MAJOR}.repo"
            TMP_REPO=$(mktemp)
            if curl -fsSL "$CUDA_REPO" -o "$TMP_REPO"; then
                sudo mv "$TMP_REPO" "$REPO_FILE"
            else
                rm -f "$TMP_REPO"
            fi
        elif [[ "$MAJOR" -ge 8 ]]; then
            CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/rhel${MAJOR}/${ARCH}/cuda-rhel${MAJOR}.repo"
            TMP_REPO=$(mktemp)
            if curl -fsSL "$CUDA_REPO" -o "$TMP_REPO"; then
                sudo mv "$TMP_REPO" "$REPO_FILE"
            else
                rm -f "$TMP_REPO"
            fi
        fi
    fi

    if [ ! -f "$REPO_FILE" ]; then
        cat <<'EOF' | sudo tee "$REPO_FILE" >/dev/null
[nvidia-container-toolkit]
name=NVIDIA Container Toolkit
baseurl=https://nvidia.github.io/libnvidia-container/stable/rpm
enabled=1
gpgcheck=1
gpgkey=https://nvidia.github.io/libnvidia-container/gpgkey
EOF
    fi
fi

if command -v rpm-ostree >/dev/null 2>&1; then
    sudo rpm-ostree install --idempotent nvidia-container-toolkit || true
else
    sudo dnf install -y nvidia-container-toolkit || true
fi

if command -v nvidia-ctk >/dev/null 2>&1; then
    sudo mkdir -p /etc/cdi /var/cdi
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml --mode=wsl || true
    sudo cp -f /etc/cdi/nvidia.yaml /var/cdi/nvidia.yaml || true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    true
elif [ -x /usr/lib/wsl/drivers/nvidia-smi ]; then
    sudo ln -sf /usr/lib/wsl/drivers/nvidia-smi /usr/local/bin/nvidia-smi
elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    sudo ln -sf /usr/lib/wsl/lib/nvidia-smi /usr/local/bin/nvidia-smi
fi

sudo mkdir -p /usr/lib/wsl
if [ ! -e /usr/lib/wsl/lib ] && [ -d /mnt/c/Windows/System32/nvidia-cuda ]; then
    sudo ln -sf /mnt/c/Windows/System32/nvidia-cuda /usr/lib/wsl/lib
fi
if [ ! -e /usr/lib/wsl/drivers ] && [ -d /mnt/c/Windows/System32/DriverStore/FileRepository ]; then
    sudo ln -sf /mnt/c/Windows/System32/DriverStore/FileRepository /usr/lib/wsl/drivers
fi

sudo udevadm control --reload || true
exit 0
'@

    $remoteScriptLf = $remoteScript -replace "`r", ""
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteScriptLf))
    Invoke-WSL -Distro $MachineName -Command "bash -lc 'set -euo pipefail; echo $encoded | base64 -d >/tmp/configure-gpu.sh; chmod +x /tmp/configure-gpu.sh; sudo /tmp/configure-gpu.sh'" | Out-Null
}

function Test-PodmanGpu {
    Write-Host "🔍 Checking GPU devices inside the machine..." -ForegroundColor Yellow
    $cmd = 'ls -l /dev/dxg 2>/dev/null; ls -l /dev/nvidia* 2>/dev/null; nvidia-smi || true'
    Invoke-WSL -Distro $MachineName -Command $cmd
}

$script:PodmanImageCacheRoot = Get-PodmanImageCacheRoot -OverrideRoot $CacheRoot

Confirm-PodmanCli

if ($ConvertImage) {
    Assert-Administrator
    $convertedPath = Convert-RockyImage -ImageSpec $ImagePath
    if ($convertedPath) {
        Write-Host "ℹ️  Prepared archive ready at '$convertedPath'. Re-run without -ConvertImage to provision the Podman machine." -ForegroundColor Cyan
    }
    return
}

Reset-PodmanMachine
Initialize-PodmanMachine
Install-BasePackages

# Bounce the distro to apply wsl.conf (systemd)
if (-not $SkipReboot.IsPresent) {
    Write-Host "🔄 Restarting WSL distro to finalize systemd setup..." -ForegroundColor Cyan
    & wsl.exe --terminate $MachineName 2>$null | Out-Null
    Start-Sleep -Seconds 1
}
$osInfo = Get-OsRelease
$machineId = if ($osInfo.ContainsKey('ID') -and $osInfo['ID']) { $osInfo['ID'] } elseif ($osInfo.ContainsKey('ID_LIKE') -and $osInfo['ID_LIKE']) { $osInfo['ID_LIKE'] } elseif ($osInfo.ContainsKey('PRETTY_NAME') -and $osInfo['PRETTY_NAME']) { $osInfo['PRETTY_NAME'] } else { 'unknown' }
if ($machineId -notlike 'rocky*') {
    Write-Warning ("Machine reports ID='{0}'. Script was validated against Rocky Linux 10; adjust steps manually if your image differs." -f $machineId)
}

Write-Host "⚙️  Installing NVIDIA container runtime bits inside '$MachineName'..." -ForegroundColor Cyan
Install-NvidiaToolkit

if (-not $SkipReboot.IsPresent) {
    Write-Host "🔄 Restarting WSL distro to finalize toolkit installation..." -ForegroundColor Cyan
    & wsl.exe --terminate $MachineName 2>$null | Out-Null
    Start-Sleep -Seconds 1
}

if ($CreatePodmanContext.IsPresent) {
    Initialize-SSHKeyAndContext
}

Test-PodmanGpu

Write-Host "✅ GPU configuration routine completed. Re-run your container helper (run.ps1 -GPUCheck) to validate from the workload container." -ForegroundColor Green
