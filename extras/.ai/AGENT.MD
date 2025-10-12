# Agent quickstart: vLLM Development with GPU + Patches

This document is the operator-facing playbook for the complete vLLM development
workflow: GPU-capable Podman backend on Windows WSL2 + Rocky Linux 10, patch
management for CUDA/vLLM compatibility, and editable container development.

**Context**: This fork maintains patches for recent CUDA (13.x) compatibility,
experimental vLLM features, and development tooling. The challenge is managing
patches that must be applied at different build stages:
- **Pre-CUDA compile**: Core compatibility (e.g., `cumem.py` for CUDA 13)
- **Post-CUDA, pre-vLLM compile**: CUDA toolkit integration patches
- **Post-vLLM compile**: Runtime/experimental patches (future)

It complements the detailed Patch Workflow doc in `extras/docs/patch-workflow.md`.

---

## TL;DR

1) Provision the Rocky 10 WSL distro and enable GPU + Podman

```powershell
pwsh extras/tools/enable-podman-wsl-gpu.ps1 -MachineName podman-machine-default -CacheRoot C:\vllm-cache
```

2) (Optional) Create a Windows podman remote context that targets the WSL podman socket over SSH

```powershell
pwsh extras/tools/enable-podman-wsl-gpu.ps1 -MachineName podman-machine-default -CreatePodmanContext -PodmanContextName wsl-podman-default -SkipReboot
podman context use wsl-podman-default
podman info
```

3) Validate GPU availability from inside the distro

```powershell
wsl -d podman-machine-default -- bash -lc "ls -l /dev/dxg; nvidia-smi || true"
```

---

## What the helper does

- Downloads the Rocky Linux 10 UBI container rootfs (or uses `-ImagePath`) and imports it into WSL2
- Enables systemd via `/etc/wsl.conf` and gracefully restarts the distro
- Installs base packages: `podman`, `openssh-server`, `fuse-overlayfs`, `slirp4netns`, `iptables`
- Creates a default `podman` user for rootless Podman; enables `podman.socket` in the user session
- Installs NVIDIA Container Toolkit and generates a CDI spec for WSL GPUs
- (Optional) Sets up a Windows `podman` context that connects to the WSL user socket over SSH

**Note**: Some warnings are expected (systemd not running during initial setup, 
nvidia-container-toolkit package resolution issues) but the final GPU test should succeed.

---

## Common options

- `-MachineName <name>`: WSL distro name (default: `podman-machine-default`)
- `-ImagePath <file-or-url>`: Override the Rocky UBI image; `.tar.xz` or `.tar` supported
- `-CacheRoot <dir>`: Cache downloads/import data outside of `%LOCALAPPDATA%`
- `-SkipReboot`: Skip the WSL restarts (you can manually `wsl --terminate <name>` later)
- `-Reset`: Unregister the distro before importing
- `-CreatePodmanContext`: Create a Windows `podman` context (defaults: user `podman`, context `wsl-<name>`) 

---

## Troubleshooting

**WSL/Import Issues**:
- Import fails with `.tar.xz` not supported
  - Decompress to `.tar` and re-run: `tar -xf Rocky-10-Container-UBI.latest.x86_64.tar.xz`
- `nvidia-smi` missing
  - Ensure latest NVIDIA Windows driver with WSL support; re-run the helper to regenerate CDI
- Systemd not active after restart
  - `wsl --terminate <name>` and open a new shell to re-enter the distro; check `/etc/wsl.conf`

**Container/Build Issues**:
- "Image missing. Use --build." 
  - Run `extras/podman/run.ps1 --build` to build the dev container first
- Patch application failures
  - Check `extras/patches/python-overrides.txt` for syntax errors
  - Verify source files haven't moved upstream (sync issue)
- CUDA version mismatches
  - Pre-CUDA patches (like `cumem.py`) should be in the overlay system
  - Check container build logs for compilation errors

**Podman Context Issues**:
- `podman` cannot connect from Windows
  - Re-run with `-CreatePodmanContext` or manually `podman system connection ls`
- SSH connection refused
  - Check if sshd is running: `wsl -d <name> -- systemctl status sshd`

---

## Dev loop after provisioning

The complete development cycle with patch management:

1. **Sync upstream changes** (preserves `extras/` and patches):
   ```powershell
   pwsh extras/tools/fork-sync/sync_with_upstream.ps1
   ```

2. **Build container with pre-build patches**:
   ```powershell
   # The entrypoint applies patches from extras/patches/python-overrides.txt
   # before CUDA compilation (e.g., cumem.py for CUDA 13 compatibility)
   extras/podman/run.ps1 --build
   ```

3. **Enter development mode**:
   ```powershell
   # Interactive shell with editable vLLM install
   extras/podman/run.ps1 -Interactive
   
   # Or run with GPU validation
   extras/podman/run.ps1 -GPUCheck
   ```

4. **Inside container - test your changes**:
   ```bash
   # The workspace is mounted as /workspace, vLLM is editable-installed
   cd /workspace
   python -c "import vllm; print(vllm.__version__)"
   
   # Test GPU + recent CUDA
   python -c "import torch; print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name()}')"
   ```

**Patch Management Notes**:
- Pre-CUDA patches: Add to `extras/patches/python-overrides.txt` 
- Post-CUDA patches: Can be applied manually in interactive mode or via future automation
- Repository stays clean: patches are applied to overlay directories, not the working tree

---

## Clean-up / reset

```powershell
# Remove the Windows podman context (optional)
podman system connection rm wsl-podman-default -f

# Unregister the WSL distro
pwsh extras/tools/enable-podman-wsl-gpu.ps1 -MachineName podman-machine-default -Reset
```

---

## Related docs

- **Patch workflow and container overlays**: `extras/docs/patch-workflow.md`
- **Fork sync helpers**: `extras/tools/fork-sync/README.md`
- **Current patches**: `extras/patches/python-overrides.txt` (pre-CUDA compile patches)
- **Build automation**: `extras/ci/build.yaml`, `extras/configs/build.env`

## Quick reference

**Full cycle from scratch**:
```powershell
# 1. Provision WSL2 + GPU
pwsh extras/tools/enable-podman-wsl-gpu.ps1 -MachineName podman-machine-default -CacheRoot C:\vllm-cache

# 2. Sync upstream (if needed)  
pwsh extras/tools/fork-sync/sync_with_upstream.ps1

# 3. Build with patches
extras/podman/run.ps1 --build

# 4. Develop
extras/podman/run.ps1 -Interactive
```

**Reset everything**:
```powershell
# Remove WSL distro
pwsh extras/tools/enable-podman-wsl-gpu.ps1 -MachineName podman-machine-default -Reset

# Remove Windows podman context (if created)
podman system connection rm wsl-podman-default -f
```
