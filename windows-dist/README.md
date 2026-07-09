# vLLM Windows ROCm Distribution

Pre-built `_C.pyd` binary + installers for vLLM on Windows with AMD ROCm.

## Contents

| File | Size | Purpose |
|------|------|---------|
| `_C.pyd` | 13.8 MB | Pre-built C++ extension (no compilation required) |
| `install.ps1` | — | PowerShell installer (auto-detects ROCm + vLLM path) |
| `setup.bat` | — | Batch installer (simpler, for cmd.exe users) |
| `build-harness/` | — | Source files to rebuild `_C.pyd` |
| `INSTALL.md` | — | Full installation guide |
| `README.md` | — | This file |

## How to Distribute

Users need to:

1. **Install prerequisites** — Python 3.12, PyTorch+ROCm, git
2. **Clone vLLM Windows port** — `git clone https://github.com/Maxritz/vllm-windows.git`
3. **Run installer** — from the cloned repo or extracted zip:
   ```
   setup.bat
   ```

The installer copies `_C.pyd` into place, sets up `sitecustomize.py`, and verifies PyTorch.

## What's NOT included

- Python interpreter — download from python.org
- PyTorch with ROCm — install with:
  ```
  pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/
  ```
- vLLM Python source — clone from:
  ```
  git clone https://github.com/Maxritz/vllm-windows.git
  ```
- ROCm SDK — download from https://rocm.docs.amd.com

The installer only handles what must be pre-compiled: the `_C.pyd` C++ extension.
