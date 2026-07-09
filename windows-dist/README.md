# vLLM Windows ROCm Distribution

Pre-built `_C.pyd` (13.8 MB) — the only part of vLLM that requires C++ compilation.
Drop it in and skip the 80s build + Visual Studio + ROCm toolchain.

## What's in the Box

| File | Size | What It Does |
|------|------|------|
| `_C.pyd` | 13.8 MB | vLLM C++ extension — 21 source files, pre-compiled for ROCm 7.13 |
| `setup.bat` | — | One-shot installer for `cmd.exe` |
| `install.ps1` | — | One-shot installer for PowerShell |
| `build-harness/` | — | Source + scripts to rebuild `_C.pyd` yourself |

## Prerequisites

| Requirement | Version | Install |
|-------------|---------|---------|
| **Windows** | 11 23H2+ | — |
| **AMD GPU** | RDNA3+ (RX 7000/9000) | — |
| **VRAM** | 8 GB min, 16 GB rec | — |
| **Python** | 3.12 | `winget install Python.Python.3.12` |
| **PyTorch + ROCm** | torch 2.11+rocm7.13 | `pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/` |
| **Git** | latest | `winget install Git.Git` |
| **HIP_PATH** | set to ROCm dir | Auto-configured by installer |

> The AMD repo `https://repo.amd.com/rocm/whl/gfx120X-all/` provides both PyTorch wheels AND ROCm runtime packages in one `pip install`.

## Quick Start

```powershell
# 1. Install prerequisites
winget install Python.Python.3.12 Git.Git

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install PyTorch with ROCm (inside the venv)
pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/

# 4. Get vLLM Windows port
git clone https://github.com/Maxritz/vllm-windows.git
cd vllm-windows
git checkout WINDOWS-PORT

# 5. Install vLLM Python source in the venv
pip install -e .

# 6. Extract this zip into the repo, then run the installer
setup.bat

```

## How to Run

After `setup.bat` finishes, the installer created `sitecustomize.py` so `HIP_PATH` is already set. Just:

```powershell
# 1. Activate your virtual environment
.venv\Scripts\activate

# 2. Start the server
python -m vllm.entrypoints.openai.api_server --model <path-to-your-model> --enforce-eager --dtype float16 --port 8001
```

Then open **http://localhost:8001/** in your browser.

**Don't have a model yet?** Download one that works great on 16 GB VRAM:

```powershell
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-3B-Instruct', local_dir=r'F:\VLLM-Models\Qwen2.5-3B-Instruct')"
```

Then run the server with `--model F:\VLLM-Models\Qwen2.5-3B-Instruct`.

That's it. Two commands. Chat UI at localhost:8001.

## How It Works

1. User installs prerequisites (Python, PyTorch+ROCm, git)
2. User clones vLLM source from GitHub
3. User runs `setup.bat` — copies `_C.pyd` into `vllm/`, creates `sitecustomize.py`

The installer does not download anything — it only places the pre-built binary.

## Why the `_C.pyd` Binary?

Building `_C.pyd` from source requires:
- Visual Studio 2022 (Desktop C++ workload, ~6 GB)
- ROCm 7.13 SDK (~3 GB SDK install)
- ninja build system
- 80+ seconds compile time
- Correct source paths and build flags (21 files)

This distribution eliminates all of that — just drop in the binary.

## Files

| File | Size | What It Does |
|------|------|------|
| `_C.pyd` | 13.8 MB | Pre-built C++ extension — 21 source files, compiled for ROCm 7.13 |
| `setup.bat` | — | One-shot installer for `cmd.exe` |
| `install.ps1` | — | One-shot installer for PowerShell |
| `build-harness/` | — | Source + scripts to rebuild `_C.pyd` yourself |

## What Comes From the Internet (Not in This Zip)

| Thing | Where |
|-------|-------|
| Python 3.12 | python.org / winget |
| PyTorch + ROCm | `repo.amd.com/rocm/whl/gfx120X-all/` |
| vLLM Python source | `github.com/Maxritz/vllm-windows.git` |

## Upstream PR

https://github.com/vllm-project/vllm/pull/48139

## License

Apache 2.0
