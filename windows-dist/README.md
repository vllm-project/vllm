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

## How It Works

1. User installs **Python 3.12**, **PyTorch with ROCm**, and **git** from the internet
2. User clones `https://github.com/Maxritz/vllm-windows.git`
3. User runs `setup.bat` — it copies `_C.pyd` into place and configures paths

That's it. The installer does not download anything — it only places the pre-built binary.

## Why This Exists

Building `_C.pyd` from source requires:
- Visual Studio 2022 (Desktop C++ workload, ~6 GB)
- ROCm 7.13 SDK (~3 GB)
- ninja build system
- 80+ seconds compile time
- 21 `.cu`/`.cpp` source files with correct paths and flags

This distribution eliminates all of that.

## What the User Still Gets From the Internet

| Thing | Where From |
|-------|-----------|
| Python 3.12 | python.org or `winget install Python.Python.3.12` |
| PyTorch + ROCm | `pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/` — this repo has both PyTorch wheels AND ROCm runtime packages |
| vLLM Python source | `git clone https://github.com/Maxritz/vllm-windows.git` |

## Quick Start

```powershell
# 1. Prerequisites
winget install Python.Python.3.12
pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/

# 2. Get vLLM
git clone https://github.com/Maxritz/vllm-windows.git
cd vllm-windows
git checkout WINDOWS-PORT

# 3. Install the pre-built binary
#    Extract vllm-windows-rocm-dist.zip into the repo, then:
setup.bat

# 4. Run
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\7.13"
python -m vllm.entrypoints.openai.api_server --model F:\VLLM-Models\Qwen2.5-3B-Instruct --enforce-eager
```

## Upstream PR

https://github.com/vllm-project/vllm/pull/48139

## License

Apache 2.0
