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

### Option A: API Server (with Chat UI)

```powershell
# Set ROCm path (or let sitecustomize.py do it)
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\7.13"

# Start the server
python -m vllm.entrypoints.openai.api_server `
    --model F:\VLLM-Models\Qwen2.5-3B-Instruct `
    --port 8001 `
    --enforce-eager `
    --dtype float16
```

Then open **http://localhost:8001/** for the dark-theme chat UI, or use the OpenAI API at `http://localhost:8001/v1`.

### Option B: Python Script

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="F:/VLLM-Models/Qwen2.5-3B-Instruct",
    dtype="float16",
    enforce_eager=True,
    max_model_len=4096,
)

sp = SamplingParams(temperature=0, max_tokens=64)
output = llm.generate(["What is the capital of France?"], sp)
print(output[0].outputs[0].text)
```

### Option C: Benchmark

```powershell
python scripts/benchmark_model.py --model F:\VLLM-Models\Qwen2.5-3B-Instruct --num-prompts 10 --max-tokens 64
```

### Expected Performance (RX 9070 XT, 16 GB VRAM)

| Model | Tokens/s | VRAM |
|-------|----------|------|
| OPT-125M | ~90 | <1 GB |
| Qwen2.5-3B-Instruct | ~24 | 5.8 GB |

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
