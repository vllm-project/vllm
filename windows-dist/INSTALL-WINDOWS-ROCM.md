# vLLM Windows + AMD ROCm — Quick Install

## Prerequisites

| Component | Version | Download |
|-----------|---------|----------|
| Python | 3.12 | python.org |
| ROCm for Windows | 7.13.0 | [AMD ROCm](https://rocm.docs.amd.com) |
| Visual Studio 2022 | 17.10+ | visualstudio.microsoft.com (Desktop C++ workload) |
| Git | latest | git-scm.com |

## 1. Install ROCm 7.13.0

Install to `E:\ROCM-7.13.0-Windows` (or adjust paths below).

Verify HIP compiler works:
```powershell
& "E:\ROCM-7.13.0-Windows\bin\hipcc.exe" --version
```

## 2. Set Up Python Environment

```powershell
# Install uv (fast Python package manager)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create and activate venv
uv venv --python 3.12
.venv\Scripts\activate

# Install PyTorch with ROCm support
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.13

# Install vLLM with pre-built extensions
uv pip install ninja
```

## 3. Create sitecustomize.py

Create `Lib\site-packages\sitecustomize.py` in your venv:

```python
import os
os.environ.setdefault("HIP_PATH", r"E:\ROCM-7.13.0-Windows")
```

## 4. Install vLLM Windows Port

```powershell
# Clone the Windows port
git clone https://github.com/Maxritz/vllm-windows.git
cd vllm-windows
git checkout WINDOWS-PORT

# Install vLLM (Python-only, no C extension build)
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

## 5. Install Pre-built `_C.pyd`

Copy the pre-built extension:

```powershell
# From the dist/ folder in this repo
copy dist\_C.pyd vllm\vllm\_C.pyd

# OR build from source (takes ~80s):
cd dist\build-harness
$env:HIP_PATH = "E:\ROCM-7.13.0-Windows"
python build_c_win.py
```

## 6. Verify Installation

```powershell
$env:HIP_PATH = "E:\ROCM-7.13.0-Windows"
$env:PYTHONPATH = "C:\path\to\vllm"

python -c "import vllm._C; import torch; print('_C ops:', [x for x in dir(torch.ops._C) if not x.startswith('_')])"
```

Expected output:
```
_C ops: ['gptq_gemm', 'gptq_shuffle', 'rms_norm', 'silu_and_mul', 'static_scaled_fp8_quant', ...]
```

## 7. Run a Model

```powershell
$env:HIP_PATH = "E:\ROCM-7.13.0-Windows"

python -m vllm.entrypoints.openai.api_server `
    --model F:\VLLM-Models\Qwen2.5-3B-Instruct `
    --port 8001 `
    --enforce-eager `
    --dtype float16
```

Then open http://localhost:8001/ for the chat UI.

## Environment Variables (Required)

| Variable | Value | Why |
|----------|-------|-----|
| `HIP_PATH` | `E:\ROCM-7.13.0-Windows` | ROCm SDK root |
| `PYTHONPATH` | `path\to\vllm` | vLLM source root |
| `VLLM_NO_USAGE_STATS` | `true` | Optional |

## Troubleshooting

**"HIP_PATH not set"**: Create `sitecustomize.py` as in step 3.

**"No module named vllm._C"**: Copy `_C.pyd` to `vllm/` directory.

**"aten::new_zeros failed"**: GPU out of memory. Reduce `--max-model-len` or use a smaller model.

**"hipblaslt error"**: Usually an OOM symptom. Lower `--gpu-memory-utilization`.

**Model too large**: Qwen2.5-3B-Instruct (~24 tok/s) is the sweet spot for 16 GB VRAM. 7B models OOM.
