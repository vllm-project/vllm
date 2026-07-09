# vLLM Windows ROCm — Manual Install

This guide covers installing from the distribution zip (`vllm-windows-rocm-dist.zip`).

## Requirements

- **OS:** Windows 11 23H2+
- **GPU:** AMD Radeon RX 9000 series (RDNA4, gfx120X) or RX 7000 series (RDNA3, gfx110X)
- **VRAM:** 8 GB minimum, 16 GB+ recommended
- **RAM:** 16 GB+ (64 GB recommended for large models)
- **Disk:** 10 GB free for ROCm SDK + models

## Step 1 — Install ROCm 7.13

Download and install ROCm for Windows from:
https://rocm.docs.amd.com

The SDK typically installs to `C:\Program Files\AMD\ROCm\7.13\` or similar.

ROCm pip wheels are at:
```
https://repo.amd.com/rocm/whl/gfx120X-all/
```

Verify installation:
```powershell
# Find hipcc
Get-ChildItem -Path "$env:ProgramFiles\AMD\ROCm\*\bin\hipcc.exe" -Recurse

# Test HIP compiler
& "C:\Program Files\AMD\ROCm\7.13\bin\hipcc.exe" --version
```

## Step 2 — Install Python & Dependencies

```powershell
# Install uv (fast package manager)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create venv
uv venv --python 3.12
.venv\Scripts\activate

# Install PyTorch with ROCm support
uv pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/

# Install build tools
uv pip install ninja
```

Note: If the AMD repo doesn't have the right wheel, fall back to:
```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.13
```

## Step 3 — Clone vLLM Windows Port

```powershell
git clone https://github.com/Maxritz/vllm-windows.git
cd vllm-windows
git checkout WINDOWS-PORT
```

## Step 4 — Install vLLM (Python)

```powershell
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

This installs vLLM's Python code without building C extensions.

## Step 5 — Install Pre-built `_C.pyd`

### Option A: From the distribution zip (recommended)

```powershell
# Extract the zip
Expand-Archive -Path vllm-windows-rocm-dist.zip -DestinationPath . -Force

# Copy the binary to vLLM
Copy-Item -LiteralPath "_C.pyd" -Destination "vllm\_C.pyd" -Force
```

### Option B: Run the installer script

```powershell
# Auto-detects ROCm and vLLM paths
.\install.ps1

# Or specify paths explicitly
.\install.ps1 -VllmDir "C:\Users\me\vllm-windows" -HipPath "C:\Program Files\AMD\ROCm\7.13"
```

### Option C: Build from source

```powershell
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\7.13"
cd build-harness
python build_c_win.py
# Output: _C.pyd — copy to vllm/
Copy-Item _C.pyd ..\vllm\ -Force
```

## Step 6 — Create sitecustomize.py

Create `Lib\site-packages\sitecustomize.py` in your venv:

```python
import os
os.environ.setdefault("HIP_PATH", r"C:\Program Files\AMD\ROCm\7.13")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "true")
```

This ensures `HIP_PATH` is set before any torch import.

## Step 7 — Get a Model

Download a compatible model — Qwen2.5-3B-Instruct works well for 16 GB VRAM:

```powershell
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-3B-Instruct', local_dir=r'F:\VLLM-Models\Qwen2.5-3B-Instruct')"
```

## Step 8 — Run

```powershell
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\7.13"
$env:PYTHONPATH = "C:\Users\me\vllm-windows"

python -m vllm.entrypoints.openai.api_server `
    --model F:\VLLM-Models\Qwen2.5-3B-Instruct `
    --port 8001 `
    --enforce-eager `
    --dtype float16
```

Open http://localhost:8001/ for the chat UI.

## Distribution Contents

| File | Size | Purpose |
|------|------|---------|
| `_C.pyd` | 13.8 MB | Pre-built C++ extension (21 source files) |
| `build-harness/` | — | Source to rebuild _C.pyd |
| `install.ps1` | — | Auto-installer (detects ROCm, vLLM, venv) |
| `INSTALL-WINDOWS-ROCM.md` | — | Detailed install guide |
| `INSTALL.md` | — | This file |
| `README.md` | — | Distribution overview |

## Verification

```powershell
python -c "import vllm._C; import torch; print('OK, ops:', [x for x in dir(torch.ops._C) if not x.startswith('_')])"
```

Expected output includes: `silu_and_mul`, `rms_norm`, `gptq_gemm`, `static_scaled_fp8_quant`, etc.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `HIP_PATH not set` | Create sitecustomize.py (Step 6) |
| `No module named vllm._C` | Copy _C.pyd to vllm/ directory |
| `hipErrorOutOfMemory` | Reduce `--max-model-len` or use a smaller model |
| `aten::new_zeros failed` | GPU OOM — lower `--gpu-memory-utilization` |
| `hipblaslt error` | Usually OOM — reduce context length |
| 7B model OOM | Use Qwen2.5-3B (~24 tok/s) instead — 16 GB limit |
