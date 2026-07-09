# vLLM Windows ROCm Distribution

Pre-built binaries and build tools for vLLM on Windows with AMD ROCm.

## Contents

| File | Size | Description |
|------|------|-------------|
| `_C.pyd` | 13.8 MB | Pre-built vLLM C++ extension (21 source files, 80s build time) |
| `build-harness\build_c_win.py` | — | Build script for `_C.pyd` |
| `build-harness\win_c_bindings_shim.cpp` | — | Custom bindings shim (stable ABI) |
| `build-harness\win_c_bindings.cu` | — | Alternative CUDA bindings |
| `build-harness\win_rocm_bindings.cu` | — | ROCm-specific bindings |
| `install.ps1` | — | One-click install script |
| `INSTALL-WINDOWS-ROCM.md` | — | Full installation guide |

## Files

| File | Purpose |
|------|---------|
| `_C.pyd` | Pre-built vLLM C++ extension (13.8 MB) |
| `install.ps1` | Auto-installer — detects ROCm, vLLM, and venv paths |
| `INSTALL.md` | Manual installation guide |
| `INSTALL-WINDOWS-ROCM.md` | Full step-by-step guide |
| `PR_DESCRIPTION.md` | Ready-to-copy PR description for upstream |
| `README.md` | This file |
| `build-harness/` | Source to rebuild `_C.pyd` |
| `vllm-windows-rocm-dist.zip` | Complete distribution archive (2.9 MB) |

## Quick Start

```powershell
# 1. Install ROCm 7.13.0
#    Download from https://rocm.docs.amd.com
#    PIP wheels: https://repo.amd.com/rocm/whl/gfx120X-all/

# 2. Clone vLLM Windows port
git clone https://github.com/Maxritz/vllm-windows.git
cd vllm-windows
git checkout WINDOWS-PORT

# 3. Set up venv
uv venv --python 3.12
.venv\Scripts\activate
uv pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# 4. Install pre-built _C.pyd
.\windows-dist\install.ps1

# 5. Run a model
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\7.13"
$env:PYTHONPATH = "C:\path\to\vllm-windows"
python -m vllm.entrypoints.openai.api_server --model <your-model-path> --enforce-eager
```

## Build From Source

If the pre-built `_C.pyd` doesn't work with your ROCm version:

```powershell
$env:HIP_PATH = "E:\ROCM-7.13.0-Windows"
cd dist\build-harness
python build_c_win.py
copy _C.pyd ..\..\vllm\
```

## Ops Included

The `_C.pyd` registers these ops under `torch.ops._C`:

- `silu_and_mul`, `rms_norm`, `fused_add_rms_norm`, `rotary_embedding`
- `gptq_gemm`, `gptq_shuffle` (GPTQ 4-bit inference)
- `static_scaled_fp8_quant`, `dynamic_scaled_fp8_quant`
- `static_scaled_int8_quant`, `dynamic_scaled_int8_quant`
- `rms_norm_dynamic_per_token_quant`, `permute_cols`
- Cache ops: `swap_blocks`, `reshape_and_cache`
- Custom all-reduce, cuda_utils, sampler ops

## Benchmark (RX 9070 XT, 16 GB VRAM)

| Model | Tokens/s | VRAM |
|-------|----------|------|
| Qwen2.5-3B-Instruct | ~24 | 5.8 GiB |
| Qwen3-1.7B-Coder | ~31 | 3.8 GiB |
| OPT-125M | ~90 | <1 GiB |

## Verified GPUs

- AMD Radeon RX 9070 XT (gfx1201, RDNA4) — ROCm 7.13.0

## License

Apache 2.0 (same as upstream vLLM)
