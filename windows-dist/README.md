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

## Quick Start

```powershell
# 1. Install ROCm 7.13.0, Python 3.12, and clone vllm-windows
git clone https://github.com/Maxritz/vllm-windows.git
cd vllm-windows
git checkout WINDOWS-PORT

# 2. Set up venv and install vLLM
uv venv --python 3.12
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.13
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# 3. Run installer (from this dist/ folder)
.\dist\install.ps1 -VllmDir . -HipPath "E:\ROCM-7.13.0-Windows"

# 4. Run a model
$env:HIP_PATH = "E:\ROCM-7.13.0-Windows"
python -m vllm.entrypoints.openai.api_server --model F:\VLLM-Models\Qwen2.5-3B-Instruct --enforce-eager
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
