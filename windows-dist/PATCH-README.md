# vLLM Windows + AMD ROCm Port

A complete port of [vLLM](https://github.com/vllm-project/vllm) to run natively on Windows with AMD ROCm GPUs (RDNA3/RDNA4). Tested on RX 9070 XT (gfx1201, RDNA4) with ROCm 7.13.0.

## Prerequisites

### Software

| Component | Version | Notes |
|-----------|---------|-------|
| Windows | 11 23H2+ | |
| ROCm for Windows | 7.13.0 | Install from `E:\ROCM-7.13.0-Windows` |
| Python | 3.12 | Use `uv` or a venv — never system Python |
| PyTorch | 2.11.0+rocm7.13.0 | `pip install torch --index-url https://download.pytorch.org/whl/rocm7.13` |
| HIP SDK | matching ROCm | bin at `E:\ROCM-7.13.0-Windows\bin\hipcc.exe` |
| ninja | latest | `pip install ninja` |
| triton-windows | custom | Fork of OpenAI Triton ported to Windows |
| CMake | 3.28+ | Required for ROCm builds |
| Visual Studio 2022 | 17.10+ | With "Desktop development with C++" |

### Environment Setup

```powershell
# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create venv
uv venv --python 3.12
.venv\Scripts\activate

# Install PyTorch with ROCm
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.13

# Set HIP_PATH (must be forced, ROCm installer may not set it globally)
$env:HIP_PATH = "E:\ROCM-7.13.0-Windows"
```

A `sitecustomize.py` in `Lib\site-packages\` is required to force `HIP_PATH` and apply stubs before any torch import:

```python
import os
os.environ.setdefault("HIP_PATH", r"E:\ROCM-7.13.0-Windows")
```

## Changes Made (39 files, +2571/-97 lines)

### 1. Platform Layer — ROCm on Windows

**`vllm/platforms/rocm.py`** — Attention backend override:
- Default ROCm attention backend `ROCM_ATTN` requires `_rocm_C.pyd` with `paged_attention` op which is not built on Windows.
- Swapped priority: `TRITON_ATTN` is tried first, falling back to `ROCM_ATTN` only if Triton unavailable.

**`vllm/platforms/__init__.py`** — Platform detection:
- Added Windows + AMD GPU detection path.
- Enables CUDA platform path when ROCm/HIP is detected on Windows.

**`vllm/platforms/cuda.py`** — CUDA platform tweaks:
- Handles Windows-specific HIP module loading.

**`vllm/platforms/rocm_dist_stubs.py`** — *New file*:
- Stubs for `torch.distributed` functions that are missing/unstable on Windows ROCm.
- Provides fallback implementations for `all_reduce`, `broadcast`, `barrier`.

### 2. Distributed / IPC Fixes

**`vllm/v1/engine/core.py`** — Process spawn fix:
- Windows uses `spawn` (not `fork`) for multiprocessing.
- Added `freeze_support()` guard and proper serialization of engine args.
- ZMQ IPC path generation handles Windows named pipes.

**`vllm/v1/engine/utils.py`** — IPC path resolution:
- `get_open_zmq_ipc_path()`: Windows uses `tcp://127.0.0.1:{port}` instead of Unix sockets.
- `kill_proc_tree()`: Windows uses `taskkill /F /T /PID` instead of `kill -9`.

**`vllm/distributed/utils.py`** — Distributed init:
- Skips NCCL init attempts on Windows (NCCL not available on Windows ROCm).
- Uses Gloo backend for CPU collectives.

**`vllm/rocm_dist_fixes.py`** — *New file*:
- Comprehensive stubs and workarounds for ROCm distributed operations.
- Monkey-patches `torch.distributed` for missing functions.
- Handles `can_init_distributed`, `init_process_group`, etc.

### 3. C Extension Build System

**`build_c_win.py`** (`experiments/vllm_c_ext/`) — *New file*:
- Custom build harness for `_C.pyd` (the vLLM C++ extension) on Windows.
- Compiles 21 source files from `csrc/libtorch_stable/`.
- Uses `torch::stable::Tensor` ABI (`STABLE_TORCH_LIBRARY_FRAGMENT`).

**`CMakeLists.txt`** — Stub support:
- Added ROCm stub headers for missing CUDA runtime functions.
- Added `strip-msvc-flags.cmake` to sanitize MSVC flags for HIP compilation.

**`rocm-windows-toolchain.cmake`** — *New file*:
- Cross-compilation toolchain for Windows → HIP/ROCm.
- Sets `hipcc` as the compiler, passes Windows-specific flags.

**`strip-msvc-flags.cmake`** — *New file*:
- Removes incompatible MSVC flags before passing to HIP compiler.

**`setup.py`** — Build config:
- Modified to support Windows HIP builds.
- Added conditional source file inclusion for Windows.

### 4. Build Fixes for `_C.pyd`

**`csrc/libtorch_stable/torch_utils.h`**:
- Added `_USE_MATH_DEFINES` guard for Windows math constants (`M_SQRT1_2`, etc.).

**`csrc/cumem_allocator.cpp`**:
- Added `_USE_MATH_DEFINES` before includes.

**`csrc/spinloop.cpp`**:
- Added Windows implementation for spinloop (uses `_mm_pause()` equivalent).

**`csrc/attention/dtype_fp8.cuh`**:
- Fixed template instantiation for Windows compilation.

**`win_c_bindings_shim.cpp`** (*new*, `experiments/vllm_c_ext/`):
- Drop-in replacement for `csrc/torch_bindings.cpp` that only registers compiled ops.
- Uses `STABLE_TORCH_LIBRARY_FRAGMENT` and `torch::stable::Tensor`.
- Registers: `silu_and_mul`, `rms_norm`, `fused_add_rms_norm`, `rotary_embedding`,
  `gptq_gemm`/`gptq_shuffle`, `static_scaled_fp8_quant`, `dynamic_scaled_fp8_quant`,
  `static_scaled_int8_quant`, `dynamic_scaled_int8_quant`,
  `rms_norm_dynamic_per_token_quant`, `permute_cols`,
  cache ops (`swap_blocks`, `reshape_and_cache`), custom all-reduce,
  cuda_utils, sampler ops.

### 5. ROCm Stubs

**`csrc/rocm-stubs/`** — *New directory*:
- `cuda_runtime.h`: Maps CUDA runtime calls to HIP equivalents.
- `cublas_v2.h`: Stubs for cuBLAS API (unused on ROCm).
- `torch/csrc/stable/macros.h`: Defines `STABLE_TORCH_LIBRARY_FRAGMENT` macro.
- `torch/headeronly/core/Dispatch.h`: Dispatch key stubs for stable ABI.

### 6. Triton Backend

**`vllm/triton_utils/importing.py`**:
- Fixed `PackageNotFoundError` from `version("vllm")` — was caught by wrong handler.
- Added Windows path check for `triton-windows` package.

### 7. Utilities

**`vllm/utils/network_utils.py`**:
- Added Windows-compatible network interface detection.

**`vllm/utils/system_utils.py`**:
- Windows-compatible memory stats (psutil fallback).
- Process affinity handling.

**`vllm/v1/worker/block_table.py`**:
- Fixed slot mapping logic for Windows memory allocator differences.
- Added fallback when slot map returns unexpected values.

**`vllm/logger.py`**:
- Thread-safe logging initialization on Windows.

### 8. Headers / Frontend

**`vllm/entrypoints/openai/frontend/index.html`** — *New file*:
- Full dark-theme chat UI with streaming SSE, model dropdown, parameter sliders,
  history sidebar, code highlighting, keyboard shortcuts.

**`vllm/entrypoints/openai/api_server.py`**:
- Default port changed to 8001 (8000 taken by system process on this system).
- CORS headers for frontend development.

### 9. Scripts

**`scripts/convert_gguf_to_hf.py`** — *New file*:
- Converts GGUF models (Q8_0, F16, etc.) to HuggingFace safetensors format.
- Handles llama, qwen2, qwen3, gemma2, phi3, starcoder2, falcon architectures.
- Uses `gguf.quants.dequantize()` for proper handling of quantized formats.
- Dynamic metadata mapping with arch-prefix fallback.

**`scripts/fix_converted_config.py`** — *New file*:
- Infers missing config values from tensor shapes.
- Auto-detects hidden_size, num_attention_heads, etc.

**`scripts/benchmark_model.py`** — *New file*:
- Benchmarking script for latency and throughput.
- Supports warmup runs, different token lengths, batch sizes.

## Building `_C.pyd` (The Hard Part)

### Build Command

```powershell
$env:HIP_PATH = "E:\ROCM-7.13.0-Windows"
cd experiments/vllm_c_ext
python build_c_win.py
```

### What It Compiles

21 source files from `csrc/libtorch_stable/`:
- `attention/attention_kernels.cu`, `activation_kernels.cu`, `gptq_kernels.cu`
- `quantization/*.cu` (fp8, int8, block quant)
- `cache_kernels.cu`, `moe_kernels.cu`, `sampling_kernels.cu`
- `custom_all_reduce.cu`, `cuda_utils.cu`
- Bindings: `win_c_bindings_shim.cpp`

### Key Build Flags

| Flag | Purpose |
|------|---------|
| `-D_USE_MATH_DEFINES` | Enables `M_SQRT1_2`, `M_SQRT2` on MSVC |
| `-DENABLE_FP8` | Enables `namespace vllm::fp8` body (empty without it) |
| `-DTORCH_USE_STABLE_ABI` | Use `torch::stable::Tensor` ABI |
| `--@std` / `-std:c++17` | C++17 required by torch headers |

## Benchmark Results

| Model | Size | Tok/s | VRAM | Load Time | Notes |
|-------|------|-------|------|-----------|-------|
| OPT-125M | 125M | ~90 | ~1 GB | 5s | Baseline |
| Qwen2.5-3B-Instruct | 3B | ~24 | ~5.8 GB | 26s | Eager mode, 64 tokens |
| Qwen3-1.7B-Coder-SFT (GGUF→HF) | 1.7B | ~31 | ~3.8 GB | 26s | Tokenizer mismatch (WIP) |
| CodeGemma-7B-it-GPTQ | 7B | N/A | 5.23 GB | 62s | HIP error during inference (low KV cache) |

**Hardware**: RX 9070 XT (gfx1201, RDNA4), 16 GB VRAM, 64 GB RAM
**Mode**: `--enforce-eager` (CUDAGraphs not yet stable)

## Known Issues & Limitations

1. **CUDAGraphs disabled** — Requires fully functional `_C.pyd` and CUDA graph capture which may hang on ROCm Windows.
2. **No ROCm attention kernel** — `_rocm_C.pyd` is not built; falls back to Triton attention.
3. **NCCL unavailable** — Distributed inference uses Gloo backend only.
4. **GPTQ models crash** — CodeGemma 7B GPTQ loads but fails during inference with hipblaslt error.
5. **Token-based offloading** — Not yet implemented (manual 60/40 split pending).
6. **Open WebUI** — Dependency conflicts in this environment.
7. **Port 8000** — Taken by system process; API server uses port 8001.

## How to Run

### Basic Inference

```powershell
$env:HIP_PATH = "E:\ROCM-7.13.0-Windows"
$env:PYTHONPATH = "C:\path\to\vllm"

python -m vllm.entrypoints.openai.api_server `
    --model F:\VLLM-Models\Qwen2.5-3B-Instruct `
    --port 8001 `
    --enforce-eager `
    --dtype float16
```

### API Server (with Chat UI)

```powershell
python -m vllm.entrypoints.openai.api_server `
    --model F:\VLLM-Models\Qwen2.5-3B-Instruct `
    --port 8001 `
    --enforce-eager `
    --dtype float16 `
    --served-model-name Qwen2.5-3B-Instruct
# Open http://localhost:8001/ for the chat UI
```

### Benchmark

```powershell
python scripts/benchmark_model.py `
    --model F:\VLLM-Models\Qwen2.5-3B-Instruct `
    --num-prompts 10 `
    --max-tokens 64
```

## Credits

- [ThePie88](https://github.com/ThePie88) — Original vLLM-ROCm-Windows fork, CK tiled FMHA, ROCm build system.
- [Maxritz](https://github.com/Maxritz) — Windows port patches, `_C.pyd` build harness, Triton backend wiring, GGUF converter, IPC fixes, frontend.
- [LLVM Project](https://llvm.org/) — Upstream vLLM.

## License

Same as upstream vLLM — Apache 2.0.
