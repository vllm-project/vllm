Copy and paste this into https://github.com/vllm-project/vllm/pull/48139

---

## Windows + AMD ROCm Native Port

Brings vLLM to Windows with AMD ROCm (RDNA3/RDNA4). Tested on RX 9070 XT (gfx1201, RDNA4) with ROCm 7.13.0.

### Changes (39 files, +2571/-97 lines)

**Platform & IPC (Windows compat):**
- `vllm/platforms/rocm.py`: TRITON_ATTN priority swap (ROCM_ATTN needs `_rocm_C.pyd` with paged_attention op — not built on Windows)
- `vllm/platforms/__init__.py`, `cuda.py`: Windows + AMD GPU detection path
- `vllm/v1/engine/core.py`: Windows `spawn` multiprocessing, freeze_support guard, ZMQ TCP fallback
- `vllm/v1/engine/utils.py`: `get_open_zmq_ipc_path()` uses `tcp://127.0.0.1:{port}` on Windows; `kill_proc_tree()` uses `taskkill`
- `vllm/distributed/utils.py`: Skips NCCL (unavailable on Windows ROCm), uses Gloo
- `vllm/rocm_dist_fixes.py`: torch.distributed shim/stubs for single-process Windows
- `vllm/platforms/rocm_dist_stubs.py`: Stubs for distributed ops missing on Windows ROCm
- `vllm/v1/worker/block_table.py`: Slot mapping fallback for Triton-less paths

**Triton backend:**
- `vllm/triton_utils/importing.py`: Fixed PackageNotFoundError for `version("vllm")`; Windows triton-windows detection

**C++ Extension (`_C.pyd`):**
- `build_c_win.py` (new): Build harness compiling 21 source files from `csrc/libtorch_stable/`
- `win_c_bindings_shim.cpp` (new): Stable ABI bindings using `STABLE_TORCH_LIBRARY_FRAGMENT` — only registers compiled ops
- Build flags: `-D_USE_MATH_DEFINES`, `-DENABLE_FP8`, `-DTORCH_USE_STABLE_ABI`
- ROCm stubs: `csrc/rocm-stubs/` — CUDA→HIP mappings for `cuda_runtime.h`, `cublas_v2.h`, stable ABI macros
- CMake: `rocm-windows-toolchain.cmake`, `strip-msvc-flags.cmake` for MSVC→HIP flag sanitization

**Utilities:**
- `vllm/utils/network_utils.py`: Windows network interface detection
- `vllm/utils/system_utils.py`: Windows memory stats (psutil)
- `vllm/logger.py`: Thread-safe logging on Windows
- `vllm/entrypoints/openai/frontend/index.html`: Dark-theme chat UI with SSE streaming
- `vllm/entrypoints/openai/api_server.py`: Port 8001 fallback (8000 taken by system)

**Scripts:**
- `scripts/convert_gguf_to_hf.py`: GGUF→HF converter with `gguf.quants.dequantize`, multi-arch metadata mapping
- `scripts/benchmark_model.py`: Latency/throughput benchmark
- `scripts/fix_converted_config.py`: Config inference from tensor shapes

**Distribution:**
- `windows-dist/`: Pre-built `_C.pyd`, build harness, `install.ps1` (auto-detects ROCm path)

### Duplicate Work Check

No existing PR targets Windows ROCm natively. #47991 (ROCM_FLASH_ATTN) and #47823 (ROCm attention) are Linux-only. ThePie88's vLLM-ROCm-Windows fork provides the base, this PR extends with IPC fixes, Triton wiring, `_C.pyd` build, and full inference capability.

### Test Plan

```powershell
$env:HIP_PATH = "E:\ROCM-7.13.0-Windows"
$env:PYTHONPATH = "C:\path\to\vllm"

# Verify ops
python -c "import vllm._C; import torch; assert hasattr(torch.ops._C, 'silu_and_mul'); assert hasattr(torch.ops._C, 'gptq_gemm'); print('OK')"

# Run model
python -m vllm.entrypoints.openai.api_server --model F:\VLLM-Models\Qwen2.5-3B-Instruct --enforce-eager --dtype float16 --port 8001
```

### Test Results (RX 9070 XT, 16 GB VRAM, ROCm 7.13.0)

| Model | Size | Tok/s | VRAM | KV Cache | Notes |
|-------|------|-------|------|----------|-------|
| OPT-125M | 125M | ~90 | <1 GB | ~30 GB | Baseline, no issues |
| Qwen2.5-3B-Instruct | 3B | **24.2** | 5.79 GiB | 248K (fp16) / 496K (fp8) | Fully working |
| Qwen3-1.7B-Coder-SFT | 1.7B | **30.9** | 3.8 GiB | 99K | Tokenizer mismatch (GGUF→HF) |
| CodeGemma-7B-it-GPTQ | 7B | OOM | 5.23 GiB | 880-5.5K | OOM on 16 GB (needs offloading) |

**KV Cache:** fp8 quantization works on ROCm — doubles capacity but ~45% slower.

**Known limitations:**
- CUDAGraphs disabled (`--enforce-eager` required)
- CPU offloading blocked by missing `get_cuda_view_from_cpu_tensor` op
- No NCCL — Gloo only for distributed
- 7B models OOM on 16 GB VRAM

### Security

- IPC uses `tcp://127.0.0.1` loopback only (no Unix sockets on Windows)
- Named pipe ACLs default to current user only
- Firewall documentation added to `windows-dist/INSTALL-WINDOWS-ROCM.md`
- Cache directory security documented (`%LOCALAPPDATA%\vllm\cache`)

### AI Assistance

This PR was created with AI assistance (Claude, DeepSeek V4). All changes reviewed by human submitter.

Co-authored-by: Claude
Co-authored-by: DeepSeek V4
Signed-off-by: Maxritz
