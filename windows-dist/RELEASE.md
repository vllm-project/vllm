# vLLM Windows + AMD ROCm — Release v1.0

Native Windows port of vLLM for AMD Radeon GPUs (RDNA3/RDNA4). Tested on RX 9070 XT (gfx1201).

## Download

- **[vllm-windows-rocm-dist.zip](https://github.com/Maxritz/vllm-windows/releases/download/v1.0/vllm-windows-rocm-dist.zip)** — 2.9 MB
  - Pre-built `_C.pyd` (13.8 MB binary, 21 source files compiled)
  - Build harness + bindings shim
  - Auto-installer (`install.ps1`)
  - Full documentation

## What Works

- ✅ **Full inference pipeline** — tokenization → model forward → sampling → detokenization
- ✅ **Qwen2.5-3B-Instruct**: ~24 tok/s on RX 9070 XT
- ✅ **OPT-125M**: ~90 tok/s baseline
- ✅ **Triton attention backend** — replaces ROCm attention (not built on Windows)
- ✅ **`_C.pyd` with 21 ops** — silu_and_mul, rms_norm, gptq_gemm, fp8/int8 quant, cache ops, all-reduce
- ✅ **fp8 KV cache** — doubles capacity (248K→496K tokens for 3B model)
- ✅ **Chat UI** — dark theme, SSE streaming, model/param controls at `/`
- ✅ **OpenAI-compatible API** — port 8001 (8000 taken by system), CORS enabled
- ✅ **GGUF→HF converter** — Q8_0, F16, Qwen3, llama, gemma2, phi3, starcoder2

## Requirements

- **GPU:** AMD Radeon RX 9000 (RDNA4, gfx120X) or RX 7000 (RDNA3, gfx110X)
- **VRAM:** 8 GB min, 16 GB recommended (3B models fit, 7B OOM)
- **ROCm:** 7.13.0 for Windows
- **Python:** 3.12
- **OS:** Windows 11 23H2+

## Quick Install

```powershell
# 1. Install ROCm 7.13 + Python 3.12
# 2. Clone repo
git clone https://github.com/Maxritz/vllm-windows.git
cd vllm-windows
git checkout WINDOWS-PORT

# 3. Set up environment
uv venv --python 3.12
.venv\Scripts\activate
uv pip install torch --index-url https://repo.amd.com/rocm/whl/gfx120X-all/
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# 4. Install pre-built _C.pyd
Expand-Archive windows-dist\vllm-windows-rocm-dist.zip -DestinationPath . -Force
Copy-Item _C.pyd vllm\ -Force

# 5. Run
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\7.13"
python -m vllm.entrypoints.openai.api_server --model F:\VLLM-Models\Qwen2.5-3B-Instruct --enforce-eager
```

Or use the auto-installer:
```powershell
.\windows-dist\install.ps1
```

## Benchmarks (RX 9070 XT, 16 GB VRAM)

| Model | Size | Tokens/s | VRAM |
|-------|------|----------|------|
| OPT-125M | 125M | ~90 | <1 GB |
| Qwen2.5-3B-Instruct | 3B | **24.2** | 5.8 GB |
| Qwen3-1.7B-Coder | 1.7B | **30.9** | 3.8 GB |

## Known Limitations

- CUDAGraphs disabled (use `--enforce-eager`)
- No CPU offloading (missing `get_cuda_view_from_cpu_tensor` op)
- NCCL unavailable — Gloo only for multi-GPU
- 7B+ models OOM on 16 GB VRAM
- No `_rocm_C.pyd` (ROCm attention kernel not built)
- GPTQ 4-bit inference works but model size limited by VRAM

## Files

| File | Size | Description |
|------|------|-------------|
| `_C.pyd` | 13.8 MB | Pre-built C++ extension |
| `install.ps1` | 2 KB | Auto-installer |
| `build-harness/` | — | Source for rebuilding _C.pyd |
| `INSTALL.md` | — | Manual install guide |
| `PATCH-README.md` | — | Full technical documentation of all changes |
| `PR_DESCRIPTION.md` | — | Upstream PR description (for vllm-project/vllm#48139) |

## Upstream PR

https://github.com/vllm-project/vllm/pull/48139

## Credits

- [vLLM Project](https://github.com/vllm-project/vllm) — Upstream framework
- [ThePie88](https://github.com/ThePie88) — Original vLLM-ROCm-Windows fork, CK tiled FMHA
- [AMD ROCm](https://rocm.docs.amd.com) — GPU compute stack
- [Maxritz](https://github.com/Maxritz) — Windows port patches, distribution

## License

Apache 2.0
