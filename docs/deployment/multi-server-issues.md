# Multi-Server vLLM Deployment Issues and Solutions

## Summary

This document catalogs deployment issues encountered across 4 GPU servers (RTX 4090/5090) running vLLM v0.22.1-v0.23.0, covering CUDA toolkit compatibility, JIT compilation failures, attention backend validation, and memory management issues.

## Server Environment

| Server | GPU | vLLM Version | CUDA | Issues |
|--------|-----|--------------|------|--------|
| hn01:30823 | RTX 4090 24GB | 0.22.1 | 13.0 | FlashInfer JIT, lib64 path |
| js02:30107 | RTX 4090 24GB | 0.23.0 (source) | 12.8 | OOM compilation, attention backend |
| ah01:30145 | RTX 5090 32GB | 0.22.1 | 12.9 | Marlin shape, CUDA graph OOM |
| sh01:30438 | RTX 4090 24GB | 0.22.1 | 12.8 | System environment (apt broken) |

---

## Issue 1: FlashInfer JIT Compilation - Missing curand.h

### Server: hn01:30823

**Symptom:**
```
FlashInfer JIT compilation failed: fatal error: curand.h: No such file or directory
```

**Root Cause:**
CUDA 13.0 installation at `/usr/local/lib/python3.12/dist-packages/nvidia/cu13/` missing `include/curand.h`. FlashInfer JIT compiler searches CUDA include paths but cannot find curand headers.

**Workaround Applied:**
```bash
# Copy from CUDA 12.8 installation
cp /usr/local/cuda-12.8/include/curand.h /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/
```

**Code Fix:**
See `vllm/utils/flashinfer.py` - improved CUDA toolkit path detection to search multiple locations.

---

## Issue 2: FlashInfer JIT Linking - Hardcoded lib64 Path

### Server: hn01:30823

**Symptom:**
```
/usr/bin/ld: cannot find -lcudart: No such file or directory
```

**Root Cause:**
`tools/build_deepgemm_C.py:72` hardcodes `-L{cuda_home}/lib64`, but CUDA 13.0 installs libraries in `lib/` not `lib64/`. FlashInfer JIT linker inherits this behavior.

**Actual Layout:**
```
/usr/local/lib/python3.12/dist-packages/nvidia/cu13/
├── lib/
│   └── libcudart.so.13  ✓
└── lib64/               ✗ (missing)
```

**Workaround Applied:**
```bash
mkdir -p /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib64
ln -s ../lib/libcudart.so.13 lib64/libcudart.so.13
```

**Code Fix:**
See `tools/build_deepgemm_C.py` - check both `lib/` and `lib64/` directories.

---

## Issue 3: Source Compilation OOM with High Parallelism

### Server: js02:30107

**Symptom:**
```
Exit code 137 (SIGKILL) during nvcc compilation
Killed: flash attention + cutlass CUDA template compilation
```

**Root Cause:**
`MAX_JOBS=96` (default on 125GB RAM system) causes parallel nvcc processes to consume >125GB. Flash attention and cutlass templates use 2-4GB per compilation unit.

**Solution:**
```bash
MAX_JOBS=8 pip install -e .
```

Reducing parallelism to 8 allows incremental compilation (47/347 targets cached from failed run).

**Documentation Fix:**
See `docs/contributing/build.md` - add memory requirements section.

---

## Issue 4: TRITON_ATTN Incompatible with TurboQuant KV Cache

### Servers: js02:30107, ah01:30145

**Symptom:**
```
ValueError: Selected backend AttentionBackendEnum.TRITON_ATTN is not valid for this configuration.
Reason: ['kv_cache_dtype not supported']
```

**Root Cause:**
TurboQuant 4-bit KV cache requires dedicated attention backend (`TURBOQUANT`), but vLLM auto-selects `TRITON_ATTN` for models with hybrid attention (e.g., DiffusionGemma). Manual override to `FLASHINFER` also fails due to head dimension constraints.

**Backend Compatibility Matrix:**

| Attention Backend | TurboQuant 4bit | DiffusionGemma | Gemma4 Unified |
|-------------------|-----------------|----------------|----------------|
| TRITON_ATTN | ❌ | ✅ | ✅ |
| FLASHINFER | ✅ | ❌ (head_dim) | ⚠️ |
| FLASH_ATTN | ✅ | ❌ (head_dim) | ❌ |
| TURBOQUANT | ✅ | ✅ | ✅ |

**Solution:**
Do not manually specify `--attention-backend` when using `--kv-cache-dtype turboquant_4bit_nc`. Let vLLM auto-select the `TURBOQUANT` backend.

**Code Fix:**
See `vllm/v1/attention/backends/flashinfer.py` - improved validation error messages.

---

## Issue 5: Marlin Shape Mismatch with Gemma4 QAT

### Server: ah01:30145

**Symptom:**
```
RuntimeError: Shape mismatch: a.size(1) = 4096, size_k = 8192
```

**Root Cause:**
vLLM 0.22.1 lacks native `Gemma4UnifiedForConditionalGeneration` implementation. Generic fallback incorrectly applies Marlin quantization to `patch_dense` layers.

**Affected Model:**
`google/gemma-4-12B-it-qat-w4a16-ct`

**Solution:**
Upgrade to v0.23.0+ which includes PR #44429 + #44571 (native Gemma4 Unified support).

**Status:**
Fixed in main branch. Issue #44796 confirmed.

---

## Issue 6: CUDA Graph OOM on RTX 5090

### Server: ah01:30145

**Symptom:**
```
Silent OOM during CUDA graph capture phase
Process killed without error message
```

**Root Cause:**
`profile_cudagraph_memory()` calculates budget based on physical GPU memory (32GB), ignoring `--gpu-memory-utilization`. Default captures 512 batch sizes `[1,2,4,8,...,512]`, each consuming ~20MB. Model (8.3GB) + graphs (10GB) > 32GB.

**Solutions:**

**Option A: Limit capture sizes (recommended)**
```bash
--compilation-config '{"cudagraph_capture_sizes": [1, 2, 4]}'
```
Reduces graph memory from ~10GB to ~100MB.

**Option B: Use PIECEWISE mode**
```bash
--compilation-config '{"cudagraph_mode": "PIECEWISE"}'
```
Only graphs compatible operations, ~50% memory reduction.

**Option C: FULL_DECODE_ONLY**
```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```
Only graphs pure decode batches, suitable for P/D separation.

**Documentation Fix:**
See `docs/configuration/memory.md` - add CUDA graph memory management section.

---

## Issue 7: bitsandbytes 4bit Incompatible with v1 Engine

### Server: hn01:30823

**Symptom:**
```
bitsandbytes 4bit quantization fails with v1 engine
Continuous initialization errors
```

**Root Cause:**
vLLM v1 engine has bugs in bitsandbytes 4bit quantization support. Model: `Phi-4-mini` with `--quantization bitsandbytes`.

**Solution:**
Use v0.22.1 or switch to different quantization method (fp8, gptq).

**Workaround:**
```bash
# Use fp8 KV cache instead of 4bit model quantization
--kv-cache-dtype fp8
```

**Documentation Fix:**
See `docs/quantization/bitsandbytes.md` - add compatibility matrix.

---

## Issue 8: LD_LIBRARY_PATH Required for CUDA 13

### Server: hn01:30823

**Symptom:**
```
libcudart.so.13: cannot open shared object file
```

**Root Cause:**
CUDA 13.0 pip packages install to `/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/`, not in standard library search path.

**Solution:**
```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

**Documentation Fix:**
See `docs/getting_started/installation/gpu.md` - add CUDA 13 library path setup.

---

## Recommendations

### For Deployment Scripts

1. **Always set LD_LIBRARY_PATH** for CUDA 13+:
   ```bash
   export LD_LIBRARY_PATH=$(find /usr/local/lib/python*/dist-packages/nvidia/ -name "*.so*" | sed 's|/[^/]*$||' | sort -u | tr '\n' ':'):$LD_LIBRARY_PATH
   ```

2. **Limit compilation parallelism** on systems with <128GB RAM:
   ```bash
   MAX_JOBS=8 pip install -e .
   ```

3. **Do not manually specify attention backend** when using TurboQuant:
   ```bash
   # ✓ Let vLLM auto-select
   --kv-cache-dtype turboquant_4bit_nc
   
   # ✗ Manual override causes conflicts
   --kv-cache-dtype turboquant_4bit_nc --attention-backend FLASHINFER
   ```

4. **Limit CUDA graph capture sizes** on GPUs with <40GB VRAM:
   ```bash
   --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4]}'
   ```

### For vLLM Developers

1. **Fix lib64 hardcoding** in `tools/build_deepgemm_C.py:72`
2. **Improve FlashInfer JIT path detection** to search multiple CUDA locations
3. **Add attention backend validation** with clearer error messages
4. **Document CUDA graph memory requirements** per GPU model
5. **Add bitsandbytes compatibility matrix** for v1 engine

---

## Related Issues

- Issue #40937: CUDA graph OOM ignores gpu_memory_utilization
- Issue #44796: Marlin shape mismatch with Gemma4 QAT
- PR #44429, #44571: Gemma4 Unified native support

## Testing

All workarounds verified on:
- hn01:30823 (RTX 4090, CUDA 13.0, vLLM 0.22.1)
- js02:30107 (RTX 4090, CUDA 12.8, vLLM 0.23.0 source)
- ah01:30145 (RTX 5090, CUDA 12.9, vLLM 0.22.1)

## License

This documentation is contributed under the Apache 2.0 license.
