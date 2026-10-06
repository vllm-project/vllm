# RDNA3 HIP Prefill Attention Kernels (INT8 + INT4)

Native gfx1100 (RX 7900 XTX) paged-prefill attention kernels for quantized
KV cache. Bypass Triton entirely — cooperative WMMA with fused dequant.

## Performance (Qwen3.6-27B W4A16, TP2, RX 7900 XTX, chunked prefill ~16K)

| KV dtype | tok/s | vs FP16 | VRAM saving | Kernel |
|----------|-------|---------|-------------|--------|
| FP16 (auto) | 727 | baseline | — | Triton |
| INT8 per-token-head (Triton) | 599 | -18% | 50% | Triton |
| **INT8 per-token-head (HIP)** | **1209** | **+66%** | **50%** | This |
| **INT4 per-token-head (HIP)** | **~1150** | **+58%** | **75%** | This |

INT4 is within 1-8% of INT8 performance (zero-point centered in loader).

## Architecture

Both kernels share the same v2 cooperative architecture:

```
┌─────────────────────────────────────────────────┐
│  4 waves × 32 lanes = 128 threads per block     │
│  BLOCK_M = 64 query rows (16 per wave)          │
│  K_TILE = 16 KV tokens per iteration            │
│  HEAD_SIZE = 128                                 │
│  Grid: (num_seqs, num_query_heads, q_tiles)     │
└─────────────────────────────────────────────────┘
```

### INT8 Kernel (`paged_prefill_attn_rdna3_v2_int8.cu`)

```
K load: 128 threads cooperative, 16 bytes/thread (int8)
        → v_cvt_f16_i16_e32 (1 VALU, native) → K_lds [fp16]
V load: same pattern → V_lds [fp16, transposed]
Scale:  loaded to LDS, fused post-WMMA

Q × K:  v_wmma_f32_16x16x16_bf16_w32 (8 WMMAs)
         S *= k_scale * sm_scale
P × V:  v_wmma_f32_16x16x16_bf16_w32 (8 WMMAs)
         P *= v_scale (fused pre-WMMA)
```

### INT4 Kernel (`paged_prefill_attn_rdna3_v2_int4.cu`)

```
K load: 128 threads cooperative, 8 bytes/thread (packed nibbles)
        → unpack (AND + shift) → SUBTRACT zero-point → K_lds [signed int8]
        Zero conversion instructions! K stored as centered int8 [-15,15]
V load: same unpack → subtract v_zp → V_lds [fp16, centered]
Scale:  steganographed float32 (bits[0:3]=zp, bits[4:31]=scale)

Q × K:  v_wmma_i32_16x16x16_iu8 (signA=true, signB=true, 8 WMMAs)
         Q quantized to int8 per-row (amortized over all K tiles)
         S = i32_dot * q_scale * k_scale * sm_scale
         NO zero-point correction needed (centered in loader)
P × V:  v_wmma_f32_16x16x16_bf16_w32 (8 WMMAs)
         P *= v_scale (fused pre-WMMA)
         NO Pv_zp_sum correction (centered in loader)
```

**Key optimization**: zero-point subtracted during load (16 integer subs/thread)
eliminates ~96 VALU/tile from the inner loop (shuffles + reductions + corrections).

## Why INT8/INT4 > FP8 on RDNA3

| Conversion | gfx1100 ISA | Cost |
|---|---|---|
| INT8 → FP16 | `v_cvt_f16_i16_e32` | **1 VALU** |
| INT4 nibble unpack | `v_and_b32` + `v_lshrrev_b32` | **2 VALU** |
| FP8 → FP16 (HIP) | 20 VALU + 3 branches | **~20 cycles** |

gfx1100 has NO native FP8 hardware. INT8/INT4 conversion is native.

## Two-Phase Design

Both kernels handle mixed cached + fresh tokens:

- **Phase 1 (cached prefix)**: Reads quantized KV from paged cache.
  Uses INT8 WMMA (INT4) or bf16 WMMA (INT8) with fused scale.
- **Phase 2 (current chunk)**: Reads fp16 K/V from current forward pass
  (not yet quantized). Uses standard bf16 WMMA, scale=1.

## Files

| File | Purpose |
|------|---------|
| `paged_prefill_attn_rdna3_v2_int8.cu` | INT8 kernel (~620 lines) |
| `paged_prefill_attn_rdna3_v2_int4.cu` | INT4 kernel (~530 lines) |
| `paged_prefill_attn_rdna3.cuh` | Shared WMMA wrappers (bf16, fp16, iu8, ii8) |
| `vllm/v1/attention/ops/rdna3_int8_prefill.py` | INT8 Python wrapper |
| `vllm/v1/attention/ops/rdna3_int4_prefill.py` | INT4 Python wrapper |
| `vllm/v1/attention/backends/triton_attn.py` | Dispatch gates |

## Dispatch Conditions

The HIP kernels auto-dispatch when ALL conditions are met:

- Platform is ROCm
- `torch.ops._C.paged_prefill_attn_rdna3_{int8,int4}` available (gfx1100 build)
- Pure prefill with continuation chunks (reads from paged cache)
- No alibi, sliding window, sinks, softcap

Falls through to Triton `triton_per_token_head_prefill` otherwise.

## Usage

```bash
# INT8 per-token-head (dynamic scales, works with all models, 50% VRAM saving)
vllm serve <model> --kv-cache-dtype int8_per_token_head

# INT4 per-token-head (asymmetric + RHT, 75% VRAM saving)
vllm serve <model> --kv-cache-dtype int4_per_token_head
```

## Build

```bash
# Inside ROCm container:
cd /path/to/vllm
MAX_JOBS=$(nproc) PYTORCH_ROCM_ARCH=gfx1100 python3 setup.py build_ext --inplace

# Verify:
python3 -c "import torch; import vllm._C; \
  print('int8:', hasattr(torch.ops._C, 'paged_prefill_attn_rdna3_int8')); \
  print('int4:', hasattr(torch.ops._C, 'paged_prefill_attn_rdna3_int4'))"
```

## Correctness

- INT8: Exact match vs PyTorch reference (symmetric quant, no precision loss in dequant)
- INT4: Cosine similarity 0.999995 vs reference (Q quantized to int8 introduces ~0.001 error)
- Both produce coherent model outputs at temperature=0 on code, math, translation, QA

## Microbenchmark (ctx=8000, ql=512, Qwen3.6 shape: 32Hq, 8Hkv, d=128)

| Kernel | Time | vs Triton |
|--------|------|-----------|
| Triton INT8 per-token-head | ~25 ms | baseline |
| **HIP INT8** | **3.03 ms** | **8.3×** |
| **HIP INT4** | **3.05 ms** | **8.2×** |
