# INT4 per-token-head KV Cache: RDNA3 Performance Report

## Hardware
- 2x AMD Radeon RX 7900 XTX (gfx1100, 96 CUs each, 24GB VRAM)
- Tensor Parallel = 2

## Model
- Qwen3.6-27B-GPTQ-W4A16-G128 (compressed-tensors)
- head_size=256, num_q_heads=28, num_kv_heads=4 (GQA 7:1)
- Hybrid architecture: 16 full_attention + 48 Mamba layers

## Results (TP2, cudagraph enabled)

| Metric | INT4 | INT8 | Gap | Notes |
|--------|------|------|-----|-------|
| **Short decode** | 51.9 tok/s | 55.7 tok/s | **-7%** | Attention visible (short context) |
| **32K decode** | 40.0 tok/s | 40.0 tok/s | **0%** | GEMM-bound, attention fully hidden |
| **32K prefill** | 1732 tok/s | ~1290 tok/s | **+34%** | HIP fused prefill + fp16 WMMA |
| **Multi-turn (5 turns)** | 50.2 tok/s | ~50.8 tok/s | **-1%** | Effectively tied |
| **Max context (cudagraph)** | 65536 | 65536 | = | INT4 uses half KV cache memory |

## Key Findings

1. **At 32K context, INT4 = INT8 in decode speed.** The model's GEMM layers
   (W4A16 matmuls) dominate decode time; attention is fully hidden behind them.

2. **INT4 prefill is 34% faster than INT8** due to the fused HIP prefill kernel
   reading half the cache bytes with bf16 WMMA.

3. **The 7% short-decode gap** comes from the INT4 Triton kernel's extra work
   (nibble unpack + zero-point correction + 2-stream dot) which is only visible
   when the attention is NOT hidden behind GEMMs (short context ≤ ~2K tokens).

4. **INT4 uses 52% of INT8's KV cache memory** (68 vs 132 bytes per token per
   head), enabling higher batch sizes or longer contexts in memory-constrained
   deployments.

## Optimizations Applied

- `logits_soft_cap` gate bug fix (fast-paths were never activating)
- Pre-computed dispatch flags and INT4 scale
- Continuation prefill fast-path (bypasses ~15 Python conditions per layer)
- HIP fused prefill kernel for HS=256 (`#pragma unroll 1` for VGPR recycling)
- Preload scales+zp into LDS (eliminates redundant global reads)
- ROCm Triton tuning: `waves_per_eu=1, matrix_instr_nonkdim=16, kpack=2, num_stages=1`
- BLOCK_KV=32 for INT4 decode (halves tile loop iterations)
- fp16 cast before `tl.dot` for QK and PV (activates `v_wmma_f32_16x16x16_f16`)

## Dead Ends

- NF4 (NormalFloat4) LUT dequant: 16 `tl.where` comparisons slower than linear cast
- INT8 WMMA for INT4 (quantize Q to i8): Q quantization overhead > WMMA benefit
- HIP decode v2 for HS=256: crashes with cudagraph (q_rot buffer realloc)
- HIP reshape for HS=256: subtle rounding diffs degrade model quality
- num_warps=2: Triton cache artifact, regresses from clean deploy
- BLOCK_KV=64: short decode -15%, no 32K gain
- Skip Softmax in Triton: runtime `if` doesn't truly skip on RDNA3 (predicated)
- NUM_KV_SPLITS reduction: no impact (not launch-bound)
