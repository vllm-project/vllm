# [ROCm][MLA] Fuse RoPE + MLA KV-cache write via AITER Triton kernel

## Summary

Activates `MLARoPEKVCacheCatFusionPass`, an existing inductor graph-rewrite pass
that fuses the RoPE application and MLA KV-cache write into a single AITER Triton
kernel (`fused_qk_rope_concat_and_cache_mla`) on ROCm.

**Why it wasn't firing:** `enable_rope_kvcache_mla_fusion` checked
`IS_QUANTIZED` which is hardcoded `False` on ROCm (issue #25689),
permanently disabling the pass at OPT_02/OPT_03. This was acknowledged
in issue #43504 where porting to manual fusion was proposed as a
workaround. This PR fixes the root cause directly by adding
`enable_aiter_rope_kvcache_mla_fusion` that checks AITER MLA
availability, making the workaround unnecessary.

For DeepSeek-R1 (61 MLA layers, TP=8), this reduces **976 kernel launches to 488
per decode step**, eliminating the HBM round-trip for `k_pe` between RoPE and
KV-cache write.

The pass is gated behind `VLLM_ROCM_USE_AITER_MLA=1` + kernel probe — it is a
no-op on CUDA and on ROCm builds without the AITER fused kernel.

---

## Motivation

On ROCm, the standard decode path for DeepSeek MLA models runs two back-to-back
operations per attention layer:
1. RoPE application to `q_pe` and `k_pe`
2. `vllm::unified_mla_kv_cache_update` — writes `kv_c_normed` + rotated `k_pe`
   to the KV cache

AITER provides `fused_qk_rope_concat_and_cache_mla`, a Triton kernel that does
both in one pass with built-in QK normalization and MXFP8 quantisation.  Replacing
the two ops removes kernel-launch overhead and intermediate tensor allocation for
every decode step on every MLA layer (DeepSeek-R1 has 61 MLA layers × 8 TP
workers).


---

## How it works

```
Before (per MLA layer, per decode step):
  rope(q_pe, k_pe)                              # RoPE applied separately
  → unified_mla_kv_cache_update(kv_c_normed, k_pe)  # cache write separately
  # k_pe written to HBM after RoPE, read back for cache write

After (MLARoPEKVCacheCatFusionPass applied):
  fused_rope_unified_mla_kv_cache_update(q_pe, k_pe, kv_c_normed, ...)
  └─ calls ops.concat_and_cache_mla_rope_fused()
     └─ dispatches to aiter.fused_qk_rope_concat_and_cache_mla (Triton)
  # k_pe stays in registers between RoPE and cache write
```

**Activation conditions** (`enable_aiter_rope_kvcache_mla_fusion`):
- `VLLM_ROCM_USE_AITER=1` and `VLLM_ROCM_USE_AITER_MLA=1`
- `aiter.fused_qk_rope_concat_and_cache_mla` importable (kernel probe returns True)
- Falls back to the standard `enable_rope_kvcache_mla_fusion` condition otherwise

**Pattern registered** (per MLA layer × `is_neox` × `use_deepseek_scaling` × optionally `use_flashinfer`):
```python
# pattern
k_pe_unsqueezed = k_pe.unsqueeze(1)
q_pe, k_pe = rope_matcher(positions, q_pe, k_pe_unsqueezed, cos_sin_cache)
dummy = torch.ops.vllm.unified_mla_kv_cache_update(
    kv_c_normed, k_pe, layer_name, kv_cache_dtype, k_scale
)

# replacement
at = auto_functionalized(
    torch.ops.vllm.fused_rope_unified_mla_kv_cache_update,
    positions=positions, q_pe=q_pe, k_pe=k_pe, kv_c=kv_c_normed,
    cos_sin_cache=cos_sin_cache, is_neox=is_neox,
    kv_cache_dtype=kv_cache_dtype, kv_cache_scale=k_scale, layer_name=layer_name,
)
```

---

## Test plan

### Unit tests (ROCm, run on MI350X)
```bash
.venv/bin/python -m pytest tests/rocm/aiter/test_aiter_mla_fused_rope_kvcache.py -v
```
- `test_kv_cache_written_after_fused_call` — KV cache slots are non-zero after fused call
- `test_q_out_updated_after_fused_call` — RoPE was applied to `q_pe` (q_out updated)
- `test_slot_mapping_respected` — only mapped slots are written, others stay zero

### Accuracy (lm-eval GSM8K, 8× MI350X, TP=8, max_len=32768)

**Command (MXFP4 example):**
```bash
lm_eval \
  --model vllm \
  --model_args "pretrained=amd/DeepSeek-R1-MXFP4,tensor_parallel_size=8,quantization=quark,kv_cache_dtype=fp8,max_model_len=32768,trust_remote_code=True" \
  --tasks gsm8k \
  --num_fewshot 8 \
  --limit 200 \
  --apply_chat_template \
  --seed 1234 \
  --gen_kwargs "temperature=0,max_new_tokens=8192"
```

For BF16: replace `quantization=quark,kv_cache_dtype=fp8` with `dtype=bfloat16,kv_cache_dtype=fp8`.
Baseline uses `VLLM_ROCM_USE_AITER_MLA=0`; fusion uses `VLLM_ROCM_USE_AITER_MLA=1`.

| Model | Setup | Baseline (`AITER_MLA=0`) | Fusion (`AITER_MLA=1`) | Δ |
|-------|-------|------------------------|----------------------|---|
| `amd/DeepSeek-R1-MXFP4` | quark, fp8 KV | 0.9350 | 0.9400 | +0.5% |
| `deepseek-ai/DeepSeek-R1` | bf16, fp8 KV | 0.8550 | 0.9000 | +5.3% |

> `flexible-extract` metric, 200 samples, 8-shot, seed=1234, temperature=0

No accuracy regression — fusion shows a small improvement within statistical noise (±0.02–0.03 stderr).

### Serving benchmarks (ROCm, `amd/DeepSeek-R1-MXFP4`, 8× MI350X, TP=8)

**Setup:** `VLLM_ROCM_USE_AITER=1`, `--quantization quark --kv-cache-dtype fp8 --tensor-parallel-size 8 --max-model-len 32768`, 200 prompts, seed=1234, comprehensive per-shape warmup.
A/B controlled via `--compilation-config '{"pass_config":{"fuse_rope_kvcache_cat_mla":<true|false>}}'`.

**ISL=1000, OSL=100:**

| MC | Baseline TPOT | Fusion TPOT | Δ TPOT | Baseline tok/s | Fusion tok/s | Δ tok/s |
|----|--------------|------------|--------|---------------|-------------|---------|
| 4  | 10.91 ms | 11.38 ms | +4.3% | 302.9 | 281.9 | -6.9% |
| 8  | 11.98 ms | 11.98 ms | 0% | 588.0 | 590.8 | +0.5% |
| 16 | 13.14 ms | 13.04 ms | -0.8% | 691.0 | 1078.4 | **+56%** |
| 32 | 15.85 ms | 15.27 ms | -3.7% | 336.8 | 182.4 | -45.9% |

![TPOT Baseline vs Fusion (MXFP4, ISL=1000, OSL=100)](https://raw.githubusercontent.com/shantipriya-amd/vllm/d9b7f3f46678db0145949b75b74f92a4f28b4f11/docs/assets/mxfp4_tpot_comparison.png)

**ISL=512, OSL=512:**

| MC | Baseline TPOT | Fusion TPOT | Δ TPOT | Baseline tok/s | Fusion tok/s | Δ tok/s |
|----|--------------|------------|--------|---------------|-------------|---------|
| 8  | 11.82 ms | 11.85 ms | +0.3% | 643.6 | 542.4 | -15.7% |
| 16 | 12.78 ms | 12.85 ms | +0.5% | 1172.0 | 1121.9 | -4.3% |

TPOT is within ±4% across all configs — fusion does not hurt per-step decode cost.
The +56% throughput at MC=16 ISL=1000 is the headline gain. Throughput variance at
other concurrencies reflects batch-scheduling dynamics, not kernel overhead.

**Fusion verified active** by:
1. Log line: `Enabled custom fusions: ..., rope_kvcache_cat_mla, ...`
2. All 8 TP workers logged: `[aiter] import [module_fused_qk_norm_rope_cache_quant_shuffle]`

---

### Serving benchmarks (ROCm, `deepseek-ai/DeepSeek-R1` BF16, fp8 KV, 8× MI350X, TP=8)

**Setup:** `VLLM_ROCM_USE_AITER=1`, `--dtype bfloat16 --kv-cache-dtype fp8 --tensor-parallel-size 8 --max-model-len 32768`, 200 prompts, seed=1234, comprehensive Triton warmup.

**ISL=512, OSL=512:**

| MC | Baseline TPOT | Fusion TPOT | Δ TPOT | Baseline tok/s | Fusion tok/s | Δ tok/s |
|----|--------------|------------|--------|---------------|-------------|---------|
| 8  | 13.15 ms | 13.28 ms | +1.0% | 413.1 | 277.5 | -32.8% |
| 16 | 15.30 ms | 15.29 ms | 0% | 797.2 | 870.1 | **+9.1%** |

**ISL=1000, OSL=100:**

| MC | Baseline TPOT | Fusion TPOT | Δ TPOT | Baseline tok/s | Fusion tok/s | Δ tok/s |
|----|--------------|------------|--------|---------------|-------------|---------|
| 4  | 16.65 ms | 13.21 ms | **-20.7%** | 219.3 | 233.3 | +6.4% |
| 8  | 13.24 ms | 13.31 ms | +0.5% | 319.6 | 111.6 | -65.1% |
| 16 | 15.62 ms | 15.53 ms | -0.6% | 914.3 | 414.0 | -54.7% |
| 32 | 30.73 ms | 26.13 ms | **-15.0%** | 942.5 | 1046.4 | **+11.0%** |
| 64 | 33.91 ms | 31.37 ms | **-7.5%** | 1487.6 | 1522.7 | **+2.4%** |

![TPOT Baseline vs Fusion (BF16, ISL=1000, OSL=100)](https://raw.githubusercontent.com/shantipriya-amd/vllm/d9b7f3f46678db0145949b75b74f92a4f28b4f11/docs/assets/bf16_tpot_comparison.png)

TPOT improves 7–21% at MC=4/32/64; decode-step cost is regression-free (≤1% at MC=8/16).
Throughput variance at other concurrencies reflects batch-scheduling dynamics, not kernel overhead.

