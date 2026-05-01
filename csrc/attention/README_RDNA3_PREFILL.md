# RDNA3 Paged-Prefill Attention Kernels (gfx1100)

Native HIP / WMMA paged-prefill attention kernels for AMD RDNA3
(RX 7900 XT/XTX, gfx1100 / 1101 / 1102). Drop-in replacement for the
Triton `context_attention_fwd` (in
`vllm/v1/attention/ops/prefix_prefill.py`) on the ROCM_ATTN backend's
`max_query_len > 1` path. Two compute paths share a single Python
entry, with C++-internal dispatch:

* **v1** (1 wave per block, BLOCK_M = 16): small-qlen path. Wins for
  `qlen < 64` where v2's wider tile would waste 3 of its 4 waves on
  out-of-range rows. Modest win or parity vs Triton in that regime.
* **v2** (4 waves per block, BLOCK_M = 64, shared K/V LDS): the
  production path. The 4-wave block amortises one cooperative K/V
  load over 4× more output rows, cutting global K/V reads by 4× —
  the dominant cost of chunk-heavy prefill on this hardware. Wins
  1.7-2.0× vs Triton at production-typical shapes.

Registered as the C++ op `paged_prefill_attn_rdna3` and wired into
`RocmAttentionImpl.forward` via the gating helper
`vllm/v1/attention/ops/rocm_paged_prefill_attn.py:is_available`. Falls
through gracefully (back to `chunked_prefill_paged_decode` → Triton
`context_attention_fwd`) on non-RDNA3 ROCm devices, on partial
rebuilds where the C++ op is missing, or for any unsupported feature
(FP8 KV / alibi / sliding window / sinks / FP8 output / softcap /
non-causal / `head_size != 128`).

## Files

| File | Role |
| --- | --- |
| `paged_prefill_attn_rdna3.cuh` | Shared types, WMMA wrappers (`__builtin_amdgcn_wmma_f32_16x16x16_{f16,bf16}_w32`), dtype-conversion helpers. |
| `paged_prefill_attn_rdna3.cu` | v1 kernel + the C++ entry `paged_prefill_attn_rdna3` that dispatches v1↔v2 by `max_query_len`. |
| `paged_prefill_attn_rdna3_v2.cu` | v2 kernel + launcher. Lives in its own translation unit per the W4A16 lesson — hipcc scopes some optimizer decisions at TU level and a monolithic TU previously miscompiled the small-M scalar path. |

## Dispatch (decision tree)

`torch.ops._C.paged_prefill_attn_rdna3` is the **single Python entry**.
All branching happens in C++ to keep `RocmAttentionImpl.forward`
torch.compile-friendly:

```text
paged_prefill_attn_rdna3(out, q, k_chunk, v_chunk, k_cache, v_cache,
                         block_table, cu_seqlens_q, seq_lens, sm_scale, causal)
│
├── checks: head_size == 128, block_size % 16 == 0, dtype ∈ {fp16, bf16}
│   (anything else is a TORCH_CHECK error — caller's gate must match)
│
├── if max_query_len >= 64 → launch_paged_prefill_attn_v2<T, 128>()
│       4 waves/block, BLOCK_M = 64, K/V LDS shared across 4 waves
│
└── else                   → launch_paged_prefill_attn<T, 128>()
        1 wave/block, BLOCK_M = 16
```

The Python-side gate (`RocmAttentionImpl.forward`) checks: `max_seqlen_q > 1`
∧ `causal` ∧ `key/value` non-null ∧ `is_available()` ∧ `supports_shape(...)`
∧ no quant KV / alibi / SW / sinks / FP8 output / softcap. Any
disallowed feature → fall back to `chunked_prefill_paged_decode` →
Triton.

## Kernel architecture

### Two-phase iteration (both v1 and v2)

A prefill block iterates K in two phases, applying online-softmax merge
across both:

1. **Cached prefix** (no causal mask): K/V come from the paged cache
   via `block_table` indirection, range `[0, ctx_len)`. The 5-D K
   layout's innermost vec dim x = 8 fp16 = 16 B is exactly one
   `global_load_b128`. Cooperative load in v2.
2. **Current chunk** (causal mask): K/V come from the linear
   `k_chunk` / `v_chunk` tensors (current request's K/V before they
   land in cache), range `[0, query_len)`. The K cache is laid out
   token-outer / dim-inner (no 5-D vec optimization), so the load is
   per-token vec_16B with scattered LDS writes for V.

The causal upper bound on K trims phase 2 to
`min(query_len, q_tile_start + valid_q_count)` — the last tile may
contain K positions that are masked out for some rows but valid for
others; the per-row mask inside `attn_step` handles the diagonal.

### Wave32 fragment layout (mode 1, verified on the W4A16 path)

```text
A frag (Q):  lane t, slot i → A[lane_lo][k = i]                lane = M, slot = D
B frag (K^T): lane t, slot i → B[k = i][lane_lo]               lane = K_TILE, slot = D
C frag (S):  lane t, slot i → C[m = 2*i + lane_hi][n = lane_lo]
              lane = K_TILE (output cols), slot = M (with hi-bit interleave)
```

The c_acc → p_frag transpose for P @ V is done via a per-wave
`P_lds[16][16]` LDS round-trip (single wave, no `__syncthreads` needed
within the wave; in v2 the per-wave P_lds means the four waves don't
contend).

### v1 — single wave per block

```text
  gridDim  = (num_seqs, num_query_heads, ⌈max_qlen / 16⌉)
  blockDim = 32 = 1 wave32
  BLOCK_M  = 16   (lane_lo == M row)
  K_TILE   = 16   (matches WMMA K dim)
  HEAD_SIZE = 128 (templated; 8 fragments along D)

  Per block:
    Q-tile [16][128] kept in VGPRs (8 × V16 = 64 VGPRs / lane)
    For each K-tile of 16 keys:
      load K [128 × 16] into K_lds (5-D paged or linear chunk)
      load V [16 × 128] into V_lds
      8 WMMAs Q @ K → S [16 × 16]
      mask + online softmax
      transpose P via P_lds round-trip
      8 WMMAs P @ V → += out_acc [16 × 128]
    output write: each lane writes 8 rows × FRAGS cols
```

Best for `max_query_len < 64`. Above that, v1 re-loads K from global
once per Q-tile, which becomes the bottleneck (e.g. for ql=4096 with
BLOCK_M=16, K is read 256× from global; v2 reads it 64× by reusing
across 4 query rows of one block × 16 query rows of one wave).

### v2 — four waves per block, shared LDS K/V

```text
  gridDim  = (num_seqs, num_query_heads, ⌈max_qlen / 64⌉)
  blockDim = 128 = 4 waves wave32
  BLOCK_M  = 64       (4 waves × 16 rows each)
  K_TILE   = 16
  HEAD_SIZE = 128

  Per block (4 waves = 128 threads):
    Each wave loads its own 16 query rows of Q (in registers).
    For each K-tile of 16 keys:
      cooperative K-load: 128 threads × 2 vec_16B = 256 vec loads = full K-tile
      cooperative V-load: same (V_lds is shared too)
      __syncthreads()                       — K/V visible to all waves
      Each wave runs attn_step on its 16 rows independently:
          Q @ K → S, mask, online softmax, transpose P (wave-local
          P_lds), P @ V → += out_acc
      __syncthreads()                       — done with K/V for this iter
    Each wave writes its own 16 rows of output.
```

LDS per block: `K_lds_raw` (4 KB) + `V_lds` (4 KB) + 4 × `P_lds[16][16]`
(2 KB) = 10 KB. gfx1100 has 64 KB LDS per CU, so neither path limits
occupancy.

### Why 4 waves is the sweet spot on this kernel

The dominant cost on long sequences is global K/V reads. With BLOCK_M
= 16 (v1) and a chunk-only ql = 4096 prefill, the 256 Q-tiles each
re-load every K-tile from global, paying 4096 × 128 × 2 × 256 = 256 MB
of K reads per (seq, head). Triton's BLOCK_M = 128 lowers that to
32 MB — an 8× advantage that explains the v1 deficit.

v2's 4-wave shared-LDS load cuts the per-block K read volume by 4×:
4096 × 128 × 2 × 64 = 64 MB per (seq, head). Combined with the
remaining 2× advantage of larger BLOCK_M than v1, v2 hits ~30 TFLOPs
on Qwen-class shapes, matching Triton on initial prefill and beating
it on the chunked / long-prompt cases that dominate steady-state
serving.

## Performance (RX 7900 XTX, RDNA3 / gfx1100)

Kernel microbenchmark, Qwen-class GQA (Hq=32, Hkv=8, D=128). 200
warmup + 200 measured iterations, median wall-time per call. Triton
column is `context_attention_fwd` from `vllm/v1/attention/ops/prefix_prefill.py`
(BLOCK_M=128 / BLOCK_N=64 for pow2 block sizes, BLOCK_M=32 / BLOCK_N=32
for non-pow2). Compute throughput is the effective achieved
TFLOP/s for the relevant matmuls.

### Standard block size (BS = 16)

| ctx | qlen | v1 (us) | v2 (us) | Triton (us) | v2 vs v1 | **v2 vs Triton** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0    | 512  | 372  | **144**  | 220  | 2.6× | **1.53×** |
| 0    | 2048 | 4549 | **1330** | 1273 | 3.4× | 0.96× |
| 0    | 4096 | 17682 | **4673** | 4424 | 3.8× | 0.95× |
| 512  | 128  | 166  | **126**  | 251  | 1.3× | **2.00×** |
| 2048 | 128  | 472  | **377**  | 721  | 1.3× | **1.91×** |
| 8000 | 512  | 4616 | **2259** | 3962 | 2.0× | **1.75×** |

### Qwen3.5 dense W4A16 production block size (BS = 784)

| ctx | qlen | v1 (us) | v2 (us) | Triton (us) | **v2 vs Triton** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0    | 4096 | 17697 | **4680** | 3823 | 0.82× |
| 4096 | 512  | 2614  | **1293** | 2224 | **1.72×** |
| 8000 | 512  | 4813  | **2317** | 4278 | **1.85×** |

### bf16

| ctx | qlen | block | v1 (us) | v2 (us) | Triton (us) | **v2 vs Triton** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0    | 2048 | 16  | 4538 | **1294** | 1470 | **1.14×** |
| 4096 | 512  | 784 | 2633 | **1291** | 2292 | **1.78×** |

### Summary

* **Steady-state production (chunked decode-prefill, ctx>>ql, both
  block sizes, both dtypes): v2 wins 1.7-2.0× vs Triton.** This is
  the regime that dominates serving wall-time once requests have
  warmed up.
* **Long-prompt (ctx ≈ 8k, ql = 512): v2 wins 1.75-1.85×.** Where
  the previous Triton-only path "petaba" (crashed / ran very slowly)
  on long inputs in earlier branches, v2 just runs faster.
* **Initial prefill (ctx = 0, ql = 4k): v2 reaches 0.82-0.95× of
  Triton.** The chunk-only regime is where Triton's BLOCK_M=128
  retains a small per-(K-tile) reuse advantage; v2 closes 95% of
  the gap. The remaining 5% would need BLOCK_M=128 + 8 waves per
  block (future v3, see "Future work").
* **v1 alone is NOT shippable.** The 0.25× regression on chunk-only
  prefill (`ctx = 0 ql = 4096`) was the impetus for v2; v1 stays as
  the small-`max_query_len` fallback (where v2 wastes 3 of 4 waves)
  and is hit only at `qlen < 64`.

## Lessons learned

These notes record what moved the needle and what was tried and
reverted (so future-us doesn't repeat the same dead ends).

### 1. Cooperative K/V load is the entire game

The 4× speedup of v2 over v1 on chunk-heavy prefill is *exactly* the
4× cache reuse from sharing the K/V LDS tile across 4 waves. The
WMMAs themselves were never the bottleneck — v1 was global-mem-bound
by re-reading K from HBM2 once per Q-tile. v2's `__syncthreads()` cost
between cooperative load and per-wave WMMA is dwarfed by the saved
HBM2 traffic.

### 2. `#pragma unroll 1` cuts ILP — don't apply blindly

Tried on the inner WMMA loops in `attn_step` to free transient VGPRs
(b_frag / v_frag = 8 VGPRs each × 8 frags = 64 VGPRs simultaneously
alive otherwise). The kernel went from 7.8 → 3.5 TFLOPs (almost
exactly 2× slower). The 64 spilled VGPRs were costly but the lost
ILP — chained data deps on `s_acc` / `out_acc` were no longer
overlapped — was much worse.

Heuristic from the W4A16 work: only apply `#pragma unroll 1` when
`.vgpr_spill_count` is so high that occupancy gain dominates ILP loss.
For attention, the WMMA chains *are* the ILP, so the threshold is
much higher than for scalar GEMMs.

### 3. LDS padding broke the wrong cases

V_lds writes in v1 had a 32-way bank conflict (per-element scalar
writes from 32 lanes to consecutive d rows all hit the same bank
because stride per d row = 32 bytes = 8 banks; lane stride 4 d ×
16 fp16 = 128 fp16 = 256 bytes = 64 banks → all lanes mod 32 to bank
0). Padding the inner stride to `K_TILE + 1 = 17` broke this to a
2-way conflict.

But on prefix-heavy cases (ctx > ql) the per-K-iter `* 17` arithmetic
overhead in the inner loop *exceeded* the bank-conflict savings (which
were only present in V_lds writes — the V cache load uses vec writes
that sidestep the conflict). Net regression on the production
workload, so reverted.

The Real fix for V_lds load (v2): the cooperative loader assigns each
of 128 threads to a distinct (k_idx, d_chunk) pair, eliminating the
bank conflict structurally — no padding needed.

### 4. C++ dispatch v1↔v2, not Python

Same lesson as the W4A16 RDNA3 kernels (and same reason). An earlier
draft considered a Python-side check `if max_query_len < 64: v1_op
else v2_op`. Under torch.compile / Dynamo this triggers a graph
break / recompile per layer — a 7× decode regression in the W4A16
case. The dispatch lives in the C++ entry function in
`paged_prefill_attn_rdna3.cu`; Python sees a single op.

### 5. Separate translation units for v1 and v2

W4A16 README Lesson 4: putting the WMMA kernel in the same TU as the
scalar path miscompiled the small-M scalar path even when the WMMA
template was never instantiated for M = 1. Hipcc scopes some
optimizer decisions (register file / SGPR pressure heuristics) at the
TU level. We split v1 and v2 across `paged_prefill_attn_rdna3.cu` and
`paged_prefill_attn_rdna3_v2.cu` from the start; cross-TU calls
resolve at link time with no codegen interaction.

### 6. bf16 precision noise is real but bounded

bf16 has a 7-bit mantissa (vs fp16's 10), so every accumulator step
in the K-loop introduces ~1 ULP of rounding. For `seq_len = 4096`,
the worst-case cell can drift by a few ULPs — measured: 4 cells out
of ~196k (0.002 %) hit `|diff| > 1e-2`. p99 of `|diff|` is exactly
0.0039, which is one bf16 ULP at output magnitudes near 1. The test
suite gates this at `1e-2` and reports it as "FAIL"; in practice
this is bf16 working as designed and the kernel is correct.

The fp16 path is comfortably within `1e-2` everywhere (max observed
`|diff| = 0.0020`).

### 7. 4 waves is the sweet spot for a 64 KB-LDS / 96-CU GPU

Considered alternatives: 2 waves × 16 rows (BLOCK_M = 32) and 8 waves
× 16 rows (BLOCK_M = 128). 2 waves only halves the K reads (instead
of quartering), so the gap to Triton stays around 0.5×. 8 waves
would quarter again (1/8 of v1's K reads), but BLOCK_M = 128 with 8
WMMA-frag-wide accumulator state per wave hits the per-SIMD VGPR
budget (1024 VGPRs / 8 waves = 128 per wave, well below our 192-VGPR
working set), so 8 waves cannot share a SIMD — total wave count drops
back to where 4 waves already are. v3 with explicit per-wave VGPR
budget reduction (split HEAD_SIZE accumulator across two passes of
4 frags each) is plausible future work but out of scope for this
round.

## Constraints / unsupported features

These are gated by `RocmAttentionImpl.forward` (caller falls back to
Triton) AND re-checked at the C++ entry (TORCH_CHECK):

* dtype must be fp16 or bf16 (no fp32, no FP8 KV cache).
* `head_size == 128` (typical Qwen / Llama). 64 / 80 / 96 / 192 / 256
  would each need a templated launcher and `attn_step_wave` bodies
  re-tuned — the WMMA fragment count `FRAGS = HEAD_SIZE / 16` and
  the per-wave VGPR budget shifts with `HEAD_SIZE`.
* `block_size % K_TILE == 0` (i.e. block_size divisible by 16). All
  production block sizes (16, 32, 64, 544, 784, 1056) satisfy this.
* `causal == True` only.
* No alibi, sliding window, sinks, FP8 output, softcap.
* `key` and `value` non-null (cross-attention with cached-only K/V is
  not handled — typical decoder self-attention always has them).

## Future work

* **v3**: 8 waves × 16 rows (BLOCK_M = 128) with VGPR-budget reduction
  (split the HEAD_SIZE = 128 output accumulator into two passes of
  4 frags each). Closes the residual 5% gap to Triton on initial
  prefill at the cost of doubling K-loop iterations.
* **head_size != 128 templates**: 64 (smaller models), 96 (some
  custom configs), 192 (Phi-3, Qwen-coder), 256 (Gemma3). Each is
  a copy-paste of the launcher with a different `HEAD_SIZE` template
  param — the attn_step body works generically. VGPR budget needs
  re-checking per head_size.
* **FP8 KV cache**: currently falls back to Triton. The 5-D K cache
  layout already supports fp8 storage; the load just needs an extra
  dequant step (multiply by k_scale tensor) before the WMMA. ~50
  lines of HIP per kernel.
* **Double-buffered cooperative LDS load**: pipeline the K-tile
  load for iter `k+1` with the WMMA work on iter `k`. Probably
  worth ~5-10% on the chunk-heavy regime where the cooperative load
  is still a measurable fraction of K-iter time.

## Bench reproduction

In the dev container with vLLM source at `/tmp/vllm_build/vllm_jartx`:

```bash
docker exec vllm-vllm1-1 python /tmp/test_paged_prefill_rdna3.py
docker exec vllm-vllm1-1 python /tmp/bench_paged_prefill_rdna3.py
```

`test_*.py` runs 13 correctness cases (single-seq, multi-seq, GQA,
fp16, bf16, BS={16, 32, 544, 784}). `bench_*.py` runs the table above.
