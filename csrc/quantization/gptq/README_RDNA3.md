# RDNA3 W4A16 GPTQ Kernels (gfx1100)

Native HIP W4A16 GPTQ kernels for AMD RDNA3 (RX 7900 XT/XTX, gfx1100/1101/
1102). Drop-in replacement for `ExllamaLinearKernel` and
`TritonW4A16LinearKernel` on the ROCm path; adds bf16 support that ExLlama
lacks. Two compute paths share a single Python entry point:

* a hand-tuned **scalar dot-product kernel** for decode and any M < 16, with
  fp16 using `v_dot2_f32_f16` (`__builtin_amdgcn_fdot2`) and bf16 widening to
  fp32 to sidestep the missing `v_pk_fma_bf16` on gfx11;
* a **`v_wmma_f32_16x16x16` matrix-instruction kernel** for bf16 prefill and
  M ≥ 16 batches, with K-split + packed-CAS atomic write-back to multiply
  wave count.

Registered as `RDNA3W4A16LinearKernel` in
`vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py` ahead of
the Triton kernel for the ROCm priority list; falls through on non-RDNA3
ROCm devices via `can_implement()` gating on
`vllm.platforms.rocm.on_gfx11()`.

## Files

| File | Role |
| --- | --- |
| `q_gemm_rdna3.cu` | Scalar dot-product kernel + C++ dispatch entry `gptq_gemm_rdna3`. Templated on dtype and M_COUNT ∈ {1, 2, 4, 8}. |
| `q_gemm_rdna3_wmma.cu` | WMMA prefill kernel (`gemm_q4_wmma_kernel`) + 3 diagnostic ops. Uses K-split (gridDim.z) with shfl_xor + packed CAS-32 atomic. |
| `qdq_4_rdna3.cuh` | int4 dequant helpers shared between the two TUs. fp16 uses the exllama `0x6400` bit-trick; bf16 widens to fp32 (no `v_pk_fma_bf16` on gfx11). |

## Dispatch (decision tree)

`torch.ops._C.gptq_gemm_rdna3` is the **single Python entry point**. All
branching happens in C++ to keep `apply_weights` torch.compile-friendly:

```
gptq_gemm_rdna3(a, b_q_weight, b_qzeros, b_scales, b_g_idx, use_v2_format)
│
├── if dtype == bf16 and M >= 16 and N % 16 == 0 and K % 16 == 0
│       → gptq_gemm_rdna3_wmma  (v_wmma_f32_16x16x16_bf16_w32, prefill)
│
└── else
        → scalar dot-product kernel
              M = 1   → M_COUNT=1 (factored scale/zb fold for bf16)
              M = 2,3 → M_COUNT=2
              M = 4-7 → M_COUNT=4
              M = 8+  → M_COUNT=8
```

The bf16-only WMMA gating is set by both microbench and end-to-end
serving:

* **fp16 scalar already wins kernel-microbench** at every M tested. The
  fp16 dequant bit-trick (`0x6400`) keeps the path memory-bound, and
  `v_pk_fma_f16` is full rate on gfx11 — WMMA can't improve on that
  without a structural rewrite (multi-wave-per-block, LDS-shared A
  across waves).
* **fp16 WMMA dispatch was tested end-to-end** and came out a wash with
  fp16 scalar (`445.7` vs `440.7` tk/s at max-num-seqs=32, within
  run-to-run variance) despite a 47% kernel microbench advantage. See
  Lesson 10 — most kernel time in real serving is decode at M < 16
  (which falls back to scalar regardless of the dispatch guard) so the
  WMMA compute-density advantage doesn't translate to TPS. The
  `gptq_gemm_rdna3_wmma` op stays callable for fp16 in case a future
  workload makes it worthwhile.
* **bf16 scalar pays a tax** for the missing `v_pk_fma_bf16` (the bf16
  scalar overload of `dot22_8_f` widens to fp32 and uses `v_fma_f32`),
  and WMMA bypasses per-element FMAs entirely. WMMA wins consistently
  at M ≥ 16.

## Kernel architecture

### Compute paths overview

```
================================================================================
                    RDNA3 GPTQ W4A16 COMPUTE PATHS (gfx1100)
================================================================================

                       [ C++ ENTRY: gptq_gemm_rdna3 ]
                            (single Python entry)
                                      │
                ┌─────────────────────┴─────────────────────┐
                │  dtype == bf16                            │
                │  AND M >= 16                              │
                │  AND N % 16 == 0  AND  K % 16 == 0        │
                └────────┬─────────────────────────┬────────┘
                         │ FALSE                   │ TRUE
                         ▼                         ▼
================================================================================
        SCALAR PATH                       WMMA PATH
        gemm_q4_kernel_rdna3<T,M_COUNT>   gemm_q4_wmma_kernel<T>
        (decode + fp16 always             (bf16 prefill / batched,
         + small M < 16)                   M >= 16, compute-bound win)
================================================================================

 BLOCK STRUCTURE                          BLOCK STRUCTURE
  - 256 threads (8 waves x 32 lanes)       - 32 threads = 1 wave-group
  - all 256 lanes active                   - lanes 0..15 unique inputs
                                           - lanes 16..31 hold COPIES
                                             (RDNA3 doubled wave32 conv.)

 TILE PER BLOCK                           TILE PER BLOCK
  M_COUNT rows x 1024 N x 256 K            16 M rows x 16 N x K/K_SPLIT
  ┌─────────────────────────────┐          ┌──────────┐
M │      1024 N columns         │ K=256  M │ 16 N     │ K = full / K_SPLIT
  │   (4 N cols per thread)     │ (1/16    │  cols    │ (gridDim.z splits
  └─────────────────────────────┘  of K)   └──────────┘  K, atomic acc)

 GRID                                     GRID
  (N/1024, M/M_COUNT, K/256)               (N/16, M/16, K_SPLIT)
   gridDim.z splits K, atomic acc          K_SPLIT = 4 if K>=1024,
                                                      2 if K>=512,
                                                      1 otherwise

 CORE EXECUTION  (per outer K=32 iter)    CORE EXECUTION  (per K=16 step)
  for k in 0..256 step 32:                  for k_tile in K-segment step 16:
   ┌─────────────────────────────┐           ┌─────────────────────────────┐
   │ uint4 weight load (4 ints)  │           │ Cooperative dequant:        │
   │ bit-trick dequant 32 weights│           │   32 lanes do 32 dequants   │
   │   fp16: 0x6400 + half2 FMA  │           │     fp16: precise           │
   │   bf16: 0x4300 + fp32 FMA   │           │       (sub then mul, no     │
   │                             │           │        fp16 ULP amplif.)    │
   │ 4 dot22_8 vs LDS A          │           │     bf16: precise (same)    │
   │   fp16: v_dot2_f32_f16      │           │   -> 16 N x 16 K B-tile     │
   │   bf16: 8 fp32 FMAs         │           │      in LDS                 │
   │                             │           │ a_frag, b_frag (A row,      │
   │                             │           │   B col, mode 1)            │
   │                             │           │ v_wmma_f32_16x16x16         │
   └─────────────────────────────┘           └─────────────────────────────┘

 OUTPUT WRITE                             OUTPUT WRITE
  K/BLOCK_KN_SIZE blocks contend per        K_SPLIT == 1 → direct store
  output position                           K_SPLIT  > 1 → shfl_xor pair
   atomic_add_pk4 on a 64-bit                + packed CAS-32 (halves
   word (4 fp16/bf16 lanes per CAS)          atomic count, no intra-block
                                             contention; inter-block only)
```

### RDNA3 wave32 fragment layout (lane × slot mapping)

Hardcoded as **mode 1: A row-major + B col-major**, output
`C[m=2*i+lane_hi][n=lane_lo]`. Identified empirically with the
`gptq_gemm_rdna3_wmma_probe` op (still registered, see "Diagnostic ops"
below) — diagonal-A tests pass for both row/col B because A=I makes the
K-axis sum collapse, masking the bug; only random-A tests reveal mode 1
as the unique correct choice.

```
A frag input (16 fp16/bf16 elements per lane):
═══════════════════════════════════════════════
  lane t holds:   a_frag[i] = A[lane_lo][k = i]
                  with lane_lo = t & 15

  Lane axis encodes M (A's row index)
  Slot axis encodes K (depth axis aligned with B)


B frag input (16 elements per lane, COL-major):
════════════════════════════════════════════════
  lane t holds:   b_frag[i] = B[k = i][lane_lo]
                  with lane_lo = t & 15

  Lane axis encodes N (B's column index)
  Slot axis encodes K (same as A — enables per-lane inner products)


C frag output after WMMA (8 fp32 values per lane):
═══════════════════════════════════════════════════
  lane t, slot i  →  C[m = 2*i + lane_hi][n = lane_lo]
  with lane_hi = t >> 4 ∈ {0, 1}, lane_lo = t & 15

  Lane axis encodes N (output column)
  Slot axis encodes M (output row, with hi-bit interleave)

  L 0..15 (lane_hi=0)  → EVEN rows {0, 2, 4, ..., 14}
  L 16..31 (lane_hi=1) → ODD  rows {1, 3, 5, ..., 15}
```

### Concrete scaling — qkv-square (K = N = 4096) on a 96-CU GPU

```
Decode (M=1):
  Scalar: gridDim = (4, 1, 16) = 64 blocks × 8 waves = 512 waves
          → ~512/3072 wave slots used = 17 % of peak wave-slot occupancy
  WMMA:   gridDim = (256, 1, 4) = 1024 blocks × 1 wave = 1024 waves
          → still wastes 15/16 of the 16-row M tile per block on M=1
  → SCALAR wins cleanly on tiny M.

Prefill / batched (M=64):
  Scalar: 4 col-blocks × 8 row-blocks (M_COUNT=8) × 16 K-splits
          = 512 blocks × 8 waves
  WMMA:   256 col-blocks × 4 row-blocks × 4 K-splits (K_SPLIT=4)
          = 4096 blocks × 1 wave
  → WMMA wins on bf16 (compute-bound).
  → On fp16, scalar+fdot2 ties WMMA at end-to-end serving (memory-bound,
    bit-trick keeps scalar dequant essentially free).
```

## Performance (RX 7900 XTX, Qwen3.6-27B-GPTQ-W4A16-G32, tp=2)

End-to-end throughput on `evalscope perf` against the OpenAI-compatible
endpoint, openqa dataset, 50 requests, concurrency 100. Avg input tokens
~30, avg output tokens ~1800, so output dominates wall-time ~60:1 and
the steady-state decode regime determines TPS.

```
evalscope perf \
  --url "http://127.0.0.1/v1/chat/completions" \
  --parallel 100 --number 50 \
  --model INCCODER --api openai \
  --dataset openqa --stream
```

| Kernel                         | dtype  | max-num-seqs=8 | max-num-seqs=32 |
| ---                            | ---    | ---:           | ---:            |
| Triton W4A16                   | bf16   | 82.4 tk/s      | —               |
| Triton W4A16                   | fp16   | 83.2 tk/s      | —               |
| **ExLlama** (no bf16 support)  | fp16   | 255.0 tk/s     | 382.5 tk/s      |
| **RDNA3 W4A16** (this PR)      | bf16   | 205.3 tk/s     | 345.6 tk/s      |
| **RDNA3 W4A16** (this PR)      | fp16   | **270.2 tk/s** | **445.7 tk/s**  |
| RDNA3 W4A16 (fp16 → WMMA op)   | fp16   | —              | 440.7 tk/s      |

* **RDNA3 fp16 vs ExLlama fp16:** +5.96 % at seqs=8 (270.2 / 255.0),
  **+16.52 % at seqs=32** (445.7 / 382.5). ExLlama is the previous-best
  fp16 GPTQ kernel on ROCm; this PR beats it at both concurrencies
  measured.
* **RDNA3 vs Triton at seqs=8:** 3.25× on fp16 (270.2 / 83.2), 2.49× on
  bf16 (205.3 / 82.4). Triton's W4A16 path is 4-5× slower at this
  workload.
* **bf16 has no fp16-class baseline.** ExLlama doesn't support bf16 (the
  ExLlama bench was run with `--dtype float16` only). Triton bf16 is
  ~2.5× slower than RDNA3 bf16 at seqs=8.
* **fp16 → WMMA dispatch was tested and reverted** (last row): 440.7 vs
  445.7 tk/s — within run-to-run variance despite a 47% kernel
  microbench advantage. See Lesson 10.

### Bench commands

Server invocation (one of):

```bash
# Baselines: disable RDNA3 kernel to force ExLlama or Triton.
VLLM_DISABLED_KERNELS=TritonW4A16LinearKernel,ConchLinearKernel,RDNA3W4A16LinearKernel \
  vllm serve … --max-num-seqs {8|32} --dtype float16   # ExLlama fp16
VLLM_DISABLED_KERNELS=ConchLinearKernel,RDNA3W4A16LinearKernel \
  vllm serve … --max-num-seqs 8                         # Triton bf16 (default)
VLLM_DISABLED_KERNELS=ConchLinearKernel,RDNA3W4A16LinearKernel \
  vllm serve … --max-num-seqs 8 --dtype float16         # Triton fp16

# RDNA3 kernel (this PR): no disabled kernels needed; first in priority list.
vllm serve … --max-num-seqs {8|32}                      # RDNA3 bf16 (default)
vllm serve … --max-num-seqs {8|32} --dtype float16      # RDNA3 fp16
```

The full server flags used in the benches:

```
--gpu-memory-utilization 0.95 --max_model_len 262144 -tp 2
--attention-backend TRITON_ATTN --kv-cache-dtype fp8
--enable-prefix-caching --language-model-only
```

### Note on M vs max-num-seqs

`max-num-seqs` (vLLM scheduler param) and `M` (kernel matmul batch dim)
are not the same:

* During **decode** the engine batches up to `max-num-seqs` sequences per
  step, so M ≈ steady-state running batch size, capped at `max-num-seqs`.
* During **prefill chunks** M = `chunk_token_count` (≤ the
  `--max-num-batched-tokens` budget, default 2048). Far larger than
  `max-num-seqs` even for short prompts.
* During **mixed batches** M = `decode_seqs + prefill_chunk_tokens`.

In this long-output bench (output ~60× input) decode dominates wall-time
and the steady-state M is close to `max-num-seqs`. So `max-num-seqs=32`
mostly exercises the kernel at M≈32 in decode plus brief prefill spikes
at higher M. References elsewhere in this doc to "M=8" or "M=32" for
serving benches refer to this steady-state.

### VGPR / spill audit (gfx1100, `.note.AMDGPU.metadata`)

| Kernel           | VGPR before | VGPR after | Spill before     | Spill after |
| ---              | ---         | ---        | ---              | ---         |
| bf16 M_COUNT=1   | 82          | 41         | 0                | 0           |
| bf16 M_COUNT=2   | 138         | 56         | 0                | 0           |
| bf16 M_COUNT=4   | 192 (cap)   | 82         | 38 VGPR / 156 B  | **0**       |
| bf16 M_COUNT=8   | 192 (cap)   | 130        | 144 VGPR / 580 B | **0**       |
| fp16 M_COUNT=*   | unchanged   | unchanged  | 0                | 0           |
| bf16 WMMA        | 43          | 44         | 0                | 0           |
| fp16 WMMA        | 35          | 43         | 0                | 0           |

The scalar bf16 M_COUNT=8 template (decode batched at max-num-seqs ≥ 8
and the M < 16 portion of max-num-seqs ≥ 32) went from the 192-VGPR
cap with 144 VGPRs spilled to scratch every K-iteration, to 130 VGPRs
with zero scratch traffic. Method: see Lesson 8 (col-outer +
`#pragma unroll 1`).

The fp16 scalar path was already register-clean thanks to the `0x6400`
bit-trick + `v_pk_fma_f16` (1 cycle/packed-FMA), so only the
`atomic_add_pk4` and the `v_dot2_f32_f16` swap apply there.

Audit method (no GPU required, pure static analysis of the .so):

```bash
/opt/rocm-X/llvm/bin/llvm-objdump --offloading <vllm/_C*.so> > /tmp/objdump.txt
# Then for each gfx1100 bundle, dump notes:
/opt/rocm-X/llvm/bin/llvm-readobj --notes <bundle>
# Look for .vgpr_count, .vgpr_spill_count, .private_segment_fixed_size.
```

## Lessons learned

These are the changes that actually moved end-to-end TPS, plus the
attempts that didn't and were reverted (recorded so we don't repeat).

### 1. The bf16 fp32 dequant bypass is essential

`__hfma2(bf162_t, ...)` on gfx11 lowers to a slow fallback because gfx11
has **no `v_pk_fma_bf16`** in the ISA (only landed on CDNA3 / gfx94x
and later). Per-element it runs roughly 2× the cycle count of
`v_pk_fma_f16` on the same VALU.

Two helpers in the bf16 path widen to fp32 explicitly and stay there:

* `dot22_8_f(bf162_t (&dq)[4], const bf16_t* a)` — widens both bf16
  operands via free `(bits<<16)` left-shift, accumulates with
  `v_fma_f32` (full rate on RDNA3).
* `dequant_4bit_8_bf16_f32` (and `_bf16_q_only` for the M_COUNT=1
  factored path) — produces fp32 output directly from the int4 unpack,
  bypassing a second round of slow bf16 FMAs.

Both bf16 scalar paths run fp32 throughout for compute; only the
A-vector and the output write are bf16. The fp16 path keeps
`__hfma2(half2,...)` because `v_pk_fma_f16` IS native and full rate;
widening to fp32 there would cost VGPRs without speed.

### 2. Dispatch lives in C++, not Python

An earlier attempt put `if x.size(0) >= 16: wmma_op(...) else: scalar(...)`
inside `apply_weights`. Under torch.compile / Dynamo the size-comparison
guard triggered a graph break / recompile on every layer and decode
went 7× slower. Fix: branch in `gptq_gemm_rdna3`'s C++ entry; Python
sees a single op.

The same constraint kills `print()` for in-kernel debugging — Dynamo
can't trace builtin print in fullgraph mode. Use
`process_weights_after_loading` (called outside compile) or vLLM's
startup logs.

### 3. WMMA fragment layout is mode 1, not mode 0

A diagonal-A test passes for both `B row-major` and `B col-major`
fragment loadings because A=I makes the K-axis sum collapse. Random-A
reveals only mode 1 (`A row, B col, output [m=2*i+lane_hi][n=lane_lo]`)
implements A·B; mode 0 silently computes A·Bᵀ. The
`gptq_gemm_rdna3_wmma_probe` op is still registered for re-verification
on a new toolchain.

### 4. WMMA TU lives in its own file

A monolithic TU with both scalar and WMMA kernels miscompiled the M=1
scalar path even when the WMMA template was never instantiated for
M=1. Hipcc appears to scope some optimizer decisions (register file /
SGPR pressure heuristics) at the TU level. Splitting into separate
translation units (linked via standard cross-TU calls) restored scalar
binary identity to its tuned baseline.

### 5. BLOCK_KN_SIZE = 256 is the sweet spot

Tried 512 to halve atomic CAS count per output. bf16 gained 5-10% at
large M (atomic count halved), but fp16 decode regressed up to 40% on
qkv-square (M=1, only ~8 of 96 CUs saturated due to 16-wave blocks).
Reverted; the comment block at the top of `q_gemm_rdna3.cu` records
the experiment.

### 6. Atomic accumulation via packed CAS, no fp32 buffer

gfx11 has no `v_global_atomic_pk_add_{f16,bf16}`. The scalar kernel
writes to fp16/bf16 output directly via `atomic_add_pk4_{f16,bf16}` —
an `atomicCAS`-retry loop on a 64-bit word covering all 4 output
columns per row in a single `global_atomic_cmpswap_b64`. Saves M*N*4
bytes allocation + memset + epilogue cast pass that an fp32-accumulator
design would need, and halves the atomic count vs two `b32` CAS calls
per row. Alignment is guaranteed by `can_implement()` requiring
`partition_weight_shape[1] % 8 == 0` and `n` always a multiple of 4.

The WMMA K-split path uses a different scheme: a 32-bit packed CAS on
adjacent column pairs, with `__shfl_xor` between even/odd lanes so each
pair of cols goes through a single atomic — see Lesson 9.

### 7. bf16 M_COUNT=1 decode: factor scale/zb out of dequant

The default per-col dequant computes `dq[i] = q_f32[i] * scale + zb`
(8 fp32 FMAs per int32 weight × 4 N cols = 32 dequant FMAs), then dot
adds 8 fp32 FMAs per (m, n_col). At M_COUNT=1 this reduces algebraically:

```
accum = sum_i (q_f32[i] * scale + zb) * a[i]
      = scale * sum_i (q_f32[i] * a[i]) + zb * sum_i a[i]
```

Compute `sum_a = Σa[i]` once per K=8 step (shared across all 4 N cols)
and a per-col `partial = Σ(q_f32 * a)`, then fold `scale` and `zb`
into the accumulator with 2 FMAs per col. Drops 64 → 47 FMAs per int32
weight at M_COUNT=1 (−27%). Break-even at M_COUNT=2, so guarded behind
`if constexpr (M_COUNT == 1)`.

`dequant_4bit_8_bf16_q_only` produces unscaled fp32 q-values directly
from the bf16 bit-trick (0x4300 magic) — zero FMAs in the dequant
itself. Numerically safe because the entire factored path runs in fp32
(the gfx11 widen sidestep means we already pay fp32 acc).

This trick does **not** apply to fp16: the +1024 bias in the `0x6400`
bit-trick would cause catastrophic cancellation when subtracting
`(1024+zero)*scale*sum_a` from `scale*Σ(q_h2*a)` in fp16-precision
accumulators. Keep fp16 with its native bit-trick + `v_pk_fma_f16`.

### 8. VGPR pressure trumps ILP at the cap — `#pragma unroll 1` on col-loops

The bf16 j-loop originally declared `float dq[4][8]` outside the m-loop
(all 4 cols' dequant results alive simultaneously across the m-loop).
At M_COUNT=4/8 this hit the 192-VGPR cap with 38 / 144 VGPRs spilled
to scratch, and pinned occupancy at 5 waves/SIMD.

Restructured to col-outer with `dq` declared inside the col loop —
expecting the compiler to free the registers between iterations. **It
didn't.** With `#pragma unroll`, the AMDGPU optimizer expanded the 4
cols into a straight-line block where all 4 dq arrays remained alive
simultaneously to maximize ILP across cols. VGPR count was unchanged.

Adding `#pragma unroll 1` on the col loop forces a real 4-iteration
loop; the register allocator is then bound to recycle VGPRs across
iterations. Inner loops (m, i) stay `#pragma unroll`d for FMA
pipelining within a col.

Outcome (bf16 M_COUNT=8): 192 VGPRs + 144 spilled → 130 VGPRs / 0
spill. The same trick applied to the M_COUNT=1 factored path
(`q_f32[8]` inside the col loop instead of `q_f32[4][8]` outside)
dropped VGPRs 82 → 41.

**Caveat — does not apply universally.** Tried the same col-outer +
`#pragma unroll 1` on the **fp16 scalar path** expecting the +12 VGPRs
recovered (110 → ~98 at M_COUNT=8) to bump occupancy. Wall-time
regressed 247 → 200 tk/s (−19%) in the same evalscope bench. fp16 was
register-clean (no spills), so the trade-off was pure ILP loss vs
marginal occupancy gain — and the AMD compiler had been interleaving
the 4 independent col-dot accumulator chains, which `unroll 1` killed.
Reverted. Heuristic: only apply when the audit shows
`.vgpr_spill_count > 0` or `.private_segment_fixed_size > 0`. Without
spills, trust the compiler's ILP scheduling.

### 9. WMMA single-wave block: every stall blocks the whole block

`gemm_q4_wmma_kernel` launches with `dim3 block(32)` = exactly one
wave32. There is no second wave to overlap with stalls, so long-latency
operations in the K-loop directly serialise the whole block. Two such
operations were the dominant cost on the initial WMMA path:

* **`__syncthreads()` × 2 per K-iteration.** A single-wave block has
  no inter-wave concurrency to synchronize; the explicit barrier
  emits `s_barrier` for nothing. The compiler-inserted `s_waitcnt
  lgkmcnt(0)` already orders dependent `ds_write`/`ds_read` pairs
  within a wave (including across-lane reads — e.g., lane 0 reading
  what lane 16 wrote into `b_lds[8..15][0]`). Both barriers were
  removed; comment in `q_gemm_rdna3_wmma.cu` explains why this is
  safe in single-wave mode.

* **Sequential A-row global loads.** Each lane originally loaded its
  16 fp16/bf16 A elements with 16 separate `global_load_b16`. With
  one wave there is no other wave to schedule during the wait — every
  load is a serial stall. Replaced with a 32-byte bulk
  `__builtin_memcpy` lowered to two `global_load_b128` instructions.

  *Implementation note:* memcpy into the **whole vector**
  (`__builtin_memcpy(&a_frag, ptr, 32)`), not into individual elements.
  `&a_frag[0]` on `ext_vector_type` is not reliably a valid C pointer
  across compiler versions; building this on hipcc 7.2.1 fails with
  the per-element form.

These two changes were the headline contributor to the bf16 WMMA path
being usable at high concurrency — same algorithm, same VGPR class,
just removing the serial stalls that the single-wave layout had no
way to hide.

### 10. WMMA K-split with packed atomic write — closing the wave-count gap

After Lesson 9, the WMMA bf16 path still ran with low CU saturation:
at typical Qwen-class shapes only ~512 waves were in flight vs the
fp16 scalar path's ~2048 at the same workload. The fp16 scalar kernel
already uses `gridDim.z` to split K into 16 segments with atomic
write-back per output cell; this lesson replicates the same idea in
the WMMA kernel at K_SPLIT ≤ 4 (capped to bound atomic contention).

**Mechanism.** The WMMA kernel takes `gridDim.z = K_SPLIT` and each
block processes a contiguous K-segment `[k_start, k_end) =
[blockIdx.z * K/K_SPLIT, (blockIdx.z+1) * K/K_SPLIT)`. K_SPLIT blocks
contribute a partial sum to each 16×16 output tile and the epilogue
atomically combines them. Total wave count multiplies by K_SPLIT —
for the typical Qwen-class K=4096 case, 512 → 2048 waves at
`gridDim.z = 4`.

**Atomic contention management.** gfx11 has no native packed atomic
add for fp16/bf16, so the epilogue uses a CAS-32 retry loop on a
uint32 word covering 2 adjacent fp16/bf16 lanes. Without further care
this would have intra-block contention: lanes `lane_lo` and
`lane_lo+1` would target the same uint32. To avoid this, the even
lane uses `__shfl_xor(c_acc[i], 1)` to pull the odd lane's value
across the wave, packs both values into a `half2`/`bf162`, and issues
a single atomic CAS — the odd lane skips the write. This halves the
atomic count and eliminates intra-block contention. The remaining
4-way inter-block contention from K_SPLIT=4 is the unavoidable cost.

**K_SPLIT heuristic.** `compute_wmma_k_split(K)` returns 4 if
`K ≥ 1024 and K % 64 == 0`, 2 if `K ≥ 512 and K % 32 == 0`, else 1.
K_SPLIT must divide `K/16` cleanly. Typical Qwen shapes (K ∈ {4096,
5120, 11008}) all hit K_SPLIT = 4. K_SPLIT = 1 falls back to direct
write with no atomics.

**Output zero-init.** With K_SPLIT > 1 multiple writers contend on
each cell, so the output tensor is allocated via `torch::zeros`
instead of `torch::empty`. K_SPLIT == 1 keeps `torch::empty`
(every cell assigned exactly once). The conditional adds zero-init
cost only where it's actually needed — see the `need_zero_init` line
in `gptq_gemm_rdna3_wmma`.

**Outcome.** K-split is what takes the WMMA bf16 path to its
end-to-end serving number in the table above; without it the WMMA
path was massively undersaturating CUs at high concurrency. VGPR
count is unchanged (`atomic_add_pk_*` adds little state), no spills
introduced.

**Why we don't try to close the residual fp16-scalar / bf16-WMMA gap
further.** Closing it needs a structural change — multi-wave per
block with LDS-shared A, or a larger N tile per WMMA invocation —
which is out of scope for this round. At the current bf16 numbers
(345.6 tk/s at max-num-seqs=32, vs broken/very-slow on the initial
branch) the path is shippable as-is.

### 11. fp16 scalar dot uses `v_dot2_f32_f16` directly

The fp16 `dot22_8_f` originally accumulated in `half2` and cast to
fp32 at the end:

```cpp
half2 result = {};
for (i = 0..3) result = __hfma2(dq[i], a2[i], result);
return __low2float(result) + __high2float(result);
```

That's 4 packed FMAs + 2 fp16→fp32 casts + a fp32 add per call. RDNA3
has `v_dot2_f32_f16` (intrinsic `__builtin_amdgcn_fdot2`) which does
`fp32 += a.x*b.x + a.y*b.y` in one instruction with the accumulator
staying in fp32. Replacing the loop with the intrinsic eliminates the
trailing 2× `v_cvt_f32_f16` + `v_add_f32`:

```cpp
float result = 0.0f;
for (i = 0..3) result = __builtin_amdgcn_fdot2(dq[i], a2[i], result, false);
return result;
```

**Why hipcc doesn't peephole this for us.** ISA verification on the
M_COUNT=8 kernel before the swap: 0 `v_dot2_f32_f16` instructions, 256
`v_cvt_f32_f16`, 218 `v_add_f32`. hipcc 7.2.1 does not recognise the
`__hfma2 + cast + add` pattern as fdot2-equivalent (presumably because
the half2 accumulator's bit-exact behaviour differs from fp32
accumulation when intermediates would round). The explicit intrinsic
is required.

**Wall-time outcome:** the fp16 numbers in the table above (270.2 at
seqs=8, 445.7 at seqs=32) are with this swap. The kernel-microbench
showed ~47% reduction in per-call time at M=32, K=N=4096; only ~10%
materialises in serving TPS because the kernel is not the only thing
on the critical path at this concurrency.

**Numerical bonus.** The new form keeps fp32 precision throughout the
dot. The old form accumulated 8 muladds in fp16 (10-bit mantissa)
before casting to fp32, which could lose ~3 bits of precision on
borderline-magnitude intermediates.

**Not applied to bf16.** gfx11 has no `v_dot2_f32_bf16` (CDNA3+
only). The bf16 `dot22_8_f` overload already widens to fp32 by
left-shifting the bf16 bits and accumulates with `v_fma_f32` (full
rate), which is the right shape for this hardware.

**Not applied to the fp16 → WMMA dispatch either.** Direct
microbench showed fp16 WMMA was ~47 % faster than fp16 scalar+fdot2
at M=32, but end-to-end serving was a wash (440.7 vs 445.7 tk/s,
within run-to-run variance — see the last row of the perf table).
Most kernel time in real serving is decode at M < 16 (which falls
back to scalar regardless of the dispatch guard) so the WMMA
compute-density advantage doesn't translate to TPS. The dispatch
stays bf16-only; the standalone op `gptq_gemm_rdna3_wmma` remains
callable for fp16 in case a future workload makes it worthwhile.

## Diagnostic ops (registered, callable from Python)

For kernel correctness debugging only — not used in production paths:

* `torch.ops._C.gptq_gemm_rdna3_wmma_probe(a, b, mode)` — runs one
  WMMA on fp16 16×16 inputs under `mode ∈ {0..3}` fragment-load
  hypotheses, dumps per-lane c_acc to `fp32[32, 8]`. Used to identify
  the wave32 fragment layout (mode 1: A row-major + B col-major +
  output `[m=2*i+lane_hi][n=lane_lo]`).
* `torch.ops._C.gptq_gemm_rdna3_wmma_dump(...)` — full-pipeline c_acc
  dump from a single 16×16 output tile after dequant + LDS + WMMA.
* `torch.ops._C.gptq_gemm_rdna3_wmma_lds_check(...)` — dequant +
  LDS-write only, dumps the b_lds tile back as fp16 for visual sanity
  check.

## Known limitations / future work

* **fp16 prefill is scalar.** WMMA fp16 is not auto-dispatched because
  end-to-end serving is a wash with scalar+fdot2 (Lesson 11). Reaching
  a clear win would likely need: 16M × 128N tile, LDS-shared A across
  waves, K-pipelining (double-buffered LDS), `uint4` vector loads for
  weights. Estimated 3-5× kernel-microbench improvement to translate
  into a serving win at M ≥ 64.

* **No fp16 → fp32-output-buffer experiment.** Replacing `atomicCAS`
  with fp32 atomic-add + epilogue cast might save 5-15 % on fp16
  decode if CAS retries are dominant. Untested.

* **fp16 decode CU saturation.** At `K=N=4096, M=1` the scalar fp16
  kernel launches 64 blocks against 96 CUs — only ~17 % of wave slots
  utilized (512 of 96×32 = 3072 slots). Increasing block count would
  give the load/store scheduler more outstanding requests in flight.
  Reducing `THREADS_X` from 256 to 128 (with `BLOCK_KN_SIZE=256`
  unchanged, each thread loading 2 K elements) would double
  `gridDim.x`. Not yet implemented because the M_COUNT={1,2,4,8}
  templates would need re-tuning together.
