## Problem

The RDNA3 (gfx11) W4A16 GPTQ kernels accumulate split-K partials in
bf16/fp16 via CAS atomics. The narrowing happens **before** accumulation,
and low-precision addition is not associative — so whichever split block
wins the CAS first changes the result. That numerical error is present on
**every affected forward pass**, in every decoding configuration.

Greedy decoding is merely where it becomes visible: at temperature 0 the
rounded logit wobble can flip `argmax` between repeated identical
generations (the trail in #50603). Under sampling (e.g. T=0.6) the same
error hides under the sampler's own noise while still sitting under every
reported metric.

## Root cause

```
FP32 per-block partial
  -> bf16/fp16 narrowing (8 / 11 mantissa bits)
  -> CAS-atomic accumulation in execution-dependent order
```

gfx11 has no packed bf16/fp16 atomic, so the epilogue emulated one with a
CAS retry loop — which also made the accumulation order scheduler- and
contention-dependent.

## Fix

Keep the compute kernels and dispatch unchanged; replace only the epilogue
on both paths:

```
FP32 per-split partials
  -> fixed ascending-z FP32 reduction
  -> one final bf16/fp16 cast
```

* Every path writes each output element exactly once; `k_split == 1`
  (WMMA) and `z_count == 1` (scalar) keep/store directly — no scratch, no
  reduce, no atomics.
* The FP32 partials scratch is `at::empty`: coverage is total by
  construction (every reducer-visible `(z, m, n)` slot has exactly one
  writer in all seven WMMA variants and the scalar kernel; the invariant
  is documented at `alloc_wmma_partials`), so there is no zero-fill pass.
* Scratch is per-call via the caching allocator, row-tiled so the bound is
  independent of the caller's M.

## Accuracy (W7900/gfx1100, FP32 dequantized reference)

Max abs error, synthetic uint4b8/group-128 weights, K=4096:

| path | dtype | legacy CAS | this PR | factor |
|------|-------|-----------:|--------:|-------:|
| scalar | bf16 | 0.24–0.37 | 0.0625 | 4–6x |
| WMMA | bf16 | 0.15–0.23 | 0.09–0.11 | ~2x |
| WMMA | fp16 | 0.024–0.038 | 0.011–0.018 | ~2x |
| scalar | fp16 | ~0.95 | ~0.95 | 1x* |

bf16 is the headline; fp16 carries 11 mantissa bits so the removed CAS
rounding costs it ~8x less (as expected). *scalar fp16's remaining error
is the classic exllama dequant bit-trick (per-group offset constant
rounded to fp16) — pre-existing, unchanged by this PR.

The earlier "0.028 → 0.0061" measurement is the same bf16 improvement
factor on real Muse weights. bf16 residual error now sits at the final
output cast (1 ulp); WMMA at the B-tile narrowing.

## Determinism

Fixed inputs now produce bit-identical outputs on every call: 20/20
benchmark shapes repeatable; real Muse q_proj at M=1 gives 1 distinct
result / 100 calls (previously up to 200/200 distinct). Repeatability
alone is not correctness, so the tests also check values against the FP32
reference — a stale `at::empty` scratch slot would be bitwise-repeatable
and wrong, and is caught by the tolerance analysis in the test module.

## Performance (W7900/gfx1100, median of 100 CUDA-event timings)

Rows are **bf16**; fp16 differs in dispatch (bf16 reaches WMMA at
M ≥ 16, fp16 not until M ≥ 64 — at M=16 fp16 is still on the scalar path):

| M | N | K | path | k_split | scratch MB | before µs | after µs | Δ |
|--:|---:|---:|------|--------:|-----------:|----------:|---------:|---:|
| 1 | 4096 | 4096 | scalar | 16 | 0.25 | 44.5 | 39.7 | -10.8% |
| 8 | 4096 | 4096 | scalar | 16 | 2.0 | 61.8 | 57.2 | -7.5% |
| 16 | 4096 | 4096 | wmma 16x16_1w | 4 | 1.0 | 58.8 | 53.7 | **-8.7%** |
| 64 | 4096 | 4096 | wmma 64x64_4w | 4 | 4.0 | 120.5 | 112.0 | **-7.0%** |
| 128 | 4096 | 4096 | wmma 128x64_k32 | 4 | 8.0 | 123.5 | 114.7 | **-7.2%** |
| 512 | 4096 | 4096 | wmma 128x64_k32 | 4 | 32.0 | 378.4 | 350.8 | **-7.3%** |
| 512 | 4096 | 6656 | wmma 128x64_k32 | 4 | 32.0 | 554.7 | 527.4 | -4.9% |
| 512 | 25600 | 6656 | wmma 128x64_k32 | **1** | **0** | 3070.7 | 3020.1 | -1.6% |

fp16 WMMA rows move the same way (M=64: 104.7→97.8, M=128: 116.1→107.9,
M=512: 353.1→327.6). Full tables for both dtypes incl. N/K/k_split/scratch
per row: `validation-54706-jartx-w7900/05-benchmark.md` in this branch.

Notes:
- "before" = this branch's original HEAD (zero-filled WMMA scratch);
  "after" = with the dead zero-fill removed (`at::zeros` → `at::empty`
  once total coverage was proven). Same-M/different-N matters: at M=512,
  N=4096 → k_split=4 (32 MB scratch) while N=25600 → k_split=1 (none).
- Versus a locally restored legacy-CAS control build, the deterministic
  epilogue is now at parity or **faster** for prefill shapes (bf16 M=16:
  53.7 vs 55.3 µs); only M=1 decode pays a residual ~4–10% (scratch
  round-trip at tiny output) — the price of bit-reproducibility.
- Numerics are bit-identical before/after the memset removal on all 20
  shapes (zero-init was dead traffic).

## Why not FP32 atomics?

gfx11 does have `global_atomic_add_f32`; an FP32 scratch accumulated with
native atomics plus one cast pass would remove nearly all the rounding
error with no reduce kernel. We still chose the fixed-order reduce:
FP32 addition is also non-associative, so atomic accumulation remains
order-dependent — vastly better numerically, but not bit-reproducible.
Since bit-exact reproducibility is an explicit goal of this PR, the
fixed ascending-z reduction is the design that guarantees it.

## Tests

`tests/kernels/quantization/test_rdna3_w4a16_determinism.py` — 20 tests:

* bit-repeatability through the public op (scalar/WMMA × bf16/fp16),
  with the scalar regression at K=4096 (16 concurrent writers — well
  inside the old failure regime, whose onset was 2–4 writers);
* FP32-reference correctness for scalar and WMMA, split-K and
  direct-store (k_split == 1 / z_count == 1) paths, both dtypes, with
  tolerances derived from each path's rounding structure;
* test hygiene: repo-standard `dist_init` fixture (no `__enter__` leak,
  no hardcoded MASTER_PORT).

On a locally restored legacy-CAS build, 8 of the 20 tests fail (7
repeatability + the scalar-bf16 reference check at max_abs 0.30 vs 0.10
tolerance); on this PR, 20/20 pass.

## Validation

- W7900D / gfx1100, ROCm 7.14 runtime, torch 2.14; upstream base
  `40b2f62061575905aaac8bc360eaea62a4baeb67`
- targeted tests 20/20 PASS (bf16 + fp16, scalar + WMMA)
- FP32-reference correctness + A/B legacy control: see above
- benchmark matrix: dtype × M{1,8,16,64,128,512} × N{4096,25600} ×
  K{4096,6656} with path/k_split/scratch per row (evidence directory in
  this branch)
- independent RX 7900 XT evidence from @cadamcat (A/B table in #50603)
- Muse eager ctx512: 1 unique / 8 greedy generations; 0
  same-input/different-output events across 16,640 intercepted W4A16
  GEMM calls

## Scope

gfx11 / RDNA3 W4A16 only; no attention-routing changes. The residual
Muse eager/8192 divergence after this fix was traced to ROCm paged
attention V-cache tail consumption and is fixed by #53856. Investigation
trail: #50603; full earlier evidence in the
`validation-50603/rdna3-w4a16-upstream-cleanup` tree linked there.

## AI assistance disclosure

AI assistance was used for code iteration, validation orchestration, and
drafting. I reviewed the final diff, kernel behavior, test results,
performance data, and evidence and can explain the change end to end.
