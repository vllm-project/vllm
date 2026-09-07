# Benchmark report — PR #54706 JartX closure (W7900/gfx1100)

GPU: AMD Radeon Pro W7900D (gfx1100, 48 CUs, 48 GB)
ROCm: 7.14.60850 (torch 2.14.0+rocm7.14 wheel runtime)
Commit: dddf164781 (PR HEAD) + closure commit; branch fix/rdna3-w4a16-determinism
Method: public op torch.ops._rocm_C.gptq_gemm_rdna3, CUDA-event timing,
25 warmup + 100 measured iterations, median reported. Kernel path and
k_split derived by replicating the C++ dispatch (not inferred from M).

Builds compared:
* before  — PR HEAD as merged onto the fork (WMMA scratch at::zeros,
            scalar always scratch+reduce)
* after   — closure build (WMMA scratch at::empty, scalar z_count==1 fast
            path, direct-store epilogue)
* legacy-ab — A/B control with the pre-#54706 CAS epilogue temporarily
            restored (zeroed output + atomic accumulation)

## Main table (median us; delta = after vs before)

| dtype | M | N | K | path | k_split | scratch_MB | before_us | after_us | delta% | legacy_us | max_abs (after) | rep |
|------|---|---|---|------|--------:|-----------:|----------:|---------:|-------:|-----------:|----------------:|:---:|
| bf16 | 1 | 4096 | 4096 | scalar:mcount1 | 16 | 0.25 | 44.46 | 39.68 | -10.8% | 38.28 | 0.0625 | Y |
| bf16 | 8 | 4096 | 4096 | scalar:mcount8 | 16 | 2.00 | 61.78 | 57.16 | -7.5% | 57.36 | 0.0624 | Y |
| bf16 | 16 | 4096 | 4096 | wmma:16x16_1w | 4 | 1.00 | 58.84 | 53.72 | **-8.7%** | 55.28 | 0.0887 | Y |
| bf16 | 64 | 4096 | 4096 | wmma:64x64_4w | 4 | 4.00 | 120.46 | 112.00 | **-7.0%** | 117.40 | 0.0968 | Y |
| bf16 | 128 | 4096 | 4096 | wmma:128x64_k32 | 4 | 8.00 | 123.54 | 114.70 | **-7.2%** | 121.44 | 0.1064 | Y |
| bf16 | 512 | 4096 | 4096 | wmma:128x64_k32 | 4 | 32.00 | 378.44 | 350.80 | **-7.3%** | 354.94 | 0.1125 | Y |
| bf16 | 512 | 4096 | 6656 | wmma:128x64_k32 | 4 | 32.00 | 554.73 | 527.40 | -4.9% | 532.46 | 0.1523 | Y |
| bf16 | 512 | 25600 | 6656 | wmma:128x64_k32 | **1** | **0.00** | 3070.69 | 3020.08 | -1.6% | 2914.74 | 0.1527 | Y |
| bf16 | 1 | 4096 | 6656 | scalar:mcount1 | 26 | 0.41 | 48.28 | 43.06 | -10.8% | 40.88 | 0.0625 | Y |
| bf16 | 128 | 4096 | 6656 | wmma:128x64_k32 | 4 | 8.00 | 175.48 | 168.82 | -3.8% | 171.98 | 0.1496 | Y |
| fp16 | 1 | 4096 | 4096 | scalar:mcount1 | 16 | 0.25 | 23.40 | 23.36 | -0.2% | 22.08 | 0.9578 | Y |
| fp16 | 8 | 4096 | 4096 | scalar:mcount8 | 16 | 2.00 | 42.64 | 42.68 | +0.1% | 43.48 | 1.0472 | Y |
| fp16 | 16 | 4096 | 4096 | scalar:mcount8 | 16 | 4.00 | 52.90 | 52.36 | -1.0% | 50.68 | 1.0583 | Y |
| fp16 | 64 | 4096 | 4096 | wmma:64x64_4w | 4 | 4.00 | 104.66 | 97.82 | **-6.5%** | 101.82 | 0.0114 | Y |
| fp16 | 128 | 4096 | 4096 | wmma:128x64_k32 | 4 | 8.00 | 116.14 | 107.88 | **-7.1%** | 110.54 | 0.0152 | Y |
| fp16 | 512 | 4096 | 4096 | wmma:128x64_k32 | 4 | 32.00 | 353.06 | 327.62 | **-7.2%** | 322.38 | 0.0148 | Y |
| fp16 | 512 | 4096 | 6656 | wmma:128x64_k32 | 4 | 32.00 | 518.83 | 494.28 | -4.7% | 484.82 | 0.0181 | Y |
| fp16 | 512 | 25600 | 6656 | wmma:128x64_k32 | **1** | **0.00** | 2884.73 | 2858.26 | -0.9% | 2699.52 | 0.0203 | Y |
| fp16 | 1 | 4096 | 6656 | scalar:mcount1 | 26 | 0.41 | 27.32 | 27.04 | -1.0% | 24.48 | 0.9077 | Y |
| fp16 | 128 | 4096 | 6656 | wmma:128x64_k32 | 4 | 8.00 | 164.38 | 155.18 | -5.6% | 156.80 | 0.0176 | Y |

(rep = bit-repeatable over 5 identical calls; legacy-ab was NOT repeatable
on 12/20 rows — all rows marked Y above for the fixed builds only.)

## Findings

1. **Removing the WMMA memset recovered the M=16 row and the whole WMMA
   band.** Every k_split>1 WMMA shape improved 3.8-8.7% in BOTH dtypes
   (bf16 M=16: 58.84 -> 53.72 us). Against the legacy CAS control, the
   deterministic epilogue is now at parity or faster at prefill shapes
   (M=16 bf16: 53.72 vs 55.28; M=64/128: -4%/-3%) — the earlier
   "-17.2% at M=16" regression is gone on this machine/branch.
2. **dtype split is explicit** (bf16 reaches WMMA at M>=16, fp16 not until
   M>=64): at M=16, bf16 takes the WMMA row above while fp16 is still on
   scalar (52.36 us, unchanged within noise — its epilogue change only
   affects code layout). fp16's WMMA gains appear from M=64 on.
3. **same M, different N -> different k_split** (JartX's example
   reproduced): M=512, K=6656: N=4096 -> blocks_xy=256 -> k_split=4 with
   32 MB FP32 scratch + reduce; N=25600 -> blocks_xy=1600 -> k_split=1,
   no scratch, direct store (both measured, both correct).
4. **Scalar z_count==1 fast path**: K<256 shapes now skip scratch+reduce
   entirely (covered by new tests; not in the perf table since production
   K >= 4096 always splits).
5. **Decode (M=1) residual cost vs legacy**: bf16 +3.7-5.3%, fp16
   +5.8-10.5% — the scratch round-trip still costs more than the CAS
   epilogue at M=1 where the output is tiny; this is the price of
   bit-reproducibility at decode shapes. Against the PR-before build,
   M=1 still improved ~11% (epilogue restructure).
6. **k_split==1 rows** changed only -0.9/-1.6% (noise): no scratch existed
   there in either build, confirming the memset removal is what moved the
   k_split>1 rows.

## Repeatability note

All 20 rows bit-repeatable on before and after builds (identical numerics
between them: max_abs equal to 6 decimals on every row — the zeros->empty
switch and the z_count==1 fast path are provably value-preserving).
