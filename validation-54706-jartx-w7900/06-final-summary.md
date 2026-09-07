# PR #54706 — JartX Review Closure on W7900

Review source: https://github.com/vllm-project/vllm/pull/54706#issuecomment-5555703806
All measurements on the machine described below, current branch as pushed.

## Environment
- GPU: AMD Radeon Pro W7900D (gfx1100, 48 CUs, 48 GB)
- gfx: gfx1100 (gcnArchName confirmed programmatically)
- ROCm: 7.14.60850 wheel runtime (torch 2.14.0+rocm7.14); build toolchain
  /opt/rocm 7.2.1 (hipcc/hipify compile-time only)
- OS: Ubuntu 24.04.4 LTS, kernel 6.8.0-79-generic
- PyTorch: 2.14.0+rocm7.14
- vLLM commit: dddf164781 (PR HEAD) + closure commit(s) on
  fix/rdna3-w4a16-determinism
- upstream base: 6fbb00b188 (upstream/main); merge-base 40b2f62061

## JartX Review Checklist

- [x] 1. Accuracy-first framing — PR description rewritten: Problem/Root
      cause lead with numerical error on every forward pass; greedy
      nondeterminism framed as the visible symptom. dtype called out
      everywhere; the old "WMMA range generally benefits" sentence is gone
      (replaced by the measured table where every k_split>1 WMMA row is
      negative before the fix and positive after).
- [x] 2. WMMA `at::zeros` memset investigated — hypothesis confirmed:
      removing the (provably dead) zero-fill recovered -3.8..-8.7% on all
      k_split>1 WMMA shapes in both dtypes; bf16 M=16 (the -17.2% row)
      improved -8.7% and is now faster than the legacy CAS control.
- [x] 3. Scratch coverage proven total — per-kernel argument documented at
      `alloc_wmma_partials`: every reducer-visible (z, m, n) has exactly
      one writer in all seven variants (16x16_1w, 32x16_2w, 64x16_4w,
      64x32_4w, 64x64_4w, 128x64_k16, 128x64_k32); the out-of-range guards
      skip only slots the reducer never reads. `at::zeros` -> `at::empty`.
      JartX's "store 0.0f from out-of-range lanes" turned out to be
      unnecessary (and would write out of bounds): the guards already
      align exactly with reducer visibility. Empirically: before/after
      numerics bit-identical on all 20 shapes; all reference tests pass.
- [x] 4. Correctness reference test — 10 new FP32-dequantized-reference
      tests (scalar/WMMA x split-K/direct-store x bf16/fp16) with
      tolerances derived from the kernels' rounding structure (see test
      module docstring + 03-correctness.txt).
- [x] 5. Scalar regression strengthened — K=1024 (4 writers) -> K=4096
      (ceil(4096/256) = 16 writers), verified against the real split
      geometry; sits well inside the old failure regime (onset 2-4
      writers). A/B: this test fails on the restored legacy CAS build.
- [x] 6. Context manager cleanup — `_cm.__enter__()` leak removed; tests
      use the repo-standard `dist_init` conftest fixture (real context
      manager + per-test cleanup).
- [x] 7. Dynamic test port — MASTER_PORT=29741 removed; `dist_init`
      rendezvouses via a temp file (no port at all), the repo convention
      for single-process kernel tests.
- [x] 8. dtype/M/N/K/k_split benchmarks — full matrix re-run on the
      current branch: dtype x M{1,8,16,64,128,512} x N{4096,25600} x
      K{4096,6656} with derived kernel path, k_split, scratch MB, and
      max_abs error per row (05-benchmark.md, 04-benchmark-*.csv).
- [x] 9. bf16/fp16 rerun — both dtypes measured separately; dispatch
      difference (bf16 WMMA from M>=16, fp16 from M>=64) explicit in the
      table and findings.
- [x] 10. Stale CAS wording fixed — TORCH_CHECK N%8 message now states
       the real reason (packed qzeros layout); file headers, heuristic
       comments, launcher comments, and epilogue comments distinguish
       legacy CAS behavior from current requirements; one-way-door notes
       where `empty` replaced `zeros` (legacy path needs pre-zeroed c).
- [x] 11. split-count==1 optimization — scalar `launch_gemm_q4_deterministic`
       now skips scratch+reduce when z_count == 1 (kernel direct-store
       epilogue), mirroring the WMMA k_split==1 path; new tests cover both.
- [x] 12. FP32 atomic alternative documented — PR description "Why not
       FP32 atomics?" section: gfx11 `global_atomic_add_f32` removes most
       rounding error but stays order-dependent (fp32 non-associativity),
       so it cannot guarantee bit-exact reproducibility; fixed-order FP32
       reduction chosen because bit-exactness is an explicit goal.

## Correctness

- Fixed path max_abs vs FP32 reference (K=4096, N=4096): scalar bf16
  0.0625 (== 1 output ulp — the final-cast bound), wmma bf16 0.089-0.112,
  wmma fp16 0.011-0.015, scalar fp16 ~0.95 (pre-existing exllama dequant
  noise, unchanged by this PR, documented).
- Legacy CAS A/B on this machine: bf16 max_abs 0.24-0.37 -> 0.06 (4-6x);
  wmma fp16 0.024-0.038 -> 0.011-0.018 (2x). The PR's earlier
  "0.028 -> 0.0061" is the same bf16 factor on real-weight scales.
- at::empty scratch passes all reference tests; before/after numerics
  bit-identical (zero-init was dead traffic).

## Determinism

- Fixed builds: bit-repeatable on all 20 benchmark shapes; 20/20 tests.
- Legacy A/B: nondeterministic on 12/20 shapes; 8/20 tests fail on the
  restored CAS build (7 repeatability + scalar-bf16 reference test at
  max_abs=0.3022 > 0.10 tol). Single-split tests pass under both.

## Performance (median us, before -> after)

- WMMA k_split>1 (both dtypes): -3.8% to -8.7% (memset removal).
  bf16: M=16 58.8->53.7, M=64 120.5->112.0, M=128 123.5->114.7,
  M=512 378.4->350.8. fp16: M=64 104.7->97.8, M=128 116.1->107.9,
  M=512 353.1->327.6.
- vs legacy CAS control: deterministic path now at parity or faster for
  prefill (bf16 M=16 53.7 vs 55.3); residual cost only at M=1 decode
  (bf16 +3.7..5.3%, fp16 +5.8..10.5%) — the price of bit-reproducibility.
- k_split==1 rows unchanged within noise (no scratch in either build).
- JartX's M=512 pair reproduced: N=4096 -> k_split=4 (32 MB scratch);
  N=25600 -> k_split=1 (no scratch, direct store).

## Key Result

The deterministic epilogue is now effectively free where it matters:
after removing the dead zero-fill, prefill shapes are FASTER than the old
atomic epilogue while being 2-6x more accurate vs the FP32 reference and
bit-reproducible; only M=1 decode pays ~4-10% for reproducibility.

## Remaining Risks

- M=1 decode retains a small latency cost vs the legacy CAS epilogue
  (scratch round-trip at tiny output). Acceptable for determinism; noted
  in the PR.
- scalar fp16 accuracy is bounded by the classic exllama bit-trick dequant
  (pre-existing, shared with the fp16 scalar path upstream); out of scope
  for this PR but documented in the tests.
- torch 2.14/ROCm 7.14 wheel toolchain mix (system hipcc 7.2.1 compile,
  wheel 7.14 runtime) — validated by 20/20 tests + bit-identical
  numerics before/after, but CI will build with the pinned torch 2.13.

## Files Changed

- csrc/rocm/q_gemm_rdna3.cu — three-mode epilogue (partials / direct
  store at gridDim.z==1 / legacy CAS for A/B), z_count==1 fast path,
  stale-CAS wording fixes, N%8 message, one-way-door notes.
- csrc/rocm/q_gemm_rdna3_wmma.cu — alloc_wmma_partials zeros->empty with
  documented coverage invariant; header/heuristic/launcher comment
  cleanup; stale contradictory zero-init note removed.
- tests/kernels/quantization/test_rdna3_w4a16_determinism.py —
  FP32-reference correctness tests, strengthened scalar regression
  (K=4096/16 writers), single-split coverage, dist_init hygiene.
- validation-54706-jartx-w7900/ — evidence (env, build, tests,
  correctness, benchmark CSVs + report, this summary).

## Commit

- bb6bbe5be5b6d04ca48cdc862d4b532fc614f222 "[ROCm] Address RDNA3 W4A16
  review feedback" on fix/rdna3-w4a16-determinism (+ evidence follow-up
  commit), pushed to AIwork4me/vllm. No history rewritten; no force-push.

## PR update

- PR description rewritten (accuracy-first) and pushed via gh.
- Reply to JartX posted (thanks, framing adopted, M=16 recovery, FP32
  reference tests, dtype+M/N/K+k_split reporting, hygiene fixes, invite to
  run the dispatch table on his 7900 XTX).
