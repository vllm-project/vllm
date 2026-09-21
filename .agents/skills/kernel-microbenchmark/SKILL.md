---
name: kernel-microbenchmark
description: Build, debug, and interpret vLLM GPU kernel microbenchmarks for CUDA, Triton, and CuteDSL, including CUPTI timing, correctness checks, generated-code inspection, multi-GPU measurements, and SOL sanity checks.
---

# Kernel Microbenchmark

## Workflow

1. Create an isolated repro or benchmark when the existing harness is noisy.
2. Check correctness before timing. Keep tolerances explicit.
3. Time only the operation under study. Exclude allocation, compilation, random
   input generation, logging, and host-device transfers unless those are the
   target.
4. Compare against a baseline and report enough metadata to reproduce the
   result: GPU, dtype, shape, command, branch/commit, and relevant env vars.
5. Treat explanations as hypotheses until backed by an artifact: ablation,
   generated PTX/SASS, profiler output, or controlled benchmark.
6. If the result changes the conclusion, preserve the compact lesson in a note,
   comment, benchmark table, or final summary. For experiments, a short
   `Question / Change / Correctness / Result / Observation / Next` note is
   usually enough.

## Benchmark Defaults

- Use FlashInfer CUPTI timing by default, with CUDA graph and cold L2 cache:
  `from flashinfer.testing import bench_gpu_time_with_cupti`.
- For compute-heavy kernels, report TFLOPS with the FLOP formula in the
  benchmark. For memory-heavy kernels, report estimated bytes moved and GB/s.
  For mixed kernels, report the most honest metric available and call out the
  caveats. TFLOPS and memory bandwidth should be computed from the theoretical
  best for the operation, not from a particular kernel implementation. For
  example, memory bandwidth should assume all data is read exactly once from
  global memory.
- When comparing across shapes, prefer throughput metrics such as TFLOPS or
  GB/s as the primary table columns; keep latency for absolute cost.
- If a result exceeds expected peak/SOL, first inspect units, FLOP/byte
  formulas, skipped work, sparsity, caching, and whether the baseline is doing
  the same operation.
- Force compilation/autotuning before measuring compiled kernels.
- Seed inputs when correctness comparisons matter.
- Keep metadata setup, plan construction, allocation, random input generation,
  and logging outside the timed region unless that overhead is the experiment.

## Sanity-Check Reference Numbers

Use these as rough reference points for large, well-shaped workloads, not as
gold standards, guaranteed peaks, or hard limits. Hardware SKU, clocks, shape,
precision conventions, and the FLOP/byte accounting can move the result. A
large gap is a prompt to investigate, not proof that a kernel is poor.

| Kernel regime | Hardware | Rough reference |
| --- | --- | ---: |
| Memory-bound, large batch | Blackwell | 6 TB/s |
| BF16 GEMM | Blackwell | 2 PFLOP/s |
| BF16 attention | B200 | 1.6 PFLOP/s |
| FP8 GEMM | Blackwell | 4 PFLOP/s |
| FP8 attention | B300 | 2.8 PFLOP/s |

The BF16 attention reference is approximately the 1613 TFLOP/s result reported
by the FlashAttention-4 paper. Compare kernels only with matching workload and
throughput conventions.

## Multi-GPU Benchmarks

- State whether the run is local or multi-node and report the GPU topology,
  world size, GPUs per node, collective backend, and relevant library versions.
- Compare like-for-like TP configurations. Report both per-rank and global
  dimensions, and do not compare results from different TP sizes without an
  explicit normalization or scaling question.
- Define the timed operation boundary before benchmarking. If the production
  wrapper performs input staging, flag resets, generation barriers, padding,
  or output copies, keep them in the timed region for an end-to-end comparison.
  Use a separate, clearly labeled ablation for kernel-only timing.
- Check distributed correctness before timing. Seed each rank deliberately,
  form the reference with the same collective semantics, and synchronize before
  reading or comparing outputs.
- Use a device-side barrier immediately before each measured replay. Keep this
  common synchronization outside the timed interval, but keep barriers required
  by the candidate implementation inside it.
- With CUDA graphs, warm up before capture, coordinate capture across ranks,
  rotate pointer-distinct graphs when cache reuse matters, and ensure every
  collective is issued in the same order on every rank.
- Measure every rank and reduce each sample with `MAX`; report the median of
  those per-sample maxima. A rank-local event time is not a distributed latency.
- Reset reusable symmetric-memory flags before each invocation and establish a
  device-side generation barrier before peers may signal them. Otherwise a fast
  rank can signal before a slow rank resets its flags, causing a lost arrival
  and intermittent deadlock.
- Preserve symmetric-memory handles, CUDA graphs, streams, and graph outputs for
  the full measurement lifetime. Allocate and rendezvous symmetric buffers in
  identical order and with identical shapes on all ranks.
- Treat hangs and isolated millisecond outliers as synchronization bugs or rank
  skew until disproven. Add stage markers, bounded synchronization checks, and
  per-rank diagnostics before blaming compilation or kernel performance.
- For multi-node runs, record the scheduler allocation and verify the fabric
  supports the required multicast or NVLink-domain assumptions. Do not describe
  a two-node TP run as equivalent to a local NVLink-domain run without checking.
- Stabilize GPU clocks or run enough untimed work to reach a steady state.
  Alternate candidate order so clock, thermal, and rank-skew effects are shared.

## Included Examples

- Use [benchmarks/cupti_microbenchmark.py](benchmarks/cupti_microbenchmark.py)
  as a minimal single-GPU FlashInfer CUPTI timing pattern.
- Use
  [benchmarks/multi_gpu_gemm_rs.py](benchmarks/multi_gpu_gemm_rs.py) as a
  minimal distributed CUDA-graph timing pattern.

Adapt the operation, cases, work formula, and correctness tolerances rather
than copying either example unchanged.
