---
name: kernel-microbenchmark
description: Build, debug, and interpret vLLM GPU kernel microbenchmarks for CUDA, Triton, and CuteDSL, including CUPTI timing, correctness checks, generated-code inspection, multi-GPU measurements, and SOL sanity checks.
---

# Kernel Microbenchmark

## Workflow

1. Create an isolated repro or benchmark when the existing harness is noisy.
2. Check correctness before timing and keep tolerances explicit.
3. Time only the operation under study. Exclude allocation, compilation,
   random input generation, logging, and host-device transfers unless they are
   the target.
4. Compare against a baseline and report enough metadata to reproduce the
   result: GPU, dtype, shape, command, branch or commit, and relevant env vars.
5. Treat explanations as hypotheses until backed by an artifact such as an
   ablation, generated PTX or SASS, profiler output, or controlled benchmark.
6. Summarize experiments as `Question / Change / Correctness / Result /
   Observation / Next` when a durable note is useful.

## Benchmark Defaults

- Prefer FlashInfer CUPTI timing with CUDA graph and cold L2 cache:
  `from flashinfer.testing import bench_gpu_time_with_cupti`.
- For compute-heavy kernels, report TFLOPS and show the FLOP formula. For
  memory-heavy kernels, report estimated bytes moved and GB/s. For mixed
  kernels, use the most honest metric available and state its caveats. Compute
  theoretical work from the operation rather than a particular implementation.
- Across shapes, use throughput metrics such as TFLOPS or GB/s as the primary
  comparison and retain latency for absolute cost.
- If a result exceeds expected peak or speed-of-light limits, inspect units,
  FLOP or byte formulas, skipped work, sparsity, caching, and semantic parity
  with the baseline.
- Force compilation and autotuning before measuring compiled kernels.
- Seed inputs when correctness comparisons matter.
- Keep metadata setup, plan construction, allocation, random input generation,
  and logging outside the timed region unless that overhead is the experiment.

## Multi-GPU Benchmarks

- Report whether the run is local or multi-node, plus GPU topology, world size,
  GPUs per node, collective backend, and relevant library versions.
- Compare like-for-like tensor-parallel configurations. Report per-rank and
  global dimensions; normalize explicitly when comparing different TP sizes.
- Define the timed operation boundary first. Include staging, flag resets,
  required barriers, padding, and output copies for end-to-end comparisons;
  label kernel-only timing as an ablation.
- Check distributed correctness before timing. Seed ranks deliberately, match
  the candidate's collective semantics in the reference, and synchronize
  before reading outputs.
- Put a device-side barrier immediately before each measured replay. Keep this
  common synchronization outside the timed interval, but retain barriers that
  the candidate implementation requires inside it.
- With CUDA graphs, warm up before capture, coordinate capture across ranks,
  rotate pointer-distinct graphs when cache reuse matters, and issue every
  collective in the same order on every rank.
- Measure every rank, reduce each sample with `MAX`, and report the median of
  those per-sample maxima. Rank-local event time is not distributed latency.
- Reset reusable symmetric-memory flags before each invocation and establish a
  device-side generation barrier before peers may signal them. Otherwise a fast
  rank can signal before a slow rank resets its flags, causing a lost arrival
  and intermittent deadlock.
- Preserve symmetric-memory handles, CUDA graphs, streams, and graph outputs
  for the full measurement lifetime. Allocate and rendezvous symmetric buffers
  in identical order and with identical shapes on all ranks.
- Treat hangs and isolated millisecond outliers as synchronization bugs or rank
  skew until disproven. Add stage markers, bounded synchronization checks, and
  per-rank diagnostics before blaming compilation or kernel performance.
- For multi-node runs, record the scheduler allocation and verify that the
  fabric supports required multicast or NVLink-domain assumptions.
- Stabilize GPU clocks or run enough untimed work to reach steady state.
  Alternate candidate order so clock, thermal, and rank-skew effects are shared.

## Included Examples

- Use [benchmarks/cupti_microbenchmark.py](benchmarks/cupti_microbenchmark.py)
  as a minimal single-GPU FlashInfer CUPTI timing pattern.
- Use
  [benchmarks/multi_gpu_gemm_rs.py](benchmarks/multi_gpu_gemm_rs.py) as a
  minimal distributed CUDA-graph timing pattern. It is derived from vLLM's
  `benchmarks/kernels/benchmark_kimi_k3_gemm_rs_ar.py` without its model,
  custom-kernel, or symmetric-memory dependencies.

Adapt the operation, cases, work formula, and correctness tolerances rather
than copying either example unchanged.
