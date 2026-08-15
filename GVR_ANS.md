# FP32 baseline versus FP32 GVR across batch and KV length

## Requested measurements

This report answers three questions for batch sizes 1, 8, 32, 64, 128, and 1024
and KV lengths 10K, 50K, 100K, and 200K:

1. production FP32 baseline top-k latency;
2. FP32 baseline versus FP32 GVR selector latency and speedup on real data;
3. FP32 baseline versus FP32 GVR complete model-forward latency and speedup.

All selector timings will use CUDA-graph replay. “Real data” means logits and
temporal hints produced by the real GLM-5.2-NVFP4 checkpoint and tokenized
document prompts, not random tensors. Repetition may be used to construct the
large-batch kernel grids, but any such row is labeled rather than represented
as an independently captured request distribution. End-to-end measurements
use the real model weights and actual forward passes.

## Updated result with the fixed adaptive GVR

The fixed kernel uses one three-column graph-compatible implementation but
changes the rung interpretation from the device-side runtime sequence length.
It uses `(q=.60, .35, pmean)` for `N < 32K` and `64K <= N < 128K`, and
`(q=.35, .05, .01)` for `32K <= N < 64K` and `N >= 128K`. Exact fallback and
final selection are unchanged. The same captured graph therefore supports all
lengths without host dispatch or recapture.

### Fixed-GVR selector latency

All cells use CUDA-graph replay and were checked against the exact FP32
selected-value set from `torch.topk`. B1/B8/B32 use native real captures;
B64/B128/B1024 repeat native real B32 captures to measure scaling.

| KV length | Batch | FP32 baseline | Fixed GVR | Speedup (baseline/GVR) |
|---:|---:|---:|---:|---:|
| 10K | 1 | 5.038 us | 9.727 us | 0.518x |
| 10K | 8 | 5.187 us | 9.908 us | 0.524x |
| 10K | 32 | 5.324 us | 9.804 us | 0.543x |
| 10K | 64 | 5.509 us | 10.167 us | 0.542x |
| 10K | 128 | 5.637 us | 9.935 us | 0.567x |
| 10K | 1024 | 34.187 us | 42.717 us | 0.800x |
| 50K | 1 | 10.400 us | 16.389 us | 0.635x |
| 50K | 8 | 10.506 us | 18.026 us | 0.583x |
| 50K | 32 | 13.055 us | 17.802 us | 0.733x |
| 50K | 64 | 24.424 us | 18.609 us | **1.312x** |
| 50K | 128 | 24.747 us | 18.309 us | **1.352x** |
| 50K | 1024 | 171.590 us | 98.077 us | **1.750x** |
| 100K | 1 | 10.975 us | 14.461 us | 0.759x |
| 100K | 8 | 11.538 us | 20.903 us | 0.552x |
| 100K | 32 | 14.435 us | 20.493 us | 0.704x |
| 100K | 64 | 29.761 us | 22.898 us | **1.300x** |
| 100K | 128 | 30.079 us | 25.136 us | **1.197x** |
| 100K | 1024 | 219.171 us | 140.239 us | **1.563x** |
| 200K | 1 | 12.834 us | 18.305 us | 0.701x |
| 200K | 8 | 15.128 us | 19.772 us | 0.765x |
| 200K | 32 | 24.167 us | 20.311 us | **1.190x** |
| 200K | 64 | 54.677 us | 26.004 us | **2.103x** |
| 200K | 128 | 67.016 us | 33.519 us | **1.999x** |
| 200K | 1024 | 488.980 us | 283.554 us | **1.724x** |

The speedup column is `baseline latency / GVR latency`. Thus `0.518x` at
10K/B1 does **not** mean that GVR takes 51.8% of the baseline time; it means
GVR takes `1 / 0.518 = 1.93x` the baseline time, a 93% latency increase. Across
B1/B8/B32, fixed-GVR latency relative to baseline is respectively
1.93/1.91/1.84x at 10K, 1.58/1.72/1.36x at 50K, 1.32/1.81/1.42x at 100K,
and 1.43/1.31/0.84x at 200K. The 200K/B32 cell is the sole small-batch win.

The fixed selector crosses over at B64 for every measured context of 50K or
longer, and at B32 for 200K. It repairs the previous 200K/B1024 regression
from 0.836x to 1.724x and improves 50K/B1024 from 1.400x to 1.750x without
materially changing 100K. It still cannot amortize its fixed work at 10K or at
B1-B32 for 50K/100K, so it is not unconditionally faster than the baseline.

### Separate-server forward measurements (causal attribution invalid)

The baseline and GVR servers used matching TP4 GLM-5.2-NVFP4 configurations,
real document requests, full-decode CUDA graphs, and disabled FlashInfer
autotuning. They were separate server processes rather than paired graph
variants in one process. Each value is the median rank-0 CUDA-event time around
the complete model forward; it excludes HTTP, scheduling, and sampling.

| KV length | Batch | Baseline | Fixed GVR | Speedup | Change |
|---:|---:|---:|---:|---:|---:|
| 10K | 1 | 7.065 ms | 7.537 ms | 0.937x | -6.26% |
| 10K | 8 | 7.658 ms | 8.125 ms | 0.943x | -5.75% |
| 10K | 32 | 9.518 ms | 9.663 ms | 0.985x | -1.50% |
| 10K | 64 | 11.543 ms | 11.666 ms | 0.989x | -1.06% |
| 10K | 128 | 14.289 ms | 14.334 ms | 0.997x | -0.32% |
| 10K | 1024 | 52.289 ms | 51.714 ms | 1.011x | +1.11% |
| 50K | 1 | 7.186 ms | 7.222 ms | 0.995x | -0.50% |
| 50K | 8 | 7.828 ms | 7.874 ms | 0.994x | -0.58% |
| 50K | 32 | 9.980 ms | 9.932 ms | 1.005x | +0.48% |
| 50K | 64 | 12.535 ms | 12.203 ms | 1.027x | +2.72% |
| 50K | 128 | 15.741 ms | 15.781 ms | 0.997x | -0.25% |
| 50K | 1024 | 62.879 ms | 59.484 ms | 1.057x | +5.71% |
| 100K | 1 | 7.220 ms | 7.262 ms | 0.994x | -0.58% |
| 100K | 8 | 7.958 ms | 7.968 ms | 0.999x | -0.12% |
| 100K | 32 | 10.385 ms | 10.248 ms | 1.013x | +1.34% |
| 100K | 64 | 13.798 ms | 12.858 ms | 1.073x | +7.31% |
| 100K | 128 | 17.235 ms | 16.623 ms | 1.037x | +3.68% |
| 100K | 1024 | 74.715 ms | 69.581 ms | 1.074x | +7.38% |
| 200K | 1 | 7.307 ms | 7.651 ms | 0.955x | -4.51% |
| 200K | 8 | 8.183 ms | 8.140 ms | 1.005x | +0.53% |
| 200K | 32 | 11.191 ms | 10.927 ms | 1.024x | +2.42% |
| 200K | 64 | 15.575 ms | 14.098 ms | **1.105x** | **+10.48%** |
| 200K | 128 | 20.513 ms | 19.074 ms | 1.075x | +7.54% |
| 200K | 1024 | 99.351 ms | 89.501 ms | **1.110x** | **+11.01%** |

These raw forward deltas must not be attributed to GVR. At 200K/B1024, the
baseline selector share estimate is `21 * 0.488980 / 99.351 = 10.34%`. A
1.724x selector speedup predicts
`99.351 - 21 * (0.488980 - 0.283554) = 95.037 ms`, or only 1.045x e2e. Even an
exact 2x selector speedup predicts only 1.054x. The observed 89.501 ms is 5.536
ms faster than the measured selector saving can explain; reproducing it from
top-k alone would require approximately a 24.5x selector speedup.

The old-to-fixed inconsistency gives the same warning. The 200K/B1024
microkernel changed from about 585 to 284 us, which would save roughly 6.3 ms
over 21 layers, but the separate-server forward changed only 0.286 ms. At 50K
the microkernel improved by about 25 us per layer while the forward was
essentially unchanged. The repeated-B32 selector capture is not representative
of the real B1024 GVR admission distribution, and the separate server runs also
permit unrelated graph/kernel, TP-rank-wait, clock, and system variation.

An output-order audit found identical selected FP32 value sets, but no evidence
that GVR created a downstream locality win: on the repeated 200K/B1024 capture,
adjacent indices stayed in the same 64-token block 13.2% of the time for GVR
versus 19.3% for baseline. This indirect check does not replace profiling, but
it makes index ordering an unsupported explanation for the extra 5.536 ms.

Therefore the kernel table remains valid for its stated captured inputs, while
the forward table is retained only as a record of separate-run observations.
A causal e2e result requires baseline and GVR kernel totals from the actual
forward rows, repeated runs, the TP critical path rather than rank 0 alone,
and preferably paired baseline/GVR CUDA graphs in the same warmed process.

## Progress log

### 2026-08-14: existing-evidence audit

- Four idle GB200 GPUs with 189,471 MiB each are available.
- The real GLM-5.2-NVFP4 checkpoint is present locally at the path used by the
  earlier evaluations.
- Existing selector captures cover six layer/step samples at native batches 1,
  8, and 32, but only at approximately 100K KV length. They are insufficient
  for the requested KV-length matrix.
- Existing end-to-end traces also cover approximately 100K only. They cannot
  establish the 10K/50K/200K results.
- `benchmarks/kernels/benchmark_gvr_captures.py` already provides exact
  FP32-baseline/FP32-GVR CUDA-graph timing and selected-value validation. It
  needs length-aware capture inputs and matrix aggregation.
- The earlier `n=batch` client pattern is not suitable for this task: large
  batches become short fork rows. The new e2e/capture workload must use
  independent concurrent requests against a common cacheable document prefix.

### 2026-08-14: native-FP16 baseline side experiment

The production baseline kernels were generalized from FP32-only input to
native FP32/FP16 input without changing their batch dispatch or launch shape.
The cooperative path templates its TMA transfers and resident score buffers;
the persistent path selects on the ordered 16-bit FP16 key with one exact
refinement round. There is no untimed FP16-to-FP32 score-tensor conversion.

The focused padded-stride correctness suite passed all 8 combinations of
FP32/FP16, cooperative/persistent, and K=512/2048. A 50-operation CUDA graph
with 50 timed replays on six real 100K captures produced:

| Batch | FP32 baseline | Native FP16 baseline | FP16 speedup |
|---:|---:|---:|---:|
| 1 | 11.173 us | 10.995 us | 1.016x |
| 8 | 11.891 us | 11.643 us | 1.021x |
| 32 | 14.800 us | 14.414 us | 1.027x |
| 128 | 32.729 us | 28.840 us | 1.135x |
| 1024 | 234.576 us | 189.486 us | 1.238x |

The 1/8/32 rows use native captures. The 128/1024 rows repeat each real
batch-32 capture to isolate kernel scaling and are not independent large-batch
request distributions. FP16 saves little in the cooperative small-batch path,
where fixed launch/synchronization/selection costs dominate. It helps the
one-CTA-per-row persistent path more, but does not approach 2x because score
traffic is only part of that algorithm's work.

### 2026-08-14: pre-fix real-data FP32 selector matrix

New FP32 logits and temporal hints were captured from the real 432.9 GiB
GLM-5.2-NVFP4 checkpoint at nominal KV lengths 10K, 50K, and 200K. The existing
100K captures came from the same checkpoint and document-prompt method. Native
batches 1, 8, and 32 used independent requests sharing the already cached
document prefix; server statistics showed 99.8-100% prefix-cache hit rates.
The 128/1024 kernel rows repeat the real batch-32 rows to isolate selector
scaling and are not represented as independently captured request
distributions.

Capture execution exposed an eager debug-mode limitation: only the first
sparse-indexer layer of each decode forward had finite score tensors; later
layers' score buffers were all NaN regardless of whether baseline or GVR was
selected. Those invalid files were excluded. The retained dataset contains
two consecutive decode steps (`call0` and `call21`) for that real sparse layer,
at every native batch and KV length. Every retained tensor is finite, all
indices are in range, and measured hint/current-exact-top-k overlap ranges from
56% to 96%. The offline benchmark recomputes masked `torch.topk` from each
logit tensor and requires GVR selected values to match it exactly.

The table below uses 50 selector operations per CUDA graph and 100 timed graph
replays for native batches. Batches 128/1024 use 20 operations per graph and
50 timed replays. “Speedup” is baseline latency divided by GVR latency, so a
value below 1.0 is a GVR slowdown.

| KV length | Batch | FP32 baseline | FP32 GVR | Speedup |
|---:|---:|---:|---:|---:|
| 10K | 1 | 5.029 us | 9.607 us | 0.523x |
| 10K | 8 | 5.206 us | 9.705 us | 0.536x |
| 10K | 32 | 5.316 us | 9.755 us | 0.545x |
| 10K | 128 | 5.639 us | 9.825 us | 0.574x |
| 10K | 1024 | 34.377 us | 42.256 us | 0.814x |
| 50K | 1 | 10.387 us | 21.112 us | 0.492x |
| 50K | 8 | 10.536 us | 23.378 us | 0.451x |
| 50K | 32 | 13.115 us | 23.206 us | 0.565x |
| 50K | 128 | 24.792 us | 23.366 us | 1.061x |
| 50K | 1024 | 171.805 us | 122.713 us | 1.400x |
| 100K | 1 | 10.756 us | 14.461 us | 0.744x |
| 100K | 8 | 11.531 us | 20.834 us | 0.553x |
| 100K | 32 | 14.518 us | 20.372 us | 0.713x |
| 100K | 128 | 30.087 us | 24.964 us | 1.205x |
| 100K | 1024 | 219.067 us | 139.254 us | 1.573x |
| 200K | 1 | 12.798 us | 26.570 us | 0.482x |
| 200K | 8 | 15.094 us | 28.978 us | 0.521x |
| 200K | 32 | 24.240 us | 30.446 us | 0.796x |
| 200K | 128 | 68.977 us | 50.804 us | 1.358x |
| 200K | 1024 | 489.199 us | 584.960 us | 0.836x |

At kernel level GVR is consistently slower through batch 32. It crosses over
at batch 128 for 50K/100K/200K and reaches its best
measured gain, 1.573x, at 100K/B1024. The 200K/B1024 regression appears on both
consecutive real captures (585.702 and 584.217 us), while the matching baseline
measurements are 489.780 and 488.618 us. It is therefore a repeatable
data/algorithm-path effect, not an isolated timing outlier.

### 2026-08-14: clarified Q1, aggregate top-k share of decode forward

GLM-5.2-NVFP4 executes 21 sparse-indexer top-k operations in each decode
forward. The measured estimate for Q1 is therefore calculated as
`21 * baseline top-k latency / baseline full-forward latency`. This is the
aggregate kernel-only top-k share; it does not include the indexer-logit GEMM
or other indexer work. B1/B8/B32 use native real captures. As noted above,
B128/B1024 repeat real B32 rows, so those two columns estimate selector share
at the requested shape rather than profiling independently captured large
batches inside the model.

| KV length | B1 | B8 | B32 | B128 | B1024 |
|---:|---:|---:|---:|---:|---:|
| 10K | 1.50% | 1.43% | 1.17% | 0.83% | 1.38% |
| 50K | 3.04% | 2.83% | 2.76% | 3.31% | 5.74% |
| 100K | 3.13% | 3.04% | 2.94% | 3.67% | 6.16% |
| 200K | 3.68% | 3.87% | 4.55% | 7.06% | 10.34% |

Thus, even replacing top-k with a zero-cost operation would improve the
measured decode forward by at most 1.01-1.12x across this matrix. The largest
ceiling is 1.115x at 200K/B1024; a real replacement necessarily realizes less
because its latency cannot be zero.

## Pre-fix reference results

### Pre-fix matched real-model end-to-end results

Both runs loaded the real 432.9 GiB NVIDIA GLM-5.2-NVFP4 checkpoint with TP4.
The inputs were tokenized vLLM documentation, and each batch consisted of
independent requests sharing a real prefix-cache entry. The server used
full-decode-only CUDA graphs captured at exactly B1/B8/B32/B128/B1024, no
speculative decoding, and `--no-enable-flashinfer-autotune`. The measurement
is a CUDA-event duration around the model forward/graph replay; it excludes
HTTP, scheduling, sampling, and the timing-file write.

Every value below is the median of 54-242 exact full-graph decode samples.
The B1024 waves used staged request admission to reach and sustain exactly
1,024 concurrent independent decodes. The 200K/B1024 prompt was 199,800
tokens to leave generation room; its accepted samples span positions
199,843-199,969. “Speedup” is baseline/GVR, so values below 1.0 are
regressions. GVR was deliberately enabled at every batch size for this test.

| KV length | Batch | FP32 baseline | FP32 GVR | Speedup | Change |
|---:|---:|---:|---:|---:|---:|
| 10K | 1 | 7.065 ms | 7.189 ms | 0.983x | -1.73% |
| 10K | 8 | 7.658 ms | 7.802 ms | 0.982x | -1.84% |
| 10K | 32 | 9.518 ms | 9.649 ms | 0.986x | -1.35% |
| 10K | 128 | 14.289 ms | 14.336 ms | 0.997x | -0.33% |
| 10K | 1024 | 52.289 ms | 51.843 ms | 1.009x | +0.86% |
| 50K | 1 | 7.186 ms | 7.219 ms | 0.995x | -0.46% |
| 50K | 8 | 7.828 ms | 8.180 ms | 0.957x | -4.30% |
| 50K | 32 | 9.980 ms | 10.248 ms | 0.974x | -2.62% |
| 50K | 128 | 15.741 ms | 15.376 ms | 1.024x | +2.38% |
| 50K | 1024 | 62.879 ms | 59.501 ms | 1.057x | +5.68% |
| 100K | 1 | 7.220 ms | 7.261 ms | 0.994x | -0.57% |
| 100K | 8 | 7.958 ms | 7.976 ms | 0.998x | -0.22% |
| 100K | 32 | 10.385 ms | 10.243 ms | 1.014x | +1.39% |
| 100K | 128 | 17.235 ms | 16.628 ms | 1.036x | +3.65% |
| 100K | 1024 | 74.715 ms | 69.536 ms | 1.074x | +7.45% |
| 200K | 1 | 7.307 ms | 7.323 ms | 0.998x | -0.23% |
| 200K | 8 | 8.183 ms | 8.158 ms | 1.003x | +0.31% |
| 200K | 32 | 11.191 ms | 10.924 ms | 1.024x | +2.44% |
| 200K | 128 | 20.513 ms | 19.085 ms | 1.075x | +7.48% |
| 200K | 1024 | 99.351 ms | 89.787 ms | **1.107x** | **+10.65%** |

The real end-to-end crossover moves with KV length: GVR is not beneficial at
B1, and it is generally a regression through B32 at 10K/50K. The useful gains
appear at B128 for 50K and longer contexts and grow to 10.65% at 200K/B1024.
Sub-1% differences should be treated as effectively neutral without repeated
whole-server runs because the within-run CUDA-event distribution does not
capture run-to-run system variation.

The 200K/B1024 e2e gain does not contradict the slower repeated-B32 kernel
row: GVR runtime is data-dependent, and that microbenchmark duplicates the
native B32 score/hint distribution rather than capturing 1,024 independent
rows. The e2e row is the authoritative real B1024 workload result.

### Follow-up: why the default 200K/B1024 kernel row regressed

A targeted replay on the same two captures shows that the row is a default
R0-threshold failure, not an inherent GVR ceiling. The `(q=.60, .35)` plus
virtual-seed default takes 584.765 us, 1,024 threads reduce it to 507.010 us,
secant-only takes 416.885 us, and the `(q=.35, .05, .01)` rung set takes
281.509 us. The last is
1.738x faster than the 489.199-us FP32 baseline. The three-rung set had lost on
the earlier six-capture 100K dataset, so the correct conclusion is that the
fixed 100K-calibrated threshold policy does not generalize to this 200K score
distribution. The main selector table intentionally records the unmodified
default; it is not the best per-length tuned result.
