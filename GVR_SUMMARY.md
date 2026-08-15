# GVR performance summary

## Corrected bottom line

GVR produces a real end-to-end decode-forward speedup when it actually
dispatches on a large, long-context batch. The corrected event-free result for
GLM-5.2-NVFP4 at 199.4K context and batch 1024 is:

| Metric | Baseline | GVR |
| --- | ---: | ---: |
| TP critical-path forward | 99.828 ms | 89.857 ms |
| 21 selector launches | 12.662 ms | 2.785 ms |
| Sparse-attention kernels | 14.164 ms | 13.970 ms |
| Indexer-logits kernels | 37.583 ms | 37.873 ms |

The observed forward speedup is **1.1110x**: 11.10% more fixed-batch
throughput, or 9.99% less latency. The baseline selector occupies **12.68%** of
the forward and GVR makes it **4.55x** faster.

Amdahl's Law predicts:

```text
99.828 / (99.828 - 12.662 + 2.785) = 1.1098x
```

The 1.1098x prediction and 1.1110x observation differ by about 0.1 ms. This is
the first result in the investigation that is both dispatch-validated and
explainable from first principles.

### Independent real-corpus reproduction

The selector result was rerun on chat 1 and the first abstention probe from the
public [Kimi Vendor Verifier BEAM corpus][beam].
The served GLM tokenizer's chat template produced the prompt, which was
left-truncated to 199,400 tokens. The matched event-free captures contain 96
exact-B1024 decode steps on each of four TP ranks.

| Median graph metric | Baseline | GVR | Speedup |
| --- | ---: | ---: | ---: |
| 21-call selector total divided by 21 | 605.204 us | 133.141 us | 4.5456x |
| TP critical-path forward | 99.814 ms | 89.682 ms | 1.1130x |

Every retained replay contains 21 expected selector kernels and zero kernels
from the opposite implementation. The original 602.939/132.639-us result is
therefore reproducible: the real BEAM prompt changes either absolute time by
less than 0.4% and reproduces the 4.546x ratio. Amdahl's Law predicts 1.1104x
from the independently measured 12.74% baseline selector share; the observed
forward result is 1.1130x.

[beam]: https://github.com/MoonshotAI/Kimi-Vendor-Verifier/tree/main/beam

## Why the previous neutral result was wrong

The former source-of-truth result claimed 99.675 ms baseline versus 99.706 ms
"GVR" at 199.8K/B1024. That comparison used a decode-logits width of 200,032.
GVR requires a width divisible by 64, but `200032 % 64 == 32`, so its
eligibility check rejected the shape.

An event-free Nsight trace reproduced the mistake. The nominal GVR graph
contained the baseline `FilteredTopKUnifiedKernel` plus 21
`_store_decode_state_kernel` launches, and contained no fused GVR selector.
Thus the near-1.0x result compared baseline top-k with baseline top-k plus
state maintenance. Lowering only the row threshold did not override the column
alignment check.

The corrected pair uses `max_model_len=200000`. Its GVR graph contains exactly
21 fused CuTe GVR launches per retained replay and no cooperative or filtered
top-k launches. The baseline contains exactly 21 filtered selectors and no GVR
launches.

## CUDA events and overlap

The earlier explanation that timing events "suppressed overlap" was also
wrong.

- CUDA events are compatible with CUDA graphs when represented as graph event
  nodes.
- Adding start/end timing around 21 selector calls adds 42 nodes and can
  materially increase graph latency, so those event-instrumented absolute
  times should not be used as production latency.
- The GLM-5.2 portable DeepSeek-V3.2 path does not put this selector on an
  auxiliary stream. In both corrected traces, all selector kernels execute on
  stream 19.
- Direct timestamp intersection found **zero selector overlap** with kernels on
  every other stream for every retained B1024 replay.

Therefore the selector sum is serial critical-path work in this workload. The
events perturb latency by adding work; there is no evidence that they reveal or
remove selector overlap.

"21 selector calls" means one top-k selection in each of GLM-5.2's 21 indexer
layers during one model forward. "42 event nodes" means one start and one end
record around each of those calls. Neither term means 21 model forwards or
concurrent selector execution.

## Context and batch dependence

Event-free graph-node attribution gives:

| Case | Baseline forward | GVR forward | Baseline selector | GVR selector | Baseline selector share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 199.8K/B1 | 7.615 ms | 7.510 ms | 0.322 ms | 0.341 ms | 4.24% |
| 10K/B1024 | 52.709 ms | 52.030 ms | 1.168 ms | 0.960 ms | 2.21% |
| 199.4K/B1024 | 99.828 ms | 89.857 ms | 12.662 ms | 2.785 ms | 12.68% |

The B1 whole-forward difference is not a causal GVR win: GVR's selector is
about 19 us slower there, while unrelated kernels differ across the two server
processes. Production dispatch retains the cooperative baseline at B1.

At 10K/B1024 the selector can save only 0.208 ms because only 2.21% of the
forward is in baseline top-k. At 199.4K/B1024 it saves 9.876 ms, and the whole
forward saves 9.974 ms. The useful ceiling therefore grows sharply with both
row count and sequence length.

These are fixed-batch GPU graph spans, not HTTP throughput. B1024 was created
as `n=1024` child sequences from one real document prompt. Only NVTX ranges
explicitly labeled with 1,024 generation sequences were retained. The long
plateaus contain 130 baseline and 146 GVR analyzed replays per rank.

## Standalone selector result

The standalone CUDA-graph benchmark remains useful for kernel crossover
analysis. At 200K context on real captured tensors:

| Constructed rows | Baseline | GVR | Kernel speedup |
| ---: | ---: | ---: | ---: |
| 1 | 12.641 us | 18.305 us | 0.691x |
| 8 | 15.133 us | 19.812 us | 0.764x |
| 32 | 24.337 us | 20.455 us | 1.190x |
| 64 | 55.402 us | 26.105 us | 2.122x |
| 128 | 66.379 us | 33.310 us | 1.993x |
| 1024 (32 native rows repeated 32 times) | 487.619 us | 283.104 us | 1.722x |

B1/B8/B32 are native captures. B64/B128/B1024 repeat native B32 rows, so their
hint distribution is not a substitute for a real B1024 model run. The
corrected end-to-end trace is authoritative for the model result.

The 1.722x standalone result and the 4.55x embedded result are not measurements
of the same selector workload. The standalone dataset retained only the first
sparse-indexer layer at two consecutive B32 decode steps (`call0` and
`call21`); an eager debug-capture limitation made the other layer tensors
invalid. Its B1024 row repeats those 32 rows 32 times.

The difference is visible before taking either ratio:

| Selector measurement at 200K/B1024 | Baseline | GVR | Speedup |
| --- | ---: | ---: | ---: |
| Repeated-B32 standalone | 487.619 us | 283.104 us | 1.722x |
| Real model selector total divided by 21 | 602.939 us | 132.639 us | 4.546x |
| Real BEAM selector total divided by 21 | 605.204 us | 133.141 us | 4.546x |

The independent BEAM trace explains the difference directly. Each row below
is the median of 384 exact-B1024 measurements from normal model execution:

| Selector | Model layer | Baseline | GVR | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 511.568 us | 293.696 us | 1.742x |
| 2 | 1 | 542.720 us | 485.040 us | 1.119x |
| 3 | 2 | 613.888 us | 1,213.680 us | 0.506x |
| 4 | 6 | 614.080 us | 44.576 us | 13.776x |
| 5 | 10 | 613.216 us | 44.416 us | 13.806x |
| 6 | 14 | 612.224 us | 44.432 us | 13.779x |
| 7 | 18 | 611.904 us | 44.384 us | 13.787x |
| 8 | 22 | 611.936 us | 44.256 us | 13.827x |
| 9 | 26 | 612.288 us | 44.544 us | 13.746x |
| 10 | 30 | 612.432 us | 44.576 us | 13.739x |
| 11 | 34 | 612.256 us | 44.416 us | 13.785x |
| 12 | 38 | 612.000 us | 44.384 us | 13.789x |
| 13 | 42 | 612.416 us | 44.544 us | 13.749x |
| 14 | 46 | 612.224 us | 44.544 us | 13.744x |
| 15 | 50 | 612.160 us | 44.384 us | 13.792x |
| 16 | 54 | 612.256 us | 44.384 us | 13.795x |
| 17 | 58 | 612.160 us | 44.480 us | 13.763x |
| 18 | 62 | 612.240 us | 44.448 us | 13.774x |
| 19 | 66 | 612.016 us | 44.544 us | 13.740x |
| 20 | 70 | 612.176 us | 44.352 us | 13.803x |
| 21 | 74 | 612.256 us | 44.480 us | 13.765x |
| **Sum of medians** | -- | **12.690 ms** | **2.793 ms** | **4.544x** |

The real first call's 1.742x matches the standalone ratio. Selector 3 regresses
to 0.506x, while selectors 4--21 individually occupy the narrow
13.739x--13.827x range. That range is measured, not extrapolated from selector
4.

Thus the standalone number is numerically valid for a repeated first-layer
input, but it was misleadingly labeled as if it predicted a real B1024
21-layer average. All 21 production calls use the same launch configuration
and separate per-layer state. Their large variation comes from GVR's
data-dependent admission/fallback path; identifying the precise rung and
candidate counts requires additional device-side counters.

## FP16 baseline experiment

Allowing the existing baseline selector to consume FP16 scores helps most at
large repeated batches:

| Batch | FP32 | FP16 | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 10.992 us | 10.767 us | 1.021x |
| 8 | 11.562 us | 11.332 us | 1.020x |
| 32 | 14.603 us | 14.174 us | 1.030x |
| 64 | 29.812 us | 26.648 us | 1.119x |
| 128 | 30.072 us | 26.776 us | 1.123x |
| 1024 | 219.113 us | 181.113 us | 1.210x |

FP16 retains 99.9714% of selected indices in the saved selector corpus, but
that is index overlap rather than an end-to-end model-quality guarantee. The
existing GSM8K run found no observed accuracy regression, but broader model
evaluation is still required before enabling FP16 by default.

## Current recommendations

1. Keep the production row guard: GVR loses to the cooperative selector at B1
   and B8 and crosses over near B32 on long contexts.
2. Require dispatch validation in every benchmark: count 21 GVR launches and
   zero fallback selectors per retained model forward.
3. Use widths divisible by 64 or implement and validate an exact tail path
   before broadening column eligibility.
4. Use event-free graph-node attribution for component accounting. Internal
   events are useful diagnostics only after separately quantifying their graph
   overhead.
5. Treat **1.1110x at 199.4K/B1024** as the corrected large-case result. Do not
   use the superseded 0.9997x result.

The detailed experiment history remains in [GVR.md](GVR.md) and
[GVR_ANS.md](GVR_ANS.md). This file is the concise source of truth.
