# GVR performance summary

## Bottom line

GVR has real standalone top-k kernel wins at large batches and long contexts,
but it does **not** provide a meaningful end-to-end model-forward speedup in
the tested GLM-5.2-NVFP4 decode workload.

With real weights and document inputs, TP4, full-decode CUDA graphs, and
FlashInfer autotuning disabled, the clean paired result ranges from
**0.9968x to 1.0016x**. That is effectively performance-neutral. The largest
apparent gain is 0.159%; the largest loss is 0.323%.

At the most important large case, 199.8K context and batch 1024:

- baseline: **99.675 ms** per model forward;
- GVR: **99.706 ms**;
- speedup: **0.9997x**, or a 0.031% slowdown.

The earlier claim of roughly 10% e2e speedup was a measurement artifact and
should not be used.

## Real end-to-end result

The table reports `baseline latency / GVR latency`. Values above 1 favor GVR;
values below 1 favor the baseline.

| Batch | 10K | 50K | 100K | 199.8K |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9989x | 0.9975x | 0.9968x | 0.9995x |
| 8 | 0.9996x | 1.0008x | 1.0010x | 0.9995x |
| 32 | 0.9998x | 0.9998x | 0.9996x | 0.9997x |
| 64 | 0.9997x | 0.9992x | 1.0002x | 0.9990x |
| 128 | 1.0009x | 1.0016x | 1.0003x | 1.0002x |
| 1024 | 1.0002x | 0.9998x | 0.9998x | 0.9997x |

These measurements forced GVR at every batch size. Production dispatch would
normally retain the baseline for some small shapes.

### Why this result is trustworthy

- Baseline and GVR were captured as two graphs in the same warmed process.
- Every decode step ran both graphs on the same model buffers.
- Baseline-first and GVR-first order alternated every step.
- Results use the maximum latency across the four TP ranks.
- Only sustained rows with the exact requested batch were retained.
- Internal selector timing events were removed from the final graphs.
- Each cell contains 56 to 230 samples after warmup and tail trimming.

## Standalone top-k result

The kernel optimization itself is real. At 200K context, CUDA-graph replay on
real captured tensors gives:

| Batch | Baseline | GVR | Kernel speedup |
| ---: | ---: | ---: | ---: |
| 1 | 12.641 us | 18.305 us | 0.691x |
| 8 | 15.133 us | 19.812 us | 0.764x |
| 32 | 24.337 us | 20.455 us | 1.190x |
| 64 | 55.402 us | 26.105 us | 2.122x |
| 128 | 66.379 us | 33.310 us | 1.993x |
| 1024 | 487.619 us | 283.104 us | 1.722x |

B1, B8, and B32 are native real-model captures. B64, B128, and B1024 repeat
native B32 rows to measure kernel scaling, so those large rows are not an
independently captured request distribution.

Across the full kernel matrix:

- GVR loses at every measured batch for 10K context.
- At 50K and 100K, the crossover is around batch 64.
- At 200K, the crossover is around batch 32.
- GVR matches the exact FP32 selected-value set from `torch.topk`.

## Why a faster kernel does not speed up the model

The standalone kernel speedup and model-forward speedup measure different
things.

First, Amdahl's Law requires the fraction of time on the **serial critical
path**:

```text
speedup = 1 / ((1 - f) + f / s)
```

Here, `s` is the selector speedup and `f` is the selector's serial fraction.
Summing selector kernel durations does not give `f` when selector work overlaps
communication or other GPU work. Making overlapped work faster may leave the
graph's critical path unchanged.

Second, the attempt to time all 21 selector calls from inside the graph changed
the graph itself. It added 42 event nodes and imposed extra ordering:

- At 10K/B1, events added 0.143 ms to baseline but 0.594 ms to GVR. That
  differential explains almost the entire apparent instrumented regression.
- At 199.8K/B1024, events added 14.857 ms to baseline and 17.350 ms to GVR.
  Again, the differential explains almost the entire instrumented regression.

The event-measured selector shares of 1.57% to 9.58% therefore are not valid
Amdahl fractions. The event-free paired forward is the authoritative e2e
measurement.

Third, repeated B32 inputs do not reproduce the hint and admission distribution
of a real B1024 decode wave. Kernel scaling rows are useful for understanding
hardware occupancy, but they cannot be substituted directly into an e2e
prediction.

## FP16 baseline experiment

Allowing the existing baseline top-k kernel to consume FP16 helps mostly at
large repeated batches:

| Batch | FP32 | FP16 | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 10.992 us | 10.767 us | 1.021x |
| 8 | 11.562 us | 11.332 us | 1.020x |
| 32 | 14.603 us | 14.174 us | 1.030x |
| 64 | 29.812 us | 26.648 us | 1.119x |
| 128 | 30.072 us | 26.776 us | 1.123x |
| 1024 | 219.113 us | 181.113 us | 1.210x |

These are 100K-context results. B64 and larger repeat native B32 captures.
FP16 versus FP32 retains **99.9714%** of selected indices, with 0.585 missing
indices out of 2048 per row on average and at most two missing. However, the
exact set changes on 40 of 82 rows, or 48.8%.

This is high index overlap, not an end-to-end model-quality result. FP16 should
not be enabled by default without a model accuracy evaluation.

## What this means

1. **Do not claim an e2e GVR speedup for this model.** The clean result is
   neutral across all tested batch sizes and context lengths.
2. **Do not enable GVR for every size based on standalone kernel results.** It
   would add complexity without measurable model-forward benefit.
3. **Keep the kernel work as useful research.** It demonstrates up to a 2.12x
   top-k win and identifies where GVR amortizes its fixed work.
4. **Future optimization must target the graph critical path.** Promising
   directions are fusing selection with its producer or consumer, reducing
   indexer GEMM/attention cost, or removing synchronization rather than only
   shortening an overlapped selector.
5. **Use event-free paired graphs for future e2e decisions.** Internal timing
   events can materially perturb multi-kernel CUDA graphs.

The detailed experiment history remains in [GVR.md](GVR.md) and
[GVR_ANS.md](GVR_ANS.md). This file is the concise source of truth for the
current conclusions.
