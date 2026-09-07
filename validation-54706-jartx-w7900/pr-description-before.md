## Problem

Identical inputs to `RDNA3W4A16LinearKernel` can produce different outputs on gfx11 because scalar and WMMA split-K paths accumulate narrowed bf16/fp16 partials with CAS atomics in execution-dependent order.

On W7900D/gfx1100 this was sufficient to change greedy token selection across repeated generations.

## Root cause

Multiple K-split blocks update the same output element.

The current path:

`FP32 partial → bf16/fp16 narrowing → low-precision CAS accumulation`

makes the final value depend on CAS completion order.

The per-split FP32 partials themselves were bit-reproducible.

For the captured workload, ascending, descending, and fixed-shuffled FP32 reductions were bit-identical.

The empirical onset in the fixed-input probe occurred between 2 and 4 concurrent writers.

## Fix

Keep the existing compute kernels and existing dispatch unchanged.

Replace only the split-K epilogue on scalar and WMMA paths:

`FP32 per-split partials → fixed ascending-z FP32 reduction → one final output cast`

`k_split == 1` keeps the original direct store.

Scratch is per-call via the PyTorch caching allocator and row-tiled; there is no persistent `thread_local` state.

## Validation

W7900D / gfx1100:

- determinism regression: 8/8 PASS
- scalar M=1, real Muse q_proj weights: 1 distinct / 100 calls
- WMMA representative: 1 distinct / 100 calls
- Muse eager ctx512: 1 unique / 8 greedy generations
- prior full-generation interception: 0 same-input/different-output events across 16,640 W4A16 GEMM calls
- bf16 + fp16 scalar/WMMA repeatability validated in the full evidence set

After rebase, validated on upstream main `40b2f62061575905aaac8bc360eaea62a4baeb67`.

## Accuracy

Against an fp32 dequantized reference, the deterministic epilogue improved max absolute error at every measured shape; e.g. M=1 improved from about 0.028 to 0.0061.

## Performance

Prior same-dispatch gfx1100 measurements:

- M=1: +4.6%
- M=8: +4.7%
- M=16: -17.2%
- M=64: -5.8%
- M=128: -3.2%
- M=512: +3.7%

The WMMA range generally benefits from removing the contended CAS epilogue.

## Tests

`tests/kernels/quantization/test_rdna3_w4a16_determinism.py`

The regression reproduces on the old atomic implementation and passes with this patch.

## Scope

gfx11 / RDNA3 W4A16 only.

This PR does not change attention routing or paged attention.

A separate residual Muse eager/8192 divergence after the W4A16 fix was traced to invalid V-cache tail consumption in ROCm custom paged attention and is eliminated by #53856:
https://github.com/vllm-project/vllm/pull/53856#issuecomment-5488571510

Addresses the RDNA3 W4A16 nondeterminism investigated in #50603.

Full validation evidence:
https://github.com/AIwork4me/HunyuanOCR-ROCm/tree/validation-50603-rdna3-upstream-cleanup/validation-50603/rdna3-w4a16-upstream-cleanup

## AI assistance disclosure

AI assistance was used for code iteration, validation orchestration, and drafting. I reviewed the final diff, kernel behavior, test results, performance data, and evidence and can explain the change end to end.

