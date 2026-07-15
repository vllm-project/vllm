# MHA Extend Kernel Repro

This is the isolated script to send to TokenSpeed kernel authors for optimizing
native MHA extend performance against AITER-style request-level attention.

```bash
OUT_DIR=/tmp/rocm_tokenspeed_mha_extend_kernel_repro \
GPU=0 \
WARMUP=10 \
ITERS=50 \
/app/tokspd/tokspd-int/tokenspeedkernelrepro/run_mha_extend_kernel_repro.sh
```

The script writes raw logs plus a markdown summary to:

```text
/tmp/rocm_tokenspeed_mha_extend_kernel_repro/summary.md
```

## Known E2E Serving Gap

The current end-to-end serving gap that motivated this kernel test bed is:

| Backend/config | Concurrency | Output tok/s | Request/s | Mean TTFT ms | Mean TPOT ms | Mean E2E ms | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ROCM_AITER_UNIFIED_ATTN` baseline | 64 | `6009.53` | `5.87` | `384.57` | `10.28` | `10895.94` | `0` |
| `ROCM_TOKENSPEED_MHA`, native extend all, corrected window | 64 | `5602.94` | `5.47` | `665.15` | `10.77` | `11687.73` | `0` |

Gap at C64 versus the accepted AITER baseline:

- output throughput: `-406.59` output tok/s, `-6.77%`
- request throughput: `-0.40` req/s, `-6.77%`
- mean TTFT: `+280.59` ms
- mean TPOT: `+0.50` ms
- mean E2E latency: `+791.79` ms

There was also a same-harness AITER rerun in the extend-comparison directory:

| Backend/config | Concurrency | Output tok/s | Request/s | Mean TTFT ms | Mean TPOT ms | Mean E2E ms | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ROCM_AITER_UNIFIED_ATTN`, same extend-compare harness | 16 | `2341.47` | `2.29` | `377.88` | `6.47` | `6994.85` | `0` |
| `ROCM_TOKENSPEED_MHA`, native extend all, corrected window | 16 | `2139.09` | `2.09` | `641.53` | `6.86` | `7656.93` | `0` |
| `ROCM_AITER_UNIFIED_ATTN`, same extend-compare harness | 64 | `6110.64` | `5.97` | `365.74` | `10.12` | `10715.93` | `0` |
| `ROCM_TOKENSPEED_MHA`, native extend all, corrected window | 64 | `5602.94` | `5.47` | `665.15` | `10.77` | `11687.73` | `0` |

Against the same-harness AITER C64 rerun, TokenSpeed native extend all is
`-507.70` output tok/s (`-8.31%`) and `+971.80` ms mean E2E latency.

## What It Runs

| Workload | Purpose |
|---|---|
| `extend_full_q1` | One-token TokenSpeed native MHA extend request compared directly with AITER unified attention. |
| `extend_full_q8` | Multi-token full-attention TokenSpeed native MHA extend compared directly with AITER unified attention. |
| `extend_sliding128_q8` | Sliding-window multi-token TokenSpeed native MHA extend compared directly with AITER. The vLLM semantic window is `128`; TokenSpeed receives `127`. |
| `mixed_balanced_native_extend` | Request-level mixed decode/extend/prefill shape compared with `ROCM_AITER_UNIFIED_ATTN`. |
| `mixed_balanced_sliding128_native_extend` | Same mixed shape with vLLM semantic sliding window `128`; TokenSpeed decode/extend receives `127` and AITER receives `(127, 0)`. |
| `mixed_prefill_heavy_native_extend` | Larger near-chunk-limit mixed shape compared with `ROCM_AITER_UNIFIED_ATTN`. |

## How To Read The Output

For every benchmark table, values above `1.000` in `Relative to AITER` mean the
TokenSpeed native-extend path is slower than AITER for the same synthetic
request shape.

## Important Scope

This repro does not modify `tokenspeed_kernel_amd`. It calls the current vLLM
wrapper ops:

- `rocm_tokenspeed_mha_extend`
- `rocm_tokenspeed_mha_decode`

For sliding-window workloads, the scripts use the corrected wrapper convention:

```text
vLLM semantic sliding_window = 128
TokenSpeed decode/extend sliding_window = 127
AITER window_size = (127, 0)
```

This avoids benchmarking the old raw-window convention that was already shown
to be numerically mismatched for sliding-window attention.

The expected kernel-author takeaway is not that native extend is always slower.
The actionable cases are:

- TokenSpeed native extend trails AITER in the saved C64 end-to-end serving run;
- these scripts isolate native extend full/sliding and mixed request shapes
  against AITER so the kernel author can optimize the TokenSpeed kernels toward
  that target;
- this package is intended as a kernel optimization test bed, not as evidence
  that TokenSpeed native extend is the source of the full-model accuracy drift.
  Later `VLLM_ROCM_USE_AITER=0` reruns point the accuracy issue at shared AITER
  non-attention/runtime kernels instead.
  
The benchmark shapes are based on the gpt-oss-120b path we were testing:

- heads=64
- kv_heads=8
- head_dim=64
- block_size=64
- dtype=bf16
- mixed decode/extend/prefill shapes around the vLLM chunked-prefill serving path
- sliding-window case uses vLLM semantic sliding_window=128, passed to TokenSpeed as 127
