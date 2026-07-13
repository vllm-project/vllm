# MHA Extend Kernel Repro

This is the isolated script to send to TokenSpeed kernel authors when discussing
native MHA extend performance:

```bash
OUT_DIR=/tmp/rocm_tokenspeed_mha_extend_kernel_repro \
GPU=0 \
WARMUP=10 \
ITERS=50 \
/app/tokspd/tokenspeedkernelrepro/run_mha_extend_kernel_repro.sh
```

The script writes raw logs plus a markdown summary to:

```text
/tmp/rocm_tokenspeed_mha_extend_kernel_repro/summary.md
```

## What It Runs

| Workload | Purpose |
|---|---|
| `extend_full_q1` | Directly tests a one-token native MHA extend request. This should be decode-like and exposes high fixed request-level overhead in the extend kernel. |
| `extend_full_q8` | Multi-token full-attention extend. This is included to show native extend is not universally slower in isolated kernel timing. |
| `extend_sliding128_q8` | Sliding-window multi-token extend, matching the attention pattern used by the evaluated model path. |
| `mixed_balanced_native_extend` | Request-level mixed decode/extend/prefill shape compared with `ROCM_AITER_UNIFIED_ATTN`. |
| `mixed_prefill_heavy_native_extend` | Larger near-chunk-limit mixed shape compared with `ROCM_AITER_UNIFIED_ATTN`. |

## How To Read The Output

For `Native Extend vs Decode-Decomposed Extend`, values below `1.000` in
`Relative to native extend` mean the decode-shaped alternative is faster than
native MHA extend. Use the `decode_decomposed_*` rows for equivalent extend
comparisons; `pure_decode_same_rows` is included only as a decode-kernel lower
bound for the same number of query rows.

For `Mixed Batch vs ROCM_AITER_UNIFIED_ATTN`, values above `1.000` in
`Relative to AITER` mean the TokenSpeed path is slower than
`ROCM_AITER_UNIFIED_ATTN` for the same synthetic request mix.

## Important Scope

This repro does not modify `tokenspeed_kernel_amd`. It calls the current vLLM
wrapper ops:

- `rocm_tokenspeed_mha_extend`
- `rocm_tokenspeed_mha_decode`
- `rocm_tokenspeed_mha_prefill`

The expected kernel-author takeaway is not that native extend is always slower.
The actionable cases are:

- one-token extend can be much slower than equivalent decode-decomposed work;
- mixed request-level serving shapes with native extend have still trailed
  `ROCM_AITER_UNIFIED_ATTN` in our measurements;
- the safe integration keeps native extend off by default because full-model
  native-extend runs previously regressed GSM8K accuracy.
