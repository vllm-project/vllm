# [Perf][Kimi K3] Support DS conv-state layout in fused KDA decode kernel

## Summary

The fused KDA decode kernel (`ops.fused_kda_decode`) previously required the
SD conv-state cache layout, so any deployment that pins
`VLLM_SSM_CONV_STATE_LAYOUT=DS` — required for NIXL point-to-point state
transfer in P/D-disaggregated serving — silently bypassed the fused path and
fell back to the multi-op Triton decode chain for all 69 KDA layers of the
model.

This PR makes the conv-state inner strides launch parameters instead of
compile-time constants, so both SD and DS layouts run through the single fused
kernel (in-proj GEMM → fused KDA decode → out-proj GEMM per layer and step).
Numerics are bit-identical between the two layouts for identical inputs.

## Motivation

- Deployment reality: production K3 serving pins
  `VLLM_SSM_CONV_STATE_LAYOUT=DS` (NIXL PD transfer), and several recent fixes
  treat DS as the supported production layout. Under DS, the fused kernel was
  never used.
- Measured cost of the fallback chain on B200 (TP8 shape, H_local=12,
  K=V=128): 1.16-1.58x slower than the fused kernel depending on batch, plus
  ~3x launch count in the per-layer decode chain (8-9 vs 3 kernels).

## Changes

- `csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu`: `KdaDecodeStrides`
  gains `conv_channel`/`conv_tap`; the 6 conv-state accesses index with both
  strides; host check accepts SD `(1, 3*dim)` or DS `(conv_width-1, 1)`; the
  stride tuple passed to the extern-C launcher grows from 5 to 7 entries.
  `conv_segment_bytes = dim * stride(1) * 2` already yields the correct
  per-plane spacing for both layouts.
- `csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel_rocm.cu`: mirrored
  changes (compile-level parity only; not runtime-tested here).
- `vllm/models/kimi_k3/nvidia/kda.py`: drop the DS exclusion in
  `is_fused_kda_decode_supported`.
- `tests/models/kimi_k3/test_kda.py`: parameterize
  `test_fused_kda_decode_correctness` over conv layout.

No behavior change for SD deployments, speculative decoding, or deepspark
paths (they keep the existing fallback chain).

## Correctness

- `tests/models/kimi_k3/test_kda.py -k fused_kda_decode`: 11/11 pass on B200
  (5 SD + 5 DS layout cases against the Triton reference chain, plus the
  speculative-decode rejection case). DS cases reuse the reference chain
  because `causal_conv1d_update` accepts arbitrary conv-state strides.
- SD vs DS with identical inputs: outputs, conv states, and recurrent states
  are bit-identical (max|diff| = 0.0).
- E2E greedy A/B (24 prompts x 128 tokens, temperature 0, production-shaped
  16xB200 TP8+PP2 deployment): first-token logprobs are bit-identical on
  24/24 prompts; greedy texts share a prefix of 42 tokens on average
  (median 48) before normal bf16 drift flips a borderline token, after which
  the two runs are no longer comparable (different conditioning).
- GSM8K on the patched serving stack (lm_eval 0.4.12 `local-chat-completions`,
  5-shot default, chat template, temperature 0, max_tokens 1500, full
  1319-sample test split): exact_match 0.9689 +/- 0.0048 (flexible-extract),
  0.9697 +/- 0.0047 (strict-match) — matches the frontier-level GSM8K expected
  of Kimi K3 (~97%), so no accuracy regression from running decode on the
  fused kernel with the DS conv-state layout.

## Performance

### Kernel microbenchmark (CUDA graph replay, B200, H_local=12, K=V=128)

| Batch | fallback (us) | fused (us) | Speedup |
|------:|--------------:|-----------:|--------:|
|     1 |          8.97 |       6.15 |   1.46x |
|     8 |         11.00 |       8.96 |   1.23x |
|    32 |         17.99 |      11.42 |   1.58x |
|    64 |         28.24 |      20.54 |   1.37x |
|   128 |         56.33 |      45.39 |   1.24x |
|   256 |        104.41 |      88.87 |   1.17x |
|   512 |        199.06 |     171.60 |   1.16x |

### E2E: 2 nodes x 8xB200, TP8+PP2, FP8 KV cache, DS layout, `FULL_DECODE_ONLY` cudagraphs

`vllm bench serve`, random dataset, 128 input / 512 output tokens,
128 requests, max-concurrency 64, mean of 2 runs per side (no profiler):

| Metric | Baseline (fallback chain) | This PR (fused kernel) | Change |
|--------|--------------------------:|-----------------------:|-------:|
| Mean TPOT | 28.50 ms | 27.85 ms | **-2.28%** |
| Median TPOT | 28.60 ms | 27.91 ms | -2.42% |
| Output throughput | 2130.9 tok/s | 2177.8 tok/s | **+2.20%** |
| Mean TTFT | 807.6 ms | 808.6 ms | +0.1% (prefill untouched, as expected) |

Run-to-run noise is <0.3% on both sides.

### Kernel trace verification

With the DS layout pinned, the patched serving stack runs
`kda_decode_fusion_many_heads_kernel` for every KDA decode layer (5076 calls
in the profiled window, 7.2us/call average, 1.26% of window kernel time); the
decode-side `_causal_conv1d_update_kernel`, `fused_recurrent_kda_packed_decode`
triton kernels and the separate gated-norm kernel disappear from the decode
hot path (prefill-only `_causal_conv1d_fwd_kernel` and FlashKDA prefill remain,
as intended). In the baseline trace the fallback trio costs ~3.5% of kernel
time.

## Test plan

```
pytest tests/models/kimi_k3/test_kda.py -k fused_kda_decode -v
```

Server (as used for the E2E numbers above):

```bash
# on two 8xB200 nodes (rank0/rank1)
VLLM_SSM_CONV_STATE_LAYOUT=DS vllm serve /path/to/Kimi-K3 \
  --served-model-name Kimi-K3 --trust-remote-code \
  --max-model-len 1048576 --gpu-memory-utilization 0.84 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank {0,1} --master-addr <rank0-ip> --master-port 29512 \
  --max-num-seqs 128 --max-num-batched-tokens 32768 \
  --max-cudagraph-capture-size 128 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```
