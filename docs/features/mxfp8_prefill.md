# MXFP8 causal prefill

`MXFP8_PREFILL` is an opt-in attention backend for Blackwell (SM100/SM103)
that runs full, uncached prefills with MXFP8 (E4M3 values, E8M0 scales)
matrix products. The KV cache stays BF16 and decode uses FlashAttention.

```bash
pip install "nvidia-cudnn-frontend>=1.33"
vllm serve Qwen/Qwen3-4B --dtype bfloat16 --attention-backend MXFP8_PREFILL
```

## How it works

MXFP8 shares one scale across 32 elements of the reduction axis. For the PV
product that axis is the sequence, so a V scale covers 32 consecutive tokens.
In causal attention a query in the middle of a block would read a scale chosen
using later tokens, so future tokens could change its output.

For query position `i`, let `b = 32 * floor(i / 32)`. Attention is split into
two phases:

1. **Normal phase** (keys `j < b`): cuDNN Frontend's
   `sdpa_mxfp8(causal_exclude_own_block=32)`
   ([NVIDIA/cudnn-frontend#2563](https://github.com/NVIDIA/cudnn-frontend/pull/2563)).
   Every V block it reads lies entirely before the query.
2. **Boundary phase** (keys `b <= j <= i`): Triton/Gluon kernels divide each
   V token by a token-local power-of-two amplitude, fold that amplitude into
   the attention weight, and run the MXFP8 product. The result is merged with
   the normal phase by log-sum-exp.

All QK and PV products use block-scaled E4M3 MMAs with FP32 accumulation;
outputs are BF16.

## When it applies

The MXFP8 path runs for a single position-zero prefill whose length is a
multiple of 128, with head size 128, BF16 activations, the default softmax
scale and no speculative decoding.

Everything else uses FlashAttention: decode, prefix-cache hits, chunked
prefill continuations, batches with more than one request, ALiBi, sliding
windows, sinks, soft caps, KV sharing, batch invariance and context
parallelism. Full-prefill CUDA graph capture also uses FlashAttention.

With chunked prefill or prefix caching enabled, fewer prefills take the MXFP8
path. For benchmarking, disable both
(`--no-enable-chunked-prefill --no-enable-prefix-caching`) in every arm.

## Dependencies

The normal phase needs `nvidia-cudnn-frontend>=1.33`. vLLM pins it below 1.33
for FlashInfer, so install it separately. Without it the backend is reported
as unsupported at selection time. cuDNN Frontend and the boundary kernels are
imported only when an eligible prefill runs.

## Tests

```bash
pytest tests/v1/attention/test_mxfp8_prefill.py -v
```

The tests check that changing any token from a cut point onward leaves all
earlier outputs bitwise unchanged, compare against BF16 SDPA, and run a real
model request followed by native decode.
