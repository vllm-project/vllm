# Automatic Prefix Caching

## Introduction

Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

APC reuses only the longest run of full blocks at the start of the prompt that is already cached. Each block's hash covers every token before it, so a prompt that differs at one token recomputes that block and every block after it, even if the later text is identical. Technical details on how vLLM implements APC can be found [here](../design/prefix_caching.md).

## Enabling APC in vLLM

Set `enable_prefix_caching=True` in vLLM engine to enable APC. Here is an example:

[examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py](../../examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)

## Hybrid Mamba models

With prefix caching enabled, Mamba state is stored only on the Mamba block grid, so a prefix-cache hit can resume only at a block boundary. `--enable-mamba-shared-prefix-checkpoint` also stores a checkpoint at the shared-prefix junction, the point where an earlier request with the same prefix stopped. Requests whose shared prefix ends inside a block can then reuse it.

This helps when many requests share a long system prompt and then diverge. It is off by default, and takes effect only when all of the following hold:

- prefix caching enabled
- `--prefix-match-unit` smaller than the Mamba block size

```bash
vllm serve <hybrid-model> \
    --prefix-match-unit 64 \
    --enable-mamba-shared-prefix-checkpoint
```

`--prefix-match-unit` is required. It sets the granularity at which prefix-cache keys are computed. When unset it defaults to the greatest common divisor of the prefix-cacheable KV cache group block sizes. For hybrid Mamba models that is the block size itself, so no sub-block boundary exists and the flag has no effect.

Choose a value that divides the block size of every prefix-cacheable KV cache group, matches the block size of any group that requires block-aligned lookups (such as sliding-window attention), and is a multiple of the per-state compression ratio for models that use one, such as sparse MLA. vLLM validates these at startup and fails if an explicit value would disable partial cache hits. Read the served block size from the startup log. 64 is a reasonable starting point.

### Superseded Mamba state eviction

With prefix caching enabled, vLLM automatically deprioritizes superseded Mamba checkpoints in multi-turn workloads with no extra configuration:

- **Eviction order**: `uncached -> deprioritized (FIFO) -> normal LRU cached`. Deprioritized blocks keep their prefix hashes while free blocks remain and are only evicted ahead of normal LRU-cached blocks under memory pressure.
- **One-turn fallback protection**: when a finished request advances a conversation's cached Mamba tip, the immediate predecessor (`t-1`) stays in normal LRU to shield against small tail rollbacks (such as tool retries), while states two or more turns behind (`t-2` and older) are deprioritized.
- **Shared-prefix protection**: shared-prefix junctions and multi-branch fork points keep normal LRU priority.

### FlashInfer ReplaySSM

FlashInfer ReplaySSM caches recent inputs to reduce full-state writes during Mamba2 decoding. It is disabled by default. Both Model Runners V1 and V2 support prefix caching with ReplaySSM, including MTP speculative decoding.

This example enables ReplaySSM and prefix caching with three MTP draft tokens:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --tensor-parallel-size 2 \
    --max-model-len 12288 \
    --dtype bfloat16 \
    --mamba-backend flashinfer \
    --mamba-ssm-cache-dtype float16 \
    --enable-mamba-cache-stochastic-rounding \
    --mamba-cache-philox-rounds 5 \
    --use-replayssm \
    --replayssm-buffer-len 16 \
    --enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

`--enable-prefix-caching` also caches Mamba states at block boundaries. To use ReplaySSM without prefix caching, replace this flag with `--no-enable-prefix-caching`. To disable speculative decoding, remove `--speculative-config`.

FlashInfer ReplaySSM has these limits:

- Prefix caching requires BF16 model inputs for its replay rings. The SSM state dtype is independent. The example uses FP16 state with stochastic rounding.
- The logical replay buffer cannot exceed 16 tokens. For speculative decoding, it must contain at least `1 + num_speculative_tokens` tokens.
- Pipeline parallelism, microbatching, and KV connectors are not supported.

With `--mamba-backend triton` (for example on ROCm), ReplaySSM also supports speculative decoding with or without prefix caching. It has the same speculative-decoding buffer requirement and parallelism limits, has no 16-token buffer cap, and does not support stochastic rounding.

Prefix reuse still requires a shared prefix with a saved Mamba checkpoint. The matching rules for hybrid Mamba models above also apply.

### Gated DeltaNet ReplaySSM

Qwen3.5 and Qwen4Exp models (Gated DeltaNet hybrids) support ReplaySSM for standard decoding with `--mamba-backend triton`, with or without prefix caching:

```bash
vllm serve nvidia/Qwen3.5-122B-A10B-NVFP4 \
    --mamba-backend triton \
    --use-replayssm \
    --enable-prefix-caching
```

Speculative decoding, `--enable-mamba-cache-stochastic-rounding`, and a `--replayssm-buffer-len` above 32 are not supported. ReplaySSM pays off at serving batch sizes. On one GB300, Qwen3.5-122B-A10B-NVFP4 decodes 1.1–1.3x faster at 64–512 concurrent requests, and Qwen3.8-Flash-Next-NVFP4 1.1–1.2x faster at 256–512; with a single request both are 3–7% slower.

The replay rings add 8–9% of an FP32 SSM state to each request's state with the default 16-token buffer. Where the existing padding of the Mamba page cannot hold them, the attention block grows and the KV cache holds up to 7% fewer tokens. With BF16 models the rings keep the cached inputs in FP16, which is more precise than BF16 but has a smaller range.

With `--prefix-match-unit`, a request can resume inside a Mamba block, so ReplaySSM also flushes its exact state at every prefix match unit. A unit that is a multiple of the buffer length adds no flushes.

## Example workloads

We describe two example workloads, where APC can provide huge performance benefit:

- Long document query, where the user repeatedly queries the same long document (e.g. software manual or annual report) with different queries. In this case, instead of processing the long document again and again, APC allows vLLM to process this long document *only once*, and all future requests can avoid recomputing this long document by reusing its KV cache. This allows vLLM to serve future requests with much higher throughput and much lower latency.
- Multi-round conversation, where the user may chat with the application multiple times in the same chatting session. In this case, instead of processing the whole chatting history again and again, APC allows vLLM to reuse the processing results of the chat history across all future rounds of conversation, allowing vLLM to serve future requests with much higher throughput and much lower latency. This only works while earlier messages stay token-identical between rounds and new content is appended at the end.

## Measuring the hit rate

Divide the increase in `vllm:prefix_cache_hits` by the increase in `vllm:prefix_cache_queries` (both count tokens).

For per-request counts, start the server with `--enable-prompt-tokens-details` to get `usage.prompt_tokens_details.cached_tokens` from `/v1/chat/completions` and `/v1/completions` (streaming needs `stream_options: {"include_usage": true}`). `/v1/responses` always reports `usage.input_tokens_details.cached_tokens`.

## Measuring cache effectiveness

The Prometheus-compatible `/metrics` endpoint exposes two cumulative token
counters:

- `vllm:prefix_cache_queries`: prompt tokens queried in the prefix cache.
- `vllm:prefix_cache_hits`: prompt tokens found in the prefix cache.

Their ratio measures token reuse, rather than the fraction of requests that had
at least one hit. For example, this PromQL query calculates the aggregate hit
ratio over a five-minute window:

```promql
sum(rate(vllm:prefix_cache_hits[5m]))
/
clamp_min(sum(rate(vllm:prefix_cache_queries[5m])), 1)
```

Use counter deltas over an isolated interval when comparing two fixed
workloads. Keep request ordering, prompt and output lengths, concurrency, and
the initial cache state constant. A fresh engine provides the cleanest empty
cache. In a development-only environment, the cache-management API can reset
the prefix cache when no requests are active; do not enable development
endpoints in production.

Report the token hit ratio together with time to first token (TTFT), request
latency, and token throughput. A high hit ratio does not imply the same
percentage improvement in end-to-end performance: APC removes shared prefill
work, while queueing, scheduling, and decoding still take time. Cache capacity,
eviction, request routing across replicas, and full-block alignment can also
change the observed ratio.

## Limits

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).
