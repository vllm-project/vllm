# Automatic Prefix Caching

## Introduction

Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

!!! note
    Technical details on how vLLM implements APC can be found [here](../design/prefix_caching.md).

## Enabling APC in vLLM

Set `enable_prefix_caching=True` in vLLM engine to enable APC. Here is an example:

[examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py](../../examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)

## Hybrid Mamba models

Under `--mamba-cache-mode align`, Mamba state is stored only on the Mamba block grid, so a prefix-cache hit can resume only at a block boundary. `--enable-mamba-fine-grained-prefix-cache` also stores a checkpoint at the shared-prefix junction, the point where an earlier request with the same prefix stopped. Requests whose shared prefix ends inside a block can then reuse it.

This helps when many requests share a long system prompt and then diverge. It is off by default, and takes effect only when all of the following hold:

- `--mamba-cache-mode align`
- EAGLE/MTP speculative decoding on the Mamba group
- `--prefix-match-unit` smaller than the Mamba block size
- the model does not use multi-module MTP

```bash
vllm serve <hybrid-model> \
    --mamba-cache-mode align \
    --prefix-match-unit 64 \
    --enable-mamba-fine-grained-prefix-cache
```

`--prefix-match-unit` is required. It sets the granularity at which prefix-cache keys are computed. When unset it defaults to the greatest common divisor of the prefix-cacheable KV cache group block sizes. Under `align` that is the block size itself, so no sub-block boundary exists and the flag has no effect.

Choose a value that divides the block size of every prefix-cacheable KV cache group, and that is a multiple of the per-state compression ratio for models that use one, such as sparse MLA. vLLM validates both at startup and names the offending sizes in the error. Read the served block size from the startup log. 64 is a reasonable starting point.

## Example workloads

We describe two example workloads, where APC can provide huge performance benefit:

- Long document query, where the user repeatedly queries the same long document (e.g. software manual or annual report) with different queries. In this case, instead of processing the long document again and again, APC allows vLLM to process this long document *only once*, and all future requests can avoid recomputing this long document by reusing its KV cache. This allows vLLM to serve future requests with much higher throughput and much lower latency.
- Multi-round conversation, where the user may chat with the application multiple times in the same chatting session. In this case, instead of processing the whole chatting history again and again, APC allows vLLM to reuse the processing results of the chat history across all future rounds of conversation, allowing vLLM to serve future requests with much higher throughput and much lower latency.

## Limits

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).

## Observing cache usage

The prometheus endpoint always reports cache effectiveness via
`vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total`.

Per-request cache usage in API responses is **opt-in**: start the server with
`--enable-prompt-tokens-details` (default: off). With the flag enabled:

- OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`)
  populate `usage.prompt_tokens_details`, e.g.
  `{"cached_tokens": 752, "created_cache_tokens": 0}` on a request whose
  prefix was served from cache (`created_cache_tokens` is a vLLM extension).
- The Anthropic-compatible endpoint (`/v1/messages`) derives its cache
  accounting from the same data: `cache_read_input_tokens` and
  `cache_creation_input_tokens` are populated, and `input_tokens` excludes
  cached tokens per the Anthropic usage contract (requires vLLM >= 0.24.0).

Without the flag, `usage.prompt_tokens_details` is `null` and `/v1/messages`
omits the cache fields entirely — even while the prometheus counters record
hits. If your metrics show cache hits but API responses report zero cached
tokens, check this flag first.

## Prefix-cache retention for sliding-window and Mamba models

For models with sliding-window or Mamba (linear-attention) KV-cache groups,
rolling state means old prefix blocks are continuously freed as the window
moves, which can flush cached prefixes of earlier requests out of the pool.
`prefix_cache_retention_interval` controls how densely reusable checkpoints
are retained for such groups:

- `0` (default): retain only semantic checkpoints (the latest replay boundary
  and shared-prefix junctions).
- A positive value: additionally retain periodic checkpoints every N tokens;
  must be a multiple of the scheduler block size.
- `None`: retain checkpoints densely.

The setting applies only to sliding-window and Mamba cache groups; setting a
positive value for a model with neither (e.g. a pure full-attention model)
fails at startup with a `ValueError` explaining it has no effect there. The
older `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` environment variable is
deprecated and will be removed in v0.29; use the
`prefix_cache_retention_interval` engine argument instead.
