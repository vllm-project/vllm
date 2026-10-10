# Automatic Prefix Caching

## Introduction

Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

!!! note
    Technical details on how vLLM implements APC can be found [here](../design/prefix_caching.md).

## Enabling APC in vLLM

Set `enable_prefix_caching=True` in vLLM engine to enable APC. Here is an example:

[examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py](../../examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)

## Hybrid Mamba models

Under `--mamba-cache-mode align`, Mamba state is stored only on the Mamba block grid, so a prefix-cache hit can resume only at a block boundary. `--enable-mamba-shared-prefix-checkpoint` also stores a checkpoint at the shared-prefix junction, the point where an earlier request with the same prefix stopped. Requests whose shared prefix ends inside a block can then reuse it.

This helps when many requests share a long system prompt and then diverge. It is off by default, and takes effect only when all of the following hold:

- `--mamba-cache-mode align`
- EAGLE/MTP speculative decoding on the Mamba group
- `--prefix-match-unit` smaller than the Mamba block size
- the model does not use multi-module MTP

```bash
vllm serve <hybrid-model> \
    --mamba-cache-mode align \
    --prefix-match-unit 64 \
    --enable-mamba-shared-prefix-checkpoint
```

`--prefix-match-unit` is required. It sets the granularity at which prefix-cache keys are computed. When unset it defaults to the greatest common divisor of the prefix-cacheable KV cache group block sizes. Under `align` that is the block size itself, so no sub-block boundary exists and the flag has no effect.

Choose a value that divides the block size of every prefix-cacheable KV cache group, and that is a multiple of the per-state compression ratio for models that use one, such as sparse MLA. vLLM validates both at startup and names the offending sizes in the error. Read the served block size from the startup log. 64 is a reasonable starting point.

## Example workloads

We describe two example workloads, where APC can provide huge performance benefit:

- Long document query, where the user repeatedly queries the same long document (e.g. software manual or annual report) with different queries. In this case, instead of processing the long document again and again, APC allows vLLM to process this long document *only once*, and all future requests can avoid recomputing this long document by reusing its KV cache. This allows vLLM to serve future requests with much higher throughput and much lower latency.
- Multi-round conversation, where the user may chat with the application multiple times in the same chatting session. In this case, instead of processing the whole chatting history again and again, APC allows vLLM to reuse the processing results of the chat history across all future rounds of conversation, allowing vLLM to serve future requests with much higher throughput and much lower latency.

## Monitoring

vLLM exports two Prometheus counters for the local prefix cache, both counted in tokens:

- `vllm:prefix_cache_queries` - prompt tokens looked up in the prefix cache.
- `vllm:prefix_cache_hits` - prompt tokens found in the prefix cache.

Like other vLLM counters, they are exposed with a `_total` suffix, so the hit rate over the last 5 minutes is:

```text
sum(rate(vllm:prefix_cache_hits_total[5m]))
/
sum(rate(vllm:prefix_cache_queries_total[5m]))
```

`sum` combines the engines of a data parallel deployment. Add a `model_name` matcher to limit it to one model. This is the same ratio as the "Prefix cache hit rate" in vLLM's periodic log output, except that the log covers the most recent 1000 requests instead of a time window. The [Prometheus and Grafana example](../../examples/observability/prometheus_grafana/README.md) dashboard includes a panel for it.

When reading the hit rate:

- Hits are counted in full blocks (in `--prefix-match-unit` steps on hybrid models that set it), and the last prompt token is always recomputed to produce logits. With a block size of 16, a request whose first 2,010 tokens match a cached prompt gets 2,000 hit tokens.
- If requests share only a common system prompt, the hit rate cannot exceed the system prompt length divided by the prompt length. A 2,048-token system prompt followed by 256 unique tokens tops out at 2048 / 2304, about 0.89, even when every request after the first one hits.
- Each request is counted once, when it is first scheduled. Requests rescheduled after preemption, and requests that skip the cache lookup (for example, because they ask for prompt logprobs), are not counted.
- A falling hit rate while `vllm:kv_cache_usage_perc` stays near 1 can mean that cached blocks are evicted before they are reused.

`vllm:prompt_tokens_by_source` splits prompt tokens by a `source` label with the values `local_compute`, `local_cache_hit` and `external_kv_transfer`. With a KV connector, `vllm:external_prefix_cache_queries` and `vllm:external_prefix_cache_hits` report the external cache the same way.

## Limits

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).
