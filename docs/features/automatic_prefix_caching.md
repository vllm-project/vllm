# Automatic Prefix Caching

## Introduction

Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

APC reuses only the longest run of full blocks at the start of the prompt that is already cached. Each block's hash covers every token before it, so a prompt that differs at one token recomputes that block and every block after it, even if the later text is identical. Technical details on how vLLM implements APC can be found [here](../design/prefix_caching.md).

## Enabling APC in vLLM

APC is enabled by default. vLLM disables it automatically for models and platforms that do not support it. To disable it, pass `--no-enable-prefix-caching` to `vllm serve`, or set `enable_prefix_caching=False` in `LLM`. Here is an example:

[examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py](../../examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)

For hybrid Mamba models, see [Hybrid Mamba Models](hybrid_mamba_models.md).

## Example workloads

We describe two example workloads, where APC can provide huge performance benefit:

- Long document query, where the user repeatedly queries the same long document (e.g. software manual or annual report) with different queries. In this case, instead of processing the long document again and again, APC allows vLLM to process this long document *only once*, and all future requests can avoid recomputing this long document by reusing its KV cache. This allows vLLM to serve future requests with much higher throughput and much lower latency.
- Multi-round conversation, where the user may chat with the application multiple times in the same chatting session. In this case, instead of processing the whole chatting history again and again, APC allows vLLM to reuse the processing results of the chat history across all future rounds of conversation, allowing vLLM to serve future requests with much higher throughput and much lower latency. This only works while earlier messages stay token-identical between rounds and new content is appended at the end.

## Measuring the hit rate

Divide the increase in `vllm:prefix_cache_hits` by the increase in `vllm:prefix_cache_queries` (both count tokens).

For per-request counts, start the server with `--enable-prompt-tokens-details` to get `usage.prompt_tokens_details.cached_tokens` from `/v1/chat/completions` and `/v1/completions` (streaming needs `stream_options: {"include_usage": true}`). `/v1/responses` always reports `usage.input_tokens_details.cached_tokens`.

## Limits

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).
