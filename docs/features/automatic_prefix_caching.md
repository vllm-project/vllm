# Automatic Prefix Caching

## Introduction

Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

!!! note
    Technical details on how vLLM implements APC can be found [here](../design/prefix_caching.md).

## Enabling APC in vLLM

Set `enable_prefix_caching=True` in vLLM engine to enable APC. Here is an example:

[examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py](../../examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)

## Observing cache usage

When server-side statistics logging is enabled (the default), the Prometheus
endpoint reports cache effectiveness with the `vllm:prefix_cache_queries` and
`vllm:prefix_cache_hits` counters. Prometheus exposes these counter names with
the `_total` suffix.

For Chat Completions, Completions, and Anthropic Messages, per-request cache
usage is opt-in. Start the server with `--enable-prompt-tokens-details`
(default: off). With the flag enabled:

- `/v1/chat/completions` populates `usage.prompt_tokens_details` with
  `cached_tokens` and the vLLM extension `created_cache_tokens`.
- `/v1/completions` populates `usage.prompt_tokens_details.cached_tokens`, but
  does not currently report `created_cache_tokens`.
- `/v1/messages` maps the same details to `cache_read_input_tokens` and
  `cache_creation_input_tokens`. Its `input_tokens` value excludes those
  cached and newly cached tokens, following the Anthropic usage contract.

For streaming Chat Completions and Completions requests, these details appear
in the final usage chunk. Request that chunk with
`stream_options.include_usage: true`, or enable it for every request with
`--enable-force-include-usage`.

Without `--enable-prompt-tokens-details`, non-streaming Chat Completions and
Completions responses set `usage.prompt_tokens_details` to `null`, and
`/v1/messages` omits its cache fields. The Prometheus counters still record
hits when server-side statistics logging is enabled.

## Hybrid Mamba models

With prefix caching enabled, Mamba state is stored only on the Mamba block grid, so a prefix-cache hit can resume only at a block boundary. `--enable-mamba-shared-prefix-checkpoint` also stores a checkpoint at the shared-prefix junction, the point where an earlier request with the same prefix stopped. Requests whose shared prefix ends inside a block can then reuse it.

This helps when many requests share a long system prompt and then diverge. It is off by default, and takes effect only when all of the following hold:

- prefix caching enabled
- EAGLE/MTP speculative decoding on the Mamba group
- `--prefix-match-unit` smaller than the Mamba block size
- the model does not use multi-module MTP

```bash
vllm serve <hybrid-model> \
    --prefix-match-unit 64 \
    --enable-mamba-shared-prefix-checkpoint
```

`--prefix-match-unit` is required. It sets the granularity at which prefix-cache keys are computed. When unset it defaults to the greatest common divisor of the prefix-cacheable KV cache group block sizes. For hybrid Mamba models that is the block size itself, so no sub-block boundary exists and the flag has no effect.

Choose a value that divides the block size of every prefix-cacheable KV cache group, and that is a multiple of the per-state compression ratio for models that use one, such as sparse MLA. vLLM validates both at startup and names the offending sizes in the error. Read the served block size from the startup log. 64 is a reasonable starting point.

### FlashInfer ReplaySSM

FlashInfer ReplaySSM caches recent inputs to reduce full-state writes during Mamba2 decoding. It is disabled by default. Both Model Runners V1 and V2 support prefix caching with ReplaySSM, including MTP speculative decoding.

This example enables ReplaySSM and prefix caching with three MTP draft tokens:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
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

`--enable-prefix-caching` selects Mamba cache mode `align`. To use ReplaySSM without prefix caching, replace this flag with `--no-enable-prefix-caching`. This selects mode `none`. To disable speculative decoding, remove `--speculative-config`.

FlashInfer ReplaySSM has these limits:

- `align` requires BF16 model inputs for its replay rings. The SSM state dtype is independent. The example uses FP16 state with stochastic rounding.
- The logical replay buffer cannot exceed 16 tokens. For speculative decoding, it must contain at least `1 + num_speculative_tokens` tokens.
- Pipeline parallelism, microbatching, and KV connectors are not supported.

Prefix reuse still requires a shared prefix with a saved Mamba checkpoint. The matching rules for hybrid Mamba models above also apply.

### Retaining sliding-window and Mamba checkpoints

`prefix_cache_retention_interval` controls how densely vLLM retains reusable
checkpoints for sliding-window and Mamba cache groups:

- `0` (default) retains request replay boundaries and detected shared-prefix
  junctions.
- A positive value also retains periodic checkpoints every N tokens. The
  accepted alignment is layout-dependent and is validated at startup.
- `None` retains checkpoints densely.

Full-attention and chunked-local cache groups ignore this setting. Setting a
positive value for a model with no sliding-window or Mamba group raises a
startup `ValueError` because the setting would have no effect.

## Example workloads

We describe two example workloads, where APC can provide huge performance benefit:

- Long document query, where the user repeatedly queries the same long document (e.g. software manual or annual report) with different queries. In this case, instead of processing the long document again and again, APC allows vLLM to process this long document *only once*, and all future requests can avoid recomputing this long document by reusing its KV cache. This allows vLLM to serve future requests with much higher throughput and much lower latency.
- Multi-round conversation, where the user may chat with the application multiple times in the same chatting session. In this case, instead of processing the whole chatting history again and again, APC allows vLLM to reuse the processing results of the chat history across all future rounds of conversation, allowing vLLM to serve future requests with much higher throughput and much lower latency.

## Limits

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).
