# Automatic Prefix Caching

## Introduction

Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

!!! note
    Technical details on how vLLM implements APC can be found [here](../design/prefix_caching.md).

## Enabling APC in vLLM

Set `enable_prefix_caching=True` in vLLM engine to enable APC. Here is an example:

[examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py](../../examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)

## Diagnosing prefix-cache misses

Prefix-cache diagnostics are disabled by default. For a supported configuration,
enable request-associated DEBUG logs with:

```bash
VLLM_PREFIX_CACHE_DIAGNOSTICS=1 VLLM_LOGGING_LEVEL=DEBUG \
    vllm serve Qwen/Qwen3-0.6B --enable-prefix-caching
```

Each `prefix_cache_diagnostic` record identifies the request and cache group,
cached token count, lookup limit, and one of these outcomes:

- `evicted`: the first missing block key was recently evicted by this pool.
- `no_match`: the lookup missed and the key is not in the bounded eviction
  history. This includes never-cached prefixes, changed tokens or extra keys,
  and evictions older than the history. It does not prove the prompt changed.
- `lookup_limit`: every block actually eligible for lookup hit, or the lookup
  limit permitted no complete block. The next block was not probed. Recomputing
  the last prompt token for logits can produce this outcome even if the entire
  prompt was cached.

The history retains at most `num_gpu_blocks` distinct keys in eviction order.
A successful prefix-cache reset clears it. Diagnostics do not issue additional
resident-cache probes or change block allocation. When disabled, no history is
allocated and no diagnostic records are produced.

The first version supports one full-attention cache group, equal hash and
physical block sizes, and no context parallelism, EAGLE/MTP prefix-block
dropping, or lookahead hashing. Unsupported configurations produce a startup warning and
no diagnostic records. Requests that skip prefix-cache reads are not classified.
If an eligible block has no computed hash, no miss reason is inferred either.

`request_extra_key_kinds` lists only the presence of request metadata such as
`cache_salt`, `lora`, `mm`, and `prompt_embeds`; it is not a per-block explanation
of which metadata caused a miss. Logs do not include token contents, decoded
prompts, cache-salt values, adapter identifiers, or block hashes. A request
retried by the scheduler may produce more than one lookup record. These records
are debug logs, not new fields in the OpenAI response.

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

`--enable-prefix-caching` selects Mamba cache mode `align`. To use ReplaySSM without prefix caching, replace this flag with `--no-enable-prefix-caching`. This selects mode `none`. To disable speculative decoding, remove `--speculative-config`.

FlashInfer ReplaySSM has these limits:

- `align` requires BF16 model inputs for its replay rings. The SSM state dtype is independent. The example uses FP16 state with stochastic rounding.
- The logical replay buffer cannot exceed 16 tokens. For speculative decoding, it must contain at least `1 + num_speculative_tokens` tokens.
- Pipeline parallelism, microbatching, and KV connectors are not supported.

With `--mamba-backend triton` (for example on ROCm), ReplaySSM also supports speculative decoding with or without prefix caching. It has the same speculative-decoding buffer requirement and parallelism limits, has no 16-token buffer cap, and does not support stochastic rounding.

Prefix reuse still requires a shared prefix with a saved Mamba checkpoint. The matching rules for hybrid Mamba models above also apply.

## Example workloads

We describe two example workloads, where APC can provide huge performance benefit:

- Long document query, where the user repeatedly queries the same long document (e.g. software manual or annual report) with different queries. In this case, instead of processing the long document again and again, APC allows vLLM to process this long document *only once*, and all future requests can avoid recomputing this long document by reusing its KV cache. This allows vLLM to serve future requests with much higher throughput and much lower latency.
- Multi-round conversation, where the user may chat with the application multiple times in the same chatting session. In this case, instead of processing the whole chatting history again and again, APC allows vLLM to reuse the processing results of the chat history across all future rounds of conversation, allowing vLLM to serve future requests with much higher throughput and much lower latency.

## Limits

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).
