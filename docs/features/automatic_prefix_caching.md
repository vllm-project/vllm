# Automatic Prefix Caching

## Introduction

Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

!!! note
    Technical details on how vLLM implements APC can be found [here](../design/prefix_caching.md).

## Enabling APC in vLLM

Set `enable_prefix_caching=True` in vLLM engine to enable APC. Here is an example:

[examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py](../../examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)

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

Prefix reuse still requires a shared prefix with a saved Mamba checkpoint. The matching rules for hybrid Mamba models above also apply.

### Resuming past the block grid

In multi-turn and agentic workloads, each turn's prompt extends the previous turn's prompt and output. With prefix caching alone, the next turn resumes at the last block boundary the previous prompt reached. It recomputes the rest of that prompt and everything the previous turn generated. Two options, both off by default, cut this work:

- `--mamba-frontier-checkpoint` caches a finished request's state at its exact end and at the last block boundary its output crossed. The next turn resumes where the previous one stopped.
- `--prefix-cache-eviction gds` evicts cached states by the prefill time they save per block, instead of least recently used first. It needs `--prefill-cost-coefficients D E`, the model's prefill cost: `D` seconds per token plus `E` seconds per token per 100,000 tokens of context. Only the ratio `E / D` changes the order.

```bash
vllm serve <hybrid-model> \
    --mamba-frontier-checkpoint \
    --prefix-cache-eviction gds \
    --prefill-cost-coefficients <D> <E>
```

To fit `D` and `E` for a model and GPU, time the prefill of `n` tokens after `c` tokens of context for a few pairs. For example, use chunks of 8,192 tokens at contexts from 0 to the longest prompts served. Then fit `t = D * n + E * n * (2 * c + n) / 200000` by least squares.

With `--prefix-cache-leaf-last`, GreedyDual-Size also evicts every state that has a deeper cached state on its path before any session's latest state, where its next turn resumes. Such a state is an earlier turn's state or the block-boundary state under a frontier. `--prefix-cache-leaf-idle-seconds S` lets a latest state unused for `S` seconds go with the earlier ones.

The options need `--prefix-cache-retention-interval 0`, so they cannot be combined with `--enable-mamba-shared-prefix-checkpoint`. A `--prefix-match-unit` smaller than the block size is supported. Frontier checkpoints are not supported with KV cache events or context parallelism. With speculative decoding, they need `"truncate_acceptance_at_stop": true` in `--speculative-config` and, for MTP and EAGLE drafters, `"disable_eagle_block_drop": true`, and no KV connector. Otherwise a KV connector, such as the CPU offloading of `--kv-offloading-size`, must use `kv_role` `"kv_both"` and `kv_load_failure_policy` `"recompute"`. vLLM checks every requirement at startup and names the one that is missing. The full list is in [Hybrid State Checkpoints](../design/hybrid_state_checkpoints.md#requirements).

The `vllm:prefix_cache_frontier_hits` counter reports the prefix-cache hit tokens that frontier checkpoints served past the block-aligned hit. How the options work is described in [Hybrid State Checkpoints](../design/hybrid_state_checkpoints.md).

## Example workloads

We describe two example workloads, where APC can provide huge performance benefit:

- Long document query, where the user repeatedly queries the same long document (e.g. software manual or annual report) with different queries. In this case, instead of processing the long document again and again, APC allows vLLM to process this long document *only once*, and all future requests can avoid recomputing this long document by reusing its KV cache. This allows vLLM to serve future requests with much higher throughput and much lower latency.
- Multi-round conversation, where the user may chat with the application multiple times in the same chatting session. In this case, instead of processing the whole chatting history again and again, APC allows vLLM to reuse the processing results of the chat history across all future rounds of conversation, allowing vLLM to serve future requests with much higher throughput and much lower latency.

## Limits

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).
