# DeepSeek-V4.1 specialized decode on two GPUs

The experimental `latency_specialized_dep2` executor adapts the NVIDIA
specialized decode to TP1, DP2 and EP2. Each rank owns its request rows;
expert dispatch and combine still use the standard modular all-gather and
reduce-scatter implementation. Local routed experts retain the original
TRTLLM MoE implementation.

## Enable

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 1 --data-parallel-size 2 \
  --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --additional-config '{"decode_backend":"latency_specialized_dep2"}' \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --kv-cache-dtype fp8 \
  --engram-config '{"cpu_offload":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
```

Other model limits, tokenizer and memory settings remain deployment choices.
Set `decode_backend` back to `auto` to disable the executor.

## Supported Scope

- Two SM103 GPUs, TP1/DP2/EP2, CP1/PP1, and modular TRTLLM MoE with linear
  expert placement. No sequence parallelism, EPLB or microbatching.
- DSpark5 with adaptive verification disabled. Sampling is unchanged; the
  TP-sharded Markov sampler is not used across independent DP requests.
- FULL decode graphs, FP8 KV, and the sparse MXFP4 indexer shown above.
- No LoRA, KV transfer, sleep, weight updates or batch-invariant mode.

Unsupported configurations fail at initialization. Pure decode graphs up to
128 token rows per rank use the executor; prefill, eager execution and larger
graphs retain the original layer loop. Request counts and token rows differ:
DSpark5 can produce six target rows and five draft rows per request.

The executor reuses the specialized attention/indexer and Engram prefetch
paths, adds local mHC boundaries without TP collectives, and fuses top-k with
activation quantization before expert dispatch. A rank with no active requests
must still participate in the normal expert collectives.

Performance depends on the per-rank row count and context distribution. This
does not change the default backend or claim a uniform speedup at larger batches.
