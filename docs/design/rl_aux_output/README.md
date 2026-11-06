# Aux Outputs for RL

RL trainers need per-token data that only the inference engine computes. This folder collects the design work on delivering those "aux outputs" from vLLM to the trainer at scale:

- the experts each MoE layer routed to (R3, routing replay);
- sparse-attention indexer selections;
- top-k logprobs and the sampling support set;
- the hidden state before `lm_head`, for full-vocabulary distillation.

| Document | Contents |
|---|---|
| [Scenarios](scenarios.md) | Two sizing scenarios (single node, frontier-scale agentic RL) and DeepSeek-V4.1-Flash measurements: bytes per token, cache and pool sizes, bandwidth, object counts |
| [Protocol](protocol.md) | What the inference side, the training side, the engine and the store each do: request/response, object keys, read/release, in-flight cap, guarantees |
| [Design space](design_space.md) | The design dimensions (cache segment, write granularity, keys, storage tiers, codecs) and the five end-to-end designs that were compared |
| [Experiments](experiments.md) | Experiment matrix, setup, and results on DeepSeek-V4.1-Flash, with a status snapshot |

## Requirements

- **Requested means delivered.** If a request asks for an aux output, the trainer can read it. A full store makes the engine wait; it never drops rows.
- **Freed once used.** Data is stored per session and deleted as soon as the session is consumed.

## Conclusions so far

1. **Split the problem in two.**
   - *Cache segment*: rows for prefix-cache hits must live as long as the KV they belong to.
   - *Delivery segment*: rows live from computation until the trainer has consumed them.
2. **Delivery is written per block, keyed per session.**
   - Each full KV block is written as soon as it fills, to `<job>/<session>/<block_hash>`.
   - At request end the engine writes the tail and a manifest, and only then responds.
   - A sample reads one manifest. Release deletes the session prefix in one call (0.2–1 ms).
3. **"Requested means delivered" comes from flow control, not from dropping.**
   - The writer retries until the write lands.
   - The driver caps in-flight sessions so the pool cannot fill.
   - A deadlock fallback aborts the newest sessions.
   - Result: 0 missing rows and 0 leaked keys in every run, down to a 16 MB pool.
4. **The first computation wins.**
   - vLLM does not cache unfinished blocks, so the next turn re-prefills them, and that prefill can route a few rows differently from the decode that sampled those tokens.
   - Keeping the rows the session already delivered made all five designs deliver rows bitwise equal to the engine's inline rows.
5. **Rows for prefix hits should live and die with the KV.**
   - An engine-local block-row cache allocated with the KV costs ~1% of the KV for R3 on DeepSeek-V4.1-Flash (~1.6 GB per TP4 engine).
   - Capping hits and recomputing only adds second computations.
6. **Sizes on DeepSeek-V4.1-Flash.**
   - R3: 480 B/token.
   - Indexer: 16 KB/token raw, 2.5–3.6 KB after 20-bit packing and zstd.
   - Hidden states: ~5 KB/token as fp8, with full-vocabulary KL < 1e-3.
7. **Check batch invariance before relying on deduplication.** With `deep_gemm_mega_moe`, the sequence-parallel reduce-scatter was not batch-invariant on CUDA; it is fixed by a one-line change.

## Status

The delivery path and the experiments run as an overlay on a vLLM build (commit `da8922bf0f`) with Mooncake. Indexer and hidden-state delivery through the same path is in progress. Related RFCs:
- #64380: delivery protocol;
- #64383: R3 as a KV-cache layer.
