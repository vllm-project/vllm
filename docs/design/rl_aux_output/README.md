# Aux Outputs for RL

An RL trainer recomputes the logprobs of tokens an inference engine sampled. Some of what the engine computed along the way cannot be recomputed by the trainer and has to be shipped with the rollout:

- the experts each MoE layer routed every position to (R3, routing replay);
- sparse-attention indexer selections;
- top-k logprobs and the sampling support set;
- the hidden state before `lm_head`, for full-vocabulary distillation.

This folder specifies how vLLM delivers these "aux outputs" to the trainer at frontier scale, and records the evidence behind it.

| Document | Contents |
|---|---|
| [Consumer contract](consumer.md) | What a trainer gets from `read_sample`, what every mask and sentinel means, and what to do with missing or no-replay cells |
| [Protocol](protocol.md) | What a row is, the invariants, and what the engine, the store, the driver and the trainer each do, event by event |
| [Design space](design_space.md) | The alternatives for each mechanism and why the protocol picks what it picks |
| [Scenarios](scenarios.md) | Sizing: bytes per token, pools, bandwidth and object counts, from a single node to frontier scale |
| [Experiments](experiments.md) | Measurements on DeepSeek-V4.1-Flash: capacity, correctness per design, scale, and the correctness hardening |

## The requirement, from first principles

The trainer replays a row to reproduce the computation that produced a sampled token. So for every position of a trajectory, the right row is the one from **the computation whose result the trajectory actually used**: the forward that produced the KV the later tokens attended to, or that sampled the next token. Everything else follows:

1. **First computation wins, per trajectory.** A position is often computed more than once: the next turn re-prefills an unfinished block, a preempted request recomputes, speculative verification and a later prefill overlap. Only the first computation fed the trajectory, so it is the one delivered; recomputations never overwrite it.
2. **Rows carry their weight version.** Trajectories span many weight updates. The manifest lists, per position range, the version of the weights that computed the delivered rows.
3. **Same-version rows are interchangeable; different versions are not.** Under the train-inference consistent build (#62365), every computation of the same tokens at the same weight version is bitwise identical, so GRPO siblings and forks can share one stored copy of a block. A computation at another version is a different row and is never substituted.
4. **Requested means delivered, or the request fails.** If a request asks for an aux output, the trainer can read every row; if the store cannot keep them, the request fails with an error instead of delivering part of it. A row is never silently replaced by a different computation.
5. **Freed once used.** Rows are stored per session and deleted as soon as the session is consumed.

## Conclusions

- **Delivery is block-wise and session-scoped.** Each full KV block is written when it fills, to `<job>/<session>/<block_hash>`; at request end a tail and a manifest are written, and only then is the response sent. Release deletes the session prefix in one call (0.2–1 ms).
- **The delivery pool doubles as the session's prefix cache (P-S).** A prefix hit is allowed only up to what the session already published, so rows for hits are always in the pool. Cross-session hits (shared system prompts) are truncated, at <1% of prefill. No engine-side row cache is needed.
- **Objects live on the store's local SSD; memory is a cache.** Each Mooncake holder offloads evicted objects to its own NVMe directory, so a listed row is never lost and the store's memory no longer has to hold every session in flight. The driver admits sessions by free disk, releases with force removal, and aborts the step when a holder dies or puts stop landing. At 512 sessions × 14.6K context, an 8 GB memory segment over SSD ran 1–2% slower than a 96 GB pinned pool, with nothing missing. Offload at put time loses objects silently and is not used. Without SSD, objects are pinned and admission follows the pool, as before; every run either way had 0 missing rows.
- **Engine events need explicit handling, and each was found and fixed by test.** Preemption, in-flight weight updates (`pause` + `reset_prefix_cache`), write failures, prefix hits that end mid-block, forks, and speculative decoding each break a naive implementation; the protocol lists what the engine does in each case.
- **Verify against an independent reference.** For prefix-hit positions the engine's own inline rows are read back from the same store, so "delivered = inline" cannot see delivery errors there. Correctness is checked against each trajectory's own first computation, dumped by the engine.
- **Sizes on DeepSeek-V4.1-Flash.** R3 480 B/token; indexer 16 KB raw, 2.5–3.6 KB after 20-bit packing and zstd; hidden 16 KB per generated token stored exactly in the LM head's float32 input (default), or 4.1 KB as optional fp8 (logprob error up to ~0.3).
- **At scale.** 512K context, 108 turns, 13 replicas on one shared Mooncake pool over RDMA: 0 missing and 0 wrong of 3.48e8 rows. Every hop is far below its NIC and memory ceilings; the one slow hop was decoding the indexer on the trainer's CPU, now on the GPU.
- **Engine overhead.** On DeepSeek-V4.1, R3 as built costs about 41% of throughput at the top scenario (512K, 51 sessions per replica), because every new prompt row runs through the replay layers. Partial capture keeps those layers on trailing rows and delivers only the deep cells sampled tokens depend on, as per-request patches (protocol §6.11); with it and CUDA graph `FULL`, R3 in the response has no measurable cost. The further cost of the indexer and hidden states was not their bytes but per-request capture work done under the output path's lock on the writing TP rank, which starved that rank's GPU; capture now runs on its own thread behind one ordered queue (protocol §6.9). On one node, with all arms on the same node, all outputs then cost 2.2% at 512K against 0.5% without the indexer, and 0.3–1.9% over no indexer at 128K. At the top scenario (512K, 51 sessions, 5 replicas, two interleaved rounds), all outputs cost 4.0% and all but the indexer 2.1%; a single top run varies by about 2 points, so only interleaved repeats resolve one change. Delivered cells are bitwise identical to the untrimmed run, and trainer replay of them gives bitwise identical logprobs. The overlay is not yet on this branch.

## Status

The delivery path runs as an overlay on a vLLM build (commit `da8922bf0f`) with Mooncake. Related RFCs: #64380 (delivery protocol), #64383 (R3 as a KV-cache layer).
