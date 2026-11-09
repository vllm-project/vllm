# Aux Outputs for RL

An RL trainer recomputes the logprobs of tokens an inference engine sampled. Some of what the engine computed along the way cannot be recomputed by the trainer and has to be shipped with the rollout:

- the experts each MoE layer routed every position to (R3, routing replay);
- sparse-attention indexer selections;
- top-k logprobs and the sampling support set;
- the hidden state before `lm_head`, for full-vocabulary distillation.

This folder specifies how vLLM delivers these "aux outputs" to the trainer at frontier scale, and records the evidence behind it.

Read in this order:

| Document | Contents |
|---|---|
| [Scenarios](scenarios.md) | Sizing: bytes per token, pools, bandwidth and object counts, from a single node to frontier scale. §1 (the two scenarios) and §8 (constraints on the design) are enough for a first read |
| [Protocol](protocol.md) | What a row is, the invariants, and what the engine, the store, the driver and the trainer each do, event by event; inline mode for small scale |
| [Consumer contract](consumer.md) | What a trainer gets from `read_sample`, what each field and failure means, and what the trainer must do |
| [Experiments](experiments.md) | Measurements on DeepSeek-V4.1-Flash: capacity, correctness per design, overhead, scale, the store, and the correctness hardening |
| [Design space](design_space.md) | The alternatives for each mechanism, why the protocol picks what it picks, and what was not adopted |

## The requirement, from first principles

The trainer replays a row to reproduce the computation that produced a sampled token. So for every position of a trajectory, the right row is the one from **the computation whose result the trajectory actually used**: the forward that produced the KV the later tokens attended to, or that sampled the next token. Everything else follows:

1. **First computation wins, per trajectory.** A position is often computed more than once: the next turn re-prefills an unfinished block, a preempted request recomputes, speculative verification and a later prefill overlap. Only the first computation fed the trajectory, so it is the one delivered; recomputations never overwrite it.
2. **Rows carry their weight version.** Trajectories span many weight updates. The manifest lists, per position range, the version of the weights that computed the delivered rows.
3. **Same-version rows are interchangeable; different versions are not.** Under the train-inference consistent build (#62365), every computation of the same tokens at the same weight version is bitwise identical, so GRPO siblings and forks can share one stored copy of a block. A computation at another version is a different row and is never substituted.
4. **Requested means delivered, or the request fails.** If a request asks for an aux output, the trainer can read every row; if the store cannot keep them, the request fails with an error instead of delivering part of it. A row is never silently replaced by a different computation.
5. **Freed once used.** Rows are stored per session and deleted as soon as the session is consumed.

## Conclusions

- **Delivery is block-wise and session-scoped.** Each full KV block is written when it fills, to `<job>/<session>/<block_hash>`; at request end a tail and a manifest are written, and only then is the response sent. Release deletes the session prefix in one call (0.2–1 ms).
- **The delivery pool doubles as the session's prefix cache (P-S).** A prefix hit is allowed only up to what the session already published, so rows for hits are always in the pool and store mode needs no engine-side row cache. Cross-session hits (shared system prompts) are truncated: each session computes a shared prefix once itself.
- **Inline mode for small scale.** Rows come back in the response, `[aux_start, rows)` per turn, with no store, session or trajectory. Rows for prefix hits come from an engine-local block-row cache allocated and evicted with the KV (c1), per replica and lost on restart (protocol §12).
- **Objects live on the store's local SSD; memory is a cache.** Each Mooncake holder offloads evicted objects to its own NVMe directory, so a listed row is never lost and the store's memory no longer has to hold every session in flight. The driver admits sessions by free disk, releases with force removal, and aborts the step when a holder dies or puts stop landing. At 512 sessions × 14.6K context, an 8 GB memory segment over SSD ran 1–2% slower than a 96 GB pinned pool, with nothing missing. Offload at put time loses objects silently and is not used. Without SSD, objects are pinned and admission follows the pool, as before; every run either way had 0 missing rows.
- **Engine events need explicit handling, and each was found and fixed by test.** Preemption, in-flight weight updates (`pause` + `reset_prefix_cache`), write failures, prefix hits that end mid-block, forks, and speculative decoding each break a naive implementation; the protocol lists what the engine does in each case.
- **Verify against an independent reference.** For prefix-hit positions the engine's own inline rows are read back from the same store, so "delivered = inline" cannot see delivery errors there. Correctness is checked against each trajectory's own first computation, dumped by the engine.
- **Sizes on DeepSeek-V4.1-Flash.** R3 480 B/token. Indexer 16 KB raw; after 20-bit packing and zstd, 2.5–3.6 KB up to 128K context and ~5.6 KB at 512K. Top-k logprobs at k = 128: 1 KB per generated token. Hidden states 16 KB per generated token, stored exactly as the LM head's float32 input.
- **Top-k logprobs are inline for now.** Returned in the response, they made rollouts 46.8% longer on the one-node 14.6K workload; a store path like that of hidden states is planned (protocol §1).
- **At scale.** 512K context, 108 turns, 13 replicas on one shared Mooncake pool over RDMA: 0 missing and 0 wrong of 3.48e8 rows. Every hop is far below its NIC and memory ceilings; the one slow hop was decoding the indexer on the trainer's CPU, now on the GPU.
- **Full depth on models with sliding-window layers.** On DeepSeek-V4.1, layers past 20 attend through a sliding window that vLLM by default replays on each request's trailing rows only. Exact rows at every position under that replay cost about 41% of throughput at the top scenario. Aux capture therefore runs at full depth: replay off, every new row through all 40 layers, and the window KV prefix-cached as snapshots at prompt end and decode end (protocol §6.11). Every cell is then exact and the delivery path has no layer-specific logic. On one node at 14.6K context (512 sessions × 16 turns), full depth took 58–62% less time than bounded replay in every arm, R3 off included, with every row delivered; decode-end snapshots took another 10.5% off. Against R3 off on the same full-depth engine, all outputs took 25–27% longer; that cost is not yet broken down. Full depth requires append-only histories for multi-turn reuse, and the decode-end snapshot does not yet support speculative decoding.
- **Capture off the output path.** The cost of the indexer and hidden states was not their bytes but per-request capture work done under the output path's lock on the writing TP rank, which starved that rank's GPU; capture now runs on its own thread behind one ordered queue (protocol §6.9). The top-scenario numbers predate full depth: on the bounded-replay build with partial capture, all outputs cost 4.0% and all but the indexer 2.1% over R3 off (512K, 51 sessions per replica, 5 replicas, two interleaved rounds; experiments: top scenario). A single top run varies by about 2 points, so only interleaved repeats resolve one change. Full depth has not yet been run at the top scenario.

## Status

The delivery path runs as an overlay on a vLLM build (commit `da8922bf0f`) with Mooncake `0.3.13.post1`. The code as run in the experiments, with its drivers and tests, is on the branch `rl/aux-output-impl` of the author's vLLM fork, under `tools/rl_aux_output/`: the connector and worker modules, the store writer and the trainer-side reader (`oc_reader.py`), the hidden-state and top-k capture hooks, the decode-end sliding-window snapshot patch, and the experiment drivers and tests. It is a snapshot, not yet split into reviewable changes.

Next: the split into pull requests, with full-depth capture (protocol §6.11) as the design; partial capture is not carried over. Related RFCs: #64380 (delivery protocol), #64383 (R3 as a KV-cache layer).
