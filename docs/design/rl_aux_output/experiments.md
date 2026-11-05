# Aux Output Experiments

Measurements behind the [protocol](protocol.md). The designs are described in [design space](design_space.md).

## Setup

- Model: DeepSeek-V4.1-Flash (40 layers + 3 MTP, 384 experts with top-6, 8 indexer source layers with top-512, 9 KV cache groups), public config.
- vLLM at commit `da8922bf0f`, Model Runner V2, TP4 + EP4 on 4× GB200 per replica.
- Batch invariance on: `VLLM_BATCH_INVARIANT=1`, `VLLM_BI_MXFP8_GEMM=tensor_core`, `VLLM_BI_MXFP8_PREQUANT=1`, `VLLM_BI_BF16_GEMM=tensor_core`. Note this is batch invariance, not the full train-inference consistent configuration of #62365.
- Engine: `kv_cache_dtype=nvfp4_ds_mla`, `attention_backend=FLASHMLA_SPARSE_DSV41`, prefix caching on, MoE backend `auto` unless noted.
- Store: Mooncake (one master; segments on the same node, or one shared pool over RDMA).
- Delivery path: block-wise writes keyed by `<job>/<session>/<name>`, tail and manifest at request end, release by session prefix; built on the block-keyed aux output connector (#45635).

## How correctness is checked

Each request is read back through its manifest and compared three ways:

- **Inline:** the rows the engine returned with the response.
- **Reference:** each trajectory's own first computation, dumped by the engine per step after the rejected speculative suffix is dropped. For a fork, positions before the fork point come from the parent. Rows of drafts the scheduler later truncated are skipped.
- **Version segments:** the manifest's segments must match the weight versions the trajectory saw.

The inline check is blind at prefix-hit positions: the engine reads those inline rows back from the same store, so "delivered = inline" holds even if both are another trajectory's rows. Errors in the hardening section below were only visible against the reference.

Engine events that do not occur on their own in a short test are triggered through the engine's real code paths:

- **Preemption:** one decode-time block allocation fails every 20 scheduler steps, so the scheduler takes its own KV-shortage branch (`running.pop()` + `_preempt_request`). Natural KV pressure does not preempt on this model: admission requires the whole prompt to fit.
- **In-flight weight update:** a thread calls `pause_scheduler("keep", clear_cache=True)` then `resume_scheduler` every 20 s while `generate` runs.
- **Write failures:** the writer fails one write in N.
- **Weight versions:** an in-place weight update after every step perturbs routers, norms and indexer weights; a probe confirms R3, indexer and hidden rows change between versions.

## Workloads

- X1: multi-turn agent (16 sessions × 4 turns), GRPO (4 groups × 8), forks (8 parents, 8 children): 112 requests.
- GRPO regression: 4 groups of 4 trajectories sharing a session, 8 turns, 64K context: 128 requests, 4.65M rows.
- Fork regression: trajectories 1–3 fork from trajectory 0 after its first turn (or after 2 more parent turns, "delayed"): 116 requests, 4.54M rows.
- Preemption regression: 12 sessions × 6 turns, 64K context: 72 requests, 2.71M rows.

## Experiment matrix

| ID | What | Pass criterion | Result |
|---|---|---|---|
| X0 | Output sanity; KV capacity; where an R3 KV layer can attach | real numbers | pass |
| X1 | Correctness per design | 0 missing, 0 wrong | all five designs pass |
| X1b | Row determinism across schedules | bitwise identical | `auto` pass; `deep_gemm_mega_moe` failed, fixed |
| X2 | Throughput with aux outputs on; engine-local store vs Mooncake at 8 replicas; each hop vs its ceiling | measured | no measurable engine cost; trainer-side decode is the slow hop |
| X4/X5 | Requested means delivered; freed once used | 0 missing, 0 leaks | pass down to a 16 MB pool |
| X6 | Indexer compression | ratio, speed | 4.6–6.7× |
| X7 | Hidden states fp8 vs bf16 | KL | fp8 chosen |
| X9 | Trainer logprob recompute with R3 replay | aligned rows | pass |
| Scale | 512K, 108 turns, shared pool | 0 missing, 0 wrong | pass |
| Hardening | Preemption, GRPO, forks, delayed forks, in-flight weight updates, write failures, indexer gaps, writer memory, speculative decoding | 0 missing, 0 wrong vs the reference | pass (P-S) |

## Designs compared (X1)

| Design | Rows for prefix-cache hits come from |
|---|---|
| c1 | an engine-local block-row cache allocated and evicted with the KV |
| c0 | nowhere: hits capped at `aux_start` and recomputed |
| P-S | the delivery pool, as the session's cache |
| P-B | R3 as an extra KV-cache layer (#64383) |
| P-A | Mooncake keyed by the global block hash (#57968) |

| Design | Missing rows | Delivered = inline | Left after release |
|---|---|---|---|
| c1 | 0 | 112/112 | 0 |
| P-S | 0 | 112/112 | 0 |
| P-B | 0 | 112/112 | 0 |
| P-A | 0 | 112/112 (R3 row hashes equal to c1) | 0 |

Without "first computation wins", c0, P-B and c1 each had requests where 1–177 rows differed: exactly `[start of the last unfinished block, aux_start)`, re-prefilled by the next turn. A full prefill from position 0 matched decode bitwise; a prefill resuming after a hit did not always. Generated tokens can therefore differ between designs (41/112 between P-S and c1), since each design changes where prefills resume.

## Capacity (X0)

- KV capacity of a TP4 engine at 0.65 GPU memory utilization: 3.39M tokens.
- Block-row cache sized to the KV: R3 ~1.6 GB per engine (~1% of the KV); indexer ~54 GB raw, ~10 GB compressed.
- R3 as a KV layer reduced KV capacity by 0.21%.

## Always delivered, freed after use (X4/X5)

96 sessions × 4 turns; the trainer reads each finished session and releases it.

| Design | Pool | In-flight cap | Missing | Wrong | Leaked keys | Release p50 |
|---|---|---|---|---|---|---|
| c1 | 64 MB | 46 | 0 | 0 | 0 | 0.4 ms |
| c0 | 64 MB | 46 | 0 | 0 | 0 | 0.4 ms |
| P-S | 64 MB | 46 | 0 | 0 | 0 | 0.3 ms |
| P-S | 16 MB | 11 | 0 | 0 | 0 | 0.2 ms |

## Throughput and overhead

### Designs on one node (X2)

| Design | Agent | GRPO | Fork |
|---|---|---|---|
| c1 | 654 | 474 | 321 |
| P-S | 624 | 428 | 348 |
| P-B | 675 | 394 | 375 |
| P-A | 662 | 434 | 353 |

Output tokens/s on X1; differences are within run-to-run noise.

### Engine-local store vs Mooncake, 8 replicas

Eight TP4 replicas and one pool node. The three arms ran in sequence on the same nodes, with the same workload (128K, 32 turns, 24 sessions per replica, a weight update every step) and the same concurrency.

| Arm | Generation time per replica (median) | Output tok/s, 8 replicas | Put p50 / p99 | Trainer read per replica |
|---|---|---|---|---|
| Engine-local store, R3 returned in the response | 804.7 s | 3,876 | — | in the response |
| Mooncake, R3 | 798.6 s | 3,845 | 0.3 / <1.8 ms | 1.0–1.2 s |
| Mooncake, R3 + indexer + hidden | 799.1 s | 3,846 | 0.3–0.4 / <0.71 ms | 99–108 s |

The engine pays nothing measurable for delivering through the store: the differences are within ±3% run-to-run noise. The full outputs are 5.9 KB per token, about 23 MB/s per replica.

### Each hop against its ceiling

Two nodes, 4× ConnectX-7 NDR (400 Gb/s) per node.

| Hop | Measured | Ceiling | Assessment |
|---|---|---|---|
| NIC (`ib_write_bw`, 1 MB) | 48 GB/s per port, 192 GB/s for 4 ports | 50 GB/s per port | line rate |
| GPU → pinned host | 67 GB/s at 4 MB, 182 GB/s at 134 MB | NVLink-C2C | tens of µs per step for R3 |
| Host block assembly | 19–23 GB/s | one-core memcpy | at the ceiling |
| Mooncake put, 4–16 MB, 8 threads | 57–70 GB/s | 192 GB/s | 30–36% of the node, above one port |
| Mooncake put, one thread | ≤7 GB/s | one port | not zero-copy: the payload is copied into the registered buffer first |
| Mooncake get | ≤14 GB/s at any size or thread count | 192 GB/s | bound by copying into Python bytes |
| Mooncake op, 60 KB (one R3 block) | 200 µs put, 110 µs get | ~30–50 µs (two RPCs + 1 µs transfer) | 4–6× above, irrelevant at ~300 objects/s per replica |
| zstd-3, one thread | 180 MB/s | — | the hop that matters on the write side |
| Indexer decode on the trainer, numpy | 110–210 MB/s | memory bandwidth | the hop that matters overall |

At the top scenario (512K, 51 sessions in flight per replica, 8 replicas) each replica writes ~45 MB/s, 0.2% of the pool node's NICs. Per replica, measured inside the run:

- **One TP rank writes:** put calls equal successes; there are no redundant puts.
- **Writer thread:** 50% busy, 46% of it in zstd. R3 blocks queue ~1.5 s behind the indexer encoding, and the engine paused scheduling on writer backlog for 78–102 s, about 3% of the run. Moving indexer encoding to a 4-thread pool cut the writer's encode wait by 4.1–4.6× on the single-node regression.
- **Engine host time:** 3.1 ms per step to commit outputs (median), and ~10 MB per step from GPU to host.
- **Trainer read:** 857–914 s to read back the 51 sessions (~164 GB), about 184 MB/s, bound by the indexer decode. Decoding on the GPU where replay runs removes this hop (protocol §7).

## Indexer (X6) and hidden states (X7)

A token's indexer row is 8 source layers × top-512 = 4,096 positions.

| Indexer encoding | Size per token | Ratio vs int32 |
|---|---|---|
| int32 | 16 KB | 1× |
| 20-bit packing (on the GPU) | 10 KB | 1.6× |
| 20-bit packing + zstd-3, contexts up to 128K | 2.5–3.6 KB | 4.6–6.7× |
| 20-bit packing + zstd-3, 512K context | ~5.6 KB | ~2.9× (zstd itself ~1.8×) |

Compression depends on context length. At long contexts the selected positions spread over the whole compressed sequence, so zstd gains less, but it still halves the stored bytes at 512K. The encoder stores the packed bytes uncompressed when zstd does not reach 90% of them.

Selections are bitwise identical across runs under batch invariance. Through the delivery path (112 requests) the indexer matched an independent in-engine capture bitwise (551 B stored per token thanks to block sharing).

| Hidden states | Result |
|---|---|
| fp32 `lm_head(hidden)` vs vLLM logits | max abs diff 2.1e-4, not bitwise |
| bf16 | max abs diff 0.062 |
| fp8 e4m3 + per-token scale | full-vocabulary KL mean 4.4e-4–8.7e-4, max 1.9e-3–2.8e-3; argmax agreement 98–100% |

Through the delivery path every hidden row was the exact fp8 requantization; KL mean 7.7e-4, max 0.012.

## Trainer-side replay (X9)

R3 rows for 16 sessions were replayed in a BF16 trainer (TP1/EP16), forward only.

- Alignment: 2% of tokens differ in their expert set at layer 0 when aligned, ~100% when shifted by ±1 position or compared against the wrong layer.
- The real routing gap grows with depth (16% at layer 1, 40–85% deep).

| | Mean abs logprob diff | Max |
|---|---|---|
| Replay off | 0.175 | 8.15 |
| Replay on | 0.082 | 3.09 |

## Scale

P-S, R3 + indexer (GPU packing) + hidden, a weight-version bump after every step, one TP4 replica per 4× GB200, all replicas on one shared Mooncake pool.

| Run | Scale | Requests | R3 / indexer rows | Missing / wrong | Version segments | Hidden | Released |
|---|---|---|---|---|---|---|---|
| 512K | 13 replicas × 51 sessions × 108 turns, 768 GB pool, RDMA over InfiniBand | 71,604 | 3.48e8 each | 0 / 0 | 71,604, 0 failures | last request per session: 42,432 rows, logprob error 0.041 | 663/663 |
| 512K, high fidelity (real text, weights perturbed every step) | 5 replicas × 51 sessions × 108 turns | 27,540 | 1.33e8 each | 0 / 0 | 27,540, 0 failures | 119,630 rows, 0 bad | 255/255 |
| 512K, hardened code, 51 sessions in flight | 8 replicas × 51 sessions × 16 turns, 700 GB pool node + 160 GB per replica node, hard-pinned | 816 per replica | — | 0 / 0 per session, 0 evictions | — | — | 51/51 per replica |

The first two runs predate the hardening below. Their sessions hold one trajectory each; branches and engine events are covered by the regression tests.

The third run exposed two store requirements, now in protocol §9:

- Without pinning, an in-flight cap sized for short-context indexer bytes let 1.3 TB into a 700 GB pool, and Mooncake evicted 1.81M keys.
- Taking down a finished replica's segment removed objects of the replicas still reading.

## Correctness hardening

Each row is a defect found by test or review, the test that exposes it, and the result before and after the fix, on the same workload. "Wrong" is against the reference unless noted.

| Defect | Test | Before | After |
|---|---|---|---|
| A preemption was handled as a finish | forced preemption, 323 preemptions | 57,678 missing, 753 wrong of 2.65M; hidden 14 bad | 0 / 0 of 2.71M; 323 stashed, 323 resumed |
| GRPO samples of one session spliced each other's tails | GRPO, weights updated every step | 19,135 wrong (inline) | 0 |
| In-flight weight update (pause + reset) | reset every 20 s | 511 wrong, hidden 26 bad | 0 / 0 over 59 resets and 598 preemptions (4.65M rows) |
| Write failures: a lost block counted once per later turn; a session-wide error marker failed every sibling; a recompute republished in place of a lost block carried another version's rows under the lost one's labels | 1 in 500 and 1 in 97 writes fail | up to 5,504 wrong; whole sessions failed | 0 wrong; failed manifests fail only their request, at once; lost blocks reported missing |
| Siblings of one session computing a block at another version received the first copy | fork 2 parent turns later | 440,160 wrong | 0 |
| Indexer blocks with rows the request never captured (a hit ending mid-block) were published with pad as valid | GRPO | 118 such blocks per run; invisible to range checks | not published; the computing request publishes them |
| A fork took the parent's latest tail instead of the tail at the fork point | delayed fork | 5,376 wrong (inline) | 0 |
| Writer bookkeeping grew without bound | GRPO with session release | 7,422 version records left | 184 (sessions released after the last snapshot) |
| Speculative decoding: hidden states dropped (1 + num_draft logits rows per request), positions taken from the scheduler's run-ahead token start, hash-lagging blocks and a capture ending on a block boundary left out of the tail, multi-block tails not spliced, the previous request's computation of `aux_start` replaced, a fork's parent-computed block replaced | GRPO and forks with DSpark (`num_speculative_tokens=4`) | up to 504 missing, 46,243 wrong; hidden 127 requests bad | R3 0 missing, 0 wrong of 4.65M; indexer 0 missing; hidden 0 bad; version segments 0 failures |
| A block starting before `aux_start` was shared without a version check, so a block spanning the fork point could hand a fork another trajectory's rows | spec-decoding fork, per-trajectory copies forced | 861 wrong | 0 |
| The fork-point tail was looked up by start position; a later, shorter parent turn overwrote it | spec-decoding fork, per-trajectory copies forced | 7 wrong | 0 |
| Same-version sharing compared sets of versions, so a mixed v0/v1 block matched an all-v1 computation; identical bytes skipped the version check | review; exact per-position comparison | — | unit cases; regression unchanged |
| A failed own copy (`hash~tag`) could be rewritten under its name; a failed indexer object blocked the block's R3 sharing | review; write-failure injection | — | 166 injected failures, all reported, 0 wrong |
| A delayed fork's own copy of a block whose first rows came from a prefix hit was withheld (pad rows), and no other request published it | delayed fork | 3,384,265 indexer rows missing | 0; 112 blocks filled from the session's object |
| Rows a fork served from a prefix hit were labelled with the version inherited from the parent | fork | 10 version-segment failures; 1.69M indexer rows missing | 0 |
| `invalid_from` was lost across a preemption; a parked finish used only the first block of a multi-block tail; hidden chunk metadata assumed full chunks; an abort and a preemption in the same step left a stash | review; forced preemption | — | 89–167 preemptions per run, 0 missing, 0 wrong; 72/72 hidden chunk records exact |
| In-flight weight update combined with preemption | one weight change per step while paused, ~110 preemptions | — | 0 wrong of 4.53M with per-trajectory copies |

KV-load rewind (block-level failure recovery) cannot occur on this model: vLLM refuses block-level recovery with 9 KV cache groups. For single-group models the protocol reports rows from the rewind point as invalid.

## Finding: same-version rows differ without the consistent build

On this build (batch invariance, not #62365), GRPO siblings prefilling the same prompt in different chunks route differently: in one run 2,253 of 5,842 positions differed, and ~4,800 blocks per GRPO run had same-hash, different-content copies. Against the reference, sharing the first copy leaves ~224K rows of a GRPO run (and ~243K of a fork run) delivered from a sibling's computation. Under the train-inference consistent build these computations are bitwise identical and the protocol shares them; keeping a copy per trajectory instead (`OC_ALT=1` in the overlay) delivers every trajectory its own computation.

## Finding: `deep_gemm_mega_moe` broke batch invariance

With `moe_backend="deep_gemm_mega_moe"`, the same prompt alone vs batched gave different R3 rows and greedy tokens. The backend enables MoE sequence parallelism; on CUDA under `VLLM_BATCH_INVARIANT=1`, the batch-invariant branch of `CudaCommunicator.reduce_scatter` is gated on `current_platform.is_rocm()`, so a plain NCCL ring reduce-scatter runs, whose summation order depends on which rank owns the output slice. The four per-rank inputs to the layer-0 post-attention reduce-scatter were bitwise equal while its output was not. Dropping the `is_rocm()` condition made alone / 8 copies / mixed batches bitwise identical in R3 rows, greedy tokens and logprobs. Any configuration enabling MoE sequence parallelism (e.g. DP>1 with TP>1 and EP) should be affected.

## Smaller bugs found building the path

- The manifest was deferred to the next engine step; an idle engine never wrote it.
- Keys used the engine-internal request id (random suffix); use the caller-visible id.
- Duplicate puts from every TP rank retried forever; "key already exists" is success.
- Async scheduling computes one more position after the last sampled token; readers truncate to `L−1`.
- Some MoE implementations call `route` directly, bypassing a router-level replay hook.
- numpy integers in the manifest broke JSON encoding, and the writer only logged the error, so readers timed out; manifests are built from Python ints.
