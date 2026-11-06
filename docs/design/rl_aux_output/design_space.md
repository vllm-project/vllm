# Design Space

The [protocol](protocol.md) is one point in this space. This page lays out the dimensions, the candidates, and why the protocol picks what it picks. Measurements are in [experiments](experiments.md); sizes in [scenarios](scenarios.md).

## Two segments

- **Cache segment**: for positions served by a prefix-cache hit, the rows must exist as long as the KV they belong to. Only outputs recorded at every position need one (R3, indexer, and hidden states when a teacher scores a student's tokens as prompt).
- **Delivery segment**: from the moment rows are computed until the trainer has consumed them. Every output needs one.

| Output | Positions | Cache segment? | Bytes/token (DeepSeek-V4.1-Flash) | Can the trainer recompute it instead? |
|---|---|---|---|---|
| R3 | all | yes | 480 B | no, it must match the rollout's routing |
| top-k + mask | generated | no | 2 KB + ≤ 1 KB | no, it is the sampling-time distribution |
| indexer | all | yes | 16 KB raw, 2.5–3.6 KB compressed | only under BI with the same weights |
| hidden (full-vocab distillation) | positions the student is trained on | when they are prompt positions of a teacher request | 10 KB bf16, ~5 KB fp8 | no, the trainer has no teacher forward |

## Cache segment: where rows for prefix hits come from

| Option | Approach | Assessment |
|---|---|---|
| C0 | No cache segment; cap hits at `aux_start` and recompute the rest | Simplest. GRPO siblings lose prefix sharing (35% throughput loss measured on a 30B MoE). Recomputation creates second computations of the same positions (see "first computation wins"). |
| C1 | Engine-local block-row cache, allocated and evicted with the KV; blocks missing from the session are written at request end | Chosen. ~1% of the KV for R3 on DeepSeek-V4.1-Flash. Every design delivered exact rows with it. |
| C2 | R3 as an extra KV-cache layer (RFC #64383) | "KV hit ⇒ rows present" across all KV tiers, including offload. On DeepSeek-V4.1-Flash it cost 0.21% of KV capacity. Needs a host for the layer in hybrid KV groups. |
| C3 | Store keyed by global block hash (#57968) | Single copy; but release by batch loses rows still shared by other sessions, and pool pressure silently drops them. |
| C4 | Extra room in GPU KV pages | Spends GPU KV; not recommended. |
| C5 | C1 + C2: a capped KV layer with truncation beyond it | Correct but degrades to C0 behavior when full. |
| C6 | The delivery pool doubles as the session's cache: allow a hit only if this session already published the block | No extra memory. Same-trajectory, fork and GRPO-sibling hits are covered; cross-session hits (shared system prompts) are truncated, costing <1% of prefill at frontier scale. |

### Which prefix hits need a cache segment in RL

1. **Next turn of the same trajectory.** These hits are within `aux_start`; the rows were already delivered.
2. **Forks and tree search.** Under session-scoped keys the parent's rows already sit in the delivery pool, pinned until the session is released.
3. **GRPO siblings.** Treat the group as one session (or use one `n>1` request), and they become case 2.
4. **Across sessions** (a common system prompt). Only these need a real cache segment, and truncating them costs little.

## Delivery segment

### Write granularity

| Option | Approach | Assessment |
|---|---|---|
| W0 | Write everything at request end | Rejected: a write spike at the end, the response waits, host memory is unbounded. |
| W1 | Write each full KV block when it fills | Chosen for KV-following outputs; aligns with session-scoped block keys. |
| W2 | Batch N blocks per object (1–4 MB) | 16–128× fewer objects; rows sit longer in host memory. An option if object rate becomes a limit. |
| W3 | Chunk sampling-following outputs by generated tokens | Chosen for top-k, mask and hidden states. |

### Keys and deduplication scope

| Option | Key | Deduplication | Release |
|---|---|---|---|
| K1 | per request, plus `aux_start` from the client | within a trajectory | per batch, no counting |
| K2 | per session: `<session>/<block_hash>` | within a trajectory tree, without client bookkeeping | delete the session prefix |
| K3 | global block hash | across trajectories | cannot release per batch without reference counting (rows were lost when tried) |

K2 is chosen. Cross-trajectory sharing (common system prompts) is worth ~0.24 GB per node for R3: not worth reference counting.

### Data path

- GPU → host → background put: the current path, used in the experiments.
- With R3 as a KV layer, rows are already in host memory and can be put directly when a block fills.
- GPU staging → RDMA put: the candidate for the indexer.

### Storage tiers

- **Host DRAM pool** (Mooncake segments): R3, top-k, mask, compressed indexer, fp8 hidden.
- **NVMe/SSD**: the overflow tier for indexer and hidden states.
- **Shared-filesystem copy-out**: once a trajectory finishes, copy its blocks to packed files and delete them from the pool. This shrinks the pool and survives trainer checkpoint restore, at the cost of an extra read and write. Cluster-wide write load is ~7 GB/s without the indexer and ~90 GB/s with it.

### Encodings

- R3: uint16 ids (384 experts), or 9-bit packing for a 44% saving.
- Top-k: 17-bit ids plus bf16 logprobs. Mask: CSR.
- Indexer: 20-bit packing plus zstd (measured 4.6–6.7×). Delta encoding does not help on top of zstd.
- Hidden: fp8 with a per-token scale.
- Every field carries a codec name.

### Requested means delivered

1. Backpressure: retry until the write lands, never time out. The engine pauses scheduling when pending bytes exceed a cap.
2. An in-flight cap at the trajectory or group level: `C ≈ 0.75 × pool / max session bytes`. Start from a conservative bound and only lower it from observed sizes. The indexer makes sessions large, which is why it must be compressed or tiered.
3. Deadlock fallback: abort the newest sessions and regenerate them.

### Freed once used

- Delete the session prefix after the step commits, or after copy-out.
- After each weight sync, sweep generations older than the oldest live trajectory.
- The engine deletes request-private objects of undeliverable responses.
- Report the number of keys swept at job end as a leak metric.

### Correctness corner cases

- **First computation wins.** vLLM does not cache unfinished blocks, so the next turn re-prefills `[unfinished block start, aux_start)`. A prefill resuming after a cache hit can route a few rows differently from the decode that sampled those tokens. Keep the rows the session already delivered.
- **Preemption.** A recomputed block rewrites the same key with the same bytes (under BI).
- **MTP rejection.** Rows of rejected tokens are dropped.
- **Stop strings.** The tail is truncated.
- **Prefill/decode disaggregation.** Both sides must use the same block-hash configuration.
- **Multimodal prompts.** Block hashes include image hashes.
- **Weight sync.** `reset_prefix_cache(reset_running_requests=True)` with R3 on must not crash the engine.
- **R3 as a KV layer on hybrid KV groups.** Attach the layer to a group with one slot per token (compression ratio 1).

### Hidden states for full-vocabulary distillation

- Take the `lm_head` input: after the final hyper-connection merge and final norm. Any other tensor makes the trainer's logits mismatch.
- Positions:
  - If the teacher scores the student's tokens by prefill, those are prompt positions of the teacher request. They follow the KV and get prefix hits, the same as R3.
  - If the teacher is the rollout model, they are generated positions.
- Training side: read the hidden states, multiply by the teacher's `lm_head` sharded by vocabulary across TP, and never gather full logits on one rank.

## The five designs compared

All five share the same delivery segment: W1/W3 writes, K2 keys, backpressure, the in-flight cap, and session release. They differ in the cache segment.

| Design | R3 cache segment | Result |
|---|---|---|
| c1 | C1: engine-local block-row cache tied to the KV, with missing blocks back-filled into the session | Exact |
| c0 | C0: cap hits at `aux_start` | Exact after "first computation wins". Loses prefix sharing for GRPO siblings. |
| P-S | C6: the delivery pool as the session cache | Exact after "first computation wins". Output tokens differ from other designs because hit patterns differ. |
| P-B | C2: R3 as a KV-cache layer (on the GPU in this port; 0.21% of KV) | Exact after "first computation wins" |
| P-A | C3: global block hash (#57968), keys-only response | Rows identical to c1 for all requests. Release semantics are unresolved for shared blocks. |

### How each design fails the requirements

| Failure | c1 / c0 / P-S / P-B | P-A |
|---|---|---|
| Delivery pool full | Backpressure; nothing dropped | Cache objects and delivery objects are the same copy. They can be neither evicted (pinned) nor deleted (still hit), so the pool can only back-pressure. |
| Writer process dies | Only that process's undelivered rows are lost; the driver regenerates the trajectory | Every engine sharing that prefix loses the rows |
| Trainer is done | Delete the session; space returns immediately | Blocks may still be hit by other sessions; deleting them causes missing rows, so this needs reference counting or a pool that only grows |

## Not adopted

- **Write at request end, hard-pin forever, fail the request on write errors** (the SGLang output-store approach): equivalent to C0 + W0 + K1. Its good parts are kept:
  - self-describing references;
  - all-or-nothing writes;
  - the engine deleting undeliverable objects;
  - codecs;
  - pre-registered read buffers.
- **Distributed reference counting:** the driver already knows when a session is consumed.
- **Engine-side admission:** the engine cannot tell a new trajectory from the next turn of an old one, so blocking the latter deadlocks.
- **Per-field keys:** they multiply master operations by 3–5×; range reads within one object achieve the same.
