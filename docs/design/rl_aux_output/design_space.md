# Design Space

The [protocol](protocol.md) is one point in this space. This page lays out each decision, the alternatives, and why the protocol picks what it picks. Measurements are in [experiments](experiments.md); sizes in [scenarios](scenarios.md).

## Two segments

A row has to exist for two different reasons:

- **Cache segment.** For positions served by a prefix-cache hit, the rows must exist as long as the KV they belong to. Only outputs recorded at every position need one (R3, indexer, and hidden states when a teacher scores a student's tokens as prompt).
- **Delivery segment.** From the moment rows are computed until the trainer has consumed them. Every output needs one.

| Output | Positions | Cache segment? | Bytes/token (DeepSeek-V4.1-Flash) | Can the trainer recompute it? |
|---|---|---|---|---|
| R3 | all | yes | 480 B | no, it must match the rollout's routing |
| top-k + mask | generated | no | 2 KB + ≤ 1 KB | no, it is the sampling-time distribution |
| indexer | all | yes | 16 KB raw, 2.5–3.6 KB compressed | only under BI with the same weights |
| hidden | generated, or teacher prompt positions | in the teacher case | 16 KB float32 (exact) | no |

## Cache segment: where rows for prefix hits come from

Which hits need a cache segment in RL:

1. **The next turn of the same trajectory.** The rows were already delivered by the previous turn.
2. **Forks and tree search.** The parent's rows sit in the delivery pool until the session is released.
3. **GRPO siblings.** Same session, so the same as 2.
4. **Across sessions** (a shared system prompt). Only these need a separate cache.

| Option | Approach | Assessment |
|---|---|---|
| C0 | No cache; cap hits at `aux_start` and recompute | Loses prefix sharing (35% throughput loss measured on a 30B MoE); creates second computations |
| C1 | Engine-local block-row cache allocated and evicted with the KV; blocks missing from the session written at request end | Exact; ~1% of the KV for R3 on DeepSeek-V4.1-Flash; ~54 GB raw for the indexer per engine |
| C2 | R3 as an extra KV-cache layer (#64383) | "KV hit ⇒ rows present" across KV tiers including offload; 0.21% of KV capacity; needs a host group with one slot per token |
| C3 | Store keyed by global block hash (#57968) | One copy, but per-batch release loses rows still shared, and pool pressure drops them |
| C6 | **The delivery pool doubles as the session's cache**: a hit is allowed only up to what the session already published | **Chosen.** No extra memory, works for the indexer too. Cases 1–3 are covered; case 4 is truncated (<1% of prefill at frontier scale) |

## Write granularity

| Option | Approach | Assessment |
|---|---|---|
| W0 | Everything at request end | A write spike, the response waits, unbounded host memory |
| W1 | **Each full KV block when it fills** | **Chosen** for KV-following outputs |
| W2 | N blocks per object | Fewer objects, rows stay longer in host memory; an option if the object rate limits |
| W3 | **Chunks of generated positions** | **Chosen** for top-k, mask and hidden |

## Keys and sharing

| Option | Key | Sharing | Release |
|---|---|---|---|
| K1 | per request | within a trajectory, client bookkeeping | per batch |
| K2 | **per session: `<session>/<block_hash>`** | **within a session** | **delete the session prefix** |
| K3 | global block hash | across sessions | needs reference counting |

K2 is chosen. Sharing across sessions is worth ~0.24 GB per node for R3, not worth reference counting.

**Sharing inside a session.** A block hash names tokens, not a computation. Two trajectories of one session can compute the same tokens differently in two ways:

- **Same weight version.** Under batch invariance without the consistency guarantees of #62365, a different chunk split routes differently: on DeepSeek-V4.1-Flash, a GRPO group's siblings prefilling the same prompt in different chunks differed in ~40% of the rows of the affected blocks. Under the train-inference consistent build these computations are bitwise identical, so one copy serves all trajectories. The protocol relies on that build.
- **Different weight version.** A trajectory computing the same tokens after a weight update has different rows, and its sampling used them. Sharing the earlier copy would deliver another version's routing. The trajectory keeps its own copy.

| Option | What a trajectory reads for a shared hash | Assessment |
|---|---|---|
| S0 | the session's first copy, always | Wrong across versions |
| S1 | its own computation, always (one copy per trajectory) | Correct, but stores identical rows many times under the consistent build |
| S2 | **the first copy if same version, otherwise its own** | **Chosen.** One copy per version |

Within one trajectory the rule is simpler: the first computation wins, and any recompute of a block the trajectory already holds is dropped.

## Weight versions

Rows are labelled with the version of the forward that computed them, stamped when the forward is prepared (a weight update can land between prepare and commit). The manifest lists segments `[start, version]`; for shared blocks the segments come from the stored object's record, so the label always describes the bytes the reader gets. Deleting by generation is not an option: trajectories span many versions (~107 at frontier scale).

## Failure policy

| Event | Option | Assessment |
|---|---|---|
| Memory full | drop rows / fail the request / block / **offload to SSD** | **Offload** on eviction (never at put, which loses objects); without SSD, **block** with an in-flight cap |
| Holder dies or disk fails | restart and continue / **abort the step** | A dead holder's objects are listed but unreadable, and a failing disk makes every put fail: **abort** on holder death or when puts stop landing |
| Any write fails | report the missing rows / substitute a later recompute / **fail the request** | A recompute can be of another version and its label would be wrong. Reporting missing rows keeps the rest of the request but needs per-row bookkeeping on both sides; failures are rare, so for now the request fails as a whole, by an **error marker per request** (a session-wide marker would fail every sibling) |
| Preemption | finish the request / **stash and resume** | Finishing loses rows computed before the preemption: **stash** |
| KV-load rewind | crash / **mark invalid from the rewind point** | Rows computed on invalid KV are reported, never delivered |

## Data path and tiers

- GPU → host → background put: the current path. With R3 as a KV layer, rows are already on the host when a block fills. GPU staging → RDMA put is the candidate for the indexer.
- **Store: local SSD under a memory cache (adopted).** Each Mooncake holder's segment is a memory cache over its own NVMe directory, with offload on eviction, for every output. Pinned host memory alone was the earlier design; it caps in-flight sessions at a fraction of host memory and cannot hold the indexer at frontier scale (scenarios §4). Offload at put time, the other offload mode, evicts objects before their disk write completes and was rejected (experiments: aux objects on SSD).
- A shared-filesystem copy-out once a trajectory finishes would survive checkpoint restore; it is not needed for delivery and is not built.

## Encodings

- R3: uint16 ids (384 experts); 9-bit packing would save 44%.
- Top-k: 17-bit ids plus bf16 logprobs; mask: CSR.
- Indexer: 20-bit packing plus zstd (4.6–6.7×); packing on the GPU, zstd on the host. Delta encoding does not help.
- Hidden: the exact `lm_head` input, uncompressed.
- Every field carries a codec name.

## Hidden states for full-vocabulary distillation

- Take the `lm_head` input: after the final hyper-connection merge and the final norm.
- With speculative decoding a request has one logits row per verified position; capture all logits rows of the batch with per-request offsets and keep the accepted ones.
- Training side: multiply by the teacher's `lm_head` sharded by vocabulary across TP; never gather full logits on one rank.

## The five designs compared

All five share the delivery segment (W1/W3, K2, backpressure, admission cap, session release) and differ in the cache segment.

| Design | Cache segment | Result |
|---|---|---|
| c1 | C1 | exact |
| c0 | C0 | exact with "first computation wins"; loses prefix sharing |
| P-S | C6 | exact with "first computation wins"; the design hardened and run at scale |
| P-B | C2 | exact with "first computation wins" |
| P-A | C3 | rows identical to c1; release semantics unresolved for shared blocks |

| Failure | c1 / c0 / P-S / P-B | P-A |
|---|---|---|
| Delivery memory full | offload to SSD (backpressure without SSD), nothing dropped | cache and delivery are one copy: neither evictable nor deletable |
| Writer process dies | only that process's undelivered rows; the trajectory is regenerated | every engine sharing the prefix loses them |
| Trainer done | delete the session | shared blocks need reference counting |

## Not adopted

- **Write at request end, hard-pin forever, fail the request on write errors** (the SGLang output-store approach). Its good parts are kept: self-describing references, the engine deleting undeliverable objects, codecs, pre-registered read buffers.
- **Distributed reference counting:** the driver already knows when a session is consumed.
- **Engine-side admission:** the engine cannot tell a new trajectory from the next turn of an old one; blocking the latter deadlocks.
- **Per-field keys:** 3–5× more master operations; range reads within one object do the same.
- **Recomputing to fill gaps:** recomputation is never a substitute for the first computation (different versions, different chunking, invalid KV).
