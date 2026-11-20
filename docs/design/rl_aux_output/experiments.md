# Experiments

This page has four parts: how the design is validated, the results so far, the items still open, and earlier measurements that still inform it. The design is in the [protocol](protocol.md); alternatives in [design space](design_space.md).

## Validation

### Setup

- Hardware: GB200 nodes.
- Models: Qwen3-30B-A3B with batch invariance, so rows are compared bitwise; DeepSeek-V4-Flash at TP4 with MTP (one draft token), whose sparse sliding-window attention has no batch-invariant mode on this base, so its rows are checked structurally (row counts, and every position's rows of the hash-routed layers, which depend on the tokens only); gpt-oss with batch invariance for the version-label runs.
- Store: Mooncake in standalone store mode for objects, on its own master.
- Engine: prefix caching on, prefix replay off, `prefix_cache_retention_interval` set to `null`.

### Checks

Each run reads back every request's objects through `aux_output_keys` and checks:

| Check | Pass criterion |
| --- | --- |
| Coverage | The keys of every response cover `[0, num_tokens - 1)`; every key is readable. |
| Names | The versions in `aux_version_segments` equal the versions the harness applied to those positions: the stated versions for reused positions, the computing version for computed ones. |
| Rows | Every object equals a reference: the same tokens computed from position 0 without hits from other requests, at the weight version of each segment. Bitwise with batch invariance, structural otherwise. |
| One copy | The number of objects under the key prefix equals the number of distinct names written. |
| Hits | Prefix tokens served after each update from the GPU cache or a connector, against the tokens that could be served. |
| Release | After a pause, removing a session by its key pattern leaves no object of the session, and none appears later. |

The same properties are checked in unit tests against an in-memory store: a test engine stamps every row with its position and the weight version that computed it, and asserts at the end that responses cover their positions and, unless the test computes positions again under the names they keep, that every key was written with one payload and that each object's rows carry the versions its name states. Every fix on the branch adds a test that fails without it.

### Plan

| ID | What |
| --- | --- |
| V1 | Connector matrix: multi-turn sessions, weight version changed between turns, prefixes reused from each connector keyed by block hashes |
| V2 | Sessions across replicas: two replicas, one object pool and one KV store, sessions alternating replicas |
| V3 | Keep and reset: pause in each mode, prefix-cache reset with running session requests, tiers cleared with sessions continuing |
| V4 | Overhead at the top scenario, sessions on against off, including single-sequence prefill at 200K and 1M and long-context decode, with all fields |
| V5 | RL weight-update loop: pause, sleep, weight update, `update_weight_version`, wake, next turns with their segments, trainer replay |
| V6 | Stress: preemption, asynchronous KV loads across an update, MTP, siblings across an update, aborts, tier eviction, a full object pool |

### Results

All runs are on the full stack.

| ID | Result |
| --- | --- |
| V1 | Qwen3-30B-A3B with batch invariance and DeepSeek-V4-Flash, each with the offloading connector (a CPU tier, and a CPU tier with a file-system tier), simple CPU, the Mooncake store connector with and without `save_decode_cache`, and multi. After every weight update every prefix token hit: 122,880 of 122,880, and 120,832 without the decode cache, which recomputes the last decode block of each turn ([protocol §6](protocol.md#6-hits-kv-tiers-and-weight-updates)). Every key read back, no key written twice, rows exact with batch invariance and the hash-routed layers exact without. Hidden states and top-k logprobs are byte-identical to a fresh computation. |
| V2 | Turns sent to a second replica hit the KV the first replica stored. |
| V3 | Every arm of V1 ran a `keep` pause, a sleep and a prefix-cache reset. A reset renamed exactly the objects it made the engine compute again: 960 per phase on Qwen3-30B-A3B (8 sessions × 120 objects) and 480 on DeepSeek-V4-Flash. On a real Mooncake store, a pause followed by `remove_by_regex` with `aux_session_pattern` removed all 193 objects of the session, and none appeared later. |
| V4 | DeepSeek-V4-Flash, TP4: 51 sessions growing from 72–88K to 205–406K positions. Without sleeps, makespan about +5%, with first-turn TTFT and single-sequence prefill TTFT at 200K and 1M within 1%. With a sleep and a weight-version change every 12 s, makespan +4.8% to +5.8%. The extra cost grows with resets × the positions computed again: after a reset a session request sends the hashes from its first computed block on, and `salt_from_hit` hashes again from its hit. The pause drain added about 0.3 s per update in this scenario and nothing measurable in short runs. |
| V5 | Bitwise runs with a checker that compares the version that computed every row with the version its object's name states: gpt-oss with batch invariance and DeepSeek-V4-Flash without, 0 stale labels. Workload time equal to the build without the session layer, within run-to-run noise. |
| V6 | Regression scenarios pass: a last token that completes a block, preemption across two weight updates, and a reset right after requests finish. A store that refuses writes and recovers is covered ([Mooncake behavior](#mooncake-behavior)); a pool that stays full until the stall deadline is an open item. |

**Pause.** The writer's drain itself costs about 5 µs when nothing is queued, on top of one worker round trip. With an in-memory store it took 3 ms for 64 queued objects of 8 KB and 27 ms for 512; with 1 ms per put, 0.07 s and 0.54 s. The wait is the time to write what is queued.

### Why same-node interleaving

Two sessions-off runs of one build differed by 4.1% in TTFT, and an earlier comparison that always ran the sessions-on arm first showed TTFT +5–11% that disappeared when the arms alternated. Arms of an overhead comparison run on one node, alternating; differences of a few points between single runs are not conclusions.

## Open items

- **Prefix-cache collapse under backlog.** Under a backlog of requests the prefix-hit rate collapses, with aux outputs on and off. The offloading connector's tier holds only about 1M tokens for DeepSeek-V4-Flash, because each key takes the slot of a full block.
- **Sleep during connector loads.** A sleep while KV connector loads are in flight raises; callers must retry it.
- **A full object store** stalls writes; when nothing frees it (a memory-only pool without releases, or failing SSD writes), the engine drops objects from the stall deadline on (protocol §5). Renaming after a pause that clears the cache and changes the version multiplies the footprint: the objects computed again are written under new names, and the old ones stay until their session is released, and on SSD until the job ends.
- **Cost of a reset.** The hashes and `begin_step` work for the positions computed again (V4).
- **Late decode blocks in the Mooncake store.** With `save_decode_cache`, session requests and asynchronous scheduling, in runs with two engines per node, the next turn of an earlier revision missed the last block of the previous turn in 7 of 11 runs. Without aux outputs 0 of 3 runs missed; with asynchronous scheduling off, or 60 s between turns, every run hit. The blocks were stored and their hashes at store and lookup were equal; they became visible late. The V1 runs with the decode cache hit every token.

## Earlier measurements

These come from earlier implementations of aux output delivery; each entry says which. They are kept where the property measured does not depend on how objects are named.

### Mooncake behavior

| Behavior | Measured | Used in |
| --- | --- | --- |
| The same key written twice | Both puts return success. Sequential writes keep the first object; concurrent writes keep one of them, not determined which, never a mix. On an SSD-first store, 1,833 puts of keys already on SSD or being offloaded all returned success and kept the first value. | immutable objects (protocol §5) |
| Removal right after a read | Without force nothing is removed (read lease, error −706); with force everything is. | release with force (protocol §7) |
| `remove_by_regex` during writes | Silently skips keys still being written and returns only the count it removed. Scans every key: 36–74 ms at 200K keys. | release only after a pause (protocol §7) |
| A holder process killed | The master still lists its keys; every get returns empty; puts fail. | missing objects fail the read; failed puts are retried, since a replaced holder takes them (protocol §5) |
| Master killed, or a full store, with the previous implementation | The request whose write fails gets an engine error; nothing is lost silently. | historical: the writer now retries and then drops instead (protocol §5) |
| A store that refuses writes for 90 s, then recovers | With a 180 s stall deadline the engine kept running; over 6 steps all 25,116 keys were present and matched a run without the stall bit for bit (tokens and objects). A reader with a 60 s timeout reported objects missing while writes were refused. | read timeout of 600 s, no shorter than the stall deadline (consumer contract) |
| A client created before the engine in one process | CUDA initialization fails in the engine's workers. | consumer contract |
| Reader after the master dies | Hangs for more than 70 s. | reads need their own timeout |
| Removal on an SSD-first store (0.3.14) | Removing half the objects and writing the same amount again grew disk use from 6.28 to 9.42 GB; releasing sessions and continuing to write grew it from 4.52 to 8.39 GB. `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` had no effect with disk eviction off. | SSD sized by bytes written, disk eviction off (protocol §8) |
| SSD offload at put | Lost 2,976 of 100K small objects (3%) with no error. Offload on eviction kept 100K of 100K. | SSD offload only on eviction, with objects not hard-pinned (protocol §8) |

The removal and offload tests used Mooncake 0.3.13.post1, the disk-space tests 0.3.14; the same-key, full-store and client-ordering tests ran with the previous session implementation, except the SSD-first rewrites in the same-key row and the store that refuses writes for 90 s, which ran with the current one.

### Previous session implementation

The previous design put the weight version into the object key and kept per-block version labels at the GPU cache and at the offloading connector's boundary (design space, option V2). On DeepSeek-V4-Flash, TP4, with the offloading connector's CPU tier:

- Top scenario: 51 sessions in flight, about 400K positions per session, 10 turns: 0 errors over 20.4M checked positions. Two same-node pairs against aux off: −0.1% and +3.4% wall time (mean +1.6%), ITL +0.4 ms on average.
- Session isolation: the same prompt in a second session hit 0 tokens.

### Engine findings that still apply

- **Retention interval.** On DeepSeek-V4-Flash with the default `prefix_cache_retention_interval` (0), sliding-window and compressed groups keep state only at the end of each prompt, so the next turn hits only the previous prompt, not its output, from the GPU or a tier. With `null` or the block size, turn 2 hits the previous prompt and output. Measured without aux outputs.
- **Prefix replay cost.** With an implementation that delivered exact outputs under prefix replay on DeepSeek-V4.1-Flash, forcing every new position through the replay stages cost about 41% of throughput at the top scenario (512K context, 51 sessions per replica). With replay off, a one-node workload at 14.6K context (512 sessions × 16 turns) took 58–62% less time in every arm, aux off included. This is why sessions require prefix replay off.
- **Batch invariance and MoE sequence parallelism.** With the `deep_gemm_mega_moe` MoE backend, the same prompt alone and batched gave different routed experts. The backend enables MoE sequence parallelism, and under `VLLM_BATCH_INVARIANT=1` the batch-invariant branch of `CudaCommunicator.reduce_scatter` was gated on ROCm, so a plain NCCL ring ran whose summation order depends on the output rank. Dropping that condition made the outputs bitwise identical. Check any configuration with MoE sequence parallelism before relying on batch invariance.
- **Top-k through the frontend.** Returning k = 128 top-k logprobs through vLLM's per-token logprobs frontend made rollouts 46.8% longer on a one-node 14.6K workload; through the store they cost nothing measurable. This is why top-k is an object field.
- **Hidden states.** Recomputing sampled-token logprobs from the delivered LM-head input gave an error of at most 7.8e-4 (mean 3e-6), from the checker's own matmul. Logits recomputed from hidden states are not bitwise equal to vLLM's (max 2.1e-4 in float32), so the behavior logprob comes from the inline value.
- **KV offload to an NVMe tier.** With a CPU tier too small for the workload (4 GB, 128K context, 48 sessions), adding an NVMe tier through the offloading connector's tiering spec cut generation time by 23% and recompute from 5.69M to 0.61M tokens. The residual recompute came from promotion failures: every NVMe hit passes through the CPU tier, which had no room. Size the CPU tier for promotion, and offload generated tokens too, not only prompts.
- **Indexer cell order.** In an earlier implementation the engine's top-k emitted each indexer cell as ascending positions with `-1` pads last, and trainer replay expected that order. A decoder that put the pads first passed every set-based check, and indexer replay gained nothing until it was fixed. Compare indexer cells as sequences.
