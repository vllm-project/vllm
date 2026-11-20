# Experiments

This page covers the validation of the session stack at its base, how overhead is measured, the store failure tests, the items still open, and earlier measurements that still inform the design. The design is in the [protocol](protocol.md); alternatives in [design space](design_space.md). The final validation of both designs, session-wise and request-wise, on eight scenarios is in [request-wise evidence](request_wise.md#evidence).

## Validation

### Setup

- Hardware: nodes with 4 GPUs each.
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

The same properties are checked in unit tests against an in-memory store: a test engine stamps every row with its position and the weight version that computed it, and asserts at the end that every key was written with one payload, that each object's rows carry the versions its name states, and that responses cover their positions. Every fix on the branch adds a test that fails without it.

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

All runs are on this branch.

| ID | Result |
| --- | --- |
| V1 | Qwen3-30B-A3B with batch invariance and DeepSeek-V4-Flash, each with the offloading connector (a CPU tier, and a CPU tier with a file-system tier), simple CPU, the Mooncake store connector with and without `save_decode_cache`, and multi. After every weight update every prefix token hit: 122,880 of 122,880, and 120,832 without the decode cache, which recomputes the last decode block of each turn ([protocol §7](protocol.md#7-prefix-cache-hits-and-kv-connectors)). Every key read back, no key written twice, rows exact with batch invariance and the hash-routed layers exact without. Hidden states and top-k logprobs are byte-identical to a fresh computation. |
| V2 | Turns sent to a second replica hit the KV the first replica stored. |
| V3 | Every arm of V1 ran a `keep` pause, a sleep and a prefix-cache reset. A reset renamed exactly the objects it made the engine compute again: 960 per phase on Qwen3-30B-A3B (8 sessions × 120 objects) and 480 on DeepSeek-V4-Flash. On a real Mooncake store, a pause followed by `remove_by_regex` with `aux_session_pattern` removed all 193 objects of the session, and none appeared later. |
| V4 | DeepSeek-V4-Flash, TP4: 51 sessions growing from 72–88K to 205–406K positions. Without sleeps, makespan about +5%, with first-turn TTFT and single-sequence prefill TTFT at 200K and 1M within 1%. With a sleep and a weight-version change every 12 s, makespan +4.8% to +5.8%. The extra cost grows with resets × session length: after a reset the block hashes of every live request are sent to the workers again, and the workers' `begin_step` and the scheduler's `salt_from_hit` run over them. The pause drain added about 0.3 s per update in this scenario and nothing measurable in short runs. |
| V5 | Bitwise runs with a checker that compares the version that computed every row with the version its object's name states: gpt-oss with batch invariance and DeepSeek-V4-Flash without, 0 stale labels. Workload time equal to the build without the session layer, within run-to-run noise. |
| V6 | Regression scenarios pass: a last token that completes a block, preemption across two weight updates, and a reset right after requests finish. A full object pool is an open item. |

**Pause.** Draining the writers at a pause costs about 5 µs when nothing is queued. With an in-memory store it took 3 ms for 64 queued objects of 8 KB and 27 ms for 512; with 1 ms per put, 0.07 s and 0.54 s. The wait is the time to write what is queued.

### Measuring overhead

- **Same node, arms alternating.** Two sessions-off runs of one build differed by 4.1% in TTFT, and an earlier comparison that always ran the sessions-on arm first showed TTFT +5–11% that disappeared when the arms alternated.
- **Generated tokens per second from single runs is not usable.** In the closed-loop simulator (DeepSeek V4.1 Flash, keep-only, 120 versions), two aux-off runs of the same build on the same node differed by 16% in generated tokens per second in the steady-state window, because the mix of long and short trajectories in the window varies. Differences of a few points, or even 10 to 15 points, between single runs are not results.
- **Prefill computed per generated token is stable.** The same two aux-off runs gave 29.9 and 30.1. It shows where extra compute goes even when throughput is noisy.
- **What it showed for session-wise.** Session-wise gave 40.9 and 41.3 (+37%) in two pairs run in opposite order. All of the excess was in first turns: the scenario's sessions come in pairs that send the same first prompt, and with the session in the cache salt from position 0 the second session computed the prompt again. First turns computed 69.5% of their prompt positions, against 28.9% with aux outputs off. The fork fix and its measurement are in [comparing the two designs](comparing.md#performance-and-integration).

### Store failures

The writer change on both branch heads keeps the engine running when the store fails: it retries refused puts until the stall deadline, then drops what the store does not take ([fault tolerance](request_wise.md#store-errors-in-the-engine)). Unit tests added with it fail on the code before it.

Kill tests with request-wise engines, synchronous training, 64 samples per step, 4 steps; a holder or the master killed during the second step's rollout and restarted:

| Kill | Steps completed | Engine errors | Masked samples in the step with the kill | Later steps |
| --- | --- | --- | --- | --- |
| holder (`client_ttl` 30 s) | 4 of 4 | none | 8 of 64; the trainer's read waited out its 600 s deadline | 0 masked, step time as before the kill |
| master | 4 of 4 | none | 8 of 64; the same wait | 0 masked, step time as before the kill |

Before the change, the same kills stopped the engine within about a second (the failed put raised at the next step), and the run failed.

## Open items

- **Prefix-cache collapse under backlog.** Under a backlog of requests the prefix-hit rate collapses, with aux outputs on and off. The offloading connector's tier holds only about 1M tokens for DeepSeek-V4-Flash, because each key takes the slot of a full block.
- **Sleep during connector loads.** A sleep while KV connector loads are in flight raises; callers must retry it.
- **A full object store.** Writes stall; when nothing frees the store for the stall deadline, the writer drops objects and the trainer masks the samples that need them (on the branch heads; at the base the engine stopped). Renaming after a pause that clears the cache multiplies the footprint: the objects computed again are written under new names, and the old ones stay until their session is released.
- **Cost of a reset.** Sending the hashes of live requests again, and the `begin_step` and `salt_from_hit` work that follows (V4).
- **Late decode blocks in the Mooncake store.** With `save_decode_cache`, session requests and asynchronous scheduling, in runs with two engines per node, the next turn of an earlier revision missed the last block of the previous turn in 7 of 11 runs. Without aux outputs 0 of 3 runs missed; with asynchronous scheduling off, or 60 s between turns, every run hit. The blocks were stored and their hashes at store and lookup were equal; they became visible late. The V1 runs with the decode cache hit every token.

## Earlier measurements

These come from earlier implementations of aux output delivery; each entry says which. They are kept where the property measured does not depend on how objects are named.

### Mooncake behavior

| Behavior | Measured | Used in |
| --- | --- | --- |
| The same key written twice | Both puts return success. Sequential writes keep the first object; concurrent writes keep one of them, not determined which, never a mix. | immutable objects (protocol §5) |
| Removal right after a read | Without force nothing is removed (read lease, error −706); with force everything is. | release with force (protocol §10) |
| `remove_by_regex` during writes | Silently skips keys still being written and returns only the count it removed. Scans every key: 36–74 ms at 200K keys. | release only after a pause (protocol §10) |
| A holder process killed | The master still lists its keys, and keeps the holder's segments for `client_ttl` (600 s by default): puts placed on them fail and readers retry against the dead holder. | missing objects are masked by the trainer; a short `client_ttl` (fault tolerance) |
| Store full, or master killed | With the base code, the request whose write fails gets an engine error and the engine stops; nothing is lost silently. | replaced on the branch heads by retry, then drop and mask (fault tolerance) |
| A client created before the engine in one process | CUDA initialization fails in the engine's workers. | consumer contract |
| Reader after the master dies | Hangs for more than 70 s. | reads need their own timeout |
| SSD offload at put | Lost 2,976 of 100K small objects (3%) with no error. Offload on eviction kept 100K of 100K. | SSD offload not used with pinned objects |

The removal and offload tests used Mooncake 0.3.13.post1; the same-key, full-store and client-ordering tests ran with the previous session implementation.

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
