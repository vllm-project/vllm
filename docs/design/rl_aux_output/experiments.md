# Experiments

This page holds the evidence for the protocol. Its own validation matrix is pending; the results below ran on earlier code, and each says which. What carries over: the writer and its behavior under store failures, the object format, the fields, the Mooncake and transport measurements do not depend on how objects are named. Naming, listing and release do: results on earlier code show that the methods and checks work and what each sharing costs, not that this code is correct.

## Code the results ran on

| Label | Protocol | Code |
| --- | --- | --- |
| U | This protocol: `aux_namespace`, `aux_prompt_start`, `aux_objects` | `a3f5ba9e0e`; fixes after the design review: engine `054544cc20` to `2f22028d5c`, trainer to `36d25ec` |
| S-seg | Session-wise with version segments: weight versions in the salt, cumulative key lists | `de72f5cf5a` for the validation runs; branch head `957235e726` adds writer degradation |
| S-seg-owner | S-seg with an owner session in every segment, so sessions can fork each other | `fd965eb0d8`; with writer degradation `da0bd138db` |
| R-seg | Request-wise with `<request id>@<weight version>` salts, `aux_session` as release unit, engine removal of undelivered outputs | `7a3f62ba55` for the validation runs; branch head `2b2087c16e`; kill tests at `35f26d0096` |
| R-span | Request-wise prototype: `aux_spans` with an owner per position, objects `<prefix>/<owner>/<block index>`, release per owner | `fa0f6430d5` |

## Validation of this protocol

**Unit tests.** The engine's unit and integration suites run in the nightly image: 896 passed, with 3 failures and 34 errors that its base shares (pipeline-parallel world size and model downloads). The tests added with each change fail on the code before it, apart from controls and guards that hold on both. The Rust frontend's engine-core client, text and server crates pass. The verl integration's aux-output suite passes (156 tests). These counts are for `a3f5ba9e0e` and the trainer at `ed8c95d`; the fixes after the design review add their own tests.

**Validation matrix: pending.** The matrix runs every scenario with both granularities on DeepSeek V4.1 Flash without batch invariance, CUDA graphs on, one seed per cell, with overhead measured against products off in alternating runs on one node:

| Cell | Scenario | Group | Trajectory |
| --- | --- | --- | --- |
| M0 | Weight updates that keep the prefix cache | pending | pending |
| M1 | Sleep and prefix-cache reset at every update, 40 versions, 1M-token sequences | pending | pending |
| M2 | Abort pause about every 6 s, 160 versions | pending | pending |
| M3 | KV cache limited to 4 GiB (preemption) | pending | pending |
| M4 | Sub-agents and compaction | pending | pending |
| M5 | GRPO groups of 8 | pending | pending |
| M6 | Asynchronous and synchronous KV load failures injected (recompute) | pending | pending |
| M0 overhead | Products on against off, four runs on one node | pending | pending |

Every cell checks:

| Invariant | Check |
| --- | --- |
| INV-COV | Every listing covers `[aux_prompt_start, end)` without gap, overlapping only where a preempted request's tail comes before the object that holds its positions again, and every object holds as many positions as its entry says |
| INV-ABUT | Spliced multi-turn trajectories abut across turns, hits, weight updates and preemptions, and hold `L − 1` positions |
| INV-NS | Every listed key lies under the request's namespace; no tail key is listed twice |
| INV-FP | Every spliced position equals the first stored copy |
| INV-LEAK | After release, no key is left under the prefix, and nothing of a live namespace is removed |
| INV-EVICT | The store evicts nothing |
| INV-TRAIN | End to end with a trainer: missing objects mask their samples, a store fault drops the samples of every namespace open during it, a fully masked step skips its update, every object is released |

The group cells also count positions listed from a computation at another weight version than the request's own, the case group granularity accepts ([protocol §12](protocol.md#12-limits)).

## Group and trajectory sharing on earlier code

Both sharings were measured on earlier code with the same isolation: one GRPO group per session on S-seg-owner (`da0bd138db`), and one owner per trajectory on R-span. Same workload and configuration in both: DeepSeek V4.1 Flash without batch invariance, CUDA graphs on, the M0 scenario of the validation set.

**Correctness.**

| | Group (S-seg-owner, one session per GRPO group) | Trajectory isolation (R-span) |
| --- | --- | --- |
| Positions checked against the first stored copy | 31.76M, 165 trajectories, 0 mismatches | 49.65M, 249 trajectories, 0 mismatches |
| Leaks after release | 0 | 0 |
| Evictions | 0 | 0 |
| Planted faults | missing key and premature release caught; the removal-by-pattern fault had nothing left to hit | missing key and premature release caught |

An earlier group run on a node with less free disk had its admission refuse 1,781 polls; it checked 20.9M positions with 0 mismatches before it was cut short, and was rerun on a node with more free disk.

**Overhead.** Four runs on one node in the order products on, off, off, on; generated tokens per second:

| | Group | Trajectory isolation |
| --- | --- | --- |
| On, off, off, on | 638.6, 621.9, 657.8, 629.2 | 653.4, 760.8, 703.9, 646.9 |
| Mean on / mean off | 633.9 / 639.9 | 650.2 / 732.4 |
| Products on against off | −0.9%, within the ±2.8% spread of the off runs | −11.2% |
| Prefix hit ratio, on / off | — | 86.3% and 86.8% / 90.4% and 89.3% |
| Positions computed per second, on / off | — | 25,847 / 22,684 (+14%) |

The trajectory cost is lost prefix hits: with products off, the samples of a group share the prompt's KV through the prefix cache, and per-trajectory salts make each sample compute it. The two sharings ran on different nodes, so only the ratios compare.

## Validation of the earlier protocols

### Method

Both earlier protocols were validated with a closed-loop simulator. A driver runs agentic sessions against one engine, updates the weights on a timer, reads every delivered object as a trainer would, and releases objects by the rules of each design. After every response it reads back the newly delivered products. It keeps a digest of the first products delivered for each position and compares every later delivery of that position with it (the FP check). Products a request computed and replaced before any response delivered them are never read, so this check does not see them. Planted faults (dropped keys, double releases, removals of another session's objects, shifted entries) confirm that each class of failure is detected.

A run is accepted only if all of the following hold:

- Every position a trajectory takes has, bit for bit, the products first delivered for that position.
- The objects of outputs no client received are gone after the next pause.
- No release removes an object of another live session.
- The store evicts nothing.
- Every session that is not a planted fault is consumed and released by the end.
- Every difference between the sampled logprob and the stored top-k logprob of the same token is classified.

### DeepSeek V4.1 Flash, without batch invariance (S-seg and R-seg)

Both earlier protocols passed all eight scenarios. Each scenario runs 120 weight versions unless noted, with trajectories of up to 1M tokens and up to 150 versions of staleness.

| Scenario | S-seg: trajectories (positions) | R-seg: trajectories (positions) |
| --- | --- | --- |
| Keep updates only | 179 (34.6M) | 187 (37.8M) |
| Sleep and prefix-cache reset at every update, 40 versions | 41 (8.9M) | 43 (9.4M) |
| Abort pause at every update, 160 versions | 80 (14.0M) | 130 (20.3M) |
| KV cache limited to 4 GiB (preemption) | 140 (28.8M) | 143 (29.7M) |
| Sub-agents and compaction | 198 (19.4M) | 207 (19.8M) |
| GRPO groups of 8 | 102 (21.0M) | 98 (18.9M) |
| Asynchronous KV load failures injected (recompute) | 93 (19.1M) | 145 (29.2M) |
| Synchronous KV load failures injected (recompute) | 196 (36.6M) | 156 (30.7M) |

- **In every accepted run:** no position differed from the products first delivered for it; no undelivered object was left; no other session's object was removed; nothing was evicted; every planted fault was caught.
- **One R-seg run of the abort-pause scenario was not accepted.** 6 removed objects reappeared (Mooncake, see [removed objects that reappear](protocol.md#10-failures)), and 8 objects of one session were still in the store 9 minutes after the client dropped them, with no release message for that session. The table shows its rerun. Two reruns with per-key logging did not reproduce the 8 objects; the cause is not known.
- **The branch heads were not run again.** They add writer degradation, and on R-seg the fix for a synchronous `LLMEngine` abort and the `aux_prompt_start` field. They are covered by unit tests that fail on the code before them.
- **Sampled against stored logprobs.** The sampled logprob and the stored top-k logprob come from two different computations, and without batch invariance they differ by up to 3.4e-2 (0.0341 in one S-seg run, above the 0.0236 of every other run; not explained).

### Qwen3-30B-A3B, with batch invariance (S-seg and R-seg)

Five scenarios each (keep, sleep and reset, aborts, mixed with preemption, sub-agents and compaction), 50 to 80 versions.

- The FP check found 0 mismatches: 24.9M positions for S-seg, 25.5M for R-seg.
- A reference engine replaying each trajectory under its listed versions matched bit for bit, except positions whose KV was reused across another version history: 6 trajectories for S-seg, 80 for R-seg. A reference that replays the KV of the producing request reproduced those, except 2 fork trajectories in one R-seg run that were not attributed.

### Storage

With routed experts, top-128 logprobs and hidden states, one position of DeepSeek V4.1 Flash takes 11,744 bytes: 480 for the experts (40 layers × 6 × 2 B), 1,024 for the top-k, and 10,240 for the hidden states. In the eight scenarios above:

| | S-seg | R-seg |
| --- | --- | --- |
| Unreleased bytes, peak per scenario | 112 to 271 GB | 61 to 188 GB |
| SSD use, peak per scenario | 155 to 287 GB | 69 to 234 GB |
| SSD still used at the end, as a share of bytes written | 0.22 to 0.95 | 0.06 to 0.51 |
| Removed objects that Mooncake registered again | 6 keys in an affected run | 6 to 9 keys per affected run |

Mooncake 0.3.14 does not reclaim the disk of removed objects, so the SSD peak is live bytes plus removed bytes not reclaimed yet. Part of the S-seg excess came from positions computed again under new version names, which this protocol no longer writes ([scenarios §9](scenarios.md#9-storage-on-the-earlier-protocols)).

### Overhead on the earlier protocols

**How it is measured.**

- **Same node, arms alternating.** Two products-off runs of one build differed by 4.1% in TTFT, and an earlier comparison that always ran the products-on arm first showed TTFT +5–11% that disappeared when the arms alternated.
- **Generated tokens per second from single runs is not usable.** Two products-off runs of the same build on the same node differed by 16% in the steady-state window (788 and 662).
- **Prefill computed per generated token is stable.** The same two runs gave 29.9 and 30.1.

**Results.**

- **R-seg, one pair.** Generated tokens per second −2.6% in the steady-state window, which by the method above is not a result. Over the whole run the products-on arm generated 293.7 tokens per second against 479.7 (−38.8%, 3,190 s against 1,462 s), because of 828 refused admissions on a full memory pool and 222 s after the last pause. Prefill per generated token 32.9 against 30.1 (+9%, not attributed). The two arms ran at different average concurrency (2.9 and 5.0).
- **S-seg, two pairs in opposite order, on another node.** Generated tokens per second −14.1% and −6.7%. Prefill per generated token 40.9 and 41.3 (+37%), from first turns: sessions came in pairs that send the same first prompt, and with the session in the salt the second session computed it again.
- **S-seg-owner, with an explicit fork between the two sessions of a pair,** four runs on one node in the order fork, off, off, fork:

| Run, in order | Generated tokens per second | Prefill computed per generated token | First-turn prompt positions hit | Later-turn positions hit |
| --- | --- | --- | --- | --- |
| Fork 1 | 562.1 | 34.85 | 57.8% | 88.9% |
| Off 1 | 661.6 | 30.14 | 84.0% | 90.3% |
| Off 2 | 723.7 | 30.14 | 81.0% | 89.7% |
| Fork 2 | 654.3 | 33.05 | 73.8% | 89.0% |

  The fork arm averaged 608.2 and the off arm 692.7 generated tokens per second, −12.2%; the two off runs differ from each other by 9%. The excess prefill fell from +37% to +12.6%; the rest is first turns that no longer hit other sessions' prompts by chance.

- **Trainer reads** took 9.4 s at p50 and 18.1 s at p99 with R-seg, 5.4 s and 11.7 s with S-seg.
- **The simulator's own read-back** of every delivered position cost another 11% (R-seg) and 34% (S-seg) of generated tokens per second. A deployment does not pay it.

**Trainer integration (R-seg).** A trainer integration outside vLLM keeps the plans and reference counts, reads each object once per node, and masks samples whose objects are missing. Against the same training loop with R3 off: synchronous training +3.2% per step (returning R3 inline: +11.3%); fully asynchronous training within noise. For reference, Miles' output store measured +4.3% per step on the same synchronous workload, and +8.7% inline.

## Store failures

The writer keeps the engine running when the store fails: it retries refused puts until the stall deadline, then drops what the store does not take ([protocol §10](protocol.md#10-failures)). Unit tests added with it fail on the code before it.

**Kill tests (R-seg).** Synchronous training on the Mooncake backend, 64 samples per step. During the second step's rollout the test killed a store holder in one run and the master in the other, then restarted it. A third run deleted the last object of every trajectory before the trainer read it, so that every sample of every step was masked.

| Run | Steps, exit | Masked in the step with the kill | Reads in that step | That step | Later steps |
| --- | --- | --- | --- | --- | --- |
| Holder, `client_ttl` 30 s | 4 of 4, no engine error | 8 of 64 | waited out the 600 s read deadline (the trainer did not fail transport errors fast yet) | generation 119 to 121 s (normal 59 to 61 s) | 0 masked, 118 to 122 s |
| Master | 4 of 4, no engine error | 8 of 64 | the same wait | generation 119 to 121 s | 0 masked, 118 to 122 s |
| Holder, `client_ttl` 600 s, fast failure on transport errors | 5 of 5, exit code 0 | 8 of 64 | 7.07 s; the masked samples attached in 0.02 to 0.14 s | 713 s, generation 645 s (normal about 120 s) | 0 masked, 119 to 121 s |
| Master, fast failure on transport errors | 5 of 5, exit code 0 | 8 of 64 | 7.3 s | 184 s, generation 119 s including the master's downtime | 0 masked, 120 s |
| Every sample masked | 3 of 3, exit code 0 | 64 of 64 in every step | returns with every sample masked | every step skips its update | — |

Every step released all its objects. Before writer degradation the same kills stopped the engine within about a second. Earlier rounds found three problems outside the engine, since fixed in the trainer integration: a step whose samples were all masked raised instead of skipping (a shared prompt object lost with the master masked every sample); reads kept retrying a dead holder while the master listed it; objects written before a master restart were gone. This protocol has not been through a kill test. Its rule differs from the one measured here: a holder or master fault drops every sample not yet consumed of every namespace open during it, detected by the store's fault epoch, where these runs masked only the samples whose objects could not be read ([protocol §10](protocol.md#10-failures)).

## Base of the earlier protocols

These runs validated the session layer both earlier protocols were cut from, with version-based names, on nodes with 4 GPUs: Qwen3-30B-A3B with batch invariance (bitwise checks), DeepSeek-V4-Flash at TP4 with MTP (structural checks), gpt-oss with batch invariance. The results that do not depend on version names:

- **Connector matrix.** With the offloading connector (a CPU tier, and a CPU tier with a file-system tier), simple CPU, the Mooncake store connector with and without `save_decode_cache`, and multi, every prefix token hit after every weight update: 122,880 of 122,880, and 120,832 without the decode cache, which computes the last decode block of each turn again. Every key read back, no key written twice. Hidden states and top-k logprobs byte-identical to a fresh computation.
- **Replicas.** Turns sent to a second replica hit the KV the first replica stored.
- **Release.** On a real Mooncake store, a pause followed by a removal by session pattern removed all 193 objects of the session, and none appeared later.
- **Overhead at the top scenario.** DeepSeek-V4-Flash, TP4, 51 sessions growing from 72–88K to 205–406K positions: makespan about +5% without sleeps, +4.8% to +5.8% with a sleep and a version change every 12 s; first-turn TTFT and single-sequence prefill TTFT at 200K and 1M within 1%.
- **Pause.** The writer's drain costs about 5 µs when nothing is queued, on top of one worker round trip; with an in-memory store, 3 ms for 64 queued objects of 8 KB and 27 ms for 512; with 1 ms per put, 0.07 s and 0.54 s.

## Weak claims

An outside review of the evidence found these claims weaker than they were stated. They stand as follows:

| Claim | Why it is weak | Closing experiment |
| --- | --- | --- |
| This protocol is correct across the validation matrix | No run used its code; the group and trajectory results ran on earlier code with the same sharing | The matrix above on U |
| Group sharing costs nothing measurable | One ABBA set on one node, on earlier code whose salts still carried versions | The M0 overhead cells on U |
| Trainer integration overhead of a few percent per step (+3.2% to +4.7% across synchronous runs, +2.6% asynchronous) | Same-node step noise is about ±5% with 1–2 pairs; R-seg protocol, R3 only, Qwen3 | Alternating steps on U, with fetch, removal and step time logged separately |
| No leak when a client disconnects | One R-seg run left 8 objects; two instrumented reruns did not reproduce it | A disconnect stress test across scheduling, decode and the final output, asserting no key left under the prefix |
| Sampled and stored logprobs differ by at most the expected bound | One S-seg run at 0.0341 above the 0.0236 of the others, never explained | Bitwise comparison of stored and inline products on identical seeds |
| A Mooncake fix stops removed objects from reappearing | Its test did not trigger the race on unpatched Mooncake in 2 runs | A deterministic master-level test of removal racing offload |
| One writer is enough across clusters | Measured only within one rack, under 1 ms round trip; a writer that puts one object at a time reaches about 1.2 MB/s at 100 ms | Injected 50–100 ms round trip against the writer |
| Store fault tolerance | Kill tests ran 64 samples per step for 4–5 steps on R-seg, which masked by read misses; no engine crash, no master restart with rescan, no multi-replica run; the fault epoch this protocol uses is not implemented | Holder and master kill and restart on U: every namespace open during the fault dropped and released, with disk accounting |

## Mooncake behavior

Measured with earlier implementations; none of it depends on how objects are named.

| Behavior | Measured | Used in |
| --- | --- | --- |
| The same key written twice | Both puts return success. Sequential writes keep the first object; concurrent writes keep one of them, never a mix. On an SSD-first store, 1,833 puts of keys already on SSD or being offloaded all returned success and kept the first value. | immutable objects, first write wins ([protocol §5](protocol.md#5-objects-and-writes)) |
| Removal right after a read | Without force nothing is removed (read lease, error −706); with force everything is. | release with force ([protocol §8](protocol.md#8-pause-and-release)) |
| Removal by pattern during writes | Silently skips keys still being written and returns only the count it removed. Scans every key: 36–74 ms at 200K keys. | release only after a pause |
| A holder process killed | The master still lists its keys, and keeps the holder's segments for `client_ttl` (Mooncake's default 600 s; the deployment requires 30 s): puts placed on them fail and readers retry against the dead holder. | a short `client_ttl` ([protocol §10](protocol.md#10-failures)) |
| A store that refuses writes for 90 s, then recovers | With a 180 s stall deadline the engine kept running; over 6 steps all 25,116 keys were present and matched a run without the stall bit for bit. A reader with a 60 s timeout reported objects missing while writes were refused. | read deadline no shorter than the stall deadline |
| A client created before the engine in one process | CUDA initialization fails in the engine's workers. | [consumer contract](consumer.md#reading) |
| Reader after the master dies | Hangs for more than 70 s. | reads need their own timeout |
| Removal on an SSD-first store (0.3.14) | Removing half the objects and writing the same amount again grew disk use from 6.28 to 9.42 GB; releasing sessions and continuing to write grew it from 4.52 to 8.39 GB. | SSD sized by bytes written, disk eviction off |
| SSD offload at put | Lost 2,976 of 100K small objects (3%) with no error. Offload on eviction kept 100K of 100K. | SSD offload only on eviction |

The removal and offload tests used Mooncake 0.3.13.post1, the disk-space tests 0.3.14.

## Engine findings that still apply

- **Retention interval.** On DeepSeek-V4-Flash with the default `prefix_cache_retention_interval` (0), sliding-window and compressed groups keep state only at the end of each prompt, so the next turn hits only the previous prompt, not its output. With `null` or the block size, turn 2 hits the previous prompt and output.
- **Prefix replay cost.** With an implementation that delivered exact products under prefix replay on DeepSeek-V4.1-Flash, forcing every new position through the replay stages cost about 41% of throughput at the top scenario (512K context, 51 sessions per replica). With replay off, a one-node workload at 14.6K context (512 sessions × 16 turns) took 58–62% less time in every arm, products off included.
- **Batch invariance and MoE sequence parallelism.** With the `deep_gemm_mega_moe` MoE backend, the same prompt alone and batched gave different routed experts: under `VLLM_BATCH_INVARIANT=1` the batch-invariant branch of `CudaCommunicator.reduce_scatter` was gated on ROCm, so a plain NCCL ring ran whose summation order depends on the output rank. Dropping that condition made the outputs bitwise identical.
- **Top-k through the frontend.** Returning k = 128 top-k logprobs through vLLM's per-token logprobs frontend made rollouts 46.8% longer on a one-node 14.6K workload; through the store they cost nothing measurable.
- **Hidden states.** Recomputing sampled-token logprobs from the delivered LM-head input gave an error of at most 7.8e-4 (mean 3e-6). Logits recomputed from hidden states are not bitwise equal to vLLM's (max 2.1e-4 in float32), so the behavior logprob comes from the inline value.
- **Indexer cell order.** In an earlier implementation a decoder that put the `-1` pads first passed every set-based check, and indexer replay gained nothing until it was fixed. Compare indexer cells as sequences.
