# Comparing the Two Designs

Two designs deliver aux outputs. Both are implemented on branches of the same base:

- **Session-wise** (`rl/aux-output-session-wise`, the [protocol](protocol.md) of this folder). An object is named by a block hash that includes the weight version, so requests of one session share objects by name.
- **Request-wise** (`rl/aux-output-request-wise`, [request-wise design](request_wise.md)). An object is named by the request that computed it. The version that computed each position is recorded after the computation, and the trainer shares and releases objects by reference.

This page puts the two side by side on three points: correctness and naming rules, storage and faults, and performance and integration. It ends with the questions we would like reviewers to judge. The feature-by-feature table is in [the two designs](request_wise.md#the-two-designs); the measurements are in [evidence](request_wise.md#evidence).

## Correctness and naming rules

Both designs follow one principle: a trajectory gets the outputs first produced for each position, the ones of the computation that sampled or prefilled its tokens.

**Common evidence.** On DeepSeek V4.1 Flash without batch invariance, in eight scenarios (keep updates, sleep and reset, abort pauses, preemption, sub-agents and compaction, GRPO groups, injected asynchronous and synchronous KV load failures), no position differed from the output first delivered for it in any accepted run: 0 of 34.6M positions in the largest session-wise run, 0 of 37.8M in the largest request-wise run. On Qwen3-30B-A3B with batch invariance, five scenarios each, both had 0 mismatches, and a reference engine reproduced every row.

| | Session-wise | Request-wise |
| --- | --- | --- |
| What a name states | the tokens and the weight versions of the prefix, fixed before the position is computed | the request that computed the position and the version current when its id was minted; the computing version is recorded after the step |
| When names are set | at admission, at weight updates, and when a waiting request is scheduled; a running request keeps its names | when the request id is minted, and again when it is scheduled after a weight update |
| What a new engine feature needs | a naming rule for each feature that changes when a position is computed: chunked prefill, speculative decoding (MTP), hash look-ahead, a weight update while a request runs, prefill/decode disaggregation | no naming rule; `held` decides which outputs the trajectory keeps |
| Cases outside the rules | documented limits ([protocol §12](protocol.md#12-limits)) | documented limits (the branch's `request_wise.md`) |
| A position computed again after its KV was lost | renamed at scheduling, written as a second object, and the response lists the second one, not the first produced | not written again below `held`; the plan keeps the first object |
| Outputs no client received | stay under the session name until the session is released | removed by the engine that wrote them (E7), through a path from the frontend back to the engine |

Request-wise findings on the E7 path:

- Validation found final outputs that the frontend dropped and never returned to the engine. Fixed before the accepted runs.
- An audit found that a synchronous `LLMEngine` aborting a request after its final output was queued, but before the next step, never returned its keys. Fixed on the branch head.
- One run of the abort-pause scenario left the 8 objects of one session in the store 9 minutes after the client dropped them, with no release message for that session. Two reruns with per-key logging did not reproduce it. The candidates are an engine removal path that was missed, or a removal that Mooncake undid ([removed objects that reappear](request_wise.md#removed-objects-that-reappear-mooncake-0314)). The cause is not known.
- A response that `generate` yielded but the client lost, for example over HTTP, is not returned to the engine. Its objects stay until the session is released.

## Storage and faults

| | Session-wise | Request-wise |
| --- | --- | --- |
| When an object can be removed | after the whole session is consumed and a pause has completed, by key pattern | when no plan references it, at any time |
| Unreleased bytes, peak per scenario | 112 to 271 GB | 61 to 188 GB |
| SSD use, peak per scenario | 155 to 287 GB | 69 to 234 GB |
| SSD still used at the end, as a share of bytes written | 0.22 to 0.95 | 0.06 to 0.51 |
| What a lost object costs | the session: later requests that hit its KV list the same key, so the session is retired | the samples whose plans reference it |
| Store failures in the writer | retry until the stall deadline, then drop; the engine keeps running | the same code change |
| Removal calls | one key-pattern removal per session | one removal per object at reference count zero, plus the engine's removals |
| Removed objects that Mooncake 0.3.14 registers again | observed, 6 keys in an affected run | observed, 6 to 9 keys per affected run |

The SSD numbers come from the same eight scenarios. Mooncake 0.3.14 does not reclaim the disk space of removed objects, so the SSD peak is the live bytes on disk plus removed bytes not reclaimed yet, and what is left at the end depends on how late the releases come. The reappearing objects are a Mooncake bug that registers a removed object again when the removal overlaps the holder writing it to SSD; each removal can meet that window.

## Performance and integration

| | Session-wise | Request-wise |
| --- | --- | --- |
| Turns of one session | reuse the previous turn's KV and objects; the client sends the previous segments | the same; the client also sends `held` |
| Siblings (`n > 1`, GRPO groups) | share KV as requests of one session | compute the prompt once per child unless the framework forks: with `n = 8` per replica, +3.5% at 8K tokens and +28% at 64K |
| Sessions that send the same prompt | compute it once per session unless the framework forks (fix in progress, below) | compute it once per session unless the framework forks |
| Trainer state | segments per session; release by session | plans and reference counts per trajectory |
| Trainer integration measured | none | a reader outside vLLM, below |

**Session-wise sharing across sessions.** In the session-wise overhead runs, the sessions came in pairs that send the same first prompt. With aux outputs off, the second session of a pair hit the first one's prompt. With session-wise on, the session is in the cache salt from position 0, so the second session computed the whole prompt again. First turns computed 69.5% of their prompt positions, against 28.9% with aux outputs off; later turns computed the same share in both arms. Over the run, prefill computed per generated token rose by 37%. Two fixes were designed and reviewed:

- **(a) Take the session out of the salt and share automatically.** Rejected in review. The segments a client sends state versions, not ownership, so an edited history could hit another session's KV and list keys that session never wrote. Closing that needs ownership evidence for every hit: a per-replica ledger, which loses same-session reuse across replicas and after eviction, or a new token in the response.
- **(b) Name the owner in each segment and fork explicitly.** Adopted. The owner session goes into each segment's salt, and a session forks another by sending that session's segments. Sessions that never fork see no change. The implementation and its unit tests are on a local branch, not on `rl/aux-output-session-wise`; its overhead measurement is running.

**Request-wise trainer integration.** A trainer integration outside vLLM keeps the plans and reference counts, reads each object once per node, and masks samples whose objects are missing. Against the same training loop with R3 off:

- Synchronous training: +3.2% per step. Returning R3 inline in the response: +11.3%.
- Fully asynchronous training: within noise; the step is bound by the trainer.
- A missing object after the write barrier is masked in under 1 s on the critical path.

For reference, Miles' output store (session path) measured +4.3% per step on the same synchronous workload, and +8.7% inline.

**Engine throughput.** Not comparable yet. Two aux-off runs of the same build on the same node differed by 16% in generated tokens per second, so single-run results (request-wise −2.6%; session-wise −14.1% and −6.7%) are not results. Prefill computed per generated token was 29.9 and 30.1 in the two aux-off runs; session-wise 40.9 and 41.3 (+37%, cause above); request-wise 32.9 (+9%, one pair, not yet attributed).

## Open items

Session-wise:

- Each new engine feature that changes when a position is computed needs a naming rule, or becomes a documented limit.
- A position computed again after a lost KV gets a second name, and the response lists the second object, not the first produced.
- Sharing a prompt across sessions needs the explicit fork, which is on a local branch, not on the design branch.
- Engine overhead after the fork fix is not measured yet.
- No trainer integration has been measured.

Request-wise:

- Engine-side release of undelivered outputs is a path that has had leak bugs; two were found and fixed.
- 8 objects were left in one abort-pause run; not reproduced, cause not known.
- Responses lost after `generate` yielded them are not returned to the engine; their objects stay until the session is released.
- Siblings share a prompt only through the explicit fork, which the RL framework must issue.
- Engine overhead is measured in one pair only.

Both: a failed synchronous KV load is a documented limit, and the Mooncake issues in [fault tolerance](request_wise.md#fault-tolerance) apply (removed objects registered again, disk not reclaimed on removal, a dead holder's segments kept for `client_ttl`, metadata lost when the master restarts).

## Questions for reviewers

- **Which design should go upstream, and why?** In particular: is a naming rule per engine feature (session-wise) or engine-side removal of undelivered outputs plus trainer-side reference counts (request-wise) the cost vLLM and RL frameworks would rather carry?
- **Explicit fork.** Request-wise siblings, and session-wise sessions that share a prompt, share KV only through an explicit fork: the parent prefills with `max_tokens=1`, and the children send its segments. Can RL frameworks issue that, and is it the right API?
- **The object-store boundary.** A local branch makes the store pluggable: `put_batch`, `get`, `remove`, `remove_prefix` and `free_bytes`, with these properties: a reader sees no object or a whole one, a rewrite keeps the first object, a removed object never comes back, nothing is evicted silently (a write that does not fit fails), and removal by key prefix. A filesystem backend passed one validation scenario with 0 mismatches over 26.5M positions; one same-node pair against Mooncake was 7.8% lower in generated tokens per second, cause not isolated. An S3 backend is not done. Is this the right interface, and which backends matter?
- **Deployment across clusters.** Engines in one cluster could write to a store next to the trainer. Over one 400 Gb/s link, writes reached about 49 GB/s with RDMA and 3.2 to 4.1 GB/s with TCP, 245 to 1,260 times the write rate of the training workloads we measured (2.5 to 12.8 MB/s). The test ran within one rack; a wide-area round trip per put was not measured, and neither was a holder that serves TCP writers and RDMA readers at once. Does cross-cluster writing matter to you?
- **Capacity defaults.** On DeepSeek V4.1 Flash one position takes 11,744 bytes: 480 for R3, 1,024 for top-128 logprobs, 10,240 for hidden states. A 200K-token trajectory is 2.35 GB with all three and 96 MB with R3 only. A node with 11.8 TB of usable SSD (14 TiB less a 20% reserve and a local copy of the weights) holds about 5K trajectories with all fields and about 120K with R3 only. Which outputs should be on by default, and should hidden states be stored at every position?
