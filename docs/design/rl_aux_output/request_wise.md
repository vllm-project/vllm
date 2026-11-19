# Request-Wise Aux Outputs

This document describes an alternative to the session design of this folder. In the session design ([protocol](protocol.md)), an object is named by a block hash, so requests of one session share objects, and the name must predict the weight version that will compute a position. In the request-wise design, an object belongs to the request that wrote it: its name says who computed it, the version is recorded after the computation, and the trainer shares and releases objects by reference.

Code:

| Branch | Design |
| --- | --- |
| `rl/aux-output-session-wise` | Session design, all five layers of the [PR stack](README.md#pr-stack), with aborted requests that list their keys |
| `rl/aux-output-request-wise` | Request-wise design on the same base: request ids in the cache salts, the `held` rule (`aux_held`), partial objects at preemption, removal of outputs never delivered. Objects are still block-hash objects; the chunk references of this document are not implemented. The branch's own `request_wise.md` describes its code and limits. |

Assumptions: batch invariance (the same tokens at the same weight versions on the same prefix give bitwise identical outputs); keep and reset weight updates are both supported.

## Principle: the outputs first produced are the ones delivered

A trainer needs the outputs of the computation that sampled each token. Routing replay replays the experts that were used when the token was sampled. A later computation of the same position, after a reset update, an abort and retry, a preemption or a failed KV load, runs at other weights or on another prefix, and its outputs must not replace the first ones. The request-wise design follows it through `held` (E5). The session design follows it through "the store keeps the first object written under a name", except where layer 5 renames a position computed again (see the table below).

## Engine rules

- **E1 Names.** Every position a request computes is salted, through the per-segment cache salts of layer 1, with the segment id `rid@v`. `rid` is unique per request across replicas (minted at admission) and `v` is the weight version current when the id was minted.
- **E2 Admission.** Positions `[0, S')` carry the segments the client sent (`S'` = the client's end, clipped to the prompt, rounded down to a hash block). Positions from `S'` on carry the request's own id.
- **E3 Scheduling.** Let `H` be the final prefix-cache hit (local and external, block aligned). From the first segment that ends after `H` and is not `rid@v_now`, block hashes are cut back to `max(H, start)` and salted with `rid@v_now`. Versions increase monotonically, so a request re-mints its id exactly when a weight update happened since the id was minted.
- **E4 Objects.** The worker writes the outputs of the positions the request writes into chunks `<prefix>/<rid>@<v>/<k>`. Chunk `k` covers absolute positions `[kC, (k+1)C)`, where `C` is a multiple of the hit granularity (the least common multiple of the KV cache group block sizes); a chunk cut by a version change or by the finish is short. A request writes each position at most once: positions below its written frontier that it computes again (preemption resume, load-failure rollback) are not written again. `v` changes at every cut, so no key is written twice.
- **E5 First produced wins.** The client sends `held`, the length of the prefix whose outputs its plan already has: the longest common prefix of the new prompt and the plan's tokens (for a fork or compaction, of the plan it copied). The engine never writes a position below `held` or below its own written frontier, whatever version or prefix it computes that position at. `H ≤ held`, because block hashes chain the tokens and the client's segments come from that plan.
- **E6 Response.** `W = held`; `refs`, one contiguous range `(chunk prefix, start, end)` per version segment written, covering `[W, end)`; `segments`, the salts used over `[0, end)`; `labels`, the version that computed each position of `[W, end)`, recorded after each step. An aborted request returns its partial output with its refs.
- **E7 Outputs never delivered.** The frontend returns the refs of a final output it dropped (a request it no longer tracks) to the engine that served it, whose writing worker removes those chunks after its puts, retrying failed removals. When a `pause_generation` that started after the request finished returns, they are gone.
- **E8 Parallel sampling.** `n > 1` mints one `rid` per child.

## Framework rules

- **F1 Plan.** Per trajectory, a plan maps each position to (chunk key, offset), with labels, segments and tokens. On a response: `plan = plan[:W] + refs`. After the update the plan covers `[0, end)` without a gap.
- **F2 Ownership.** Each chunk counts the plans that reference it; at zero it is removed with `batch_remove`.
- **F3 Derived samples.** A fork, a compaction or a per-turn sample copies the plan (references only) and increments the counts when the derived request is submitted.
- **F4 Order.** A sample is released only after its last request's final output arrived; the segments of a released sample are never sent again; forks start only from a state whose response arrived.
- **F5 Removal.** `batch_remove` retries "not found" and "being written". A key still missing after every replica completed a `pause_generation` that started after its request finished does not exist.
- **F6 Reads.** Per training batch, deduplicate chunk keys, read once, slice per plan.
- **F7 Checks.** Reject a response whose `W` differs from the `held` sent, or whose plan update leaves a gap.

## Why it holds

- **Correct outputs.** Every plan entry was written by the computation that sampled or prefilled that token and is never replaced (E4, E5). KV under block hash `h` was computed by the request whose id is in `h`, so a hit cannot attach another request's state, and a stale copy in any KV tier is unreachable after a cut.
- **No collateral removal.** The engine removes only chunks of its own requests that were never delivered. The trainer removes a chunk only when no plan references it.
- **No leak.** Every chunk written is either delivered (counted by the trainer) or returned to the engine (E7). A crash leaves a prefix to sweep at the end of the run.
- **No resurrection.** A key is written once.

## Flows

| Case | Client sends | Hit | Plan |
| --- | --- | --- | --- |
| Multi-turn | history and the new message, previous segments, `held` = previous end | up to `H ≤ held` | `plan[:held] + refs`; `[H, held)` computed again, not written |
| Compaction, edited history | new tokens, old segments, `held` = first changed token | stops before the first changed token | copy the old plan, then `plan[:held] + refs` |
| Fork (sub-agent, tree) | parent prefix `[0, F)` and new tokens, parent segments, `held = F` | parent KV up to the block below `F` | copy the parent plan at submit, then `plan[:F] + refs` |
| GRPO | an explicit fork (parent prefill with `max_tokens=1`), or nothing | fork: as above; otherwise none | as a fork |
| Keep update | segments | old-version KV | old outputs with old labels |
| Reset update | segments, `held` | none if every tier was cleared | the prefix is computed at the new weights, not written; the plan keeps the outputs of the sampling computation |
| Other replica, restart | segments | none | as reset |

## The two designs

| | Session (`rl/aux-output-session-wise`) | Request-wise |
| --- | --- | --- |
| Name must predict the computing version | yes: names are set at admission and at weight updates, and again at scheduling (layer 5); a running request keeps its names ([protocol §12](protocol.md#12-limits)) | no: names are identities; versions are recorded after the computation |
| Chunked prefill, speculative decoding, look-ahead, a weight update while a request runs | each needs naming rules | no effect on correctness |
| Outputs first produced | the store keeps the first object under a name; with layer 5 a position computed again after a lost KV gets a new name, a second object, and the response lists the second one, which conflicts with the principle above | `held`, and no rewrite below the written frontier |
| Outputs of computations no one reads | kept until the session is released | not written; outputs never delivered removed by the engine |
| Release | by session key pattern after a pause, forced | by reference count, at any time |
| Sharing between requests of a session | implicit | explicit fork |
| Prefill/decode disaggregation | a binding of the prefill's names | each side lists its refs |
| Trainer work | send segments back, release by session | plans and reference counts |

What request-wise gives up:

1. Sharing that is not declared: GRPO siblings and `n > 1` compute the prompt each unless forked (with `n = 8` per replica: +3.5 % at 8K tokens, +28 % at 64K).
2. A cross-session prefix cache (neither design has it).
3. Reuse without sending segments back.
4. A fork from a request that is still generating.
5. The trainer keeps plans and reference counts.
6. The engine removes objects it wrote but never delivered.

## Evidence

Both branches were validated with a closed-loop simulator. A driver runs agentic sessions against one engine, updates the weights on a timer, reads every delivered object as a trainer would, and releases objects by the rules of each design. After every response it reads back the newly delivered rows and keeps a digest per position. At the end it compares the rows each trajectory takes, bit for bit, with the rows first delivered for each position. Planted faults (dropped keys, double releases, removals of another session's objects, shifted rows) confirm that each class of failure is detected.

**DeepSeek V4.1 Flash, without batch invariance.** The runs cover eight scenarios. Each runs 120 weight versions unless noted, with trajectories of up to 1M tokens and up to 150 versions of staleness. The table gives the number of trajectories checked.

| Scenario | Session | Request-wise |
| --- | --- | --- |
| Keep updates only | 179 | 187 |
| Sleep and prefix-cache reset at every update, 40 versions | running | 42 |
| Abort pause at every update, 160 versions | 80 | 155 |
| KV cache limited to 4 GiB (preemption) | 140 | 133 |
| Sub-agents and compaction | running | 207 |
| GRPO groups of 8 | 102 | 109 |
| Asynchronous KV load failures injected (recompute) | 93 | 154 |
| Synchronous KV load failures injected (recompute) | 196 | 156 |

Every completed run shows the following:

- No position's row differs from the row first produced for it. The row of a request's last prompt position, from which it samples its first token, is taken from that request. The session runs alone checked 154M positions.
- No object is left in the store after its release, including the objects of outputs no client received.
- No release removes an object of another live session.
- Every planted fault is caught.

The sampled logprob and the stored top-k logprob of the same token come from two different computations. Without batch invariance they differ by up to 3.4e-2. This is a numerical difference, not a row mismatch.

**Qwen3-30B-A3B, with batch invariance.** Both designs were run on five scenarios (keep, sleep and reset, aborts, mixed with preemption, sub-agents and compaction), each with 50 to 80 versions.

- The first-produced check found 0 mismatches, over 2,229 trajectories for the session design and 2,475 for request-wise.
- A reference engine replays each trajectory under its listed versions and matches bit for bit. The exceptions are positions whose KV was reused across another version history: 6 trajectories for the session design and 85 for request-wise. For those, a reference that replays the KV of the producing request matches exactly.
- The request-wise runs used `2f6c9bd4fe`; runs on the final head are in progress.

**Overhead.** This was measured on the request-wise branch, on one node, in the keep-only scenario, with 120 versions per run. The session design has no separate measurement yet.

- With aux outputs on, generated tokens per second in steady state drop by 2.6%. Decode speed per running request is unchanged: 93.8 against 91.5 tokens/s.
- Trainer reads take 9.4 s at p50 and 18.1 s at p99, and read 3.6 to 3.8 GB per update.
- The simulator's own read-back costs another 11%. A deployment does not have it.
- Whole-run throughput is not used for this comparison. The runs complete different numbers of trajectories, and the read-back keeps requests out of the engine.

**Deployment.** Mooncake disk eviction must be off.

- With the default bucket eviction, a quota or a full disk deletes unreleased objects, oldest bucket first. Reads then miss, but no wrong row is delivered.
- With eviction off, writes that do not fit are refused, and the engine stops at the put stall deadline.

Admission by free disk is therefore required (the session branch's protocol, section 8). Its bound, the longest context times all branches times the group size, reserved about ten times what the sub-agent scenario wrote, which lowered concurrency. A bound per live branch is future work.

**Not covered.**

- Mooncake 0.3.14 does not free SSD space when objects are removed, so both designs need a holder-side reclaim (RFC kvcache-ai/Mooncake#3220).
- Mooncake 0.3.14 can register again a key that was removed while its offload to SSD was in flight.
- A failed synchronous KV load remains a documented limit of both designs. The runs cover only the injected path.
- The request-wise branch's `request_wise.md` lists that design's other limits: outputs lost after `generate` yielded them, and the versions of listed products, which are not returned.
- The session branch's protocol lists a request that is preempted and then scheduled under another version history. Such a request lists only the rows it computed again.
