# Request-Wise Aux Outputs

This document describes an alternative to the session design of this folder. In the session design ([protocol](protocol.md)), an object is named by a block hash, so requests of one session share objects, and the name must predict the weight version that will compute a position. In the request-wise design, an object belongs to the request that wrote it: its name says who computed it, the version is recorded after the computation, and the trainer shares and releases objects by reference.

Code:

| Branch | Design |
| --- | --- |
| `rl/aux-output-session-wise` | Session design, all five layers of the [PR stack](README.md#pr-stack) |
| `rl/aux-output-request-wise` | Request-wise prototype on top of the session stack: request ids in the cache salts, the `held` rule, removal of outputs never delivered. Objects are still block-hash objects; the chunk references of this document are not implemented. |

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

## Evidence so far

- The prototype passes the unit tests of the branch. An earlier revision (request ids in salts, block-hash objects) matched a reference computation byte for byte on 831 trajectories with batch invariance on, with no leak.
- A head-to-head run of both designs on a DeepSeek V4.1 Flash workload (120 weight versions, keep and abort updates alternating, 6 to 12 turns per trajectory) measured stored objects at 2.94 times the expected count for the session design and 1.00 times for request-wise (613 GB and 211 GB of SSD). That request-wise run predates the `held` rule; it rewrote positions computed again after a reset or an abort. A run with the `held` rule is pending.
- Mooncake does not free SSD space when objects are removed. Both designs need a holder-side reclaim (RFC kvcache-ai/Mooncake#3220).
