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

## Fault tolerance

This section applies to both designs. It covers what the engine, the store and the trainer do when a write, a read, the store or the job's coordinator fails. Each part says whether it is implemented or planned.

### Publication and reads

- The engine lists a key in the response when it starts writing the object, not when the write completes. The trainer may read as soon as the response arrives.
- Mooncake publishes an object only when its put completes. A reader sees either no object or the whole object, never part of one.
- Readers poll with a deadline. `read_aux_rows` retries a missing key until its timeout, which defaults to 600 s and should be no shorter than `object_put_stall_timeout_s`.
- **Write barrier.** Take a request that has ended. Once every engine that served it has completed a pause that started afterwards (for example the pause before a weight update, or a sleep), every object the request wrote is in the store. This is the pause rule of the [protocol](protocol.md) for the session design, and E7 and F5 above for the request-wise design. A key still missing after the barrier is lost, not late.

Status: implemented in both designs.

### Failure of one request or one object

Such a failure explicitly fails the samples that need the object. The rest of the batch trains as usual.

- **Writes.** The engine retries a put the store refuses until `object_put_stall_timeout_s` (600 s by default). After that the engine stops with a write error, and its steps and pauses fail instead of hanging. Implemented.
- **Outputs no one reads.** These are the objects of aborted requests, of final outputs the frontend dropped, and of `generate` calls that were cancelled. The request-wise design removes them in the engine (E7). The session design removes them with the session. Implemented.
- **Reads.** The trainer masks a sample whose object is missing:
  - at once, if the object's request has passed the write barrier;
  - otherwise after waiting up to the read deadline, since the write may still land.

  All ranks agree on the masked samples before the step. This is implemented in a trainer integration outside vLLM.

Evidence comes from deleting objects on purpose before the trainer read them:

- **Synchronous training, 3 steps.** 17 objects were deleted, and 17 samples were masked on every rank. The read on the critical path took 0.05 to 0.53 s per step. Before reads used the write barrier, it took 85 to 90 s per step, because three read layers each waited out a 30 s deadline.
- **Fully asynchronous training, 8 steps.** 3 samples were masked, at a cost of at most 0.42 s per step.
- **Not exercised.** A trajectory that finished after the last barrier and is read at once. A missing object there waits out the deadline.

### Store failure

- **A holder is lost.** The objects in its memory segment and on its SSD are lost, so many reads miss at once.
- **The master is lost.** All metadata is lost, so no object can be found.

Planned handling:

- **Engine.** A write that cannot reach the store fails at the write deadline instead of hanging.
- **Trainer.** Missing objects in bulk count as a store failure. Signs are the share of missing samples in a batch rising above a threshold, or store connection errors. The trainer then drops the batch and generates it again, instead of waiting out each read.
- **Restart.** Start a new master and new holders on new directories, under a new key prefix, and let the engines and readers reconnect. Sessions in flight are abandoned.

Status: planned. Tests that kill a holder and that kill the master are in progress.

### Coordinator crash

The coordinator (the driver, or a ledger actor) owns plans and references that the store does not hold. If it crashes, the job restarts.

| Rule | Status |
| --- | --- |
| Each run writes under its own `object_key_prefix`. The store, its holders and their SSD directories belong to the job, and are deleted when it ends. | Implemented in the trainer integration (aux-fetch); no vLLM-side code needed |
| On resume from a checkpoint, batches whose plans reference objects of the crashed run are not replayed. They are dropped and generated again. | Implemented in the trainer integration (aux-fetch); no vLLM-side code needed |
| When the store outlives a run, the next run removes the previous run's prefix at startup, after the engines that wrote under it have stopped (`remove_by_regex(aux_prefix_pattern(prefix), True)`). | Planned |
| A lease or TTL removes objects whose owner is gone. It is only a fallback, for stores shared across runs. | Planned |

### Removed objects that reappear (Mooncake 0.3.14)

- **Cause.** This is a bug in the Mooncake 0.3.14 master, and it needs offload to SSD (`offload_on_evict`).
  - The trigger is a removal that lands after a holder has taken the object's offload task, but before the holder reports success.
  - `Remove` does not check for an offload in progress: `CheckRemovable` ignores `offloading_task`. So the removal erases the key and drops the task.
  - The holder still finishes writing the bucket and calls `NotifyOffloadSuccess`. That call finds the key gone and registers it again through `AddReplicaForRetainedClient`, with one `LOCAL_DISK` replica and no memory replica (`master_service.cpp`).
- **Evidence from a validation run.** Per-key engine logs show the engine wrote one such key once, 46 s before the removal. The removal returned 0, and 0.12 s later the key existed again with one complete `LOCAL_DISK` replica. All 21 reappeared keys whose replicas were inspected had only that replica. A new put would have created a memory replica.
- **Evidence from a minimal reproduction.** This ran without vLLM, on stock Mooncake 0.3.14. It overfills the memory segment, then removes every key while the holder writes a bucket. Each time, exactly one whole bucket came back within 0.5 s: 16 objects of 16 MB, or 4 objects of 64 MB.

  | Objects | Segment | SSD | Removed | Reappeared |
  | --- | --- | --- | --- | --- |
  | 16 MB | 1 GB | local | 1,152 | 32 |
  | 64 MB | 2 GB | local | 576 | 4 |
  | 16 MB | 1 GB | network file system | 1,152 | 16 |

- **Effect.**
  - The key is readable again, and its disk space is held until the holder's directory is deleted.
  - Readers are not affected, because no plan references a released key.
  - Its contents may not match the original object, because the holder reads the object's memory after the removal may have freed it. This case was not tested.
- **Handling.** After a removal, check the keys again and remove what reappeared.
- **Fix.** `NotifyOffloadSuccess` should not register a key that no longer exists; the holder should mark the bucket entry dead instead. This is not implemented.

## Evidence

Both branches were validated with a closed-loop simulator. A driver runs agentic sessions against one engine, updates the weights on a timer, reads every delivered object as a trainer would, and releases objects by the rules of each design. After every response it reads back the newly delivered rows and keeps a digest per position. Planted faults (dropped keys, double releases, removals of another session's objects, shifted rows) confirm that each class of failure is detected.

A run is accepted only if all of the following hold:

- Every row a trajectory takes is bit for bit the row first delivered for that position. The row of a request's last prompt position, from which it samples its first token, is taken from that request.
- The objects of outputs no client received are gone after the next pause.
- No release removes an object of another live session.
- The store evicts nothing.
- Every session that is not a planted fault is consumed and released by the end.
- Every difference between the sampled logprob and the stored top-k logprob of the same token is classified.

**DeepSeek V4.1 Flash, without batch invariance.** Both designs are valid in all eight scenarios. Each scenario runs 120 weight versions unless noted, with trajectories of up to 1M tokens and up to 150 versions of staleness.

| Scenario | Session: trajectories (positions) | Request-wise: trajectories (positions) |
| --- | --- | --- |
| Keep updates only | 179 (34.6M) | 187 (37.8M) |
| Sleep and prefix-cache reset at every update, 40 versions | 41 (8.9M) | 43 (9.4M) |
| Abort pause at every update, 160 versions | 80 (14.0M) | 130 (20.3M) |
| KV cache limited to 4 GiB (preemption) | 140 (28.8M) | 143 (29.7M) |
| Sub-agents and compaction | 198 (19.4M) | 207 (19.8M) |
| GRPO groups of 8 | 102 (21.0M) | 98 (18.9M) |
| Asynchronous KV load failures injected (recompute) | 93 (19.1M) | 145 (29.2M) |
| Synchronous KV load failures injected (recompute) | 196 (36.6M) | 156 (30.7M) |

- Request-wise ran at its head `7a3f62ba55`.
- Session ran at `de72f5cf5a`. Its head `04495f3806` adds a simplification with identical test results.
- In every run:
  - no row differs from the row first produced;
  - no undelivered object is left;
  - no other session's object is removed;
  - nothing is evicted;
  - every planted fault is caught.
- The sampled logprob and the stored top-k logprob come from two different computations, and without batch invariance they differ by up to 3.4e-2. A 1M-token request resubmitted across a weight update shows differences of the same order. In both cases the rows are the first-produced ones.

**Qwen3-30B-A3B, with batch invariance.** Both designs were run on five scenarios (keep, sleep and reset, aborts, mixed with preemption, sub-agents and compaction), each with 50 to 80 versions. The request-wise runs used its head.

- The first-produced check found 0 mismatches: 24.9M positions for session, 25.5M for request-wise.
- A reference engine replays each trajectory under its listed versions and matches bit for bit. The exceptions are positions whose KV was reused across another version history: 6 trajectories for session, 80 for request-wise. For those, a reference that replays the KV of the producing request reproduces every row exactly.
- That second reference cannot reproduce sampled logprobs when one KV block mixes versions. This is a limit of the reference, not of the rows.

**Capacity.** With routed experts, top-128 logprobs and hidden states, one position of DeepSeek V4.1 Flash takes 11,744 bytes: 480 for the experts (40 layers × 6 × 2 B), 1,024 for the top-k, and 10,240 for the hidden states.

- Bytes held by unreleased sessions peaked at 112 to 271 GB per run for session and 61 to 188 GB for request-wise.
- SSD use peaked higher. On Mooncake 0.3.14 it is the live bytes plus removed bytes that are not reclaimed yet: with nothing live at the end, 18 to 260 GB remained on disk. Plan disk by bytes written, not by live bytes.
- The live peak exceeded a bound that counts only running sessions by up to 2×, for three reasons:
  - sessions waiting for their group's release still hold objects;
  - prefix-cache resets list recomputed positions under new names;
  - sub-agents keep more sessions per group.
- A trainer that consumes trajectories many versions late holds those trajectories' objects as well; the simulator consumed them at once.

**Overhead.**

- Request-wise, measured on one node in the keep-only scenario with 120 versions per run:
  - With aux outputs on, generated tokens per second in steady state drop by 2.6%. Decode speed per running request is unchanged.
  - Trainer reads take 9.4 s at p50 and 18.1 s at p99.
  - The simulator's own read-back costs another 11%, which a deployment does not pay.
- Session: the same measurement is pending.

**Deployment.**

- Mooncake disk eviction must be off:
  - With the default bucket eviction, a quota or a full disk deletes unreleased objects, oldest bucket first. Reads then miss, but no wrong row is delivered.
  - With eviction off, writes that do not fit are refused, and the engine stops at the put stall deadline.
- Admission by free disk is therefore required (the session branch's protocol, section 8).
- Its bound (the longest context times all branches times the group size) reserved about ten times what the sub-agent scenario wrote. This lowered concurrency there. A bound per live branch is future work.

**Known limits.**

- **Objects that reappear after a release**: 6 to 9 keys per run, outside planted faults, in a few runs of either design. Mooncake 0.3.14 registers them again ([fault tolerance](#fault-tolerance)).
- Mooncake 0.3.14 does not free SSD space when objects are removed (RFC kvcache-ai/Mooncake#3220).
- A failed synchronous KV load remains a documented limit of both designs. The runs cover only the injected path.
- The request-wise branch's `request_wise.md` lists that design's other limits: outputs lost after `generate` yielded them, and the versions of listed products, which are not returned.
- The session branch's protocol lists a request that is preempted and then scheduled under another version history. Such a request lists only the rows it computed again.
