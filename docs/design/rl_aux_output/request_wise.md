# Request-Wise Aux Outputs

This branch changes who names a position and who owns an object. In the session design ([protocol](protocol.md)) a position is salted with the weight version that computes it, so requests of a session share objects and a name must predict the computing version. Here a position is salted with the id of the request that computes it, a request writes only positions whose products no one holds yet, and the trainer releases objects by reference.

This document describes the code on this branch. It replaces sections 4 (names), 5 (keys and tails), 7 (release) and the matching limits of the [protocol](protocol.md), and the release rules of the [consumer contract](consumer.md); everything else there (object contents, writes, KV tiers, pause, deployment) applies unchanged.

Principle: **the products first produced are the ones delivered.** A trainer needs the products of the computation that sampled each token (routing replay replays the experts used when the token was sampled). A later computation of the same position, after a reset update, an abort and retry, a preemption or a lost KV block, may run at other weights or on another prefix, and must not replace them.

Assumption: batch invariance (the same tokens at the same weight versions on the same prefix give bitwise identical products). Keep and reset weight updates are both supported.

## 1. Request and response

Request fields (`SamplingParams`):

| Field | Meaning |
| --- | --- |
| `aux_session` | Release unit and key namespace; required for the fields below. Derived requests (forks, compactions, retries) use the session of the request they derive from. |
| `aux_version_segments` | The `aux_version_segments` of the response whose plan the client continues: the salts of the positions it reuses. |
| `aux_held` | The number of leading positions whose products the client already holds (section 4). The engine writes no position below it. |

Response fields (the final output, and the output of an aborted request):

| Field | Meaning |
| --- | --- |
| `aux_output_keys` | The objects this request wrote, in position order. Their rows are the products of positions `[aux_held, end)`, without a gap. |
| `aux_version_segments` | The salts over `[0, end)`; `end` is the start of the last segment, whose salt is `None`. Send them back with the next request of the plan. |

The product of position `p` is the output of the forward pass whose input is token `p`; it decided token `p + 1`. After a normal stop of a request with `aux_held`, `end` is the number of tokens minus one: the last token is covered (its product is at `end − 1`) but has no product of its own. After an abort `end` can be smaller, but never smaller than `aux_held`.

Once a plan uses `aux_held`, every later request of the plan must send it. Without `aux_held`, a session request lists every full object over `[0, end)` by block hash, including those of positions it hit, and an object written by a held request holds only the rows from that request's floor, so a plan that mixes the two modes reads wrong rows. Without `aux_held` the response also lists objects other requests wrote; `AuxSampleKeys` attributes each key to the first sample that lists it.

## 2. Names

- **Segment ids.** Every position a request computes is salted, through the per-segment cache salts of layer 1, with `<request id>@<weight version>`. Request ids are unique and versions contain no `@`, so a segment id names exactly one request and version, and a block hash names the request that computed the block and the versions it computed at.
- **Admission.** Positions below the end of the client's segments (clipped to the prompt, rounded down to a KV block) carry those segments; later positions carry the request's own id at the current version.
- **Scheduling.** After the final prefix-cache hit `H`, from the first segment that ends after `H` and is not the request's own current id, block hashes are cut back to `max(H, start)` and salted with the current id. A request re-salts exactly when a weight update happened since its id was minted, or when it computes positions the client stated but whose KV is gone.
- **Hits.** A request hits only blocks whose salts it states: its own, or those the client sent back. A hit therefore never attaches KV of a request the plan does not reference, and a copy of a block in any KV tier keyed by vLLM's block hashes is unreachable once the block is salted again.

## 3. Objects

- **Full objects** cover one KV block of the full-attention group and are keyed `<prefix>/<aux_session>/<hash of the block's last hash block>`, as in the session design. A request writes only the positions from its floor (section 4), so its first object after the floor holds only the rows from the floor to the block end.
- **Tail objects** are keyed `<prefix>/<aux_session>/tail/<H>`; `H` hashes, under a tail tag, the previous block's hash, the tokens from that block's end to the object's end, the object's first position (for a request with `aux_held`) and the extra keys of those positions. A request writes one at its final stop for the positions after its last full object.
- **Partial objects at preemption.** When a request with `aux_held` is preempted, the products it captured after its last object, up to its settled end (the positions whose step outputs the scheduler has processed), are written as a tail object, and its written frontier moves to the settled end. A request preempted by a prefix-cache reset writes it at the generation change. The resumed request computes those positions again without writing them. The object's start, end and tokens enter its key, so no key is written twice.
- **Puts.** The writer puts each object on its own and counts it as progress toward the put stall deadline. A put refused for lack of memory is retried without the objects already written, so a key the trainer read and removed is never written again. Puts and removals share one queue in order, so new removals wait behind a refused put; removals the store refused earlier are retried after the 1st, 2nd, 4th, 8th and 16th refused attempt of the put and every 32nd after that.

A consumer concatenates the rows of the listed objects.

## 4. The floor

A request with `aux_held` writes no position below `floor = max(aux_held, written frontier)`. The written frontier is the end of the objects the request has written; the floor takes it whenever the request is scheduled again after a preemption.

- **Turns, forks, compactions.** Positions in `[H, aux_held)` are computed again (KV lost, prefix edited, other replica) but not written; the plan keeps the products of the computation that sampled them.
- **Preemption.** Positions below the frontier are computed again on resume but not written; the response lists the objects written before the preemption with those written after it.
- **Last prompt position (C-14).** The engine rejects `aux_held > prompt_length − 1`: the position that samples the first new token is always computed and written by this request.

## 5. Removal by the engine

- **Unlisted objects.** When a request finishes, the objects it wrote that its final response does not list (objects of positions in flight when it stopped, objects of a request aborted after a prefix-cache reset, an abort output nobody took) are removed by the worker that wrote them, after the puts queued before.
- **Responses no client received.** The frontend returns to the engine (`release_unreturned_aux_outputs`) the keys of a final output for a request it no longer tracks (aborted by the client, dropped by a pause), and of a final output still queued when its `generate` call is cancelled or closed; the engine removes them in the same way. For a held request the engine marks the response as listing only objects it wrote, so the set is exact. With data parallelism every engine of the client receives the keys; each removes them after its own puts, so the writer's removal follows the write.
- **Aborted requests.** An aborted request returns its partial output with the keys of the objects it wrote before the abort was settled; it writes no tail, because steps in flight may already have moved its rows into objects it does not list.
- **Refused removals.** A removal the store refuses is kept, logged, and retried with every later removal and every pause; "not found" counts as removed. Removals still refused at shutdown are left to the release of section 7.

Barrier: when a pause that started after a request finished returns, its unlisted objects are removed, unless the store refused the removal. For keys the frontend returns, the barrier is a pause sent through the same frontend after the frontend processed the dropped output: it sends the release to every engine right after processing the output, before it handles anything else, and does not wait for an answer. An output that arrives after the pause was sent is released at the next pause.

## 6. Contract for the trainer

- **Plan.** Per trajectory keep a plan: position → (key, row offset), with the segments and tokens. On a response, `plan = plan[:aux_held] + rows of aux_output_keys`. Check that the listed rows number exactly `end − aux_held`.
- **C-8 `aux_held` is exact.** Send `held_sent = min(longest common prefix of the new prompt and the plan's tokens, end, prompt_length − 1)` and nothing else, where `end` is the plan's end; the plan holds no product at it, and the last cap is C-14 (section 4). Too small, and the engine writes positions the plan already holds and the plan update replaces the products first produced. Too large, and the plan keeps products of tokens that changed, or has a gap where it holds nothing; the row count still matches. The engine checks only the upper bound. Truncation (compaction, dropping a suffix) is an explicit operation: release the references of the dropped suffix first, then compute `aux_held` on the truncated plan.
- **C-15 continue only from covered tokens.** Keep only the tokens the plan covers: tokens `0` to `end` of the last response (token `end` was sampled from the product at `end − 1`). After an abort, a dropped response, or a response that ends before its last token, cut the later tokens before the next request. Apply C-15 first, then compute `held_sent`.
- **Ownership.** Each key counts the plans that reference it. A derived sample (fork, compaction, per-turn sample) copies the plan by reference and increments the counts when the derived request is submitted; a derived request that fails before any output decrements them. At zero, `batch_remove(keys, force=True)`, retrying "not found" (−704) and "being written" (−703) for a bounded time.
- **Order.** Release a sample only after its last request's final output arrived. Never send the segments of a released sample again. Fork only from a state whose response arrived.
- **Reads.** Per training batch, deduplicate keys, read each once, slice per plan.

## 7. Lifecycle

- **Release unit.** `aux_session` is the fallback release unit. Derived requests inherit the unit of their parent; otherwise the trainer needs global references to release across units.
- **Fallback release.** `remove_by_regex(aux_session_pattern(prefix, session), True)` removes a unit only when all its samples are consumed, it admits no new request, and every writer that served it completed a pause since. A writer that crashed is isolated first.
- **Choose units that close.** A unit that never closes leaves only the end-of-run sweep of the prefix for its leaks.

## 8. Limits

- **Outputs lost after the frontend handed them over** are not returned to the engine: a response `generate` yielded but the client lost (for example over HTTP), or an `LLMEngine` request aborted after its final output arrived but before the next step. Their objects leak until the unit release of section 7.
- **Positions in flight at a preemption** are not settled. Their products are not in the partial object, except for full objects whose hash was already sent (prefill chunks), which are written when the step lands; the resumed request writes the others from its new computation. Under batch invariance they are equal unless a weight update or a lost prefix came between.
- **A request that stops on an output in flight while preempted, or is aborted while waiting after a preemption**, returns no keys; its objects are removed.
- **Synchronous KV load failure.** Positions after the failure are computed again on other KV under the same names; the request is not re-salted (K4).
- **Versions of listed products** are not returned. The segments name the KV each position used, which after a re-salt can differ from the version that computed a listed product (for example the products of a partial object written before a preemption).
- **Objects are block-hash objects.** Chunk references (`<rid>@<v>/<k>` objects with ranges) are not implemented.
- **Undeclared sharing.** GRPO siblings and `n > 1` compute their prompts separately unless the client forks them explicitly.
- **Prefill/decode disaggregation** is not supported.
- **Mooncake re-creation.** A removal that races with an offload already taken by a Mooncake store worker can bring the key back with its SSD replica; see [experiments](experiments.md#mooncake-behavior).
