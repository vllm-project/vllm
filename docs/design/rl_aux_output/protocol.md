# Aux Outputs for RL: Protocol

This page specifies what the engine does with a session request, from admission to release. Terms are defined in the [README](README.md#terms), alternatives in [design space](design_space.md), sizes in [scenarios](scenarios.md).

The `rl/aux-output-session-wise` branch (head `04495f3806`) carries the version of this page that matches its code: aborted session requests list their keys, sessions are admitted by free disk, and the limits include those found in validation ([evidence](request_wise.md#evidence)).

## 1. The idea

A KV block in vLLM's prefix cache is found by its block hash. The hash covers the tokens of the block, the hash of the previous block, and extra keys (LoRA name, multimodal inputs, cache salt, prompt embeddings). Every KV connector that stores KV outside the GPU uses the same hashes as keys.

The protocol adds two things to the hash of a session request:

- the **session** and the key prefix, through the cache salt, so that only requests of one session share KV;
- the **weight version of every position**, through per-segment cache salts, so that KV computed with one version is found only by a request that states that version for those positions.

The hash then names a computation: these tokens, after this prefix, with these weight versions, in this session. The protocol uses it as the name of the object that holds the rows of the block. A request that hits a KV block, wherever the block comes from, lists the object with the same name, which the computation that produced the KV wrote. A request that computes a block itself names it with the version that computes it.

## 2. Configuration and startup checks

`AuxOutputConfig` fields:

| Field | Meaning |
| --- | --- |
| `enable_return_routed_experts` | Capture routed experts. MoE models only. |
| `enable_return_indexer_topk` | Capture the sparse-attention indexer's selections. |
| `enable_return_hidden_states` | Store the hidden states the forward pass returns, as field `hidden_states` of session objects. |
| `return_topk_logprobs` | `k > 0` stores the top-k next-token ids and logprobs as fields `topk_token_ids` and `topk_logprobs` of session objects. |
| `object_key_prefix` | Enables sessions. First segment of every object key. |
| `max_bytes` | Size of the engine-local row cache of requests without a session. |

**Rejected configurations.** The engine refuses to start when:

- `object_key_prefix` is empty, contains `/`, or is set without any enabled output (`--enable-return-routed-experts` and `--enable-return-indexer-topk` enable theirs);
- `enable_return_hidden_states` or `return_topk_logprobs` is set without `object_key_prefix`: both exist only as object fields;
- `max_bytes` is set without routed experts, or together with indexer top-k;
- routed experts are enabled on a model without MoE layers;
- indexer top-k is enabled with a pipeline-parallel size above 1;
- any aux output is enabled with a non-generate runner, without prefix caching, with adaptive speculative verification, or with decode or prefill context parallelism.

With `object_key_prefix` set the engine also refuses to start unless the model has a plain full-attention KV cache group (its block size is the object size), prefix-cache hits end on KV block boundaries, block hashes look ahead at most one token, and no KV cache group uses prefix replay. Disable `CacheConfig.swa_bounded_replay` for models that use it.

For models with sliding-window KV cache groups (DeepSeek-V4 class), set `prefix_cache_retention_interval` to `null` or to the block size. With the default, a later turn's hit stops at the previous turn's prompt instead of including its output.

**Key prefix.** Objects are immutable, so a prefix is never reused. Choose a new one for every start of a deployment, including a restart of the same job, and a different one for every deployment that shares the store. Replicas of one deployment use the same prefix: they write the same names for the same computations, so each object is stored once.

**Store.** The engine connects to Mooncake with the configuration at `MOONCAKE_CONFIG_PATH`, as the Mooncake KV connector does. The mode must be `"standalone-store"`: objects placed in the memory of an embedded client would disappear when that client exits. Objects are hard-pinned and never evicted.

## 3. Request

| Field | Meaning |
| --- | --- |
| `aux_session` | The session. A non-empty string without `/`. |
| `aux_version_segments` | The `aux_version_segments` of the session's previous response whose tokens this prompt extends. Optional. |

Both are fields of `SamplingParams`; on `/inference/v1/generate` they go inside `sampling_params`. Only the Python API server accepts them.

A request with `aux_session` is rejected (HTTP 400) when `object_key_prefix` is not set, when it has `stop` strings (the frontend would end it before the engine returns its keys; use `stop_token_ids`), when it is a streaming-input request, or when `aux_version_segments` is malformed (section 4.2). `aux_version_segments` without `aux_session` is rejected. `routed_experts_prompt_start` and `indexer_topk_prompt_start` are ignored for a session request.

Requests without `aux_session` keep the inline behavior: routed experts and indexer selections come back in the response, with rows of prefix-hit positions taken from the engine-local row cache. Without routed experts and indexer top-k they return no rows. The rest of this page is about session requests.

## 4. Names

A name is fixed when its hash block is hashed. Four moments set the salts that go into the hashes.

### 4.1 Preprocessing: session salt

When the engine core preprocesses a session request, it folds the key prefix and the session into the request's cache salt. Sessions under different prefixes share no KV. A plain request whose own salt could be mistaken for a session salt gets a different prefix, so it cannot collide with a session. Block hashes computed by a frontend are dropped; the engine hashes a session request itself.

### 4.2 Admission: client segments

`aux_version_segments` is a list of `[start, version]` pairs closed by `[end, null]`:

```json
[[0, "v1"], [256, "v2"], [512, null]]
```

Positions `[0, 256)` carry `v1`, `[256, 512)` carry `v2`. `end` is where the coverage stops. Starts increase strictly from 0, adjacent versions differ, versions are non-empty strings, and only the closing pair has a `null` version. `[[0, null]]` covers nothing. Anything else is rejected.

The engine clips `end` to the position of the prompt's last token and rounds it down to an object boundary. Positions below it are salted with the client's versions, every later position with the current weight version. The salts are applied at preprocessing, before the request's first hashing, and again at admission with the then-current version; if they differ, the scheduler drops the prompt hashes from the cut on and hashes again.

The client's versions decide which stored KV the request can hit: KV computed with `v1` for `[0, 256)` is found only by a request that states `v1` there. They do not decide the names of what the request computes: positions it does not hit are named with the current version when it is scheduled (section 4.4). Segments sent with an edited history (a chat template that drops earlier reasoning, truncation, compaction) therefore cost nothing but the hits after the first changed token, which could not exist anyway. A request without segments names its whole prompt with the current version and computes it as if new to the session.

### 4.3 Weight update: unhashed positions

`update_weight_version(v)` sets the version of later steps; with `object_key_prefix` set it must be a non-empty string. Call it only while no step is in flight, that is while the engine is paused. For each live session request the engine calls `Request.add_cache_salt_segment(num_computed_tokens, v)`: the new segment starts at the first position that is neither computed nor hashed, replaces any segments from there on, and is skipped if the last segment already has `v`. Blocks hashed afterwards carry `v` for those positions; a block filled across the update states both versions, each for its own positions.

### 4.4 Scheduling: positions a waiting request computes

The stated versions are right for the positions a request reuses, but a waiting request also computes positions that are already hashed: the part of its prompt that missed the prefix cache and the connectors, everything after a preemption, and the positions after KV it loaded asynchronously. Just before such a request is scheduled, the scheduler passes the number of positions it will not compute, `H`, to the aux connector:

- at admission, `H` is the final prefix-cache hit (local and external, after any adjustment that drops part of the hit);
- when the request leaves the wait for an asynchronous KV load, `H` is the KV that arrived. A load that failed in part leaves a smaller `H`.

The connector looks for the first segment that states a version other than the current one and ends after `H`. If there is none, nothing changes. Otherwise it cuts the block hashes back to `max(H, start of that segment)`, adds a segment there with the current version, and hashes again. Positions before the cut keep their hashes, including those after `H` whose stated version is already current.

Hashes before `H` never change, so everything the scheduler took from the lookup (the hit blocks, the external token count) stays valid. The GPU prefix cache and every connector read the request's hashes afresh when they allocate, store or look up. The offloading connector, which keeps a list of keys derived from the hashes, notices the changed hash and derives its keys again from the first chunk whose hash changed.

A running request is not salted again (section 12).

## 5. Objects

**Full objects.** One per KV block of the full-attention group, keyed

```text
<object_key_prefix>/<aux_session>/<hash of the block's last hash block, hex>
```

The hash of the last hash block chains every earlier one, so the key names the whole prefix: tokens, versions, session and the other extra keys. A full object is written once the block's last position is computed and its hash has reached the worker.

**Tail objects.** When a request stops, the positions after its last listed full object go into one tail object, keyed

```text
<object_key_prefix>/<aux_session>/tail/<H>
```

`H` hashes the previous block's hash, the tail's tokens, and the block-hash extra keys of the tail positions (their versions, LoRA, multimodal inputs, prompt embeddings) under a tail domain tag. Tails of different branches, lengths or versions never share a key. A tail usually covers less than one block. It covers more when the hashes of the last blocks had not reached the worker when the request stopped, and with hash look-ahead it includes the block that the last sampled token completes.

**Contents.** A safetensors object with one array per enabled output, one row per position:

| Field | Shape | Type | Stage |
| --- | --- | --- | --- |
| `routed_experts` | `[positions, MoE layers of the stage, experts per token]` | `uint8` up to 256 experts, else `uint16`; logical expert ids | every stage, its own layers |
| `indexer_topk` | `[positions, indexer source layers, index_topk]` | `int32`, exactly as captured | pipeline-parallel size 1 only |
| `hidden_states` | `[positions, hidden_size]` | the model dtype; bfloat16 stored as `int16` bit patterns | last stage |
| `topk_token_ids` | `[positions, k]` | `int32` | last stage |
| `topk_logprobs` | `[positions, k]` | `float32` | last stage |

Without routed experts, objects carry only the other enabled fields.

**Pipeline parallelism.** Each stage writes its part to `<name>/pp<stage>`. Responses list the names without the suffix; `aux_object_names(keys, pipeline_parallel_size)` expands them.

**How the fields are computed.**

- `hidden_states` are what the model's forward pass returns for every position. For most models that is the input of the LM head; a model that applies its final normalization inside `compute_logits` returns the states before it.
- Top-k logprobs come from the model's `compute_logits` on those hidden states, log-softmax in float32, before any sampling parameter. They are computed for the rows of session requests only, in chunks of at most 1024 rows and at most 64Mi / (k + 1) rows, as for `prompt_logprobs`. The float32 logits of a chunk and the step's results are not in the memory profile: leave headroom in `gpu_memory_utilization` for about 1024 × vocabulary size × 10 bytes plus the results, session tokens of a step × k × 8 bytes, twice that while the chunks are concatenated. On the host the results are copied to pinned memory, staged per request and serialized into their objects, and up to `2 × max_num_seqs` steps of objects wait to be written, so host memory needs several times the results' size.
- A prefix-cache hit keeps the rows of the request that computed the position. Unlike `prompt_logprobs`, top-k logprobs therefore leave prefix caching on, and a teacher can score a long multi-turn prompt turn by turn.

**Writes.** The first TP rank of each stage writes. A background thread puts batches of objects; the queue holds up to `2 × max_num_seqs` batches, and enqueueing waits when it is full. Writing a key that exists succeeds, and the store keeps one of the objects written under a key, never a mix ([experiments](experiments.md#mooncake-behavior)). A failed put is logged and not retried; the engine raises `BlockObjectStoreError` at its next step or at the next pause, and stops.

## 6. Response

The final output of a stopped session request (the last chunk when streaming) carries:

| Field | Meaning |
| --- | --- |
| `aux_output_keys` | Keys of the objects that hold positions `[0, num_tokens - 1)`, in order: full objects, then the tail if any. |
| `aux_version_segments` | The segments of the names up to `end`, closed by `[end, null]`. Pass it with the session's next request. |

`end` is the smaller of the positions whose hashes were sent to the worker and the positions computed by settled steps, rounded down to an object boundary. Full objects are listed up to `end`; the rest of `[0, num_tokens - 1)` is the tail. With hash look-ahead the object that the last sampled token completes therefore stays in the tail, and the next request computes it again under its own version.

Every turn lists all its keys, about 100 bytes per KV block. A request that is aborted, or stops on an output in flight while it is preempted, lists no keys.

Writes are asynchronous: a response can arrive before its objects land. The reader waits for them (see the [consumer contract](consumer.md)).

## 7. Prefix-cache hits and KV connectors

A session request may hit up to the last KV block boundary below its last token, from the GPU prefix cache or from any KV connector that keys KV by vLLM's block hashes: the offloading connector (CPU and tiering specs), simple CPU, HiSparse, Mooncake store, UMBP, and the multi connector over them. None of them has session code: they store and look up KV by hash, and the hash contains the session and the versions. LMCache, LMCache-MP and HF3FS are out of scope.

**Why a hit lists the right object.** A KV block with a session's hash was computed by a request of that session that stated the same tokens and versions, and was named by the version that computed it (section 4.4). That request wrote the object under the same name. The hitting request lists that name and gets the rows of the computation whose KV it uses, across tiers, replicas and weight updates.

**What a miss costs.** When the KV of a stated prefix is gone (evicted from every tier, tiers cleared, another replica without a shared KV store), the request computes those positions again under the current version and writes new objects under new names. The trajectory's names stay true; only compute and pool space are spent.

**Decode blocks.** A KV connector must store the blocks that decoding completes, or the last block of every turn is missing from it and the next turn computes it again on a replica that does not hold it. For `MooncakeStoreConnector` set `save_decode_cache: true`.

**Across replicas.** A session can move to another replica that shares the object pool. If the replicas also share a KV store, the new replica hits the KV the old one stored and lists the same names.

**Across sessions.** Requests of different sessions never share KV. A system prompt shared by many sessions is computed once per session.

## 8. Weight updates, keep and reset

An RL loop pauses the engine, may release GPU memory, loads new weights, calls `update_weight_version`, and resumes. Session prefixes survive the update in the GPU prefix cache if it is kept, and otherwise only in KV connector tiers.

The deployment chooses whether the prefix cache is reset, whether the reset also clears connector tiers, and what happens to in-flight requests (pause mode `keep`, `wait` or `abort`).

- **Kept KV.** The next turn of a session sends the previous response's segments, hits its prefix under the old versions, and computes only its new positions with the new version.
- **Reset.** A prefix-cache reset with `reset_running_requests` preempts the running requests. The objects they completed before the reset are still written, and so are the objects of session requests that finished before it. A preempted request is scheduled again as in section 4.4.
- **Tiers cleared.** The next turn misses its prefix and computes it again with the new weights under the new version (section 7).

## 9. Engine events

- **Chunked prefill and async scheduling.** Names are fixed when blocks are hashed; the step in which a position runs does not change them. A position computed past the last sampled token is not listed.
- **Preemption.** The request waits again and is salted when it is scheduled (section 4.4). At the same version its hashes stay as they are.
- **Speculative decoding.** Supported. When EAGLE-style drafting drops the trailing block of a hit to compute it again, that block is named by the current version. Draft KV that multi-module MTP computes again after a weight update carries the target positions' versions; the target rows are not affected.
- **`prompt_logprobs` and `skip_reading_prefix_cache`.** The request computes its prompt again from position 0 and names it with the current version.
- **Abort.** An aborted request lists no keys. Full objects of the blocks it computed whose hashes reached the worker are still written; other requests of the session can hit them.
- **Write failure.** The engine stops (section 5). Size the pool for every object written between two releases ([scenarios](scenarios.md)).

## 10. Pause and release

**Pause.** The terminal events of a stopped request reach the workers with the next step, and objects are written in the background. When a pause completes (`pause_generation` in any mode, including the pause before a weight update, sleep or checkpoint), the engine first sends the pending terminal events to the workers and waits until their writers are idle. A completed pause therefore guarantees that every object of every request that stopped before it is in the store, including the tails of aborted requests and of requests that list no keys. The pause waits for the queued writes and fails if they do not finish within 60 seconds or a write fails. This happens only with `object_key_prefix` set; it costs microseconds when nothing is queued and otherwise the time to write what is queued.

**Release.** The engine only writes. The consumer deletes objects with a Mooncake client (`mooncake.store.MooncakeDistributedStore`) connected to the same store:

- `remove_by_regex(object_store.aux_session_pattern(object_key_prefix, aux_session), True)` for a session, once every request of the session has finished and every engine that served it has completed a pause since;
- `remove_by_regex("^" + re.escape(object_key_prefix + "/"), True)` for everything under a prefix, once the engines that write under it have shut down or stopped writing to it.

Release by session, not by the keys a response lists. A session also holds objects no final response lists: blocks hashed when a request stopped, which that response lists in its tail and the next turn lists by name; objects of aborted requests; the objects of every pipeline-parallel stage.

Force is required: a read holds a lease on its keys, and a removal without force inside the lease removes nothing. Mooncake skips objects that are still being written, even with force; they stay pinned. The pause rule above is what makes a session release complete.

Do not send another request with a released `aux_session`: the engine still lets it hit the KV of the removed objects. Continue under a new session name, which computes the prefix again.

## 11. Requirements

1. **Batch invariance.** The same name must mean the same rows: the same tokens at the same versions give bitwise identical outputs regardless of batch composition, chunking, and prefix-cache hit or miss (#62365).
2. **Identical hashing across replicas.** Replicas that share KV stores or the object pool use the same hash algorithm (`sha256`, or the same `PYTHONHASHSEED`), hash block size and look-ahead.
3. **Identical object layout across replicas.** Replicas that share the object pool use the same pipeline-parallel size and enabled outputs.
4. **Separate Mooncake masters.** The object pool does not share a master with a KV store that is reset with `remove_all(force=True)`.
5. **KV capacity.** KV tiers hold the live sessions' prefixes, including their decode blocks, across weight updates. A loss is not a correctness error but recomputes the prefix and writes it again ([scenarios §3](scenarios.md#3-kv-tier-capacity-across-weight-updates)).

## 12. Limits

A key names the tokens and the weight versions a request states for them, and the store keeps the first object written under it. Names are salted again only when a request is scheduled from the waiting state (section 4.4). A running request keeps its names, so in the cases below a position computed after a weight update is named with the earlier version and the store keeps the earlier rows. The engine does not detect or count them.

- **A failed synchronous KV load.** A load that runs inside a step and fails makes the running request compute those positions again under the names it was scheduled with. A failed asynchronous load is salted again as in section 4.4.
- **Hashed before an update, computed after it, while running.** A partly prefilled request, and a decoding request whose last sampled token completes a block, name positions hashed before a weight update and computed after it with the earlier version. So do concurrent siblings with the same prompt whose prefill progress differs at the update; the store keeps the object written first.
- **MTP draft KV** computed again after an update carries the target positions' versions (section 9).

Not supported:

- **Prefill/decode disaggregation** (NIXL, Mooncake P2P, Moriio): the decode side does not receive the prefill's segments.
- **Prefix replay**: rejected at startup.
- **LMCache, LMCache-MP, HF3FS**: out of scope.
- **The Rust frontend**: does not accept `aux_session`.
