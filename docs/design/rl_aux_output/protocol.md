# Aux Outputs for RL: Protocol

This page specifies what the engine does with a session request, from admission to release. Terms are defined in the [README](README.md#terms), alternatives in [design space](design_space.md), sizes in [scenarios](scenarios.md).

## 1. The idea

vLLM finds a KV block by its block hash, which covers the block's tokens, the previous block's hash, and extra keys (LoRA name, multimodal inputs, cache salt, prompt embeddings). Every KV connector that stores KV outside the GPU uses the same hashes as keys.

The protocol adds two things to the hash of a session request:

- the **session** and the key prefix, through the cache salt, so that only requests of one session share KV;
- the **weight version of every position**, through per-segment cache salts, so that KV computed with one version is found only by a request that states that version for those positions.

The hash then names a computation: these tokens, after this prefix, with these weight versions, in this session. It is the name of the object that holds the block's rows. A request that hits a KV block, wherever it comes from, lists the object of that name, written by the computation that produced the KV. A request that computes a block names it with the version that computes it.

## 2. A session, step by step

1. **Turn 1 at `v1`.** The request carries `aux_session`. Each block is named by its hash, which includes the session and `v1`, and its object is written in the background. The response returns the keys and the segments `[[0, "v1"], [end, null]]`.
2. **Weight update.** The trainer pauses the engine with `pause_generation(clear_cache=False)` (the default clears the prefix cache and the connector tiers), loads weights, calls `update_weight_version("v2")` and resumes. Turn 1's KV survives in the prefix cache or a connector tier.
3. **Turn 2 at `v2`.** The prompt extends turn 1 and carries turn 1's segments, so its prefix hashes are turn 1's names: it hits turn 1's KV and lists turn 1's keys. Its new positions are named and written with `v2`.
4. **Read.** The trainer reads each turn's rows with that turn's keys.
5. **Pause and release.** After the session's last request stops and a pause completes, all its objects are in the store, except those a failing store lost (section 5); the consumer removes them with one forced regex removal and retires the session name.

## 3. Request and response

| Request field | Meaning |
| --- | --- |
| `aux_session` | The session. A non-empty string without `/`, other than `tail`. |
| `aux_version_segments` | Optional. The `aux_version_segments` of the session's previous response whose tokens this prompt extends. |

Both are `SamplingParams` fields (inside `sampling_params` on `/inference/v1/generate`), accepted only by the Python API server. A session request is rejected (HTTP 400) without `object_key_prefix`, with `stop` strings (the frontend would end it before the engine returns its keys; use `stop_token_ids`), as a streaming-input request, or with malformed segments (section 4.2); segments without `aux_session` are rejected. `routed_experts_prompt_start` and `indexer_topk_prompt_start` are ignored for it.

**When to send `aux_version_segments`:**

| The session's earlier KV | Send? | Effect |
| --- | --- | --- |
| Still held (prefix cache kept, or a connector tier holds it) | Yes | The prefix hashes match the stored KV, so the request hits it and lists the existing objects. Without segments the prefix is hashed with the current version, so after a weight update it finds no KV and is computed again: nothing breaks, only reuse is lost. |
| Gone (prefix cache and tiers reset, evicted, or a replica without a shared KV store) | Yes | The lookup misses, and the positions are computed and named with the current version (section 4.4), under new names from the object (the KV block) that holds the first position computed again whose stated version is not current, and, since block hashes chain, for every object after it; objects before it, and all of them when every stated version is current, keep the names the store already holds. The cost is a lookup that cannot hit, and siblings that state lost versions other than the current one each compute the prefix (section 9). |

Always send them: a computed position is named by the version that computes it (a new object where that renames it), and a hit position runs no forward pass and lists the existing one. Leaving them out is not wrong either; it only costs reuse.

A request without `aux_session` keeps the inline behavior: routed experts and indexer selections come back in the response, prefix-hit rows from the engine-local row cache; without those outputs it returns no rows.

The final output of a stopped session request (the last chunk when streaming) carries:

| Response field | Meaning |
| --- | --- |
| `aux_output_keys` | Keys of the objects holding positions `[0, num_tokens - 1)`, in order: full objects, then the tail if any. About 100 bytes per KV block; every turn lists all its keys. |
| `aux_version_segments` | The segments of the names up to `end`, closed by `[end, null]`. Pass it with the session's next request. |

`end` is the smaller of the positions whose hashes reached the worker and the positions computed by settled steps, rounded down to an object boundary. Full objects are listed up to `end`, the rest is the tail. With hash look-ahead the object that the last sampled token completes therefore stays in the tail, and the next request computes it again under its own version. A request that is aborted, or stops on an output in flight while preempted, lists no keys. The trainer can read as soon as the response arrives: `read_aux_rows` waits for objects still being written, up to its timeout (checked between reads, so a get that blocks is not interrupted), which should be no shorter than the stall deadline (section 5; the reader's default is 600 s). A timeout means the objects did not arrive in time; by itself it does not show that they are lost: the engine may still be writing behind a deep queue, or retrying puts the store refuses. Once every engine that served the session has completed a pause that started after the request stopped, a listed object that is still missing is lost: the writer dropped it, a store process that held it died, or it is held by a store call hung past the stall deadline and may still land later (section 5). The [consumer contract](consumer.md#what-the-trainer-must-do) says what to do then.

## 4. Names

A name is fixed once its position is computed, and a position computed past the last sampled token is not listed. Four moments set the salts: preprocessing, admission, a weight update and scheduling (sections 4.1–4.4). A weight update and the scheduling of a waiting request cut hashes back, never dropping the hash of a block already computed or hit, and name the rest again (sections 4.3, 4.4).

### 4.1 Preprocessing: session salt

The engine core folds the key prefix and the session into the request's cache salt, so sessions under different prefixes share no KV. A plain request whose salt could be mistaken for a session salt gets a different prefix and cannot collide with a session. Block hashes computed by a frontend are dropped; the engine hashes a session request itself.

### 4.2 Admission: client segments

`aux_version_segments` is a list of `[start, version]` pairs closed by `[end, null]`, for example `[[0, "v1"], [256, "v2"], [512, null]]`: positions `[0, 256)` carry `v1`, `[256, 512)` carry `v2`, and coverage stops at `end`. Starts increase strictly from 0, adjacent versions differ, versions are non-empty strings, and only the closing pair is `null`; `[[0, null]]` covers nothing. Anything else is rejected.

The engine clips `end` to the prompt's last token and rounds it down to an object boundary. Positions below it are salted with the client's versions, later ones with the current version. The salts are applied at preprocessing, before the first hashing, and again at admission with the then-current version; if they differ, the scheduler drops the prompt hashes from the cut on and hashes again.

The client's versions decide only which stored KV the request can hit (KV computed with `v1` for `[0, 256)` needs `v1` stated there), not the names of what it computes, which carry the version current when it is scheduled (section 4.4). Segments sent with an edited history (a chat template that drops earlier reasoning, truncation, compaction) therefore cost only the hits after the first changed token, which could not exist anyway. A request without segments names its whole prompt with the current version, as if new to the session.

### 4.3 Weight update: positions not computed yet

`update_weight_version(v)` sets the version of later steps; with `object_key_prefix` set it must be a non-empty string, and it must be called only while the engine is paused, with no step in flight: the engine does not check this, and a call with a step in flight can give the positions that step computes, or a draft it later rejects, the wrong version. A running session request is named again from `num_computed_tokens` on: the connector cuts its block hashes back there, adds a segment with `v` and hashes again, as at scheduling (section 4.4). The dropped hashes are those of blocks no step has fully computed, so neither the GPU prefix cache nor a connector has stored anything under them, and the worker has not received them. A waiting request gets the segment from its first unhashed position, and its scheduling cuts back the rest. A block filled across the update states both versions, each for its own positions.

### 4.4 Scheduling: positions a waiting request computes

A waiting request also computes positions that are already hashed: the part of its prompt that missed the prefix cache and the connectors, everything after a preemption, and the positions after KV it loaded asynchronously. Just before it is scheduled, the scheduler passes the aux connector `H`, the number of positions it will not compute: at admission the final prefix-cache hit (local and external, after any adjustment that drops part of it); after an asynchronous KV load the KV that arrived, smaller if the load failed in part (with `kv_load_failure_policy="recompute"`).

The connector finds the first segment ending after `H` that states a version other than the current one. If there is one, it cuts the block hashes back to `max(H, start of that segment)`, adds a segment there with the current version, and hashes again; complete hash blocks before the cut, including those after `H` already at the current version, keep their hashes. Hashes before `H` never change, so what the scheduler took from the lookup (hit blocks, external token count) stays valid. The GPU prefix cache and the connectors read the hashes afresh when they allocate, store or look up; the offloading connector, which keeps keys derived from the hashes, derives them again from the first chunk whose hash changed.

### 4.5 Engine features

- **Preemption.** The request waits again and is salted when scheduled; at the same version its hashes stay.
- **Speculative decoding.** Supported. A trailing block of a hit that EAGLE-style drafting drops and computes again is named by the current version. Draft KV that multi-module MTP computes again after a weight update carries the target positions' versions; the target rows are not affected.
- **`prompt_logprobs` and `skip_reading_prefix_cache`.** The request computes its prompt again from position 0, or from `prompt_logprob_start` when that is positive (positions before it can hit the local prefix cache, not connector tiers, and only when the request sets `skip_reading_prefix_cache=False`, which `prompt_logprobs` otherwise turns on), under the current version.

## 5. Objects and writes

**Full objects.** One per KV block of the full-attention group, keyed `<object_key_prefix>/<aux_session>/<hash of the block's last hash block, hex>`. That hash chains every earlier one, so the key names the whole prefix: tokens, versions, session and other extra keys. The object is written once the block's last position is computed and its hash has reached the worker.

**Tail objects.** When a request stops, the positions after its last listed full object go into one object keyed `<object_key_prefix>/<aux_session>/tail/<H>`. `H` hashes, under a tail domain tag, the previous block's hash, the tail's tokens, and the extra keys of the tail positions (versions, LoRA, multimodal inputs, prompt embeddings), so tails of different branches, lengths or versions never share a key. A tail usually covers less than one block; it covers more when the hashes of the last blocks had not reached the worker, and with hash look-ahead it includes the block that the last sampled token completes.

**Contents.** A safetensors object with one array per enabled output, one row per position; without routed experts only the other fields:

| Field | Shape | Type |
| --- | --- | --- |
| `routed_experts` | `[positions, hidden layers, experts per token]` | `uint8` up to 256 experts, else `uint16`; logical expert ids, rows of dense layers 0 |
| `indexer_topk` | `[positions, indexer source layers, index_topk]` | `int32`, exactly as captured |
| `hidden_states` | `[positions, hidden_size]` | the model dtype; bfloat16 stored as `int16` bit patterns |
| `topk_token_ids` | `[positions, k]` | `int32` |
| `topk_logprobs` | `[positions, k]` | `float32` |

- `hidden_states` are what the forward pass returns for every position: for most models the input of the LM head, the states before the final normalization for a model that applies it inside `compute_logits`.
- Top-k logprobs come from `compute_logits` on those states, log-softmax in float32, before any sampling parameter, for session rows only, in chunks of at most 1024 rows and at most 64Mi / (k + 1) rows, as for `prompt_logprobs`. The chunk's float32 logits and the step's results are not in the memory profile: leave `gpu_memory_utilization` headroom for about 1024 × vocabulary size × 10 bytes plus the results (session tokens of a step × k × 8 bytes, twice that while chunks are concatenated). On the host the results are copied to pinned memory, staged per request, serialized and queued, so host memory needs several times their size.
- A prefix-cache hit keeps the rows of the request that computed the position. Unlike `prompt_logprobs`, top-k logprobs leave prefix caching on, so a teacher can score a long multi-turn prompt turn by turn.

**Pipeline parallelism.** Each stage passes the routing of its own and earlier MoE layers to the next with its intermediate tensors, and the last stage writes objects with every layer, so objects do not depend on the pipeline-parallel size.

**Writes.** The first TP rank of the last stage writes, from a background thread whose queue holds up to `2 × max_num_seqs` batches; enqueueing waits when it is full. Writing a key that exists returns success, and the store keeps the first object written under it, never a mix; of concurrent writes it keeps one. An SSD-first store did the same for all 1,833 rewrites of keys already on SSD or being offloaded ([experiments](experiments.md#mooncake-behavior)). A put that Mooncake refuses is retried every 0.2 s, and enqueueing and pauses wait while it is: -200, no memory (memory full while it offloads, or a pinned pool waiting for releases), and any other code, such as -800 from a holder that is gone or an RPC failure from a master that is gone, since a replaced process can take the write. A failed SSD write leaves memory full, and a store that can never fit the object (no holder mounted, or an object larger than any segment) also refuses with -200; both look like a full store.

**Store failures.** The stall deadline is `object_put_stall_timeout_s` (default 600 s), capped at half of `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (default 300 s, so 150 s), since a step that waits longer for the writer queue times out and stops the engine. When writes have been pending for the deadline without a successful put, the writer drops the rest of the batch it is writing. From then on each batch gets one attempt and is dropped if the store refuses it, until a put succeeds and retries resume. While the writer drops, or no put has succeeded for the deadline, enqueueing does not wait either: a batch that meets a full queue is dropped, so a store that stays down holds back steps for at most one deadline. A store that takes writes, only more slowly than the engine produces them, slows the engine down as before. The engine keeps running, and a response that lists a dropped object is not changed: the object is missing, and it stays missing, since a later request that hits its KV lists the same key and does not compute it again. The [consumer contract](consumer.md#what-the-trainer-must-do) says how the trainer drops the samples that need it. A put that hangs inside the store does not stop the engine either: once it has run for the deadline, a batch that meets a full queue is dropped, and a pause stops waiting for it and drops the batches queued behind it; the hung put itself can still land later. Shutdown waits for a put in progress, so a put that hangs inside the store keeps the engine from exiting. A writer error that is not a store error stops the engine at its next step, and a pause fails with it.

## 6. Hits, KV tiers and weight updates

A session request may hit up to the last KV block boundary below its last token, from the GPU prefix cache or any KV connector keyed by vLLM's block hashes: the offloading connector (CPU and tiering specs), simple CPU, HiSparse, Mooncake store, UMBP, and the multi connector over them. None has session code.

**A hit lists the right object** (section 1): the KV block was computed by a request of the session that stated the same tokens and versions and named it by the computing version (section 4.4), so the hitting request gets that computation's rows across tiers, replicas and weight updates.

**A miss costs compute and pool space.** When the KV of a stated prefix is gone (evicted from every tier, tiers cleared, another replica without a shared KV store), the request computes those positions again under the current version and writes them under new names from the object (the KV block) that holds the first position computed again whose stated version is not current, and, since block hashes chain, for every object after it; objects before it, and all of them when every stated version is current, keep the names the store already holds; the names stay true. Size the KV tiers for the live sessions' prefixes, including decode blocks, across weight updates ([scenarios §3](scenarios.md#3-kv-tier-capacity-across-weight-updates)).

**Decode blocks.** A KV connector must store the blocks decoding completes, or the next turn computes the last block of every turn again on a replica that lacks it. For `MooncakeStoreConnector` set `save_decode_cache: true`.

**Replicas and sessions.** A session can move to another replica that shares the object pool; if they also share a KV store, the new replica hits the old one's KV and lists the same names. Different sessions never share KV, so a system prompt shared by many sessions is computed once per session.

**Weight updates.** An RL loop pauses the engine, may release GPU memory, loads weights, calls `update_weight_version` (section 4.3), and resumes. Session prefixes survive in the GPU prefix cache if it is kept, otherwise only in connector tiers. The deployment chooses whether the prefix cache and the connector tiers are reset (`pause_generation` resets both unless `clear_cache=False`) and the pause mode for in-flight requests (`keep`, `wait` or `abort`).

- **Kept KV.** The next turn sends the previous segments, hits its prefix under the old versions, and computes only its new positions with the new version.
- **Reset.** A prefix-cache reset with `reset_running_requests` preempts the running requests. Objects they completed before it, and those of session requests that finished before it, are still written; a preempted request is scheduled again as in section 4.4.
- **Tiers cleared.** The next turn misses its prefix (see above).

## 7. Pause and release

**Pause.** Terminal events of a stopped request reach the workers with the next step, and objects are written in the background. Before a pause (`pause_generation` in any mode, including before a weight update, sleep or checkpoint) completes, the engine sends the pending terminal events and waits until the writers are idle, so it guarantees that every object of every request that stopped before it is in the store, including full objects of aborted requests and of requests that list no keys. Two kinds of objects are exceptions (section 5): those the writer dropped, which are lost, and those of a store call hung past the stall deadline: the pause stops waiting for it and drops the puts queued behind it, and the hung call may still land later. The pause waits while writes progress or are retried; once the writer drops, each queued batch gets one attempt. It fails with a writer error that is not a store error, or when another part of the pause fails, such as a connector reset. Only with `object_key_prefix` set, it costs one worker round trip when nothing is queued, and otherwise the time to write what is.

**Release.** The engine only writes. The consumer deletes objects with a Mooncake client (`mooncake.store.MooncakeDistributedStore`) connected to the same store:

- `remove_by_regex(object_store.aux_session_pattern(object_key_prefix, aux_session), True)` for a session, once all its requests have finished and every engine that served it has completed a pause since;
- `remove_by_regex(object_store.aux_prefix_pattern(object_key_prefix), True)` for a prefix, once the engines writing under it have shut down or stopped writing to it.

Release by session, not by listed keys: a session also holds objects no final response lists, namely blocks hashed when a request stopped (listed in its tail, and by name in the next turn) and objects of aborted requests. Force is required: a read holds a lease on its keys, and a removal without force inside the lease removes nothing. Mooncake skips objects still being written, even with force, and they stay in the store; the pause rule is what makes a session release complete. An object held by a store call hung past the stall deadline can land after the release; the release of the prefix removes it. Never send a released `aux_session` again: the engine would still let it hit the KV of the removed objects. Continue under a new session name, which computes the prefix again.

## 8. Deployment

| `AuxOutputConfig` field | Meaning |
| --- | --- |
| `enable_return_routed_experts` | Capture routed experts. MoE models only. |
| `enable_return_indexer_topk` | Capture the sparse-attention indexer's selections. |
| `enable_return_hidden_states` | Store `hidden_states` (section 5). |
| `return_topk_logprobs` | `k > 0` stores `topk_token_ids` and `topk_logprobs` (section 5). |
| `object_key_prefix` | Enables sessions. First segment of every object key. |
| `mooncake_config_path` | Mooncake configuration file of the object store; `None` reads `MOONCAKE_CONFIG_PATH`. |
| `object_hard_pin` | `False` (default): objects are not pinned, for an SSD-first store. `True`: hard-pinned, for a memory-only store. |
| `object_put_stall_timeout_s` | How long writes may be pending without a successful put before the engine drops them (default 600 s, at most half of `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, section 5). |
| `max_bytes` | Size of the engine-local row cache of requests without a session. |

**Startup checks.** The engine refuses to start when:

- `object_key_prefix` is empty, contains `/`, or is set without any enabled output (`--enable-return-routed-experts` and `--enable-return-indexer-topk` enable theirs);
- `enable_return_hidden_states`, `return_topk_logprobs` or `mooncake_config_path` is set without `object_key_prefix` (the first two exist only as object fields);
- `max_bytes` is set without routed experts or with indexer top-k;
- routed experts are enabled on a model without MoE layers, or indexer top-k with pipeline parallelism;
- any aux output is enabled with a non-generate runner, without prefix caching, with adaptive speculative verification, or with decode or prefill context parallelism;
- `object_key_prefix` is set and the model has no plain full-attention KV cache group (whose block size is the object size), hits may end inside a KV block, block hashes look ahead more than one token, or a group uses prefix replay (disable `CacheConfig.swa_bounded_replay`).

For models with sliding-window KV cache groups (DeepSeek-V4 class), set `prefix_cache_retention_interval` to `null` or the block size; with the default, a later turn's hit stops at the previous turn's prompt instead of including its output.

**Key prefix.** Objects are immutable, so a prefix is never reused: choose a new one for every start of a deployment, including a restart of the same job, and a different one for every deployment sharing the store. Replicas of one deployment share it, writing the same names for the same computations, so each object is stored once.

**Replicas.** One name must mean the same rows everywhere:

- **Batch invariance.** The same tokens at the same versions give bitwise identical outputs regardless of batch composition, chunking, and prefix-cache hit or miss (#62365).
- **Identical hashing.** Replicas sharing KV stores or the object pool use the same hash algorithm with `PYTHONHASHSEED` set to one value everywhere or unset everywhere (unset, only a cryptographic algorithm such as `sha256` gives every process the same seed), hash block size, full-attention KV block size and look-ahead. A key does not encode the object size, so different object sizes would give one key different rows.
- **Identical object layout.** Replicas sharing the object pool enable the same outputs.

**Store.** The engine reads the Mooncake configuration at `mooncake_config_path`, or at `MOONCAKE_CONFIG_PATH`, the file the Mooncake KV connector reads, when it is not set. The mode must be `"standalone-store"`: objects in an embedded client's memory would disappear when it exits. The object pool has its own master and pool, never shared with a Mooncake KV store, which evicts and can be reset with `remove_all(force=True)`; with the Mooncake KV connector, `mooncake_config_path` names another file.

**Capacity.** Every listed object must stay readable until its session is released; one that disappears can only time out its reader. The engine cannot see the master's settings, so with `object_hard_pin` `False` and no offload on eviction an evicted object is lost. Two deployments keep every object:

- **SSD-first (default).** Each holder's memory segment caches a local NVMe directory; objects leave memory by offload, never by loss, and are not pinned, since a pinned object is never evicted and so never offloaded. `mooncake_master` runs with `--enable_offload=true --offload_on_evict=true --offload_force_evict=false --enable_disk_eviction=false --client_ttl=600`.
    - **Offload on eviction only.** Offload at put evicted objects whose disk write had not finished and lost 3% of 100K small objects without an error ([experiments](experiments.md#mooncake-behavior)).
    - **Client TTL above any holder pause.** A holder silent longer than the TTL is dropped with its memory: at 10 s a 30 s pause lost every object, at 600 s none.
    - **A new directory per holder and job.** Every `mooncake_client` runs with `--enable_offload=true` and `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` set to an empty directory it owns, deleted when the job ends. Removal deletes only metadata: Mooncake 0.3.14 neither reclaims nor reuses bucket files, so each holder's disk use is the total it was written in the job, reclaimed only when the directory is deleted, and a holder restarted on an old directory registers removed keys again ([experiments](experiments.md#mooncake-behavior)). Plan SSD by the bytes written in the job, not by the bytes in flight.
    - **SSD on every holder the master can place objects on**, or objects on a memory-only holder can be evicted and lost.
    - **Admission by free disk.** Before each new group of sessions, admit it only if `free disk − segment − reserve − reserved ≥ group bytes` holds on every holder. `group bytes` is what the group can write: `longest context × bytes per position × branching factor × group size` ([scenarios §2](scenarios.md#2-bytes-per-position-and-per-trajectory)), plus one tail object per turn, and a copy of a prefix for every weight update at which its KV can be lost, since a loss after an update computes it again under new names (section 6); `reserved` counts the same for the admitted groups, less what they have written. Each holder's disk fills on its own: use its free disk and the share of these bytes the master places on it, or, when the share is not known, the holder with the least free disk. Released groups free no disk, so the headroom shrinks with every byte written in the job, not with the bytes in flight. The next turn of an admitted session is always admitted.
    - **When admission fails.** Mooncake 0.3.14 reclaims no disk, so once a holder fails the check because written bytes filled its disk, it fails for the rest of the job; a failure that only reflects reservations clears when admitted groups write less than reserved. Stop admitting, read and release the admitted sessions, and end the job segment: continue on new holder directories with a new `object_key_prefix`, or use a Mooncake release that reclaims removed objects.
    - **Disk eviction off.** Disk eviction deletes bucket files by age, including objects not yet released, and `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` has no effect without it, so disk use is bounded only by admission.
- **Memory only (`object_hard_pin` `True`).** Objects are hard-pinned until released; admit while the admitted, unreleased bytes stay below `0.75 × pool`. A full pool that nothing releases stalls writes, and from the stall deadline on the engine drops objects until a put fits again (section 5); no fallback aborts sessions to free it, so admission and release must prevent it.

**Holders stay up for the job.** Mooncake keeps one copy of each object, on the holder it was placed on. Every holder stays up until the objects on it have been read and released, and no holder is restarted within a job: a replica that stops its holder when it finishes loses the objects other replicas placed there, which the master still lists while every get returns empty ([experiments](experiments.md#mooncake-behavior)). If a holder dies anyway, generate the sessions with objects on it again under new session names, and watch holder liveness rather than waiting for read timeouts.

## 9. Limits

A key names the tokens and the versions a request states for them, and the store keeps the first object written under it. A position is named by the version that computes it (section 4). The cases below are the exceptions and limits. The engine does not count them; a failed synchronous load raises, asserts or goes unnoticed depending on the case (below).

| Limit | When | Effect | Remedy |
| --- | --- | --- | --- |
| Failed synchronous KV load | A connector reports a failed synchronous load through `invalid_block_ids` | Not supported. The report arrives after the step ran: the objects completed by that step, and by a step already in flight under asynchronous scheduling, including the tails of requests that finished, were computed from the invalid KV and are already written under their final names, and the store keeps them. With more than one KV cache group, any `invalid_block_ids` makes the scheduler raise, whatever the policy. With one group and `kv_load_failure_policy="recompute"`, recomputing the request usually stops the engine with an assertion (not under `python -O`), but nothing stops it when the request is aborted, preempted or reset (`reset_running_requests`) before it is scheduled again, when a request that shares the invalid block is not rolled back with it (for example one absent from the failing step under asynchronous scheduling), or when a request finishes on an output already in flight. With the default `"fail"` the request fails, but the objects stay and a retry or the next turn reads them | Use, for session requests, `MooncakeStoreConnector` with `load_async` at its default `true`, which loads asynchronously and reports failed loads, or `OffloadingConnector`, which raises on a failed synchronous load. Neither backend of `SimpleCPUOffloadConnector` reports a failure. The CPU backend copies in both directions on one thread, and a failed copy ends it. The disk backend (`kv_offload_backend="disk"`) loads and stores on separate threads: a failed read, or a failed GPU copy while loading, ends the load thread; a failed write, or a failed GPU copy while storing, ends the store thread. The failed transfer and every later one on that thread never finish, and the engine never goes idle. Once a store is stuck (on the CPU backend, any store after a failed copy in either direction), every later pause hangs, and with it the weight update and sleep; while only loads are stuck, pauses in `wait` or `abort` mode hang, and after an abort of the stuck request every pause in any mode. Either way there is no engine error or request failure and the blocks stay held, and shutdown ends only when the process manager kills the engine after its timeout, skipping teardown: object writes still queued at the kill are lost. The disk backend's offload file is removed only at its next start, however the engine exits. |
| MTP draft KV | Multi-module MTP computes draft KV again after an update | The draft KV carries the target positions' versions (section 4.5) | None needed: the target rows are not affected |
| Siblings that state the same lost versions | Several requests of a session state versions other than the current one whose KV is gone | Each computes the prefix: salting follows a request's prefix lookup at scheduling, so each misses the KV another computed under the current version. Names and rows stay correct; where every stated version is current, the names do not change and the siblings reuse each other's KV | Costs compute only. Keep the KV (below), or wait for the first sibling's response and submit the others with the `aux_version_segments` it returns (section 3) |

To keep session KV: raise `gpu_memory_utilization`; add a KV tier such as `OffloadingConnector` or `SimpleCPUOffloadConnector` (with `"kv_offload_backend": "disk"` when it shares the node with training; after a failed transfer pauses can hang, see the table); share a KV store across replicas; and set `save_decode_cache: true` for `MooncakeStoreConnector` (section 6). Size the tiers with [scenarios §3](scenarios.md#3-kv-tier-capacity-across-weight-updates); they mainly keep prefixes across sleep and reset.

**Aborts.** An aborted request lists no keys, but the full objects of its blocks whose hashes reached the worker are still written, and other requests of the session can hit them. Its tail object is not written: only a request that stops on its own gets a tail key.

Not supported:

- **Prefill/decode disaggregation** (NIXL, Mooncake P2P, Moriio): the decode side does not receive the prefill's segments. Support is planned.
- **Prefix replay**: rejected at startup; disable `CacheConfig.swa_bounded_replay`.
- **LMCache, LMCache-MP and HF3FS**: out of scope.
- **The Rust frontend** does not accept `aux_session`; use the Python API server.
