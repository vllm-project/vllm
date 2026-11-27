# Aux Outputs for RL

An RL trainer recomputes the logprobs of the tokens an inference engine sampled. Some values the engine computed on the way cannot be recomputed by the trainer, such as the experts each MoE layer routed a position to (routed experts, R3). They have to travel with the rollout. Returned inline, they force the client to hold every row it received and hand it to each request or trainer worker that needs the same prefix; for long multi-turn rollouts that is most of the data.

This page specifies the second path: a request names a **namespace**, the engine writes the rows once to a shared object store, and the response lists object keys instead of rows. Any process that can reach the store reads them. Requests without a namespace keep the inline behavior. The user guide is [Routed Experts Return](routed_experts.md#namespaces-delivering-routed-experts-through-a-shared-object-store).

## Terms

- **Position.** Position `p` processes token `p` and produces the logits for token `p + 1`. A sequence of `L` tokens has rows for positions `[0, L - 1)`.
- **Namespace.** The string a request sends in `aux_namespace`. It is the scope in which requests share KV, and therefore the scope in which a position can reference rows another request stored.
- **Span.** The positions one object holds, `B` of them: one hash block when prefix-cache hits can end inside a KV block (partial hash hits, below), else one KV block of the full-attention group (of group 0 if there is none).
- **Object.** An immutable entry in the store holding the rows of one span, or of the positions after a request's last full span (a tail).
- **Listing.** The `aux_objects` of a response: `[key, start, end]` entries, each saying that the rows of positions `[start, end)` are the first `end - start` rows of object `key`.

## Rows follow the KV

A request that reuses KV computed its later tokens against that KV, so the rows it needs for those positions are those of the computation that produced the KV. The design makes the name of a span identify that computation:

1. The input processor folds the namespace and the key prefix into the request's cache salt, behind a leading `@`. No other request carries such a salt: the HTTP routes reject `@` in a client salt, and with `object_key_prefix` set the engine rejects a request without a namespace whose cache salt contains `@`, offline requests included. A request therefore reuses only KV that requests of its namespace computed for the same tokens, and every KV connector keyed by block hashes follows with no connector code.
2. A full span is stored as `<object_key_prefix>/<aux_namespace>/<hash of its last hash block, hex>`. Block hashes chain, so the name identifies the whole token prefix within the namespace.
3. Every run writes every full span it computes, and the store keeps the first copy of a name. A request that hits KV lists the name under which the computation that produced the KV stored its rows.

The weight version is in neither the salt nor the name: a request that hits KV computed at an earlier version lists that computation's rows, which is what its tokens attended to. The objects carry no version: `weight_versions` in the response gives the version that sampled each token, not the version of the listed rows.

### Group and trajectory namespaces

Both are the same code; the client chooses how far requests share.

| Granularity | `aux_namespace` | Use for | Shares | Released |
| --- | --- | --- | --- | --- |
| Group | the GRPO group id, `g1` | samples of one prompt trained together (GRPO, RLOO, best-of-n) | KV and objects across every request of the group: its samples, their later turns, retries | once all its samples are settled, by one prefix removal |
| Trajectory | group and sample, `g1/s0`, or a trajectory id | samples released one by one before their group completes; one sample per prompt | KV and objects across the requests of one sample | once the sample is settled; removing `g1/` releases all its samples |

Requests of different namespaces share nothing, also when one name is under the other's prefix: `g1/s0` shares nothing with `g1`. The `/` only lets one prefix removal take several namespaces. Under group granularity the prompt is computed and stored once when one sample goes first; samples sent at once to different replicas or data-parallel ranks each compute it, and the store keeps one copy (see [Limits](#limits)). Under trajectory granularity every sample computes and stores its own prompt, which costs the prompt hits of the group: in a GRPO rollout of DeepSeek-V4.1-Flash with 4 samples per group it lowered generated tokens per second by about 9%, while group granularity left them unchanged. The user guide [chooses a namespace](routed_experts.md#choosing-a-namespace) per kind of rollout.

## Request and response

| Request field | Meaning |
| --- | --- |
| `aux_namespace` | The namespace, a non-empty string; `/` nests one namespace in another for release. Its presence opts the request in. |
| `aux_prompt_start` | How many leading positions the client already holds rows for, `0 <= S <= prompt length` (default 0). It also sets where inline rows start. `routed_experts_prompt_start` is a deprecated alias; with auxiliary outputs enabled, a request whose two values disagree (0 counts as unset) is rejected. |

Both are `SamplingParams` fields, accepted on `POST /inference/v1/generate` (inside `sampling_params`) and by the offline `LLM` API. The OpenAI routes take no `aux_namespace`. A request with `aux_namespace` is rejected (HTTP 400, `VLLMValidationError` offline) without `object_key_prefix` configured, with an empty namespace, with `n > 1` (send each sample as its own request), with `stop` strings (the frontend would end the request before the engine returns its listing; use `stop_token_ids`), and as a streaming-input request.

The final output of a request that stops or is aborted (the last chunk when streaming) carries `aux_objects`, in order of `start`. The entries cover `[S, num_tokens - 1)` for a stopped request without a gap or an overlap, one key per span. They include the spans the request took from the prefix cache, so the first entry is the object that holds position `S` and can start before it. A key is listed whether its write has landed, is pending, or was dropped. A stopped request with nothing to list gets an empty listing.

- **Aborted.** An aborted request lists the positions its returned tokens cover from `S`: full objects for the spans they complete, then a tail for the rest, or nothing (`aux_objects` is null) if they cover no position at or after `S`. One whose rows were dropped before it ran again (a prefix-cache reset while it was preempted, or a `recompute` of a failed load below its written spans) lists only its written spans, with no tail, so its listing can end before its returned tokens; continue the trajectory from the listing's end. A client that aborts its own request receives a final output with `finish_reason` `abort` and no listing (`aux_objects` is null); the engine still writes its spans and a tail that nobody lists, removed with the namespace.
- **Preempted.** A preempted request writes the full spans it computed. It resumes from its prefix-cache hit, a span boundary, and computes the positions after it again; as for inline rows, its listing holds that recomputation.
- **Error.** A request that ends with an error lists nothing.
- **Request ids.** As for every request, a client does not reuse a request id while the request that had it is still finishing.

## Objects and writes

| Object | Key | Holds |
| --- | --- | --- |
| Full | `<object_key_prefix>/<aux_namespace>/<hash of the span's last hash block, hex>` | the rows of one span |
| Tail | `<object_key_prefix>/<aux_namespace>/tail-<UUID4 hex>` | the rows of the positions after the last full object a request lists |

An object is a safetensors file with one array, `routed_experts`, of the inline rows' layout: `[positions, layer visits, experts per token]`, `uint8` up to 256 experts, else `uint16`. A tail's name is drawn at random when the engine lists it and carried by the listing; a name derived from the request id could repeat across retries or replicas.

**Partial hash hits.** When a Mamba or linear-attention group with prefix caching has a block larger than the hash block, vLLM lets prefix-cache hits end inside a KV block. Objects then span one hash block (`--prefix-match-unit` when set, else the gcd of the prefix-cacheable groups' block sizes), so every hit ends on an object boundary. Otherwise an object spans one block of the anchor group (the full-attention group, else group 0), whatever the block sizes of the other groups.

**Write rule.** A run writes every full span it computes, once the span is hashed, whatever any KV cache group keeps (the full-attention prefix cache, a sliding-window or Mamba group, a KV tier, a host tier that publishes blocks late). Every span another request can hit was therefore computed by a run that wrote it. The object is written from the run's rows once the step that computed the span's last position has delivered them: a running request writes the spans its settled steps completed; a request that finishes or is preempted also writes the spans its steps in flight complete. A tail is written only when the request stops or is aborted. A listing names full objects only up to the last span the request's runs wrote or hit; the positions after it are its tail. Writing a key that exists keeps the first copy.

**Errors and KV that fails to load.** A request that ends with an error still writes the spans it computed, so that a request that hit them, in the same step or later, finds them. Under the default `fail` policy (`kv_load_failure_policy`), the scheduler reports the requests whose KV failed to load before ending them with an error; their steps in flight computed over the failed KV and write nothing more.

**No write barrier.** Writes run in the background; a response can arrive before its objects land, and a pause does not wait for them to land. Handing them to the writer waits, as a step does, while the writer's queue is full: up to one deadline (see Writer). The writes due reach the worker with the next step. An engine core process steps while its scheduler holds finished requests whose end the worker has not seen, so the step after the last request ends carries them. An in-process engine (offline `LLM` without multiprocessing) hands them over at its next step. A pause hands them over at once, since no step runs until it resumes.

**Writer.** The first TP rank writes from a background thread. The worker queues the writes due at the start of a step and those of a step's output as one batch each. Every batch must be written within 80% of `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` of being queued; for a store that offloads to slow SSD, raise that timeout. A put the store refuses is retried until then; a put that fails otherwise is dropped at once. A batch not written by its deadline is dropped and logged: its keys stay listed and read as missing, and nothing else changes (later objects of the same namespaces and all other objects are written as before). Every drop also writes a marker `<object_key_prefix>/<aux_namespace>/dropped` holding the reason for each namespace of the batch, also for a namespace whose keys in that batch landed (a store can write part of a batch). A marker the store refuses is written again after the next put that lands. The marker is removed with the namespace. The queue holds 256 batches; a step that finds it full waits until a batch lands or misses its deadline. Once a batch has missed its deadline, and until a put lands again, steps never wait: each batch is queued with its own deadline, and one that finds the queue full is dropped. A store that keeps refusing makes steps wait one deadline once, not once per deadline, and the batches queued during a shorter outage are written when it ends.

## Hits, replicas and weight updates

A request with a namespace may hit up to the last span boundary below its last token, from the GPU prefix cache or any KV connector keyed by vLLM's block hashes (offloading, simple CPU, HiSparse, Mooncake store, and the multi connector over them). An external hit is used only if it ends on a span boundary within the hashed blocks. A connector that loads the whole prompt asynchronously is the exception to the first rule: the request computes its last prompt position again, and its run starts after the hit, so the hit's objects are listed and that position's rows are not written again.

**Replicas.** Routing all requests of a namespace to one replica is recommended, not required: names, listings and release stay correct across replicas. One replica keeps the hits and keeps two replicas from each computing a span the namespace shares. Replicas that share a KV tier or the object store hash identically: the same hash algorithm (with `PYTHONHASHSEED` set to one value everywhere or unset everywhere; unset, only a cryptographic algorithm such as `sha256` gives every process the same seed), hash block size and object span.

**Weight updates.** Names do not change at an update. An update that keeps the CPU or external KV tiers lets later requests hit KV computed at the earlier version; they list that version's rows, which is consistent. Clearing KV at an update makes later requests compute again, under names that already hold a copy (see [Limits](#limits)).

KV connectors must store the blocks decoding completes, or every turn computes its last block again (`MooncakeStoreConnector`: `save_decode_cache: true`). For models with sliding-window groups, set `prefix_cache_retention_interval` to `null` or the block size so that a later turn hits the previous turn's output.

## Turns and splicing

A trajectory holds rows for positions `[0, h)`: 0 before its first request, then whatever its listings covered. Each request sends

```text
aux_prompt_start = min(h, longest common token prefix of the new prompt and the held tokens, prompt length - 1)
```

On a turn whose prompt extends the held tokens this is `h`. After a retokenization or a rewritten history it is the first changed token. The last term makes every request list the prompt's last position, which the engine always computes again and samples from. With multimodal inputs the common prefix also ends where a placeholder is filled by another media item, since block hashes cover the media.

The trainer appends each listing with one rule: **the earlier copy wins.** With `e` the end of what the trajectory holds, cut back to `min(aux_prompt_start, R)` where `R = max(L - 1, 0)` for a trajectory of `L` tokens; then each entry `[key, start, end]` is used for positions `[max(start, e), end)` at offset `max(start, e) - start`, skipped when `end <= e`, and `e = end`. A gap, or a trajectory that ends short of `R` when trained on, masks the sample.

Example, span 16. Turn 1, `aux_prompt_start` 0: a 40-token prompt and 20 generated tokens, positions `[0, 59)`. It lists `h0 [0, 16)`, `h1 [16, 32)`, `h2 [32, 48)` and its tail `xA [48, 59)`. Turn 2 sends the 60 tokens plus a 25-token tool result, `aux_prompt_start` 59, and generates 15 tokens. It hits spans 0 to 2, computes positions 48 to 98, and lists `h3 [48, 64)`, `h4 [64, 80)`, `h5 [80, 96)` and its tail `xB [96, 99)`. Splicing with `e = 59` takes `[59, 64)` of `h3` at offset 11 and appends the rest: positions 48 to 58 stay on turn 1's tail, the computation its tokens were sampled from.

## Release

The engine only writes. The trainer removes a namespace with one prefix removal of `<object_key_prefix>/<aux_namespace>/` once every sample it expects, including those of the namespaces under its prefix, is settled (consumed, masked, failed, or dropped without being sent again), and no request of these namespaces runs or will be sent. The trailing `/` takes `g1/s0` and `g1/s1` with `g1`, and nothing of `g10`. Releasing exactly the sharing scope needs no bookkeeping; removing less, such as one settled sample's objects inside a group, needs a count of the samples that still reference each key.

Never send a released namespace again: the engine would still let it hit KV whose objects were removed. A write that lands after the removal (a store call past its deadline, a span an aborted request's step in flight completed) leaves an object behind until the run's key prefix is removed.

**Key prefix.** Objects are immutable, so a prefix is never reused: choose a new `object_key_prefix` whenever engines start, including a restart of the same job, and a different one per deployment sharing the store. Replicas started together share one.

## The store

The engine and the trainer reach the store through `AuxObjectStore` (`vllm.distributed.aux_output_connector.object_store`). The writer, the key layout and the serialization (safetensors, one dict of arrays per object) sit above it; `read_aux_rows` reads a list of keys and concatenates one field, and `aux_namespace_prefix` and `aux_prefix` give the prefixes to remove. `object_store` names the implementation: `"mooncake"` (`MooncakeAuxObjectStore`), or the qualified name of a class outside vLLM. `create_aux_object_store(name, extra_config)` connects it with the classmethod `create(**extra_config)`, on the engine's output worker and in the trainer alike. The class implements:

- `put_batch(keys, values)`. It raises `BlockObjectStoreError` (`vllm.distributed.aux_output_connector.store`) for a refusal that may succeed later, such as a full store, and the writer retries it every 0.2 s until the batch's deadline. Any other exception drops the batch at once. The engine calls `put_batch` and `close` from one writer thread.
- `get(key)`, returning `None` for a key that does not exist (yet): readers poll on it, since an object may still be in flight.
- `remove_prefix(prefix)` and `close()`.

Keys are opaque strings of `/`-separated segments that may include `.` and `..`; a store that maps keys to file paths must encode them. The store, with its deployment, must provide:

1. **Atomic publication.** A reader sees no object or the whole object.
2. **First write wins.** Writing a key that exists succeeds and keeps the first object; of concurrent writes one is kept, never a mix.
3. **Visibility at return.** Once a put has returned, every reader's get returns the object.
4. **No silent eviction.** An object stays readable until it is removed, also after the processes that wrote it exit; a write that does not fit fails and evicts nothing.
5. **Removal by prefix**, durable: a removed object is never readable again unless written again.
6. **Room for the data in flight.** The store holds every object of the namespaces not yet removed: the samples in flight and those waiting to be trained, at `get_routed_experts_bytes_per_token` per position for routed experts. There is no admission control. A full store refuses puts, the writer retries, its queue fills, steps wait, and a batch past its deadline is dropped and its samples masked; until the store takes a put again, steps no longer wait and a batch that finds the queue full is dropped.

On Mooncake: `"mode": "standalone-store"` and `"global_segment_size": 0` in the configuration file, with a master and pool of its own, never shared with a Mooncake KV store. Either keep objects on SSD (`--enable_offload=true --offload_on_evict=true --enable_disk_eviction=false`, objects not pinned) or pin them in memory (`"hard_pin": true`). Run the master with a short `client_ttl` (30 s): until then it places puts on a dead holder's segments. `MooncakeAuxObjectStore` retries a put only when Mooncake has no room for it (`-200`, `NO_AVAILABLE_HANDLE`); any other error code drops the batch at once, and a removal that returns an error code raises. On SSD, removal must free the files: up to Mooncake 0.3.14 a removal drops only the master's metadata, and with disk eviction off the SSD fills. Use a Mooncake build with SSD reclaim ([kvcache-ai/Mooncake#5711](https://github.com/kvcache-ai/Mooncake/pull/5711)), where each holder deletes the files whose objects were all removed. It checks every `MOONCAKE_OFFLOAD_RECLAIM_INTERVAL_SECONDS` (default 10) and covers its whole index every `MOONCAKE_OFFLOAD_RECLAIM_CYCLE_SECONDS` (default 120). It stays off for a holder that starts with files on disk, so give every job an empty offload directory. A removal that overlaps a holder's offload of the object to SSD can leave the key registered again; remove the prefix again after the next pause.

**Reading.** Keep the read deadline no shorter than the write deadline, and let one deadline cover a training batch, not each key. A key still missing at the deadline is lost, and every sample that needs it is masked. Once the drop marker of a namespace is written, `read_aux_rows` raises at once for a missing key of that namespace, also when the key is still in flight. Read each key once per batch; under group granularity the prompt objects are shared. Create the store client in a process that does not start an engine afterwards.

## Configuration

| `AuxOutputConfig` field | Meaning |
| --- | --- |
| `enable_return_routed_experts` | Capture routed experts. Namespace requests are rejected without it. |
| `object_key_prefix` | Enables namespaces. Non-empty, without `/`; the first segment of every key. |
| `object_store` | `"mooncake"` (default), or the qualified name of a class implementing `AuxObjectStore`, `create` included ([the store](#the-store)). |
| `object_store_extra_config` | Keyword arguments of the store's `create`. Mooncake: `"config_path"`, its configuration file (default `MOONCAKE_CONFIG_PATH`); `"hard_pin"`, pin objects in memory (default `false`). |

With `object_key_prefix` and `enable_return_routed_experts` set, the engine refuses to start without prefix caching, and when a KV cache group uses prefix replay (DeepSeek-V4.1 sliding-window bounded replay recomputes the window instead of hitting it; pass `--no-swa-bounded-replay`). Every other KV cache layout is accepted: sliding window, Mamba and linear attention, sink and R-SWA full attention, HiSparse, partial hash hits and hash look-ahead. The checks of routed experts themselves apply as before.

## Supported models

MoE models whose routers the capture binds run with the default configuration; DeepSeek-V4 models need `--no-swa-bounded-replay`, and routed experts are refused with adaptive speculative verification. Verified end to end (multi-turn rollouts under group and trajectory namespaces, every position compared with its first computation) on Qwen/Qwen3.5-397B-A17B-FP8, openai/gpt-oss-120b, ibm-granite/granite-4.0-tiny-preview (Mamba2 hybrid with partial hash hits) and deepseek-ai/DeepSeek-V4-Pro-DSpark.

## Limits

| Limit | When | Effect | Mitigation |
| --- | --- | --- | --- |
| A position computed again keeps the first copy | A span computed again under a name that holds a copy: after its KV was cleared (weight update, reset, eviction from every tier), at a turn boundary over the previous turn's tail, when a resumed request recomputes its last partial span, under the `recompute` failure policy, or when a retry's prompt ends on a span boundary | The listing names the first copy while the request's later tokens attended to its own computation. They agree under batch invariance at the same weights. | Make recomputation rare: give the KV cache as much memory as the model leaves, keep CPU or external KV tiers across weight updates, and store decode blocks (`save_decode_cache`). Replaying the stored routing on a recompute closes it (a later change). |
| Concurrent computations of one span | Two computations run before either stored it, such as samples of a group sent at once to different replicas | The name holds whichever landed first; the other sample lists it. | Send one sample of a group first and the rest after its prefill, or route the namespace to one replica. |
| A lost object | Its batch was dropped, while the span's KV stayed cached | Later hits list the missing key; their samples are masked. | Size the store for the data in flight (store requirement 6). |
| A preempted step's rows never reach the worker | A request preempted with a step in flight whose rows are never delivered | The spans that step completes are not written; a request that hit them lists missing keys and is masked. | Avoid preemption: size `max_num_seqs` and the KV cache so that running requests fit. |
| A span cached but never computed | A request preempted in the same schedule that allocated and cached blocks for it (base vLLM keeps them cached with no KV) | A later request that hits them lists keys nobody wrote and is masked. | Avoid preemption, as above. |
| Rows computed over KV that failed to load | Under `recompute`, or under `fail` for a request preempted or aborted with a step in flight before the failure was reported (asynchronous scheduling) | Objects of those positions can be written and listed, or a listing can name a key nobody writes (masked). | Keep the default `fail` policy. |

Not supported: prefix replay (refused at startup), pipeline parallelism, prefill/decode disaggregation, LMCache, LMCache-MP and HF3FS, and `aux_namespace` on the OpenAI routes, the Rust frontend and requests passed to the engine as a prebuilt `EngineCoreRequest` (deprecated).
