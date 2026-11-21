# Aux Outputs for RL: Protocol

This page specifies how products travel from the engine to the trainer through a shared object store: what a request states, what the engine writes and lists, how a trajectory is spliced from the listings, how objects are released, what the store must provide, and what happens when a write or a read fails. Terms are in the [README](README.md#terms), the trainer's side step by step in the [consumer contract](consumer.md), the reasons behind each rule and the two granularities side by side in the [design space](design_space.md), and sizes in [scenarios](scenarios.md).

## 1. Overview

A request that wants products names a namespace in `aux_namespace` and states in `aux_prompt_start` how many leading positions the client already holds products for. The namespace, the key prefix and the captured products are the cache salt, so a request reuses the KV that requests of its namespace computed for the same tokens, and nothing else. Every full KV block a request of the namespace computes is stored once, as an object named by the block hash; the positions after a request's last full block go into a tail object of that request. The response lists, in `aux_objects`, the objects that hold every position from `aux_prompt_start` to the end: blocks the request took from the prefix cache, blocks it computed, and its tail. The trainer splices the listings of a trajectory, keeping the first listing of every position, masks a sample it cannot read, drops the samples of every namespace that was open during a store fault, and removes a namespace by key prefix once its samples are settled and a pause has completed.

There is one mechanism. What the client puts in `aux_namespace` decides what is shared.

## 2. Granularity: what a namespace is

| Granularity | `aux_namespace` | Shares | Prompt | Release |
| --- | --- | --- | --- | --- |
| Group | the GRPO group id, for example `g1` | KV and objects across every request of the group: its samples, their later turns, sub-agents and retries | stored once; computed once when later samples start after it is cached (see below) | one prefix removal per group, once all its samples are settled |
| Trajectory | the group id and the sample, for example `g1/s0` | KV and objects across the requests of one sample only | computed and stored once per sample | one prefix removal per sample, once it is settled |

Requests of different namespaces never share KV or objects, whether or not one namespace is nested under another: the full namespace is in the salt. Nesting matters only for release, since removing `g1/` also removes `g1/s0/` and `g1/s1/` (section 8). Trajectory namespaces give the effect of per-request isolation, in which no sample uses another sample's computation, within the same protocol.

**Forks and sub-agents.** Under group granularity a fork or a sub-agent stays in the group's namespace and hits the KV of the prefix it shares. Under trajectory granularity it either stays in its parent's namespace, and hits the parent's KV, or gets a namespace nested under the parent's, for example `g1/s0/f1`, and computes the shared prefix again. Either way the parent's prefix removal releases it.

### Example: two samples of one prompt

Block size 4, key prefix `p`. Two samples, `s0` and `s1`, of a 9-token prompt each generate 3 tokens: 12 tokens per trajectory, products for positions `[0, 11)` (position 11 is the last token's, never fed through the model). Both send `aux_prompt_start` 0. `s1` is sent after the KV of `s0`'s prompt blocks is cached. `<x0>` and `<x1>` stand for the random names the engine gives the two tails.

**Group: both send `aux_namespace` `g1`.**

| Request | Hits | Computes | Lists |
| --- | --- | --- | --- |
| `s0` | nothing | positions 0 to 10 | `[p/g1/<b0>, 0, 4]`, `[p/g1/<b1>, 4, 8]`, `[p/g1/tail/<x0>, 8, 11]` |
| `s1` | blocks 0 and 1, which `s0` computed | positions 8 to 10 | `[p/g1/<b0>, 0, 4]`, `[p/g1/<b1>, 4, 8]`, `[p/g1/tail/<x1>, 8, 11]` |

The prompt's two full blocks are stored once and both samples list them; block 2 holds generated tokens and is a tail in each sample. The store holds 4 objects. Once both samples are consumed and a pause has completed, `remove_prefix("p/g1/")` removes all of them. Had `s1` been sent at the same time as `s0` to another replica or data-parallel rank, both would have computed blocks 0 and 1 and written the same names; the store keeps one copy, and both list it (below).

**Trajectory: `s0` sends `g1/s0`, `s1` sends `g1/s1`.**

| Request | Hits | Computes | Lists |
| --- | --- | --- | --- |
| `s0` | nothing | positions 0 to 10 | `[p/g1/s0/<a0>, 0, 4]`, `[p/g1/s0/<a1>, 4, 8]`, `[p/g1/s0/tail/<x0>, 8, 11]` |
| `s1` | nothing: its salt differs | positions 0 to 10 | `[p/g1/s1/<c0>, 0, 4]`, `[p/g1/s1/<c1>, 4, 8]`, `[p/g1/s1/tail/<x1>, 8, 11]` |

The hashes differ between the namespaces because the namespace is in the salt. Each sample computes and stores its own prompt; the store holds 6 objects. `remove_prefix("p/g1/s0/")` releases `s0` once it is consumed, `remove_prefix("p/g1/s1/")` releases `s1`, and `remove_prefix("p/g1/")` releases both at once.

**What the two give each sample.** Under trajectory granularity every position of `s1` holds a computation of `s1`'s own. Under group granularity `s1` gets, for positions 0 to 7, the products of the computation that first stored blocks 0 and 1. When `s1` hit that KV, these are the products of the computation its own tokens attended to. When `s1` computed the blocks itself (sent at once with `s0`, below, or after the prefix cache was reset at a weight update), they are those of `s0`'s computation, possibly at another weight version. Within one trajectory both granularities keep the first stored copy when the same tokens are computed again (section 12).

### Samples sent at once

A request reuses the blocks that requests of its namespace have already scheduled on the same engine and data-parallel rank when it is first scheduled, and computes the rest of its prompt itself. Under group granularity, samples sent at once can therefore each compute most of the prompt: samples that land on different replicas or data-parallel ranks, which do not share prefix caches, and samples that start while another sample's prefill is still under way. If all `n` samples of a group compute the prompt, it is computed `n` times and stored once, and `n − 1` of the samples list another sample's computation for positions they computed themselves. Under batch invariance at the same weights the computations agree; without it they can differ numerically (section 12).

To compute the prompt once, send one sample of the group first, and the others to the same replica and rank once its prefill has finished, for example after its first streamed token: they hit every full block of the prompt and compute only the block that holds its last position. Under trajectory granularity every sample computes the prompt in any case.

## 3. Request and response

| Request field | Meaning |
| --- | --- |
| `aux_namespace` | The namespace: at most 256 characters, one or more non-empty segments of `[A-Za-z0-9._-]` separated by `/`, none of them `tail`. Its presence opts the request in. |
| `aux_prompt_start` | `S`: how many leading positions the client already holds products for, between 0 and the prompt length (default 0). It applies to every product, and sets the start of the inline products of a request without a namespace as well. `routed_experts_prompt_start` and `indexer_topk_prompt_start` are deprecated aliases. |

Both are `SamplingParams` fields. `aux_namespace` is accepted on `POST /inference/v1/generate` (inside `sampling_params`) and by the offline `LLM` API only. The OpenAI completion and chat completion routes, of the Python API server and of the Rust frontend, take `aux_prompt_start` and its aliases but no `aux_namespace`, and return no listing: there `aux_prompt_start` only sets where the inline products start, and `return_token_ids` returns the prompt token ids it counts, so a client that sends text can check its common prefix against them. A request with `aux_namespace` is rejected (HTTP 400; `VLLMValidationError` offline):

- without `object_key_prefix` configured;
- with a namespace outside the format above;
- with `n > 1`: send the samples of a group as separate requests;
- with `stop` strings, since the frontend would end the request before the engine returns its listing (use `stop_token_ids`);
- as a streaming-input request.

With or without a namespace, `aux_prompt_start` outside `[0, prompt length]` is rejected, and so are aliases that disagree with it or with each other (0 counts as unset; an alias in use logs a deprecation warning once). A request without `aux_namespace` keeps the inline behavior: routed experts and indexer selections come back in the response from position `aux_prompt_start` on.

The final output of a request that stops or is aborted (the last chunk when streaming) carries:

| Response field | Meaning |
| --- | --- |
| `aux_objects` | `[key, start, end]` entries in order of `start`: the products of positions `[start, end)` are the first `end - start` entries of object `key`. `start` is always the first position the object holds. The entries cover `[S, end)` without a gap, where `end` is `num_tokens - 1` for a stopped request. They overlap only where a preempted request's tail comes before the object that holds its positions again; at equal `start` the tail comes first. The first entry is the object that holds position `S`, so it can start before `S`. |
| `weight_versions` | The weight version that sampled each generated token: sampling provenance, not the version of the listed products, whose first stored copy can come from another computation ([consumer contract](consumer.md#fields-and-versions)). |

The listing includes the blocks the request took from the prefix cache: their names follow from the block hashes, and a request of the namespace with the same token prefix wrote them. A key is listed whether its write has landed, is still pending, or was dropped (section 10). When `S` is `num_tokens - 1` the listing is empty.

**Aborted and preempted requests.** An aborted request lists the positions its returned tokens cover, from `S`: full objects for the blocks they complete, then a tail of the rest. A block that a step still in flight completes is written, since the prefix cache holds its KV, but not listed; the release of the namespace removes it. A tail with no position at or after `S` is neither listed nor written, and an aborted request whose returned tokens cover no position at or after `S` lists nothing (`aux_objects` is null). A preempted request writes a tail of the positions it computed after its last full block, under the same rule, and computes them again when it resumes. Its final listing names that tail before the objects of the later computation: the tail holds the computation the request sampled from, and the splice keeps the earlier entry (section 7). A prefix-cache reset that preempts a request keeps its tails and writes them. A client that aborts its own request receives no final output and no listing.

## 4. Names

The engine core folds the key prefix, the namespace, the captured products (routed experts, indexer selections, hidden states, the top-k count) and the request's own `cache_salt` into its cache salt. A plain request whose salt could be mistaken for a namespace salt gets a different prefix and cannot collide with one. Block hashes chain, so a block hash names the whole token prefix up to the block's end, within one namespace, under one key prefix and one set of captured products.

| Object | Key | Holds |
| --- | --- | --- |
| Full | `<object_key_prefix>/<aux_namespace>/<hash of the block's last hash block, hex>` | the products of the `B` positions of one block |
| Tail | `<object_key_prefix>/<aux_namespace>/tail/<random name>` | the products of the positions after the last full block a request computed; the random name is a UUID4 in hex |

Tails are never in the prefix cache: every request computes its own. The engine draws a tail's name at random once, when it lists the tail; the listing carries the key, and nothing derives it. A name built from the request id could repeat, since a request id is unique only within one engine and only while request-id randomization is on: a retry that keeps its id, or two replicas serving one id, would write the same key, and the store would keep the earlier request's products for both. No namespace segment is `tail`, so a tail key never reads as a key of a nested namespace. A namespace segment spelled like a block hash is not rejected, though; it would make a full key of the parent namespace a path prefix of the keys under it, which a store that keeps keys as paths cannot hold. Group ids and sample indexes never look like block hashes.

Nothing else enters a name. The weight version is in neither the salt nor the key: a request at version `v2` that reuses KV computed at `v1` lists the objects of that computation, and a position the engine computes again after its KV was lost gets the same name as before, so the store keeps the first copy ([design space](design_space.md#why-the-weight-version-left-the-salt)).

## 5. Objects and writes

**When objects are written.** A full object is written once the step that computed the block's last position has settled, that is, once the scheduler has processed the step's output, or at the next pause. A request that finishes also writes the blocks its steps in flight complete: the prefix cache already holds their KV, and a later request that hits them lists them. A tail object is written when the request stops, is aborted or is preempted. Writes run in the background; the response can arrive before its objects land.

**Failed KV loads.** A namespace request whose synchronous KV load failed is preempted, and its blocks from the first invalid one on are evicted from the prefix cache. It computes them again from a valid prefix; since objects are written only from settled steps, no object holds products computed over the failed load.

**Contents.** A safetensors object with one array per enabled output, one entry per position. Without routed experts, objects carry only the other fields.

| Field | Shape | Type |
| --- | --- | --- |
| `routed_experts` | `[positions, hidden layers, experts per token]` | `uint8` up to 256 experts, else `uint16`; logical expert ids, dense layers 0 |
| `indexer_topk` | `[positions, indexer source layers, index_topk]` | `int32`, exactly as captured |
| `hidden_states` | `[positions, hidden_size]` | the model dtype; bfloat16 stored as `int16` bit patterns |
| `topk_token_ids` | `[positions, k]` | `int32` |
| `topk_logprobs` | `[positions, k]` | `float32` |

How the fields are computed:

- `hidden_states` are what the forward pass returns for every position. For most models that is the input of the LM head; a model that applies its final normalization inside `compute_logits` returns the states before it.
- Top-k logprobs come from `compute_logits` on those states, log-softmax in float32, before any sampling parameter (temperature, penalties, logit bias, structured-output masks), so requests with different sampling parameters that share a name share one meaning. They are computed for namespace positions only, in chunks of at most 1024 positions and at most 64Mi / (k + 1) positions, as for `prompt_logprobs`. The chunk's float32 logits and the step's results are not in the memory profile: leave `gpu_memory_utilization` headroom for about 1024 × vocabulary size × 10 bytes plus the results.
- A prefix-cache hit keeps the products of the request that computed the position. Unlike `prompt_logprobs`, top-k logprobs leave prefix caching on, so a teacher can score a long multi-turn prompt turn by turn.

**Pipeline parallelism.** Each stage passes the routing of its own and earlier MoE layers to the next with its intermediate tensors, and the last stage writes objects with every layer, so objects do not depend on the pipeline-parallel size.

**Writer.** The first TP rank of the last stage writes, from a background thread whose queue holds up to `2 × max_num_seqs` batches; enqueueing waits when it is full. Objects are immutable: writing a key that exists returns success, and the store keeps the first object written under it, never a mix; of concurrent writes it keeps one (section 9). An SSD-first Mooncake store did the same for all 1,833 rewrites of keys already on SSD or being offloaded ([experiments](experiments.md#mooncake-behavior)).

**A failed write is between the engine and its store** (section 10). The engine keeps running, does not evict the prefix-cache blocks whose object was dropped, and does not change a response that lists a dropped key. The object stays missing: a later request that hits its KV lists the same key and does not compute it again. The trainer masks every sample that needs it.

## 6. Hits, replicas and weight updates

**Hits.** A request with a namespace may hit up to the last block boundary below its last token, from the GPU prefix cache or any KV connector keyed by vLLM's block hashes: the offloading connector, simple CPU, HiSparse, Mooncake store, UMBP, and the multi connector over them. None has namespace code, because the namespace is part of an ordinary cache salt. LMCache, LMCache-MP and HF3FS are out of scope.

**A hit computes nothing.** The KV of a hit block was computed by a request of the namespace with the same token prefix, which wrote the block's object under the same name. The hitting request lists that name.

**A recompute keeps the first copy.** When the KV of a block is gone (evicted, prefix cache reset at a weight update, a connector tier cleared), the next request that needs it computes it again under the same name. The put succeeds without changing the stored object, and the request lists the key. The products are those of the first computation; the recomputed KV is what later tokens attend to.

**Weight updates.** An RL loop pauses the engine, may release GPU memory, loads weights, calls `update_weight_version`, and resumes. Names do not change at an update. Whether the prefix cache and the connector tiers are reset (`pause_generation(clear_cache=...)`) and the pause mode for requests in flight (`keep`, `wait` or `abort`) change only how much is computed again.

**Replicas.** Routing all requests of a namespace to one engine replica is recommended, not required: names, listings, the write barrier and release stay correct when a namespace spans replicas. Replicas update their weights at different times and do not share their prefix caches, so one replica per namespace keeps the hits (performance) and, under group granularity, keeps two samples on two replicas from computing the same block at different weight versions, where the store would keep one copy for both (exactness, section 12). Under a trajectory namespace, a request sent to another replica computes the trajectory's prefix again; the store keeps the copy the trajectory already used, so the cost is the computation. If a replica is lost, the unfinished samples it served are dropped and their namespaces released. Data-parallel ranks of one engine update their weights together and count as one replica for versions, but each rank has its own prefix cache: a namespace spread over ranks computes a shared prefix once per rank unless a shared KV tier serves the hit.

**Decode blocks.** A KV connector must store the blocks decoding completes, or the next turn computes the last block of every turn again. For `MooncakeStoreConnector` set `save_decode_cache: true`.

## 7. Turns: `aux_prompt_start` and splicing

A trajectory holds products for positions `[0, h)`: 0 before its first request, then whatever its listings covered. Each request sends

```text
aux_prompt_start = min(h, longest common token prefix of the new prompt and the tokens the trajectory holds, prompt length − 1)
```

On a later turn whose prompt extends the previous tokens this is `h`, and the engine lists only what the trajectory does not hold yet. After a retokenization, a compaction or a rewritten history it is the first changed token: the trajectory drops the products after it, and the new listing, named by the new tokens, covers the rest. After an aborted request, `h` is what the aborted request listed, and the next request computes and lists from there.

**The last prompt position.** The engine always computes a prompt's last position again and samples the first generated token from it. The last term makes the request list that position, so a retry or a regeneration of a prompt whose products the trajectory holds takes that position from its own computation, held in its tail when the prompt ends inside a block. When the prompt ends on a block boundary, the block holding the last position is a full block the namespace stored before; the listing names that object, and the position keeps the first stored copy (section 12).

**Multimodal inputs.** Block hashes cover the media items that fill a prompt's placeholder tokens, so two prompts share a position only when their tokens and their media agree up to it. With multimodal inputs the client computes the common prefix over both: a placeholder token filled by another media item (by the client's identity for the item, such as a hash of its content) ends it. Comparing token ids alone would keep products computed for other media.

The trainer appends each listing to the trajectory with one rule: **an earlier copy wins.** With `e` the end of what the trajectory holds, cut back to the smaller of `aux_prompt_start` and `R = max(L − 1, 0)` for a trajectory of `L` tokens, each entry `[key, start, end]` is used for positions `[max(start, e), end)` and skipped when `end ≤ e`; then `e = end`. A gap, or a trajectory that ends short of `R` when it is trained on, masks the sample. The [consumer contract](consumer.md#splicing-a-trajectory) gives the steps.

The rule keeps the first stored copy of every position. A turn's tail object holds the first copy of its positions; the next turn's listing starts at the block that holds `aux_prompt_start`, usually a full object that the next turn computed again over the tail's positions, and the trainer takes it only after the tail. The same rule keeps the first copy when a later request computed earlier positions again after their KV was lost. A first request with `aux_prompt_start` 0 lists every position, the blocks it hit included, so under group granularity a late sample needs nothing from its siblings' responses.

### Example: two turns, block size 16

One trajectory, namespace `g7` (a group namespace; with a trajectory namespace such as `g7/0` only the names change).

Turn 1, request A, `aux_prompt_start` 0: a prompt of 40 tokens, 20 generated, 60 tokens in all. A computes positions 0 to 58 (position 59 is the last token's, never fed through the model).

| Object | Positions | Written by |
| --- | --- | --- |
| `p/g7/<h0>` | `[0, 16)` | A |
| `p/g7/<h1>` | `[16, 32)` | A |
| `p/g7/<h2>` | `[32, 48)` | A |
| `p/g7/tail/<xA>` | `[48, 59)` | A, at its stop |

A's listing: `[p/g7/<h0>, 0, 16]`, `[p/g7/<h1>, 16, 32]`, `[p/g7/<h2>, 32, 48]`, `[p/g7/tail/<xA>, 48, 59]`. The trajectory holds `[0, 59)`.

Turn 2, request B: A's 60 tokens plus a tool result of 25 tokens, a prompt of 85 tokens; 15 generated, 100 tokens in all. The prompt extends the held tokens, so `aux_prompt_start` is 59. B hits the KV of blocks 0 to 2. Block 3, `[48, 64)`, was not complete in A, so B computes positions 48 to 98.

| Object | Positions | Written by |
| --- | --- | --- |
| `p/g7/<h3>` | `[48, 64)` | B; positions 48 to 58 computed again |
| `p/g7/<h4>` | `[64, 80)` | B |
| `p/g7/<h5>` | `[80, 96)` | B |
| `p/g7/tail/<xB>` | `[96, 99)` | B, at its stop |

B's listing starts at the block that holds position 59: `[p/g7/<h3>, 48, 64]`, `[p/g7/<h4>, 64, 80]`, `[p/g7/<h5>, 80, 96]`, `[p/g7/tail/<xB>, 96, 99]`.

Splicing with `e = 59`: `<h3>` overlaps, so the trainer takes `[59, 64)` of it at offset 11; positions 48 to 58 stay on A's tail. The other three entries are appended. The trajectory holds `[0, 99)`, 99 = 100 − 1 positions, from 8 objects (`<h0>`, `<h1>`, `<h2>`, A's tail, `<h3>` from offset 11, `<h4>`, `<h5>`, B's tail).

Had the client rewritten A's answer from token 50 on, the longest common prefix would be 50 and `aux_prompt_start` 50: the trainer cuts A's tail at 50, B hits blocks 0 to 2 as before and computes from 48, and its listing starts with `<h3'>`, the block of the new tokens, used from offset 2.

## 8. Pause and release

**Pause.** Before a pause (`pause_generation` in any mode, including before a weight update, sleep or checkpoint) completes, the engine sends the pending terminal events and waits until the writers are idle. A completed pause therefore guarantees that every object of every request that stopped, was aborted or was preempted before it is in the store, including objects no response lists. This is the **write barrier**: once every engine that served a request of a namespace has completed a pause that started after a request ended, a key that request lists and that is still missing is lost, not late. The set of engines is the namespace's, not the request's: with a KV tier shared across replicas, a request can list a block that another replica computed for the same namespace. Two kinds of objects are exceptions: those the writer dropped, which are lost, and those of a store call hung past the stall deadline; the pause stops waiting for such a call and drops the puts queued behind it, and the hung call may still land later. The pause waits while writes progress or are retried; once the writer drops, each queued batch gets one attempt. With `object_key_prefix` set it costs one worker round trip when nothing is queued, and otherwise the time to write what is.

**Release.** The engine only writes. The trainer removes a namespace with one prefix removal of `<object_key_prefix>/<aux_namespace>/` once:

- every sample the namespace expects, including those of the namespaces nested under it, is settled: consumed (read, and its training step committed), masked, failed, or dropped without being sent again. A group released before all its samples settle would let a late sample hit KV whose objects are gone;
- no request of these namespaces is running or will be sent;
- every engine that served a request of these namespaces has completed a pause that started after their last request ended.

Under group granularity that is one removal per group, once all its samples are settled. Under trajectory granularity it is one removal per sample; removing the group's prefix instead releases all its samples at once. A store fault settles the samples of every namespace open during it as dropped (section 10), and those namespaces are released by the same rule.

The trailing `/` matters: the removal takes the namespace and every namespace nested under it (`g1/s0`, `g1/s1` under `g1`), and nothing of a namespace whose name extends it (`g10`). On Mooncake it is `remove_by_regex(object_store.aux_namespace_pattern(object_key_prefix, aux_namespace), True)` on a Mooncake client (`mooncake.store.MooncakeDistributedStore`) connected to the object store. Force is required: a read holds a lease on its keys, and a removal without force inside the lease removes nothing. Mooncake skips objects still being written, even with force; the pause rule is what makes the removal complete.

The prefix removal also takes the objects no listing names: objects of failed requests and of retries whose responses were discarded, and blocks that an aborted request's steps in flight completed. Failed requests are not released on their own; their objects wait for the namespace. On Mooncake 0.3.14 remove the prefix again after the next pauses, to take objects an offload registered again (section 10).

**Never send a released namespace again.** The engine would still let it hit the KV whose objects were removed. A new namespace name computes the prefix again. `remove_by_regex(object_store.aux_prefix_pattern(object_key_prefix), True)` removes a whole run, once the engines writing under it have stopped.

## 9. The store

The engine writes through `MooncakeAuxObjectStore` (`put_batch`, `get`, `close`); the trainer reads with `get` and removes with a Mooncake client. Any store used for objects must provide:

1. **Atomic publication.** A reader sees no object or the whole object, never part of one, and never bytes other than those written.
2. **First write wins.** Writing a key that exists succeeds and keeps the object stored first. Of concurrent writes to one key one is kept, never a mix.
3. **Visibility at return.** Once a put has returned to the writer, every reader's `get` returns the object. The write barrier relies on it.
4. **Removal is durable.** A removed object is never readable again unless it is written again, whatever runs in the background (offload, compaction, a restart of a store process). Removal succeeds while the object is being read.
5. **No silent eviction.** An object stays readable until it is removed. A write that does not fit fails, evicts nothing, leaves nothing behind, and may succeed later. Objects may be lost only through failures (section 10), and a lost object reads as missing, never as other bytes.
6. **Removal by prefix.** Removing a prefix that ends with `/` removes exactly the objects under it, except those still being written, which the write barrier accounts for.
7. **Fault epoch.** A value that changes whenever the store may have lost objects or its record of them: a holder lost, restarted or dropped after its TTL, or the master restarted. The trainer compares it at namespace open and before consumption (section 10).
8. **Capacity report (optional).** Free bytes, so that admission can keep the store from filling.

Mooncake 0.3.14 in standalone store mode meets 1, 2, 3 and 6 as measured, and 5 with hard pin or with disk eviction off; it violates 4 when a removal overlaps an offload (section 10), and it reports no fault epoch. The trainer integration derives one from a hard-pinned sentinel object on every holder, probed with `batch_probe_key`: a holder that is lost, restarted or dropped after its TTL, and a master restart, lose sentinels. That code is in progress. A pluggable interface with `remove_prefix`, per-key removal status and a filesystem backend exists on a separate local branch ([design space](design_space.md#store)).

**Error classes.** Each party treats store errors by class:

| Class | Meaning | Mooncake 0.3.14 | Writer | Reader | Remover |
| --- | --- | --- | --- | --- | --- |
| FULL | No room now; a later write may fit | −200 on put | retry | — | — |
| NOT_READY | The object is being written | −703 | — | wait up to the read deadline | retry |
| NOT_FOUND | No such object | −704 | — | before the write barrier wait; after it the object is missing | retry until the barrier, then treat as gone |
| TRANSIENT | Transport, RPC or node unreachable | −800, `RPC_FAIL`, −600 from a range read on a dead holder, −1 | retry | retry for a short bound (10 s), then the object is missing | report the key as not removed |
| FATAL | Configuration or authorization | wrong mode or an unusable configuration at connect | refuse to start | raise | report |

`MooncakeAuxObjectStore.get` returns `None` for NOT_FOUND, NOT_READY and TRANSIENT alike, since the Mooncake `get` it calls returns empty bytes on any failure; a reader that needs the class reads with Mooncake batch calls and classifies their return codes.

## 10. Failures

**Writer.** The engine never exits because of a store error.

- **Retry.** A put the store refuses is retried every 0.2 s, whatever the error: no room (a full store while it offloads, or a pinned pool waiting for releases), a holder that is gone, an RPC failure from a master that is gone.
- **Stall deadline.** `object_put_stall_timeout_s` (default 600 s), capped at half of `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (default 300 s), so 150 s in effect by default. A step that waited longer than the execute timeout for the writer queue would stop the engine.
- **Drop.** When no put has succeeded for the deadline, the writer drops the rest of the batch it is writing. From then on each batch gets one attempt and is dropped if refused, and enqueueing no longer waits for a full queue, so a store that stays down holds back steps for at most one deadline. The first put that succeeds ends this. A store that takes writes, only more slowly than the engine produces them, slows the engine down; nothing is dropped.
- **Keys stay listed.** A response keeps listing a dropped key; there is no error marker per request, and no prefix-cache block is evicted for it.
- **A store call that hangs** does not stop the engine: once it has run for the deadline, a batch that meets a full queue is dropped, and a pause stops waiting for it and drops what is queued behind it. The hung call itself can still land; the release of its namespace or prefix removes it.
- **Other writer errors** are bugs, not store errors. They stop the engine at its next step, and a pause fails with them.

**Reader.** Keep the read deadline no shorter than the stall deadline, and let one deadline cover a batch, not each key. NOT_READY and NOT_FOUND before the barrier wait up to the read deadline; after the barrier, a key still missing is lost, and every sample that needs it is masked ([consumer contract](consumer.md#failures)). TRANSIENT is retried for a short bound, then the object is missing: a transfer error means the store has a record of the object but cannot reach the node that holds it, so waiting longer does not help.

**Store faults.** A store fault is any event that can lose objects or the store's record of them: a holder lost or restarted, a holder dropped after `client_ttl`, or the master lost or restarted. After one, objects of any namespace may be missing, and a block whose object was lost can be written again by another computation of the same tokens, so a read can succeed with products other than those first stored. Read misses therefore do not detect it. The trainer records the store's fault epoch (section 9) when it opens a namespace and checks it again after reading, before it trains on the namespace's samples. If the epoch changed, every sample of the namespace not yet consumed is dropped (masked, counted as a store fault), its trajectories are not continued, and the namespace is released (section 8). Namespaces opened after the fault are not affected.

| Failure | Writer | Trainer | Mooncake 0.3.14 |
| --- | --- | --- | --- |
| One object lost (the writer dropped it, or it was removed by mistake) | nothing to do | masks the samples that need it, at once after the barrier | reads as NOT_FOUND |
| The store refuses or cannot take writes | retries, then drops | masks the samples that need a dropped key, after the barrier | a refused write leaves nothing |
| A holder is lost, restarted, or dropped after `client_ttl` | puts placed on it fail as TRANSIENT, are retried, then dropped | a store fault: drops the samples of every namespace open at the time | the master keeps a dead holder's segments registered for `client_ttl` and places new puts on them until then |
| The master is lost or restarted | puts fail with an RPC error, are retried, then dropped; the restarted master takes them | a store fault, as above. Reads fail or block while it is down (more than 70 s measured): give the reading step its own timeout | metadata does not survive a restart |
| An engine crashes | objects still queued are lost | a dead engine completes no pause: the barrier for its requests is the confirmation that its process is gone; keys still missing then are lost. The namespaces it served are released, and a restarted engine uses a new key prefix (section 11) | — |
| The trainer or coordinator crashes | — | the job restarts, with its engines, under a new key prefix (section 11); batches whose keys belong to the crashed run are dropped and generated again | — |

**`client_ttl`.** Within `client_ttl` every put the master places on a dead holder's segment fails after about 1 s and is retried, while puts on live segments succeed, so the writer never reaches its stall deadline and the engine's steps wait on its queue. In a kill test the step with the killed holder took 119 to 121 s of generation with `client_ttl` 30 s (normal 59 to 61 s), and 645 s with Mooncake's default 600 s ([experiments](experiments.md#store-failures)). Run the master with `client_ttl` 30 s, and keep holders from pausing longer than that: a holder silent for longer than `client_ttl` is dropped with its objects, which is a store fault.

**Mooncake 0.3.14 issues.**

- **Removed objects that reappear.** A removal that lands after a holder has taken the object's offload task, but before the holder reports success, erases the key; the holder then finishes writing the bucket and registers the key again, with one `LOCAL_DISK` replica. A minimal reproduction on stock Mooncake brought back one whole bucket each time. Readers are not affected, because a released namespace is not read again. Remove the prefix again after the next pauses.
- **Disk not reclaimed on removal** (RFC kvcache-ai/Mooncake#3220). SSD use is the bytes written in the job.
- **Disk eviction must be off.** With bucket eviction on, a quota or a full disk deletes unreleased objects, oldest bucket first.
- **Dead holder segments kept for `client_ttl`**, and **metadata lost on master restart** (above).

## 11. Deployment

| `AuxOutputConfig` field | Meaning |
| --- | --- |
| `enable_return_routed_experts` | Capture routed experts. MoE models only. |
| `enable_return_indexer_topk` | Capture the sparse-attention indexer's selections. |
| `enable_return_hidden_states` | Store `hidden_states` (section 5). |
| `return_topk_logprobs` | `k > 0` stores `topk_token_ids` and `topk_logprobs` (section 5). |
| `object_key_prefix` | Enables object delivery. First segment of every object key. |
| `mooncake_config_path` | Mooncake configuration file of the object store; `None` reads `MOONCAKE_CONFIG_PATH`. |
| `object_hard_pin` | `False` (default): objects are not pinned, for an SSD-first store. `True`: hard-pinned, for a memory-only store. |
| `object_put_stall_timeout_s` | How long writes may be pending without a successful put before the engine drops them (section 10). |
| `max_bytes` | Size of the engine-local cache of inline products for requests without a namespace. |

**Startup checks.** The engine refuses to start when:

- `object_key_prefix` is empty, contains `/`, or is set without any enabled output;
- `enable_return_hidden_states`, `return_topk_logprobs` or `mooncake_config_path` is set without `object_key_prefix` (the first two exist only as object fields);
- `max_bytes` is set without routed experts or with indexer top-k;
- routed experts are enabled on a model without MoE layers, or indexer top-k with pipeline parallelism;
- any aux output is enabled with a non-generate runner, without prefix caching, with adaptive speculative verification, or with decode or prefill context parallelism;
- `object_key_prefix` is set and the model has no plain full-attention KV cache group (whose block size is the object size), hits may end inside a KV block, block hashes look ahead more than one token, or a group uses prefix replay (disable `CacheConfig.swa_bounded_replay`).

For models with sliding-window KV cache groups (DeepSeek-V4 class), set `prefix_cache_retention_interval` to `null` or the block size. With the default, a later turn's hit stops at the previous turn's prompt instead of including its output.

**Key prefix.** Objects are immutable, so a prefix is never reused. A prefix belongs to one start of the engines: choose a new one whenever engines start, including a restart of the same job and a replica restarted on its own, and a different one for every deployment sharing the store. Replicas started together share one prefix. A key begins with its prefix, so the trainer releases a namespace under the prefix of the keys it listed, and removes a previous start's prefix whole once the engines that wrote under it have stopped.

**Identical object layout.** Replicas sharing the object pool use the same hash algorithm (with `PYTHONHASHSEED` set to one value everywhere or unset everywhere; unset, only a cryptographic algorithm such as `sha256` gives every process the same seed), the same hash block size and the same full-attention block size. Replicas that capture different products get different salts and so different names; they share no KV and no objects.

**Store.** The Mooncake mode must be `"standalone-store"`: objects in an embedded client's memory would disappear when it exits. The object pool has its own master and pool, never shared with a Mooncake KV store, which evicts and can be reset with `remove_all(force=True)`; with the Mooncake KV connector, which reads `MOONCAKE_CONFIG_PATH`, set `mooncake_config_path` to the object store's file.

**Capacity.** Every listed object must stay readable until its namespace is released. Two deployments keep every object:

- **SSD-first (default).** Each holder's memory segment caches a local NVMe directory. Objects leave memory by offload, never by loss, and are not pinned, since a pinned object is never evicted and so never offloaded. `mooncake_master` runs with `--enable_offload=true --offload_on_evict=true --offload_force_evict=false --enable_disk_eviction=false --client_ttl=30`.
    - **Offload on eviction only.** Offload at put lost 3% of 100K small objects without an error ([experiments](experiments.md#mooncake-behavior)).
    - **A new directory per holder and job.** Every holder runs with `--enable_offload=true` and `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` set to an empty directory it owns, deleted when the job ends. Each holder's disk use is the total written to it in the job.
    - **SSD on every holder** the master can place objects on.
    - **Admission by free disk.** Charge every open namespace the bytes of the objects its responses listed, each key once, plus, for each of its requests in flight, the most that request can still write: the positions from one block before its `aux_prompt_start` to its longest response, times bytes per position, plus one header per object and its tail. When a request ends its listing replaces its worst case. Admit a request only if `free disk − segment − reserve ≥ charges + its worst case` holds on every holder. Objects no response lists (a failed request's) are not charged. Released namespaces free no disk while Mooncake does not reclaim it; a holder that fails the check because its disk filled stays failed for the rest of the job.
    - **Disk eviction off.**
- **Memory only (`object_hard_pin` `True`).** Objects are hard-pinned until released. Admit a request only while the charges of the open namespaces, counted as above, stay at or below `0.75 × pool`, and size the pool at least `1/0.75 ≈ 1.33 ×` the bytes written between two releases ([scenarios §4.4](scenarios.md#44-memory-only-store)).

**Holders and master stay up for the job.** Mooncake keeps one copy of each object, on the holder it was placed on, and the master holds the only record of where it is. Neither is restarted within a job. If one dies or restarts, that is a store fault: every namespace open at the time is dropped and released (section 10).

## 12. Limits

| Limit | When | Effect |
| --- | --- | --- |
| Another computation of the same tokens | A namespace computes positions it already has an object for, or writes an object for positions a listing already named: the KV of a stored block evicted or reset (as at a weight update with a cache reset) and the block computed again; a retry or a regeneration of a prompt the namespace computed before that ends on a block boundary, which computes the block holding its last prompt position again and samples from that position while the listing names the block stored earlier (a prompt that ends inside a block lists its own tail for that position, section 7); an object the writer dropped, written later by another computation of the same block. Under group granularity also samples that compute the same prompt (section 2) or the same opening text | Each position keeps its first stored copy, while the request attended to, and may have sampled from, the new computation. Trajectory granularity removes only the cross-sample case; within one trajectory these cases remain. Under batch invariance at the same weights the computations agree; at other weights they differ, and the version of the stored products is not delivered (a version in the object header would detect it, [consumer contract](consumer.md#fields-and-versions)). How often it occurs is not measured yet |
| A hit whose object is lost | A put was dropped while the block's KV stayed cached | Later requests hit the KV and list the lost key; every sample that needs it is masked. If the KV is evicted first, the next computation of the block writes the key (first row) |
| Store faults | A holder or the master is lost or restarted, or a holder is dropped after `client_ttl`, while namespaces are open | Every sample of those namespaces not yet consumed is dropped, also where its objects survived (section 10). The fault epoch on Mooncake is in progress (section 9) |
| Removed objects that reappear | A release overlaps a Mooncake holder's offload of the object to SSD | The key is registered again from the SSD copy (section 10); readers are not affected |

Not supported: prefill/decode disaggregation; prefix replay (rejected at startup); LMCache, LMCache-MP and HF3FS; `aux_namespace` on the OpenAI routes and on the Rust frontend.
