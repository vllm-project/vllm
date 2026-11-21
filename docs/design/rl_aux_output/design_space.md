# Design Space

The [protocol](protocol.md) delivers products through a shared object store. A request names a namespace; the namespace is part of the prefix-cache salt and of every object name; the response lists objects; the trainer splices, masks what it cannot read, and removes a namespace by key prefix after a pause. The client chooses how widely products are shared through what it puts in the namespace.

This page lists each decision, the alternatives, and what the protocol chose, then puts the two granularities side by side ([group and trajectory namespaces](#group-and-trajectory-namespaces)). Sizes are in [scenarios](scenarios.md), measurements in [experiments](experiments.md).

## What the trainer needs

For every position of a trajectory the trainer needs the products of the computation that sampled the next token, or, for prompt text, of the computation that prefilled it. Three facts make that hard:

- **Positions are computed more than once.** A later turn computes again the last partial block of the previous turn; KV evicted or reset at a weight update is computed again; concurrent samples of a group compute the same prompt; a preempted request computes again after it resumes.
- **Trajectories span weight updates.** At frontier scale a trajectory lives through several updates ([scenarios §1](scenarios.md#1-scenarios)), and later turns reuse KV computed at earlier versions.
- **Without batch invariance, a second computation can differ.** Two computations of the same position at the same weights can give different products, and at different weights they usually do.

## Principle: the first stored copy

The products the store holds first for a position are the reference. A later computation of the same position never replaces them for a trajectory that already has them. Routing replay must replay the experts that were used when the token after a position was sampled, and that computation is the first one to store the position's products.

The protocol keeps the first copy in two places: the store keeps the first object written under a name, and the trainer keeps the earlier listing of a position when a later listing covers it again.

## Decisions

### What the salt states

| Option | Approach | Assessment | Used by |
| --- | --- | --- | --- |
| Version in the key | `<hash>@<version>`, stamped when the step that completes the block runs | The key does not follow the KV: a hit on KV computed at one version lists the key of whatever version the bookkeeping recorded | Built, replaced |
| Version ledger | Per block and per KV cache group, next to the KV | Needs code in every connector and cuts hits where groups disagree | Built, replaced |
| Versions per segment in the block hash | Per-segment cache salts; the client carries `aux_version_segments` between turns | Every connector follows without connector code, but every engine feature that changes when a position is computed needs a naming rule | Earlier session-wise protocol |
| Request id and version per segment | `<request id>@<version>` per segment, plus a floor `aux_prompt_start` | No prediction of the computing version, but still re-salting at updates and at scheduling | Earlier request-wise protocol |
| Owner per position | Per-segment cache salts, one owner per span the client sends back | A name never collides with another owner's; sharing needs the client's spans, cut at exactly the reused token prefix | Request-wise prototype |
| **Namespace** | The namespace, with the key prefix and the captured products, is an ordinary cache salt | Requests of a namespace share KV and objects by name; nothing changes at a weight update; the client chooses what a namespace covers | **This protocol** |

The namespace keeps the property that made the per-segment salts attractive: the name is part of the block hash, so every KV connector keyed by block hashes stores and finds KV under it with no connector code, and needs only an ordinary cache salt. The captured products are in the salt so that requests capturing different products never share KV whose objects lack a field.

### Why the weight version left the salt

**The namespace already isolates.** KV under a namespace's block hash was computed by a request of that namespace for the same tokens, and the object of that computation is stored under the same name. The version in the salt added nothing to who owns a block.

**The trainer kept the first copy anyway.** With versions in the salt, a position computed again under another version history got a new name and a second object. The rule for trajectories was already to take the products of the request that sampled the next token, otherwise the first delivery, so the second object was never used. It cost store space (up to 4.5 times the objects in flight in one earlier run, 56 times with a reset at every update; [experiments](experiments.md#storage)) and a naming rule for every case.

**The birth version.** The version that matters for a product is its birth version: the weight version of the computation that first stored it. A version in the salt had to predict the computing version before the computation ran, which is what required rules at admission, at weight updates and at scheduling. Without it, the response carries only `weight_versions`, the version that sampled each generated token. That is sampling provenance: it equals the birth version only when the first stored copy came from the computation that sampled the next token, and not when it came from a block computed again after a KV reset, a preemption, or, under group granularity, another sample. An optional per-position version in the object header would record the birth version without being part of the name.

**What disappeared with it.**

- Prefills chunked across a weight update, and blocks whose positions straddle a version change, which needed two segments per block.
- Re-salting at scheduling from the prefix-cache hit (`salt_from_hit`), cutting block hashes back at a weight update, and the offloading connector deriving its keys again.
- `aux_version_segments` in requests and responses, and their validation.
- Cross-session forks with an owner session in every segment, and the release order between sessions.

**What remains.**

- The tail: the last partial block of a request, and what the next request does with it (below).
- First write wins when a position is computed again.
- Release only after a pause, the writer's retry-then-drop on store failures, and the store's error classes ([protocol §9 and §10](protocol.md#9-the-store)).

**What it gives up.** Identical tokens at different weight versions in one namespace share a name. A request that computes a full block another request of its namespace stored at an earlier version lists the first copy, also for positions it computed itself ([protocol §12](protocol.md#12-limits)). Under trajectory granularity this needs two computations within one sample; under group granularity two samples suffice. The optional header would detect it.

### Sharing scope

| Option | Sharing | Release | Used by |
| --- | --- | --- | --- |
| Global block hash | across all requests | reference counts across trajectories | Neither |
| Owner per position, spans from the client | what the client states: later turns of a trajectory and explicit forks | owner prefixes, counted when forks share them | Request-wise prototype |
| One session per GRPO group, fixed | everything within the group | the group prefix | Earlier session-wise usage |
| **Namespace chosen by the client** | everything within the namespace: a group, a trajectory, or anything the client nests | the namespace prefix, which also takes nested namespaces | **This protocol** |

One mechanism covers both ends. A group namespace stores a group's prompt once, lets a late sample, a sub-agent or a retry hit KV another request of the group computed, and releases the group at once. A trajectory namespace isolates each sample as the owner salts did: no sample lists another sample's computation, and each sample is released when it is consumed. It needs none of what the owner design needed: no spans in the request or response, no owner reference counts, no rule that spans stop at exactly the reused token prefix. A trajectory states in `aux_prompt_start` how many positions it holds, and the engine, which hashes the tokens, decides what it can reuse.

Different groups do not share in either granularity: a system prompt common to all groups is computed and stored once per group, or per sample.

### How a trajectory finds its objects

| Option | Assessment | Used by |
| --- | --- | --- |
| Every response lists keys for all its positions | About 100 bytes per block per turn, for every turn of a long trajectory | Earlier session-wise protocol |
| The response lists the objects from the first position the request computed | Positions the request hit come from an earlier response, so a late sample needs a sibling's listing | Intermediate session-wise code |
| The response lists exactly the objects the request wrote, plus cumulative spans | Each object is read whole; the spans say what to send next | Request-wise prototype |
| **The response lists the objects from the client's start on, hit blocks included** | `[key, start, end]` entries; the client states with `aux_prompt_start` what it holds, so no listing depends on another request's | **This protocol** |

Entries are `[key, start, end]`, so the trainer needs neither the block size nor the hash function. `start` is the first position the object holds, and the trainer takes an entry only from where its trajectory ends ([consumer contract](consumer.md#splicing-a-trajectory)).

`aux_prompt_start` is one field for every product, the same in the inline mode of requests without a namespace. The per-product fields it replaces, `routed_experts_prompt_start` and `indexer_topk_prompt_start`, stay as deprecated aliases.

### The tail

A request ends inside a block. Its last positions are not a full block, so they cannot be named by a block hash, and the next turn computes them again because their KV was never cached as a full block.

| Option | Assessment | Used by |
| --- | --- | --- |
| Return the tail's products inline | Products move through the frontend and the client | Neither |
| The tail belongs to its request; the next request recomputes it and does not write it | Needs per-position ownership in the salt | Request-wise prototype |
| **A tail object per request; the next turn writes the whole block; the trainer keeps the tail** | A random name per tail keeps apart samples that stop on the same tokens, retries, replicas and the tails one request writes at preemption and at its stop; the listing carries it | **This protocol** |

The next turn's full object of that block is used only after the tail's positions. Returning the full object for the tail positions would use the second computation, not the first stored copy.

### Which copy a trajectory keeps

| Option | Assessment | Used by |
| --- | --- | --- |
| Overwrite on every recompute | Replaces the products a trajectory used with a later computation's | Neither |
| A request never writes stated positions | The client states what the plan holds; the engine computes those positions if it must but does not write them | Request-wise prototype |
| **Immutable objects, first write kept, earlier listing kept** | A recompute under the same name is a no-op put; the trainer keeps the positions it already holds | **This protocol** |
| Per-trajectory copies when versions differ | Needed only when names carry neither versions nor owners and a trajectory must get its own computation | Neither; trajectory namespaces give each sample its own names instead |

Objects are immutable: a write to an existing key succeeds, and the store keeps one object per key, never a mix ([protocol §9](protocol.md#9-the-store)).

### Release

| Option | Assessment | Used by |
| --- | --- | --- |
| By the keys responses listed | Misses objects no listing names: failed requests, blocks an aborted request's steps in flight completed | Neither |
| Reference counts in the engine | Engine state across requests and replicas for a fact the trainer knows | Neither |
| The engine removes objects of outputs no client received | Needed only when the trainer cannot name every owner | Earlier request-wise protocol |
| Owner prefixes, counted when forks share them | One removal per request; counts with forks | Request-wise prototype |
| **The namespace prefix after a pause** | One removal per namespace, nested namespaces included; it takes listed and unlisted objects alike | **This protocol** |

The pause is a point every RL loop already has at each weight update, and the protocol uses it as the write barrier. Draining the writers costs a worker round trip when nothing is queued and otherwise the time to write what is queued. How long objects live follows from the granularity: until a group's slowest sample is consumed, or until each sample is.

### Write failures

| Option | Assessment | Used by |
| --- | --- | --- |
| Stop the engine on a failed put | A store outage stops generation | Earlier code before writer degradation |
| Evict the prefix-cache blocks whose object was dropped | Later requests would compute the block again and write it, but the engine would need to map every dropped key back to cache blocks, across every KV tier | Neither |
| Mark the response | The response may already have been sent when the put fails | Neither |
| **Retry, then drop; keep the key listed; the trainer masks** | The engine keeps running; a lost object masks the samples that need it, found missing after the write barrier | **This protocol** |

A write failure is between the engine and its store ([protocol §10](protocol.md#10-failures)). The engine does not evict prefix-cache blocks whose object was dropped, so a later request that hits that KV lists the same lost key.

### Store faults

| Option | Assessment | Used by |
| --- | --- | --- |
| Mask by read misses | Catches objects that stay missing, but not an object lost with a holder and written again by another computation of the same tokens: the read then succeeds with other products. Reads against a dead holder or master also wait out their bounds | Earlier protocols |
| Fence each key by store generation | The store would refuse to recreate a key lost in an earlier generation, so a recomputation could not replace it unnoticed. Needs per-key generations in the store | Neither |
| **Drop every namespace open during a fault, detected by a store epoch** | Coarse: a fault also drops samples whose objects survived. Needs one epoch per store, compared at namespace open and before consumption | **This protocol** |

A store fault loses an unknown set of objects, so the protocol drops the samples in flight instead of tracking objects one by one ([protocol §10](protocol.md#10-failures)).

### Replicas

Routing all requests of a namespace to one engine replica is recommended, not required. Replicas update their weights at different times and do not share their prefix caches. Under a group namespace, two samples on two replicas could compute one name at different weight versions, and the store would keep one copy for both. Under a trajectory namespace, moving a sample to another replica only computes its prefix again: the store keeps the copy the sample already used. Data-parallel ranks update their weights together but keep separate prefix caches, so they count as one replica for versions and as several for hits.

### `n > 1`

Requests that name a namespace reject `n > 1`: one listing per request keeps the response simple. A group is sent as separate requests, in either granularity.

### Object granularity

- **One object per full-attention KV block.** Hits end on block boundaries, so a hit always covers whole objects. All enabled outputs go into one object, so the object count does not grow with the number of outputs. With pipeline parallelism the last stage writes each object, so the count does not grow with the stages either.
- **The tail as its own object.** With the tail in the store the response carries keys only.

### Store

- **Mooncake, standalone store mode.** Objects outlive the writing and reading clients and stay until they are released ([scenarios §4](scenarios.md#4-object-pool)).
- **SSD-first, objects not pinned (default).** Memory holds recent objects; the SSD tier, with disk eviction off, holds all of them, and new namespaces are admitted by free disk. Removal does not free disk on Mooncake 0.3.14, so the SSD is sized by the bytes written in the job. Offload on eviction kept every object; offload at put lost 3% silently ([experiments](experiments.md#mooncake-behavior)).
- **Memory only.** Hard-pinned objects need room for every object written between two releases ([scenarios §4.4](scenarios.md#44-memory-only-store)).
- **Separate masters.** A KV store that is reset with `remove_all(force=True)` would also remove objects on a shared master.
- **Pluggable backends.** What any backend must provide is in [protocol §9](protocol.md#9-the-store). A local branch adds an `AuxObjectStore` interface (`put_batch`, `get`, `remove` with a status per key, `remove_prefix`, `free_bytes`) and a filesystem backend; it is not part of this branch.

### Models with sliding-window layers

On DeepSeek-V4-class models the sliding-window layers can run with prefix replay: on a hit, the window is recomputed over each request's trailing positions, and their products are then exact only near each step's end. Delivering exact products at every position under replay cost about 41% of throughput at the top scenario with an earlier implementation ([experiments](experiments.md#engine-findings-that-still-apply)). The protocol therefore refuses prefix replay.

### Fields

- **Routed experts**: logical expert ids per MoE layer.
- **Indexer top-k**: `int32` selections per indexer source layer, exactly as captured.
- **Hidden states**: what the forward pass returns. Logprobs recomputed from them had an error of at most 7.8e-4 with an earlier implementation; the behavior logprob still comes from the inline sampled logprob.
- **Top-k logprobs**: at every position, before any sampling parameter, so that a teacher can score long multi-turn prompts incrementally and so that requests with different sampling parameters that share a name share one meaning. Returning top-k through vLLM's per-token logprobs frontend instead made rollouts 46.8% longer at k = 128 with an earlier implementation.
- **Without routed experts.** With any enabled output, objects carry only the enabled fields, so a dense teacher can deliver top-k logprobs.

## Why the protocol delivers the first stored copy

- **A hit lists the object of the computation that produced the KV.** KV under a namespace's block hash was computed by a request of the namespace for the same tokens, which wrote the object under the same name; the hitting request lists that name.
- **A recompute keeps the first copy.** A block computed again gets the same name; the put is a no-op and the listing names the first object.
- **The tail keeps the first copy.** The trainer keeps a turn's tail for its positions and takes the next turn's full object only after them.
- **A rewritten history gets new names.** The client cuts `aux_prompt_start` at the first changed token; the changed tokens hash to new blocks, which the next request computes and lists.
- **What it relies on.** Under group granularity, no two samples computing the same full block at different weight versions, which routing a group to one replica avoids across replicas (or the optional header to detect it).

## Not adopted

- **R3 as a KV cache layer.** Routed experts would ride along with the KV through every tier. Every connector would have to move the extra layer, it cannot hold the indexer or hidden states at reasonable cost, and the trainer still needs a separate delivery path.
- **Storing products by KV slot.** Products keyed by the GPU slot of the KV are lost when KV moves to a tier or another replica, or when GPU memory is released at an update.
- **Manifests and per-trajectory copies.** With the namespace in the names, a name identifies the computation, and a listing replaces the manifest.
- **The weight version in the salt.** Replaced by `weight_versions` for sampling provenance and the optional object header for the birth version, for the reasons above.
- **A separate request-wise protocol.** Per-position owners from client spans; trajectory namespaces give the same isolation without spans, reference counts or the exact-prefix rule.
- **Taking the namespace out of the salt to share prompts across groups automatically.** A request could then hit KV of another namespace and list keys that namespace releases on its own schedule.

## Group and trajectory namespaces

The protocol shares KV and objects among the requests of one namespace and among nothing else, and leaves the namespace to the client. Two choices cover GRPO training:

- **Group.** `aux_namespace` is the group id. All requests of a group share.
- **Trajectory.** `aux_namespace` is `<group>/<sample>`. Each sample shares only with itself.

Both use the same engine, names, writer, store, splicing rule and release call. What the choice changes:

### Mechanisms

| | Group | Trajectory |
| --- | --- | --- |
| `aux_namespace` | `<group>` | `<group>/<sample>` |
| Shares KV and objects | every request of the group: its samples, their later turns, sub-agents and retries | the requests of one sample: its later turns and retries, and forks the client keeps in it |
| The group's prompt | stored once; computed once when later samples start after its blocks are cached; samples sent at once to different replicas or ranks, or started during another's prefill, each compute it and list the first copy stored | computed and stored once per sample |
| Listing | the blocks a sample hit are listed by the names another sample wrote | every listed object was written by a request of the sample |
| Release | one `remove_prefix` per group, once all its samples are settled and a pause has completed | one `remove_prefix` per sample, once it is settled and a pause has completed; or one per group |
| Recommended replica routing | all samples of a group to one replica | each sample to one replica; a group's samples can be spread over replicas |
| Admission charge | what the group's responses listed, each key once, plus the worst case of its requests in flight | the same, per sample |

The [two-sample example](protocol.md#example-two-samples-of-one-prompt) shows both: under group granularity the second sample lists the first sample's two prompt objects and adds its own tail, 4 objects in all; under trajectory granularity each sample writes three, 6 in all.

### Compute and storage

A group of `n` samples with a shared prompt of `P` positions:

| | Group | Trajectory |
| --- | --- | --- |
| Prompt positions computed | `P` when one sample goes first and the others hit the cached prompt; up to `n × P` for samples sent at once ([protocol §2](protocol.md#samples-sent-at-once)) | `n × P` |
| Prompt positions stored | `P` | `n × P` |
| 8 samples, 200K-position prompt, DeepSeek-V4.1-Flash, R3 only | 96 MB | 768 MB |
| The same, R3 + top-128 logprobs + hidden states | 2.35 GB | 18.8 GB |

Bytes per position are in [scenarios §2](scenarios.md#2-bytes-per-position-and-per-trajectory). Later turns, forks and retries store the same under both granularities as long as they stay in their sample's namespace.

**Throughput.** Both sharings were measured against products off, on DeepSeek-V4.1-Flash without batch invariance, CUDA graphs on, the base scenario of the validation matrix (weight updates that keep the prefix cache), with the same workload: four runs on one node, products on, off, off, on. Neither ran on the code of this protocol; each ran on earlier code with the same sharing.

| | Group | Trajectory |
| --- | --- | --- |
| Code | earlier session-wise code (`da0bd138db`), one GRPO group per session | request-wise prototype (`fa0f6430d5`), every trajectory salted with its own owner |
| Generated tokens per second, on / off (means of two runs) | 633.9 / 639.9 | 650.2 / 732.4 |
| Products on against off | −0.9%; the two off runs lie within ±2.8% of their mean | −11.2%; the two off runs lie within ±3.9% of their mean |
| Where it goes | within noise | lost prefix hits: hit ratio 86.3% and 86.8% on, against 90.4% and 89.3% off; 14% more positions computed per second (25,847 against 22,684) |

The trajectory cost is the prompt each sample computes again: with products off, samples of a group share the prompt's KV through the ordinary prefix cache, which per-trajectory salts prevent. The two columns ran on different nodes, so only the ratios compare.

### Exactness

Both give every position its first stored copy within a namespace. They differ in whose computation that copy can be.

| | Group | Trajectory |
| --- | --- | --- |
| A sample that hits the cached prompt | gets the products of the computation whose KV it attended to | not applicable: no sample hits another's KV |
| Samples sent at once on the same prompt | those that compute it list the first copy stored: `n − 1` of `n` when all compute it. Under batch invariance the copies agree; without it they can differ numerically. Sending one sample first avoids it | each lists its own |
| Two samples compute the same block at different weight versions | they share the first copy: a sample that computes the prompt again after a prefix-cache reset at an update, or two samples that generate the same full block before and after an update, list the earlier version's products for positions they computed themselves | cannot happen across samples |
| Within one sample | a block computed again after its KV was lost, and the last prompt position of a retry or a regeneration whose prompt ends on a block boundary, keep the first copy; the tail rule keeps a turn's tail | the same: trajectory granularity removes only the cross-sample cases |

How often the cross-version case occurs under group granularity is not measured yet; the validation harness of this protocol counts positions listed from another version's computation. The proposed version in the object header ([consumer contract](consumer.md#fields-and-versions)) would detect it at read time.

### Release and lifetime

| | Group | Trajectory |
| --- | --- | --- |
| When objects can go | after every sample of the group is settled | after each sample is settled |
| Removal calls | one per group | one per sample, or one per group |
| What the trainer tracks | per group: open samples, requests in flight, the last barrier | the same, per sample |
| Unlisted objects (failed requests, discarded retries, blocks an aborted request's steps in flight completed) | removed with the group | removed with the sample |

Under group granularity a long-running sample holds the prompt and the finished samples of its group in the store until it is consumed; under trajectory granularity each sample's objects leave when it does, but the group's prompt is stored `n` times while they live.

### Faults

Write failures, store faults, the store contract and the reader rules are the same for both ([protocol §10](protocol.md#10-failures)): the writer retries until the stall deadline and then drops, the engine keeps running, the trainer masks every sample that needs an unreadable object or whose listings leave a position uncovered and skips a step whose samples are all masked, and a store fault drops the samples not yet consumed of every namespace open during it.

| | Group | Trajectory |
| --- | --- | --- |
| A lost prompt object | masks every sample of the group that lists it | masks one sample |
| A store fault | drops the unconsumed samples of every group open at the time | drops every unconsumed sample open at the time |
| A lost replica | the unfinished samples of the groups it served are dropped | the unfinished samples it served are dropped |
| A crashed coordinator | the job restarts under a new key prefix | the same |

### Evidence

| | Group | Trajectory |
| --- | --- | --- |
| Code of this protocol | engine and trainer unit tests pass in container images, apart from environment failures the engine's base shares; no end-to-end run | the same code, the same status |
| Correctness on earlier code with the same sharing | one GRPO group per session, earlier session-wise code: 0 mismatches against the first stored copy over 31.76M positions, 165 trajectories, 0 leaks, 0 evictions, planted faults caught | per-trajectory owner salts, request-wise prototype: 0 mismatches over 49.65M positions, 249 trajectories, 0 leaks, 0 evictions |
| Overhead against products off | −0.9%, within noise | −11.2% |
| Validation matrix M0–M6 on this protocol | pending | pending |

Details, and the claims that are still weak, are in [experiments](experiments.md).

### Open items

- Run the validation matrix on this protocol with both granularities: correctness against the first stored copy, the count of positions listed from another version's computation, leaks and evictions, and overhead against products off in alternating runs on one node.
- Decide whether to add the version header and the counter of positions computed again without being written.
- The store fault epoch on Mooncake 0.3.14, which reports none: a hard-pinned sentinel object per holder, probed with `batch_probe_key`, is in progress ([protocol §9](protocol.md#9-the-store)).
- Forks under trajectory granularity: whether frameworks keep a fork in its parent's namespace (shares the parent's KV) or nest it (computes the shared prefix again).
- The [Mooncake 0.3.14 issues](protocol.md#10-failures) apply to both: removed objects registered again, disk not reclaimed on removal, a dead holder's segments kept for `client_ttl`, metadata lost when the master restarts.

### Questions for reviewers

- **Default granularity.** Group granularity computes and stores a group's prompt once and cost nothing measurable on earlier code, but lets samples share a block computed at another weight version. Trajectory granularity gives every sample its own computation and releases it early, at 11% of throughput in the measured workload and `n` prompt copies. Which should RL frameworks use by default, and is the cross-version sharing acceptable for routing replay at your staleness?
- **Release granularity.** Does it matter at your scale that a group's objects live until its slowest sample is consumed, with hidden states on?
- **Versions.** Is the sampling version of the next token, from `weight_versions`, enough for staleness weighting, or should objects carry a per-position version header?
- **The object-store boundary.** A local branch makes the store pluggable ([protocol §10](protocol.md#10-failures)): `put_batch`, `get`, `remove`, `remove_prefix` and `free_bytes`. Is this the right interface, and which backends matter?
- **Cross-cluster writing.** At frontier scale a node writes 1.8 MB/s with R3 only and 46 MB/s with all fields; across clusters the limit is the total bandwidth between them ([scenarios §10](scenarios.md#10-writing-across-clusters)). Does it matter to you, and which fields would cross?
- **Capacity defaults.** One position of DeepSeek V4.1 Flash takes 11,744 bytes with R3, top-128 logprobs and hidden states, 480 with R3 only. Which outputs should be on by default?
