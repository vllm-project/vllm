# Design Space

The [protocol](protocol.md) is one point in this space. This page lists each decision, the alternatives, and why the protocol picks what it picks. Sizes are in [scenarios](scenarios.md), measurements in [experiments](experiments.md).

## What the trainer needs

For every position of a trajectory the trainer needs the aux outputs of the computation the trajectory used: the forward that produced the KV later tokens attended to, or that sampled the next token. Three facts shape the design:

- **Positions are computed more than once.** The next turn computes again what the cache did not keep; a preempted request recomputes; siblings compute the same prompt.
- **Trajectories span weight updates.** At frontier scale a trajectory lives through several updates and may be up to ~100 versions stale ([scenarios §1](scenarios.md#1-scenarios)). A prefix computed with one version is reused by later turns computed with another.
- **KV leaves the GPU.** RL frameworks often release GPU memory at every update, so the KV of a session's prefix survives only in KV connector tiers, on this replica or another.

So the question is: when a request reuses KV, how does it find the aux outputs of the computation that produced that KV, whatever tier or replica the KV came from; and when it computes a position, how is that computation named?

## Where the weight version lives

| Option | Approach | Assessment |
| --- | --- | --- |
| V1 | Version in the object key (`<hash>@<version>`), stamped when the step that completes the block runs | The key does not follow the KV: the KV cache does not know versions, so a hit on KV computed with one version lists the key of whichever version the bookkeeping recorded. Mixed-version blocks and siblings across an update need extra rules. Built and replaced. |
| V2 | Version ledger next to the KV: per block and per KV cache group, in the scheduler and at the offloading connector's boundary | Correct for the GPU cache and one tier layout of one connector, but needs code in every connector and cuts hits wherever groups disagree. Built, reviewed and replaced: it could not cover every connector without wrapping each one. |
| V3 | **Version in the block hash**, through per-segment cache salts | **Chosen.** Every connector keyed by block hashes stores and finds KV by (tokens, versions) with no connector code. The object name is the block hash, so the KV and the object share one name by construction. |

Per-segment cache salts are generic: a request carries `(start, salt)` pairs and each hash block includes the pairs overlapping it, with starts relative to the block. Requests without segments hash as before, so the change is invisible outside sessions.

## Who states the versions of a prompt

| Option | Assessment |
| --- | --- |
| The engine, from its own record of which version computed each block | Needs the record on every replica and in every tier: back to V2. |
| The engine, always the current version | A prompt computed before an update would never be hit after it: every turn after an update computes its whole context again. |
| **The client, from the previous response** | **Chosen.** The engine returns `aux_version_segments` with each response; the client sends them with the next request. The engine keeps no state across requests, and a session can move between replicas. |

The client's statement decides only what the request can hit. Everything the request computes is named by the engine with the version that computes it (next section). Stale or edited segments are therefore safe: positions after the first divergence miss the cache and take the current version. An earlier plan to verify the segments against a hash in their closing entry is no longer needed.

## When names are fixed

A block is hashed as soon as its tokens exist, and connectors keep or look up the hashes they see. Hashes are therefore changed only at points where no connector holds them for positions the request is about to compute:

- **Admission.** The client's coverage is rounded down to an object boundary before the prompt's last token, so a covered object is never partly covered. Later positions get the current version.
- **Update.** A running request is cut back to its computed positions and hashed again with the new version; a waiting one gets the new version from its first unhashed position. No step has fully computed the blocks whose hashes are dropped, so no connector has stored under them.
- **Scheduling from the waiting state.** Just before a waiting request is scheduled (at admission after the prefix-cache lookup, after a preemption, after an asynchronous KV load), every position it will compute is salted with the current version, from the first segment that states another version. Hashes before the hit stay, so everything taken from the lookup remains valid; connectors that derive state from the hashes see the change at their next update.

This rule replaced two narrower ones. Cutting the hashes back at preemption ran before the lookup and did not cover a request whose prefix had been evicted, tiers that were cleared, or KV loaded while the version changed. Salting at the lookup instead covers every case where a waiting request computes a position, with one check per scheduling and a rehash only when a stated version is not current.

Rewriting a running request's hashes at an update is safe for the same reason: the engine is paused with no step in flight, so every dropped hash is that of a block no step has fully computed or stored. What remains is a synchronous KV load that fails inside a step: the step has already written objects from the invalid KV, so it is not supported for session requests (protocol §9).

## Session scope

| Option | Sharing | Release |
| --- | --- | --- |
| Global block hash | across sessions | needs reference counts across sessions |
| **Session and key prefix in the cache salt and in the key** | **within a session** | **delete the session's objects** |

The session is chosen. A shared system prompt is computed once per session, a small cost against reference counting across sessions. With the session in the salt, the session's KV in any tier was computed by a request of the session, which wrote the object under the same name: no ledger of what a session published is needed. The key prefix is in the salt too, so a session name reused under a new prefix (a new deployment start) cannot hit KV whose objects belong to the old prefix.

## Object granularity

- **One object per full-attention KV block.** Session hits end on KV block boundaries, so a hit always covers whole objects. All enabled outputs go into one object, so the object count does not grow with the number of outputs. With pipeline parallelism the stages pass the routing on with their intermediate tensors and the last stage writes each object, so the count does not grow with the stages either.
- **Tail as its own object.** The positions after the last full object are stored as one object named by a hash of the previous block, the tokens and their extra keys under a tail tag. Tails of different branches, lengths or versions never collide. Returning tail rows inline instead would move aux values through the frontend and the client; with the tail in the store the response carries keys only.

## Write policy

| Option | Assessment |
| --- | --- |
| Overwrite on every recompute | Can replace the rows a trajectory used with a later computation's. |
| **Immutable objects, first write kept** | **Chosen.** Under batch invariance a second computation under the same name has the same rows. A write to an existing key succeeds, and Mooncake keeps one object per key, never a mix. |
| Per-trajectory copies when versions differ | Needed only when names do not carry versions; obsolete with V3. |

**Failures.** A put the store refuses for lack of memory is retried every 0.2 s, since a full store frees memory by offloading to SSD or by releases; other store errors are not retried. Writes pending for `object_put_stall_timeout_s` without a successful put, including a put that hangs inside the store, stop the engine at its next step, and a pause fails with the same error. Readers wait at least as long; a read that still times out checks the engine before treating the objects as lost, since a healthy engine can land objects later behind a deep queue. Error markers per request and readers that tell partial samples apart were part of the earlier protocol and are not carried over: a write that never lands means a broken store or a sizing error, and a stopped engine makes it visible.

## Store

- **Mooncake, standalone store mode.** Objects outlive the writing and reading clients and stay until they are released ([scenarios §4](scenarios.md#4-object-pool)).
- **SSD-first, objects not pinned (default).** Memory holds recent objects; the SSD tier, with disk eviction off, holds all of them, and new sessions are admitted by free disk. Removal does not free disk, so the SSD is sized by the bytes written in the job. Offload on eviction kept every object in the measurements; offload at put lost 3% silently and is not supported ([experiments](experiments.md#mooncake-behavior)). A pinned object is never evicted and so never offloaded, which is why the default does not pin. The engine cannot see the master's settings, so `AuxOutputConfig.object_hard_pin` marks a memory-only store.
- **Memory only.** Hard-pinned objects need a pool for every object written between two releases; admission keeps the unreleased bytes below 0.75 × pool. A full pool that nothing releases stops the engine at the stall deadline: the earlier protocol's fallback of aborting the newest sessions is not carried over.
- **Separate masters.** A KV store that is reset with `remove_all(force=True)` would also remove objects on a shared master.

## Release

| Option | Assessment |
| --- | --- |
| By the keys responses listed | Misses objects no final response lists: blocks a request hashed when it stopped, aborted requests. |
| Reference counts or a per-session ledger in the engine | Engine state across requests and replicas, for a fact the trainer already knows. |
| **By session key pattern, after a pause** | **Chosen.** `aux_session_pattern` matches every object of a session. A completed pause writes every object of the requests that stopped before it, so a release after a pause removes the whole session. |

The pause is a point every RL loop already has at each weight update. Making it drain the writers costs a worker round trip when nothing is queued and otherwise the time to write what is queued. The writer queue bounds the batches held, not the wait: a pause waits while writes progress and fails at the stall deadline when none lands. Alternatives that give the consumer a per-session count of pending writes would need a new engine API and a polling loop; the pause needs neither.

## Weight updates and KV capacity

Recomputing every in-flight context at every update costs about 7% of each update interval at frontier scale ([scenarios §7](scenarios.md#7-recomputing-in-flight-work-at-weight-sync)). Sessions therefore continue across updates on their stored KV. With GPU memory released at each update, that KV must survive in connector tiers, including the blocks that decoding completes. When it does not, the request computes the prefix again under the current version, and writes new objects for the positions computed again from the object that holds the first one whose stated version is not current: the names stay true, and the cost is compute and pool space. KV capacity is a performance requirement of the deployment.

## Models with sliding-window layers

On DeepSeek-V4-class models the sliding-window layers can run with prefix replay: on a hit, the window is recomputed over each request's trailing positions. Aux outputs of those layers are then exact only near each step's end. Delivering exact outputs at every position under replay forces every new position through the replay stages. With an earlier implementation that cost about 41% of throughput at the top scenario, and turning replay off took 58–62% less time on a one-node 14.6K-context workload ([experiments](experiments.md#engine-findings-that-still-apply)). The engine therefore refuses sessions with prefix replay; the window KV is prefix-cached as snapshots instead.

## Fields

- **Routed experts**: logical expert ids per MoE layer, uncompressed.
- **Indexer top-k**: `int32` selections per indexer source layer, exactly as captured. An earlier implementation packed them to 20 bits and compressed them with zstd (4.6–6.7× up to 128K context, ~2.9× at 512K); a codec is a possible follow-up. Indexer capture requires a pipeline-parallel size of 1.
- **Hidden states**: what the forward pass returns. With an earlier implementation, logprobs recomputed from them had an error of at most 7.8e-4; the behavior logprob still comes from the inline sampled logprob.
- **Top-k logprobs**: at every position, so that a teacher can score long multi-turn prompts incrementally; `prompt_logprobs` disables prefix-cache hits, object fields keep them. Computed only for session rows, from the full logits in chunks bounded like `prompt_logprobs` (at most 1024 rows, and at most 64Mi / (k + 1) rows), with headroom left outside the memory profile. Returning top-k through vLLM's per-token logprobs frontend instead made rollouts 46.8% longer at k = 128 with an earlier implementation. A vocab-parallel path that avoids gathering full-vocabulary logits is deferred.
- **Without routed experts.** Hidden states and top-k logprobs need no MoE: with any enabled output, objects carry only the enabled fields, so a dense teacher can deliver top-k logprobs.

## Not adopted

- **R3 as a KV cache layer** (#64383). Routed experts would ride along with the KV through every tier. Every connector would have to move the extra layer, it cannot hold the indexer or hidden states at reasonable cost, and the trainer still needs a separate delivery path.
- **Storing rows by KV slot.** Rows keyed by the GPU slot of the KV are lost when KV moves to a tier or another replica, or when GPU memory is released at an update: exactly the cases this design is for.
- **Manifests, trajectories and per-trajectory copies.** The earlier protocol tracked which computation each trajectory saw and wrote a manifest per request. With versions in the names, a name identifies the computation, and a list of keys replaces the manifest.
- **Partial capture with prefix replay** (delivering only the deep cells sampled tokens depend on). Replaced by running without prefix replay.
- **Counters and warnings for stale names.** Not built; the remaining limits are documented.
