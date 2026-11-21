# Aux Outputs for RL

An RL trainer recomputes the logprobs of the tokens an inference engine sampled. Some values the engine computed on the way cannot be recomputed by the trainer, or only at a high cost, and have to be shipped with the rollout:

- the experts each MoE layer routed every position to (R3, routing replay);
- the selections of a sparse-attention indexer;
- the final hidden states;
- the top-k next-token logprobs, for example of a teacher scoring long prompts.

These are the **products** of a position. This folder specifies how vLLM delivers them for long multi-turn RL rollouts through a shared object store, in one protocol in which the client chooses how widely products are shared, and records the evidence behind it. The user-facing guide is [Routed Experts Return](../../training/routed_experts.md#namespaces-delivering-products-through-a-shared-object-store).

## Terms

- **Position.** Position `p` processes token `p` and produces the logits for token `p + 1`. A sequence of `L` tokens has products for positions `[0, L - 1)`: the last token is never fed through the model.
- **Products.** The values one forward pass computes for one position: one entry per enabled output (routed experts, indexer selections, hidden states, top-k logprobs).
- **Namespace.** The string a request sends in `aux_namespace`. Requests of one namespace share KV and objects by name; requests of different namespaces share nothing. Segments separated by `/` nest a namespace under another for release.
- **Object.** An immutable entry in the object store holding the products of a contiguous span of positions, one array per enabled output. An object never spans a KV block boundary of the full-attention KV cache group.
- **Block.** One KV block of the full-attention group, `B` positions. A full object covers one block; the positions after a request's last full block go into a tail object of that request.
- **First stored copy.** The products the store holds first for a position under a name. It is the reference for correctness: a trajectory gets, for every position, the products of the computation that first stored them, and a later computation of the same position never replaces them.
- **Listing.** The `aux_objects` of a response: a list of `[key, start, end]`, each saying that the products of positions `[start, end)` are in the object `key`.
- **Splicing.** The trainer's assembly of one products array per trajectory from the listings of the requests that produced it.
- **Write barrier.** A completed pause, started after a request ended, of every engine that served a request of its namespace. After it, every object the request lists is in the store or never will be ([protocol §8](protocol.md#8-pause-and-release)).
- **Settled.** A sample is settled once it is consumed (read, and its training step committed), masked, failed, or dropped. A namespace is released once every sample it expects is settled ([protocol §8](protocol.md#8-pause-and-release)).
- **Store fault.** A holder or the master lost or restarted, or a holder dropped after its TTL. It changes the store's fault epoch, and the samples of every namespace open during it are dropped ([protocol §10](protocol.md#10-failures)).

## The protocol in short

1. **The request names a namespace and what it holds.** `aux_namespace`, accepted on `/inference/v1/generate` and by the offline `LLM` API, opts the request in; `aux_prompt_start` says how many leading positions the client already holds products for. `aux_prompt_start` is one field for every product and also sets where inline products start; `routed_experts_prompt_start` and `indexer_topk_prompt_start` are its deprecated aliases.
2. **The namespace is the salt.** The namespace, the key prefix and the captured products are the prefix-cache salt, so KV the engine reuses and the products stored for it always come from the same namespace, and every KV connector keyed by block hashes follows with no connector code. The weight version is not in the salt.
3. **Objects are named by the block hash.** Full blocks are stored as `<prefix>/<namespace>/<block hash>`, written once; a block computed again gets the same name, and the store keeps the first object. The positions after a request's last full block are a tail object with a random name, `<prefix>/<namespace>/tail/<random name>`.
4. **The response lists objects, not products.** `aux_objects` covers every position from `aux_prompt_start` to the end, the blocks the request hit included, and names a preempted request's tails before the objects that computed their positions again; `weight_versions` gives the version that sampled each token, not the version of the listed products.
5. **The trainer splices with the earlier copy winning,** masks a sample whose objects cannot be read or leave a position uncovered, and removes a namespace with one prefix removal once its samples are settled and a pause has completed.
6. **Failures stay out of the engine.** A failed write is retried, then dropped; the key stays listed, and the trainer masks the samples that need it. A store fault during a namespace's lifetime drops all its samples not yet consumed; the trainer detects it by the store's fault epoch, compared at namespace open and before consumption, not by read misses.

The client chooses the granularity of sharing by what it puts in the namespace:

- **Group.** The GRPO group id. A group's prompt is stored once. Samples that start after it is cached hit it; samples sent at once can each compute it and then share the first stored copy, so send one sample first to compute it once. The group is released in one removal once all its samples are settled. Two samples that compute the same block at different weight versions share the first copy.
- **Trajectory.** The group id and the sample, `<group>/<sample>`. Every sample computes and stores its own prompt and is released as soon as it is settled; no sample uses another sample's products, though within a sample a block computed again still keeps its first copy. Removing the group's prefix releases all its samples at once.

[Protocol §2](protocol.md#2-granularity-what-a-namespace-is) shows both on a two-sample example; the [design space](design_space.md#group-and-trajectory-namespaces) puts their costs and evidence side by side.

## Requirements

- Prefix caching on, a plain full-attention KV cache group, hits on KV block boundaries, at most one hash look-ahead token, no prefix replay. The engine refuses to start otherwise.
- A new `object_key_prefix` whenever the engines start, including a restart of the same job; replicas started together share it ([protocol §11](protocol.md#11-deployment)).
- Replicas that share KV stores or the object pool hash identically: the same hash algorithm with `PYTHONHASHSEED` set to one value everywhere or unset everywhere (unset, only a cryptographic algorithm such as `sha256` gives every process the same seed), hash block size and full-attention KV block size (the object size).
- KV connectors store the blocks that decoding completes (for `MooncakeStoreConnector`, `save_decode_cache: true`).
- The object pool does not share a Mooncake master or pool with a KV store; it keeps every object until it is released (SSD offload on eviction with disk eviction off, or hard pin).

Recommended, not required: route all requests of a namespace to one replica, which keeps the hits and, under group granularity, keeps two replicas from computing one block of a group at different weight versions. Batch invariance is not required either: correctness is defined by the first stored copy.

## Documents

| Document | Contents |
| --- | --- |
| [Protocol](protocol.md) | Granularity with a two-sample example; request and response, names, objects and writes, hits and replicas; turns, `aux_prompt_start` and splicing with a two-turn example; pause and release; what the store must provide; failures; deployment; limits |
| [Consumer contract](consumer.md) | What the trainer sends, keeps, splices, reads, masks and releases, with an example |
| [Design space](design_space.md) | The alternatives for each decision, why the weight version left the salt, and group and trajectory namespaces side by side |
| [Scenarios](scenarios.md) | What each granularity shares and stores; bytes per position, object pool, object counts and write rates, reads, memory or SSD, writing across clusters |
| [Experiments](experiments.md) | Measured results, each labelled with the code it ran on; the pending validation matrix of this protocol; known weak claims |

## Status

- **This change** delivers routed experts (R3) through objects end to end, under group and trajectory namespaces: `aux_prompt_start`, `aux_namespace`, `aux_objects` on `/inference/v1/generate` and the offline `LLM` API, the object store and its writer, the write barrier at a pause, and the release patterns. While routed experts are the only product, the salt carries no set of captured products.
- **Follow-ups.** Indexer selections, hidden states and top-k logprobs as object fields, with the captured products in the salt; routed experts under pipeline parallelism; `aux_prompt_start` on the OpenAI routes and the Rust frontend.
- **Evidence.** [Experiments](experiments.md) labels every result with the code it ran on.
- **Store fault epoch.** Mooncake 0.3.14 reports none; a trainer can derive one from a hard-pinned sentinel object on every holder.
- **Requests without a namespace** keep the inline behavior: routed experts are returned in the response, from `aux_prompt_start` on.

RFC #64380 described an earlier protocol (manifests, per-trajectory copies, a writer-side version ledger). This folder replaces it.
