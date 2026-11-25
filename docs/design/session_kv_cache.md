# Session KV Cache

`--session-cache` is an experimental alternative to [automatic prefix caching](prefix_caching.md).
Instead of naming every KV block by a hash of its content and searching the hash table for
the longest cached prefix, the engine keeps KV in *states* that requests publish, and an
upper-layer *resolver* tells each request which states share its content. The two modes
coexist; one engine instance uses one of them, and prefix caching stays the default.

## Components

```text
 request ──► SessionResolver ──binding──► SessionKVCacheManager (store) ──► block tables
                 ▲                                   │
                 └────────── seq-numbered events ◄───┘
```

- **Resolver** (`vllm/v1/session/resolver.py`): a token radix tree over everything the store
  has published. For each request it returns a `SessionBinding`: up to four candidate
  `(node_id, length)` pairs, longest first, the longest content match (`match_length`), the
  epoch, and the resolver's last applied event `seq`. It also tracks sessions (a sequence of
  turns that extend each other) for bookkeeping; clients may pass `session_id` as a hint but
  never need to.
- **Store** (`vllm/v1/core/session_kv_cache_manager.py`): owns physical KV as a tree of
  *nodes*. A node holds, per KV cache group, the blocks covering its token span
  `[start, end)` on top of its parent. It never hashes or compares tokens: it trusts the
  resolver for content and enforces only ownership and resumability.
- **Protocol** (`vllm/v1/session/protocol.py`): plain, serializable messages so the resolver
  can later move out of the engine process. Store → resolver events: `NodePublished`,
  `NodeExtended` (also seals a hybrid node with its commit points), `NodeTruncated`,
  `RequestFinished`, `StoreReset`.

## Lifecycle of a request

1. **Resolve** (engine `add_request`): the resolver matches the prompt and attaches a binding.
2. **Attach** (`get_computed_blocks`): the store re-resolves if anything was published since
   the binding, then takes the first candidate it can still resume from, falling back to
   the candidate's ancestors. The block table is assembled along the node path; deeper
   nodes win where a child re-covers its parent's partial last block.
3. **Publish** (`allocate_slots`): each scheduled prompt chunk is published in the request's
   own node, which it extends chunk by chunk, so other requests can share the prompt from
   the step that computes it (every attention layer writes a step's KV before reading it).
4. **Finish** (`free`): the request publishes up to its computed tokens and reports the node
   that its session continues from.

## Resume rules

| Model | Where a request can resume | Why |
| --- | --- | --- |
| Attention only | Any block boundary inside the matched content | Recomputing the rest of the block is cheaper than copying it. |
| Hybrid (Mamba / GDN) or sliding window | Exactly at a node's commit point | A recurrent state exists only for the position it was saved at. The boundary attention block and the state are copied (copy-on-write) so nodes never change after sealing. |
| With EAGLE-style drafting (attention only) | The drafter's lookahead (`num_hash_lookahead_tokens`, 0 for DSpark) before the shared content ends | Draft KV at a position also reflects the next tokens. A mismatch would only lower draft acceptance. |

Hybrid nodes are published as *content-only* while the request runs (matchable, not
resumable) and become resumable when *sealed* with a recurrent state at their end:

- where the request stops (its own state), or
- at a **divergence checkpoint**: when the resolver reports that a request shares at least
  an attention block (at most 256 tokens) more content with some state than it can resume from, the store returns that
  position as `shared_prefix_boundary`, the scheduler ends a prefill chunk there, and on the
  next step the store copies the state into a new node before the request advances it in
  place. Conversations that share a system prompt therefore resume after the second one.
- Under speculative decoding the state after the last verified token lives in a
  speculative block chosen by acceptance, so a hybrid request saves a copy where its prefill
  ends instead of where it stops.

A checkpoint is not resumable in the step that fills it: the worker applies a step's block
copies as one gather-then-scatter, so chained copies in one step would read stale data.

## State identity

A node id is `NodeId(origin, local)`: `origin` is drawn at random whenever a store starts
or resets and `local` counts nodes within it. Nodes only grow at their end or shrink, so a
node id with a length always names the same tokens, and an id is never reused by another
engine, after a restart, or after a reset. A reference that went stale (the state was
evicted, or belongs to another engine) can therefore only fail to resolve; it can never
resolve to different content. This is what lets bindings, events and future KV connectors
or routers refer to states across processes without agreeing on content hashes. Events and
bindings are plain data and round-trip through msgpack.

## Invariants

1. A request reads only KV computed from its own token prefix (resolver matches content; the
   store resumes only at block boundaries or commit points of nodes on the candidate path).
2. A sealed node never changes; only its own running request extends an unsealed node, and
   only while it is still writing that node's last block.
3. Every block a node holds carries one reference for that node; a block is free exactly when
   neither a node nor a request holds it.
4. The resolver mirrors the store: it applies every event in `seq` order, and a binding older
   than the store's last publication is re-resolved before use; a binding from an older
   epoch (after a cache reset) is ignored.
5. Eviction frees only blocks of idle states, and only when that lets the allocation fit.

## Eviction

When an allocation does not fit, the store counts every block that only idle states hold,
except below where a running or attaching request reads through an ancestor. If that covers
the deficit it trims the least recently used idle nodes from their tails, block by block,
like prefix caching's block LRU (nodes of hybrid or multi-group models are removed whole). A waiting request's match
and a preempted request's chunks are not protected, again matching prefix caching.

## Releasing ended sessions

A caller that knows a session has ended (an RL trajectory, a deleted conversation) can free
it at once instead of waiting for eviction: `EngineCore.release_sessions(session_ids)`
takes the client `session_id`s the requests carried. The resolver returns every state those
sessions published and forgets the sessions; the store frees each state nothing else uses,
cuts states that other sessions' states hang below down to what those still read (a prompt
shared by a group of rollouts), and frees the rest once their last user stops. The public
API (`LLM`, OpenAI server) is not defined yet. Requests without a `session_id` are freed
only by eviction.

## Group kinds

| Group | What a node holds |
| --- | --- |
| Full attention, MLA, indexer caches | Blocks by token position. Compressed caches (`tokens_per_state > 1`) complete a state only at multiples of the ratio; unless a ring buffer holds the open group, hybrid commit points are multiples of every ratio. |
| Sliding window | References to the window's blocks before each commit point (null elsewhere). Like Mamba state, a window moves on as a request computes, so models with sliding windows resume at commit points; unlike Mamba it is written by position, so no copy is needed. |
| Mamba / GDN / KDA | One state block per commit point. |
| Ring buffers (the open group of a compressed cache) | In models that resume at commit points: a copy of the ring at each commit point, like Mamba state, so commit points need no compression alignment. In attention-only models: nothing, since block-aligned resumes find the ring empty. |
| Windows rebuilt by bounded replay | Nothing: per-request scratch, as with prefix caching. |

## Not supported yet

KV connectors (P/D, offloading, HiSparse), KV cache events,
LoRA, multimodal inputs, prompt embeddings, streaming input, pipeline parallelism, and
encoder-decoder models. Unsupported configurations fail at startup; unsupported requests are rejected.
