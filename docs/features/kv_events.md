# KV Cache Events

vLLM can publish KV cache events: which prefix-cache block hashes a server holds, and in which tier. Cache-aware routers and external KV cache indexes subscribe to these events to route requests to servers that already hold their prefix.

This page is the contract between vLLM and those consumers. It covers the transport, the encoding, when each event fires, and which changes stay compatible.

## Enabling events

Pass `--kv-events-config` to `vllm serve`. Events describe the prefix cache, so prefix caching must stay enabled (the default).

```bash
vllm serve Qwen/Qwen3-0.6B \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557", "replay_endpoint": "tcp://*:5558", "topic": "kv-events"}'
```

See [KVEventsConfig][vllm.config.kv_events.KVEventsConfig] for every field.

Each data-parallel (DP) rank runs its own publisher:

- For `tcp` endpoints, the rank is added to the port of both `endpoint` and `replay_endpoint`. With the ports above, rank 1 would publish on 5558, the replay port of rank 0, so space the two ports at least the DP size apart.
- For `inproc` endpoints, `_dp<rank>` is appended.
- Other schemes support a single rank.
- `tcp://*:0` binds the node IP (set `VLLM_HOST_IP` if it cannot be detected) on a port chosen by the OS.

`GET /kv_event_sources` returns each rank's resolved publisher configuration, keyed by DP rank. The Rust frontend reports the same sources through the gRPC `GetKvEventSources` call, with `encoding` `msgpack` and `schema_version` 1.

## Transport

The publisher sends one message per batch on a ZeroMQ PUB socket:

| Frame | Content |
| --- | --- |
| 0 | `topic`, UTF-8. Empty by default. Subscribers filter on it by prefix. |
| 1 | Sequence number: 8-byte big-endian unsigned integer. |
| 2 | Payload: one msgpack-encoded `KVEventBatch`. |

The sequence number starts at 0 when the publisher starts and increases by one per batch. It resets when vLLM restarts, so it identifies a batch only within one publisher lifetime.

The scheduler publishes at most one batch per step, after processing the step's output, and only when there are events.

### Delivery

ZeroMQ PUB/SUB does not guarantee delivery:

- The PUB socket drops batches for a subscriber that falls `hwm` batches behind.
- A subscriber receives nothing published before its subscription took effect.

Subscribers detect loss as a gap in sequence numbers.

### Replay

If `replay_endpoint` is set, a ZeroMQ ROUTER socket serves the most recent `buffer_steps` batches:

- **Request.** Send an empty delimiter frame followed by the first wanted sequence number, as an 8-byte big-endian unsigned integer. Use a DEALER socket: a REQ socket accepts only the first message of the reply.
- **Reply.** The publisher sends the retained batches with sequence numbers at or above the requested one, oldest first. Each uses the same three frames as live messages.
- **End marker.** The reply ends with an empty topic frame, a sequence frame holding -1 as an 8-byte big-endian *signed* integer (`0xFFFFFFFFFFFFFFFF`), and an empty payload.

A gap older than the retained window cannot be recovered from replay.

## Encoding

Payloads are msgpack.

### Batch

`KVEventBatch` is a positional array:

| Position | Field | Type | Meaning |
| --- | --- | --- | --- |
| 0 | `ts` | float | Wall-clock time, in seconds since the epoch, when the batch was assembled. It is not the time each event happened. |
| 1 | `events` | array | Events in emission order. |
| 2 | `data_parallel_rank` | int | DP rank of the publisher. |

### Events

Each event is a map. Its `type` key holds the event name: `BlockStored`, `BlockRemoved` or `AllBlocksCleared`.

A field marked *always* is present on every event of that type and may be nil. A field marked *when set* is omitted when it has no value.

`BlockStored` announces one or more consecutive blocks of a prefix:

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `block_hashes` | array of hash | always | Hashes of the stored blocks, in prefix order. Can be empty; see [Hybrid models](#hybrid-models). |
| `parent_block_hash` | hash or nil | always | Hash of the block that precedes the first token in `token_ids`. Nil at the start of a sequence. |
| `token_ids` | array of int | always | Tokens covered by the event, starting right after the parent. Empty for [hash-only stores](#hash-only-stores). |
| `block_size` | int | always | Tokens per hash in this event. Can be 0 when `token_ids` is empty. |
| `lora_id` | int or nil | always | Deprecated: use `lora_name`. |
| `medium` | string or nil | always | Tier that holds this copy: `GPU`, `CPU` or `STORAGE`. See [Event sources](#event-sources) for exceptions. |
| `lora_name` | string or nil | always | Name of the LoRA adapter, if any. |
| `extra_keys` | array | when set | One entry per hash in `block_hashes`, nil for a block without extra keys. See [Block hashes](#block-hashes). |
| `group_idx` | int | when set | Index of the KV cache group. |
| `kv_cache_spec_kind` | string | when set | Attention type of the group: `full_attention`, `mla_attention`, `sliding_window`, `sliding_window_mla`, `mamba`, `chunked_local_attention`, `sink_full_attention`, `encoder_only_attention`, `cross_attention` or `unknown`. |
| `kv_cache_spec_sliding_window` | int | when set | Sliding window of the group, in tokens. |
| `locality` | string | when set | `LOCAL` or `REMOTE`, relative to the publishing instance. |
| `ownership` | string | when set | Identifier of the offloading tier that owns this copy, for tiers that set one. |
| `session_id` | string | when set | Session of the request that triggered the store. It identifies a request context, not exclusive ownership of the block. |

`BlockRemoved` withdraws one copy of each listed hash:

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `block_hashes` | array of hash | always | Hashes of the removed blocks. |
| `medium` | string or nil | always | Tier the copies were removed from. |
| `group_idx` | int | when set | Index of the KV cache group. |
| `locality` | string | when set | As in `BlockStored`. |
| `ownership` | string | when set | As in `BlockStored`. |

`AllBlocksCleared` has no fields.

A *hash* is an unsigned 64-bit integer by default: the last 8 bytes of the block's digest, read big-endian. With `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`, hashes are the full digest as msgpack binary: 32 bytes for SHA-256 and 16 bytes for xxHash.

## Event semantics

### Scope and reference counting

vLLM can hold several physical copies of the same hash. For example, a request can recompute a block that is already cached, and offloading adds copies in other tiers.

Each copy is announced and withdrawn separately. Consumers count references for each publisher, DP rank, hash, `medium`, `group_idx`, `locality` and `ownership`:

- Each hash in `BlockStored` adds one reference.
- Each hash in `BlockRemoved` drops one reference.
- The entry is gone once its count reaches zero.
- A removal for a hash the consumer does not hold is a no-op.

### Stores and removals in the GPU prefix cache

- **`BlockStored`** fires when the scheduler commits a request's blocks to the prefix cache. That happens when the request is scheduled, before the forward pass computes their KV. A store means requests can now hit the hash, not that the forward pass has run.
- **`BlockRemoved`** fires when a cached block's hash leaves the prefix cache:
    - when the freed block is reallocated for new data (freeing a block alone emits nothing);
    - when a partial block is replaced or promoted to a full block;
    - when a failed KV load invalidates the block, unless failed loads are recomputed.

  Each removal withdraws one copy. The hash stays cached while another copy holds it.
- In `incremental` [report mode](#reuse-reports), stores and removals in the GPU prefix cache balance: every announced copy is withdrawn exactly once when it leaves the prefix cache, unless an `AllBlocksCleared` covers it.

### `AllBlocksCleared`

`AllBlocksCleared` fires when the prefix cache is reset, for example after a weight update. A reset succeeds only when no request holds cache blocks.

The event carries no scope, and pools differ in whether they emit one:

- The GPU pool emits one.
- The HiSparse host pool and `SimpleCPUOffloadConnector` emit their own.
- The `OffloadingConnector` CPU tier emits none.

Consumers drop every entry reported by that publisher and DP rank. Later removals for hashes they no longer hold are no-ops.

### Reuse reports

A request selects its report mode with `kv_cache_report_mode` in the request's `vllm_xargs` (offline: `SamplingParams.extra_args`):

- **`incremental`** (default): only new copies are announced.
- **`full`**: a prefix-cache hit also emits a `BlockStored` for the reused blocks, starting at the first block with a nil parent and carrying the request's `session_id`.

A `full`-mode report announces no new copy, and no removal ever matches it. Consumers that count references should treat those stores as hints, or keep requests in `incremental` mode.

### Hybrid models

Models with several KV cache groups report each group separately, tagged with `group_idx`:

- **Hashes are shared across groups.** A hash depends only on the tokens and the extra keys, not on the group, so the same hash appears in several groups.
- **Block sizes can differ.** A group whose block is larger than the hashing granularity reports the hash that ends each of its blocks. `block_size` gives the span of each hash.
- **Skipped blocks.**
    - Sliding-window and Mamba groups skip blocks they do not keep. Their `block_hashes` can list fewer hashes than `token_ids` covers, or none.
    - Such an event adds residency to hashes another store already announced.
    - Its parent may not be cached in that group.

### Hash-only stores

Some connector stores name a hash without its tokens: `token_ids` is empty. Such a store adds a copy in another tier for a hash that an earlier store announced with its tokens.

Consumers that do not know the hash ignore the store. The `OffloadingConnector` emits full payloads instead when `self_describing_kv_events` is enabled; see [KV offloading](kv_offloading_usage.md).

### Event sources

| Source | `medium` | Stores | Removals | Notes |
| --- | --- | --- | --- | --- |
| GPU prefix cache | `GPU` | yes | yes | Sets `session_id`. |
| HiSparse host pool | `CPU` | yes | yes | |
| `SimpleCPUOffloadConnector` | `CPU`, or `STORAGE` with a disk tier | yes | yes | Sets `locality` to `LOCAL`. |
| `OffloadingConnector` CPU tier | `CPU` | yes | yes | Hash-only stores unless `self_describing_kv_events` is enabled. |
| `OffloadingConnector` FS and OBJ tiers | `STORAGE` | yes | no | Sets `locality` when the tier configures it. |
| `OffloadingConnector` KVCR tier | `CPU` or `STORAGE` | yes | yes | Sets `ownership` to `kvcr`. |
| `MooncakeStoreConnector` | `cpu` | yes | no | Leaves `lora_id` and `lora_name` unset. |
| `LMCacheConnectorV1` | set by LMCache | yes | no | LMCache defines the hashes, block size and medium. |
| `FlexKVConnectorV1` | set by FlexKV | set by FlexKV | set by FlexKV | Events come from the FlexKV library. |

### Ordering

- **Within a publisher:** batches arrive in sequence order. Within a batch, GPU prefix-cache events come first, in the order they occurred, followed by connector events.
- **Across sources:** connector events lag the GPU events that caused them. An offload store can arrive after the GPU copy of the same hash was removed.
- **Across publishers:** there is no ordering between DP ranks or between servers.

## Block hashes

Prefix-cache hashes form a chain. vLLM splits the token sequence into units of the hashing block size: the KV cache block size for single-group models, and a common divisor of the group block sizes for hybrid models. Each unit's hash covers the previous unit's hash, the unit's tokens and its extra keys:

```text
hash[i] = H((hash[i-1] or NONE_HASH, tokens[i], extra_keys[i] or None))
```

`H` is set by `--prefix-caching-hash-algo`:

| Algorithm | Serialization | Digest |
| --- | --- | --- |
| `sha256` (default) | Python pickle | SHA-256 |
| `sha256_cbor` | canonical CBOR | SHA-256 |
| `xxhash` | Python pickle | xxHash 128-bit |
| `xxhash_cbor` | canonical CBOR | xxHash 128-bit |

Only the CBOR variants can be reproduced outside Python.

`NONE_HASH` is `H(seed)`:

- If `PYTHONHASHSEED` is set, it is the seed.
- If not, the SHA-256 algorithms use the fixed seed `vllm-none-hash`, and the xxHash algorithms use a random per-process seed, so xxHash hashes differ between instances.

Hash inputs tag each extra key with its source. Events publish them untagged:

| Source | Hash input | `extra_keys` entry |
| --- | --- | --- |
| LoRA | `("lora", name, path)` | `name` |
| Multimodal input | `("mm", identifier, offset)` | `[identifier, offset]` |
| Cache salt, first block only | `("cache_salt", salt)` | `salt` |
| Prompt embeddings | `("prompt_embeds", digest)` | `digest` |

Limits on recomputing hashes from events:

- Integer hashes are truncated, so a consumer that reproduces the chain must recompute it from tokens rather than continue from a published parent.
- The adapter path is not published, so hashes of LoRA blocks cannot be reproduced from events.

## Compatibility

vLLM evolves this format additively:

- New fields on existing events are optional and omitted when unset.
- `KVEventBatch` may gain trailing elements.
- Existing fields keep their name, type and meaning.
- Events keep firing under the conditions on this page.

Removing, renaming or retyping a field, or changing when an event fires, breaks consumers. Such changes must update this page and `tests/v1/core/test_kv_events_contract.py`, which pins the encoding.

Consumers stay compatible by:

- ignoring unknown fields and trailing batch elements;
- skipping events whose `type` they do not recognize.

Decoding a batch into a closed set of event types, such as `msgspec` decoding into `KVEventBatch`, rejects the whole batch when it contains a new event type. Decode events individually to skip unknown ones.
