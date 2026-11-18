# KV Cache Events

vLLM can publish KV cache events: which prefix-cache block hashes a server holds, and in which tier. Cache-aware routers and external KV cache indexes subscribe to them to send a request to a server that already holds its prefix.

This page is the contract between vLLM and those consumers: the transport, the encoding, what each event means, and which changes stay compatible.

## Enabling events

Pass `--kv-events-config` to `vllm serve`. Events describe the prefix cache, so prefix caching must stay enabled (the default).

```bash
vllm serve Qwen/Qwen3-0.6B \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557", "replay_endpoint": "tcp://*:5567", "topic": "kv-events"}'
```

See [KVEventsConfig][vllm.config.kv_events.KVEventsConfig] for every field.

The publisher binds an `endpoint` with a wildcard host, and connects to any other `tcp` endpoint. `tcp://*:0` binds the node IP (set `VLLM_HOST_IP` if it cannot be detected) on a port chosen by the OS.

Each data-parallel (DP) rank runs its own publisher:

- For `tcp` endpoints, the rank is added to the port of both `endpoint` and `replay_endpoint`, so keep the two ports at least the DP size apart.
- For `inproc` endpoints, `_dp<rank>` is appended.
- Other schemes support a single rank.

`GET /kv_event_sources` returns each rank's resolved publisher configuration, keyed by DP rank. The Rust frontend reports the same sources through the gRPC `GetKvEventSources` call, with `encoding` `msgpack` and `schema_version` 1.

## Transport

The publisher sends one message per batch on a ZeroMQ PUB socket:

| Frame | Content |
| --- | --- |
| 0 | `topic`, UTF-8. Empty by default. Subscribers filter on it by prefix. |
| 1 | Sequence number: 8-byte big-endian unsigned integer. |
| 2 | Payload: one msgpack-encoded `KVEventBatch`. |

The sequence number starts at 0 when the publisher starts and increases by one per batch. It identifies a batch only within one publisher lifetime.

The scheduler publishes at most one batch per step, after processing the step's output, and only when there are events.

### Delivery

ZeroMQ PUB/SUB does not guarantee delivery:

- The PUB socket drops batches for a subscriber that falls `hwm` batches behind.
- A subscriber receives nothing published before its subscription took effect.

Subscribers detect loss as a gap in sequence numbers. A sequence number that goes backwards means the publisher restarted, so everything the subscriber holds for it is stale.

### Replay

If `replay_endpoint` is set, a ZeroMQ ROUTER socket serves the most recent `buffer_steps` batches:

- **Request.** Send an empty delimiter frame followed by the first wanted sequence number, as an 8-byte big-endian unsigned integer. Use a DEALER socket: a REQ socket accepts only the first message of the reply.
- **Reply.** The publisher sends the retained batches with sequence numbers at or above the requested one, oldest first. Each uses the same three frames as live messages.
- **End marker.** The reply ends with an empty topic frame, a sequence frame holding -1 as an 8-byte big-endian *signed* integer (`0xFFFFFFFFFFFFFFFF`), and an empty payload.

If the first replayed batch is newer than the requested one, the missing batches are gone. The subscriber then drops what it holds for that publisher and rebuilds from the events that follow.

## Encoding

Payloads are msgpack. A decoded batch, shown as JSON:

```json
[
  1767225600.25,
  [
    {
      "type": "BlockStored",
      "block_hashes": [11, 12],
      "parent_block_hash": 10,
      "token_ids": [101, 102, 103, 104],
      "block_size": 2,
      "lora_id": null,
      "medium": "GPU",
      "lora_name": null,
      "group_idx": 0,
      "kv_cache_spec_kind": "full_attention"
    },
    {"type": "BlockRemoved", "block_hashes": [7], "medium": "GPU", "group_idx": 0}
  ],
  0
]
```

### Batch

`KVEventBatch` is a positional array:

| Position | Field | Type | Meaning |
| --- | --- | --- | --- |
| 0 | `ts` | float | Wall-clock time, in seconds since the epoch, when the batch was assembled. Events in the batch can be older. |
| 1 | `events` | array | Events in emission order. |
| 2 | `data_parallel_rank` | int | DP rank of the publisher. |

### Events

Each event is a map. Its `type` key holds the event name: `BlockStored`, `BlockRemoved` or `AllBlocksCleared`.

A field marked *always* is present on every event of that type and may be nil. A field marked *when set* is omitted when it has no value.

`BlockStored` announces one or more blocks of a prefix:

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `block_hashes` | array of hash | always | Hashes of the stored blocks, in prefix order. Can be empty; see [Skipped blocks](#skipped-blocks). |
| `parent_block_hash` | hash or nil | always | Hash of the block that precedes the first token in `token_ids`. Nil at the start of a sequence. |
| `token_ids` | array of int | always | Tokens covered by the event, starting right after the parent. Empty for [hash-only stores](#hash-only-stores). |
| `block_size` | int | always | Tokens per hash in this event. 0 in some hash-only stores. |
| `lora_id` | int or nil | always | Deprecated: use `lora_name`. |
| `medium` | string or nil | always | Tier that holds this copy: `GPU`, `CPU` or `STORAGE`. See [Event sources](#event-sources) for exceptions. |
| `lora_name` | string or nil | always | Name of the LoRA adapter, if any. |
| `extra_keys` | array | when set | One entry per hash in `block_hashes`, nil for a block without extra keys. See [Block hashes](#block-hashes). |
| `group_idx` | int | when set | Index of the KV cache group. |
| `kv_cache_spec_kind` | string | when set | Attention type of the group, such as `full_attention`, `mla_attention`, `sliding_window`, `sliding_window_mla`, `chunked_local_attention`, `sink_full_attention`, `mamba` or `unknown`. |
| `kv_cache_spec_sliding_window` | int | when set | Sliding window of the group, in tokens. |
| `locality` | string | when set | `LOCAL` or `REMOTE`, relative to the publishing instance. |
| `ownership` | string | when set | Identifier of the offloading tier that owns this copy, for tiers that set one. |
| `session_id` | string | when set | Session of the request that triggered the store: its `session_id` field, `X-Session-ID` header or `vllm_xargs.session_id`. Blocks are shared across sessions. |
| `lookahead_token_ids` | array of array of int | when set | One list per hash in `block_hashes`: the tokens after the block that its hash also covers. See [Lookahead hashes](#lookahead-hashes). |

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

### Copies and reference counting

Each hash in a `BlockStored` announces one physical copy that the publisher holds and that a lookup for that hash can hit. A block boundary without a resident copy is not listed. [Reuse reports](#reuse-reports) are the one exception: they repeat copies that were already announced.

vLLM can hold several copies of the same hash. For example, a request can recompute a block that is already cached, and offloading adds copies in other tiers. Each copy is announced and withdrawn separately.

Consumers count references for each publisher, DP rank, hash, `medium`, `group_idx`, `locality` and `ownership`:

- Each hash in `BlockStored` adds one reference.
- Each hash in `BlockRemoved` drops one reference.
- The entry is gone once its count reaches zero.
- A removal for a hash the consumer does not hold is a no-op.

How hashes are grouped into events is unspecified. An emitter can send one event per hash or list several hashes in one event. Consumers apply each listed hash in order.

### Stores and removals in the GPU prefix cache

`BlockStored` fires when the scheduler commits a request's blocks to the prefix cache:

- For tokens the server computes, that is when the request is scheduled. Later requests can hit the hash from then on, while the forward pass that fills the block may still be running.
- For KV loaded asynchronously through a connector, that is when the load completes.

`BlockRemoved` fires when a cached block's hash leaves the prefix cache:

- when a freed block is reallocated for new data (freeing a block alone emits nothing);
- when a [partial block](#partial-blocks) is replaced or promoted to a full block;
- when a failed KV load or a fault-tolerance check invalidates the block. Failed loads that are recomputed keep their blocks.

Every announced copy is withdrawn exactly once when it leaves the prefix cache, unless an `AllBlocksCleared` covers it.

### `AllBlocksCleared`

`AllBlocksCleared` fires when a pool's prefix cache is reset through `reset_prefix_cache`. A reset succeeds only when no request holds cache blocks.

The event carries no scope, so consumers drop every entry reported by that publisher and DP rank. Later removals for hashes they no longer hold are no-ops.

Not every source is reset with the GPU pool, and not every source emits the event; see [Event sources](#event-sources). A tier that keeps its copies through a reset does not announce them again.

### Reuse reports

A request selects its report mode with `kv_cache_report_mode` in the request's `vllm_xargs` (offline: `SamplingParams.extra_args`):

- **`incremental`** (default): only new copies are announced.
- **`full`**: a prefix-cache hit also emits a `BlockStored` for the reused blocks, starting at the first block with a nil parent and carrying the request's `session_id`.

A `full`-mode report announces no new copy, and no removal matches it. Consumers that count references should treat those stores as hints, or keep requests in `incremental` mode.

### Several KV cache groups

Models with several KV cache groups report each group separately, tagged with `group_idx`. A hash depends only on the tokens and the extra keys, so the same hash appears in several groups.

#### Block sizes

Groups can differ in block size. A group whose block is larger than the [hashing block size](#block-hashes) reports the hash that ends each of its blocks. `block_size` gives the span of each hash.

#### Partial blocks

A group whose block spans several hashing units can also cache a hash that ends inside a block, such as the last hash of a prompt. Its `BlockStored` lists that one hash, with `block_size` set to the hashing block size. Its parent is the hash of the previous hashing unit, which no store may have announced. When the block fills, the partial hash is removed and the hash of the full block is stored.

#### Skipped blocks

Sliding-window and Mamba groups do not keep every block. Their `BlockStored` lists only the kept blocks, so `block_hashes` can hold fewer hashes than `token_ids` covers, or none. Such an event does not say which tokens each listed hash covers, and its parent may not be cached in that group.

### Hash-only stores

Some connector stores name a hash without its tokens: `token_ids` is empty. Such a store adds a copy in another tier for a hash that an earlier store announced with its tokens.

Consumers that do not know the hash ignore the store. The `OffloadingConnector` emits full payloads instead when `self_describing_kv_events` is enabled; see [KV offloading](kv_offloading_usage.md).

### Ordering

- **Within a publisher:** batches arrive in sequence order. Within a batch, prefix-cache events come first, in the order they occurred, followed by connector events.
- **Across sources:** connector events lag the GPU events that caused them. An offload store can arrive after the GPU copy of the same hash was removed.
- **Across publishers:** there is no ordering between DP ranks or between servers.

## Event sources

Sources differ in what they emit besides `BlockStored`:

| Source | `medium` | `BlockRemoved` | `AllBlocksCleared` | Notes |
| --- | --- | --- | --- | --- |
| GPU prefix cache | `GPU` | yes | yes | Sets `session_id`. |
| HiSparse host pool | `CPU` | yes | yes | Reset together with the GPU pool, and emits its own event. |
| `SimpleCPUOffloadConnector` | `CPU`, or `STORAGE` in disk mode | yes | when the reset includes the connector | Sets `locality` to `LOCAL`. |
| `OffloadingConnector` CPU tier | `CPU` | yes | no | Hash-only stores unless `self_describing_kv_events` is enabled. |
| `OffloadingConnector` FS and OBJ tiers | `STORAGE` | no | no | Emit only with the tier's `enable_kv_events`. Set `locality` when the tier configures it. |
| `OffloadingConnector` KVCR tier | `CPU` or `STORAGE` | yes | no | Sets `ownership` to `kvcr`. |
| `UMBPStoreConnector` | `CPU` | yes | no | Sets `locality` to `LOCAL`. Leaves `lora_id` and `lora_name` unset. Announces no new store once it tracks `kv_event_max_objects` copies. |
| `MooncakeStoreConnector` | `cpu` | no | no | Leaves `lora_id` and `lora_name` unset. |
| `LMCacheConnectorV1` | set by LMCache | no | no | LMCache defines the hashes, block size and medium. |
| `FlexKVConnectorV1` | set by FlexKV | set by FlexKV | set by FlexKV | Events come from the FlexKV library. |

A source without `BlockRemoved` never withdraws a copy, so consumers need their own expiry for it.

## Block hashes

Prefix-cache hashes form a chain. vLLM splits the token sequence into units of the hashing block size: the KV cache block size for single-group models, and a common divisor of the group block sizes for models with several groups. Each unit's hash covers the previous unit's hash, the unit's tokens and its extra keys:

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

### Lookahead hashes

With EAGLE-style speculative decoding, the drafter's KV at the end of a block depends on the `L` tokens that follow it, so each hash covers those tokens as well. The extra keys in the hash input become `(extra_keys, *lookahead_tokens)`, and `lookahead_token_ids` publishes those tokens for each hash.

Lookahead hashes are on by default for the methods that need them. `disable_lookahead_block_hashes` in `--speculative-config` turns them off, and servers with different settings produce different hashes for the same tokens.

### Recomputing hashes from events

- Integer hashes are truncated, so a consumer that reproduces the chain recomputes it from tokens and cannot continue from a published parent.
- The adapter path is not published, so hashes of LoRA blocks cannot be reproduced from events.
- The hashing block size is not published. For a group whose block spans several hashing units, `extra_keys` describe the whole block.

## Compatibility

vLLM evolves this format additively:

- New fields on existing events are optional and omitted when unset.
- `KVEventBatch` may gain trailing elements.
- Existing fields keep their name, type and meaning.
- Events keep firing under the conditions on this page.

Removing, renaming or retyping a field, or changing when an event fires, breaks consumers. Such changes must update this page and `tests/v1/core/test_kv_events_contract.py`, which pins the encoding.

Consumers stay compatible by:

- ignoring unknown fields and trailing batch elements;
- skipping events whose `type` they do not recognize;
- accepting values of `medium` and `kv_cache_spec_kind` they do not recognize.

Decoding a batch into a closed set of event types, such as `msgspec` decoding into `KVEventBatch`, rejects the whole batch when it contains a new event type. Decode events individually to skip unknown ones.
