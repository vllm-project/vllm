# KV event snapshots

External cache-aware routers can rebuild their cache index after starting late,
losing events, or restarting. Enable a snapshot endpoint on the ZMQ KV event
publisher to obtain the current event-derived cache state and a position from
which to consume live events. Snapshot availability does not depend on the
bounded replay history.

## Enable snapshots

```bash
vllm serve Qwen/Qwen3-0.6B --enable-prefix-caching \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","snapshot_endpoint":"tcp://*:5559","topic":"kv-events"}'
```

Use a distinct endpoint for each service. TCP ports are offset by the data-parallel
rank. `tcp://*:0` chooses an available port. The server's `/kv_event_sources`
route reports each rank's resolved addresses; with the Rust frontend, so does
the `GetKvEventSources` gRPC call, as `snapshot_endpoint`. Snapshot sockets
bind, including when given a concrete TCP address. A bind failure fails
publisher startup, so the server never advertises an unusable snapshot service.

Run the example subscriber:

```bash
.venv/bin/python examples/features/kv_events/kv_events_snapshot_subscriber.py \
  --endpoint tcp://localhost:5557 \
  --snapshot-endpoint tcp://localhost:5559 --topic kv-events
```

It obtains a snapshot, follows live updates, and resynchronizes after a sequence
gap, publisher restart, or heartbeat timeout. It prints events; routing-index
construction belongs to the consuming router. Run one subscriber per publisher
and data-parallel rank. These sockets use the existing trusted-network ZMQ
deployment model; see [security guidance](../usage/security.md).

## Wire contract

Without `snapshot_endpoint`, the existing live and replay protocols are unchanged.
With it, live and replay frames keep their format, and each payload is an
`IdentifiedKVEventBatch`: a `KVEventBatch` with a trailing `publisher_id`
element. `KVEventBatch` decoders ignore that element, so existing consumers keep
working.

| Message | Application frames |
| --- | --- |
| Live PUB or replay data | `topic`, `sequence`, `IdentifiedKVEventBatch` |
| Snapshot request | `snapshot` (one byte-string frame from a REQ socket) |
| Snapshot response | `sequence`, `publisher_id`, zero or more `KVEventBatch` chunks |

The live/replay sequence starts at zero and is an unsigned 8-byte big-endian
integer. The 16-byte `publisher_id` changes only when the publisher restarts.
An idle publisher emits an empty batch every second; this establishes
subscription delivery and exposes a lost final batch. Numeric replay requests
and the replay end marker are unchanged.

The snapshot response sequence is a **signed** 8-byte big-endian integer naming
the last batch included. `-1` denotes a publisher that has not recorded a batch;
`-2` denotes an unavailable snapshot and carries no chunks; request again later.
Every reply includes
the publisher's 16-byte identity. A valid empty cache can have no chunks and a
nonnegative sequence. A REQ socket receives a complete snapshot as one multipart
reply. The publisher ignores any other request. Transport routing identities
are not shown above.

Live and replay payloads keep the existing `KVEventBatch` encoding. Snapshot
chunks use the same encoding and data-parallel rank. Their stores carry each
block's tokens, extra keys, LoRA, group, locality and ownership. They carry no
`session_id`, which names the request behind a live store.

## Install a snapshot and follow live updates

1. Subscribe to live events and receive a message before requesting a snapshot.
   Merely connecting a SUB socket does not establish delivery.
2. Continue buffering live messages while the snapshot request is outstanding.
3. Apply all snapshot chunks in order, as ordinary events, to an index that
   holds nothing for this publisher. Snapshot stores can include evicted
   ancestors needed to reconstruct descendants or tokenless offload entries.
   Removals in the snapshot drop that excess residency.
4. Discard buffered messages at or below the snapshot sequence. Apply the
   remaining messages only if their `publisher_id` matches the reply's and
   their sequences are consecutive, starting at `snapshot_sequence + 1`.
5. Continue consuming consecutive live messages. Treat empty heartbeat batches
   as sequence updates.
6. On a gap, identity change, timeout, malformed reply, or buffer exhaustion,
   clear this publisher's state and start again. Do not apply a partial or
   unavailable snapshot; after a `-2` reply, request again later.

`SnapshotClient` in the example implements the transport part of this procedure.
Its `bootstrap()` returns snapshot chunks followed by the validated buffered
suffix. Its `ready` flag describes transport continuity, not completion of the
caller's index construction.

The recorder keeps one record per block hash: its parent, tokens and hash
inputs. Residency is counted per scope: medium, KV cache group, locality and
ownership. A record is retained while the block is resident in any scope, while
a retained record names it as parent, and for 64 event-carrying batches after
its last residency ends, because an offload store can complete after its GPU
copy was evicted. A snapshot stores every retained block it can rebuild with
its tokens, parents first: a live block in one of its live scopes, a dead block
in its group's GPU scope. It then removes the dead blocks, stores each live
residency by hash with its exact count, and removes each live block's first
store. A consumer that counts references per scope and hash, and forgets an
engine hash once no entry holds its key, ends with the live residency of every
block the snapshot includes.
`AllBlocksCleared` from the GPU block pool clears GPU residency; offloaded
residency and its reconstruction metadata remain. Consumers must apply the same
tier semantics to the subsequent live stream.

KV cache groups with a common block size store a block under one hash with the
same inputs, so one record serves every group. A store from a group that leaves
out null or masked blocks, such as sliding-window or Mamba groups, lists fewer
hashes than its token span covers. It teaches no block; it adds residency to
blocks another store taught, and a snapshot restores that residency with
token-less stores. Offload stores can omit `extra_keys`; the recorder keeps keys
a store stated and compares them once known.

## Limits and failure behavior

The recorder starts with the publisher and receives each batch before live
publication. One separate thread owns both snapshot state and its ROUTER
socket, and serves one request at a time after folding every batch queued
before it. A requester that stops reading never blocks the recorder; it can
time out and retry.

Pending recorder input is limited to 4,096 batches. When it is full, the
publisher waits up to one second for room. Records and live references are
each limited to `snapshot_max_blocks` (one million by default), recently dead
records to 65,536 (the oldest leave early beyond that), and encoded snapshot
replies to `snapshot_max_response_bytes` (256 MiB by default). The example limits
buffered bootstrap data to 64 MiB. Snapshot work shares the engine process and
Python GIL, so it can still affect CPU use and inference latency.

Snapshots are best effort. The recorder cannot rebuild a block when no store
taught its tokens: a store whose parent or own record has already left, or a
store that leaves out blocks no other store taught. It also cannot rebuild a
hash restated with other inputs. KV cache groups with different block sizes
restate hashes this way, because one hash names a different token span in each
group. A snapshot leaves such blocks and their descendants out, and consumers
learn them when they are stored again.

Input the recorder could not queue within the wait, unsupported events and
exceeded record, reference or reply limits stop the recorder: it logs, frees
its state, and answers `-2` until the publisher restarts, while live publishing
continues. A consumer that falls behind a running recorder can recover by
requesting another snapshot without restarting vLLM.

Use incremental KV event reporting. Optional per-request `full` reporting can
re-announce existing blocks with the same `BlockStored` schema as new copies;
event reference counts cannot distinguish these cases. Snapshots preserve
event-derived state and do not resolve that upstream ambiguity.
