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
rank. `tcp://*:0` chooses an available port; the publisher's
`get_publisher_config()` returns its resolved address. Snapshot sockets bind,
including when given a concrete TCP address. A bind failure fails publisher
startup instead of advertising an unusable snapshot service.

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
Enabling it extends each live and replay **data** message's sequence frame from
8 to 24 bytes. Consumers must understand this format before snapshots are enabled.
The older replay-only subscriber example does not implement this extension.

| Message | Application frames |
| --- | --- |
| Live PUB or replay data | `topic`, `sequence + publisher_id`, `KVEventBatch` |
| Snapshot request | `snapshot` (one byte-string frame) |
| Snapshot response | `sequence`, `publisher_id`, zero or more `KVEventBatch` chunks |

The live/replay sequence starts at zero and is an unsigned 8-byte big-endian
integer. The following 16 bytes identify the publisher; they change when it
restarts or when snapshots become available again after a `-2` reply. An idle
publisher emits an empty batch every second; this establishes subscription
delivery and exposes a lost final batch. Numeric replay requests still contain
only the 8-byte starting sequence. The replay end marker remains unchanged.

The snapshot response sequence is a **signed** 8-byte big-endian integer naming
the last batch included. `-1` denotes a publisher that has not recorded a batch;
`-2` denotes an unavailable snapshot and carries no chunks. Every reply includes
the publisher's 16-byte identity. A valid empty cache can have no chunks and a
nonnegative sequence. A REQ socket handles a complete snapshot as one multipart
reply; DEALER clients can send a single request frame and receive the same reply
without an empty delimiter. Transport routing identities are not shown above.

Event payloads retain the existing `KVEventBatch` encoding, including the
data-parallel rank and each store's tokens, extra keys, LoRA, group, locality,
ownership, and session metadata.

## Install a snapshot and follow live updates

1. Subscribe to live events and receive a message before requesting a snapshot.
   Merely connecting a SUB socket does not establish delivery.
2. Continue buffering live messages while the snapshot request is outstanding.
3. Build a **new private index** from all snapshot chunks in order. Snapshot
   stores can include evicted ancestors needed to reconstruct descendants or
   tokenless offload entries. Trailing removals remove that excess residency.
4. Discard buffered messages at or below the snapshot sequence. Apply the
   remaining messages only if their identities match and their sequences are
   consecutive, starting at `snapshot_sequence + 1`.
5. Atomically install the private index in the router, then continue consuming
   consecutive live messages. Treat empty heartbeat batches as sequence updates.
6. On a gap, identity change, timeout, malformed reply, or buffer exhaustion,
   mark this publisher's routing state unavailable and start again with a fresh
   private index. Do not install a partial or unavailable snapshot.

`SnapshotClient` in the example implements the transport part of this procedure.
Its `bootstrap()` returns snapshot chunks followed by the validated buffered
suffix. Its `ready` flag describes transport continuity, not completion of the
caller's index construction.

The recorder keeps one record per block hash: its parent, tokens and hash
inputs. A record is retained while the block is resident in any tier, while a
retained record names it as parent, and for 16 event-carrying batches after
its last residency ends, because an offload store can complete after its GPU
copy was evicted. A snapshot teaches every retained block, parents first,
removes them again, and then stores each live residency by hash with its exact
count. `AllBlocksCleared` from the GPU block pool clears GPU residency;
offloaded residency and its reconstruction metadata remain. Consumers must
apply the same tier semantics to the subsequent live stream.

## Limits and failure behavior

The recorder starts with the publisher and receives each immutable encoded batch
before live publication. One separate thread owns both snapshot state and its
ROUTER socket. Snapshot requests process a finite FIFO cut and coalesce up to
32 waiting requests. A slow snapshot requester does not block the publisher on
its socket; the requester can time out and retry.

Pending recorder input is limited to 4,096 batches and 64 MiB. When it is full,
the publisher waits up to one second for room. Records and live references are
each limited to one million, recently dead records to 65,536 (the oldest leave
early beyond that), and encoded snapshot replies to 256 MiB. The example limits
buffered bootstrap data to 64 MiB. Snapshot work shares the engine process and
Python GIL, so it can still affect CPU use and inference latency.

A block the recorder cannot rebuild, such as a store whose parent or own record
has already left, or a hash restated with other inputs, taints its record.
While any tainted record is retained, requests receive `-2`; snapshots become
available again once none is, without a restart. If a `-2` was sent during
that period, the publisher identity changes when it ends, so consumers that
fell back to live events alone bootstrap again. A removal of a hash without a
record is counted and logged; a consumer bootstrapped after the record left
fails on it once and bootstraps again.

Input the recorder could not queue within the wait, undecodable batches,
unsupported events and exceeded record, reference or reply limits invalidate
the recorder: requests receive `-2` until the publisher restarts, while live
publishing continues. Every invalidation is logged. A stopped service instead
causes request timeouts. A consumer that falls behind an otherwise healthy
recorder can recover by requesting another snapshot without restarting vLLM.

Use incremental KV event reporting. Optional per-request `full` reporting can
re-announce existing blocks with the same `BlockStored` schema as new copies;
event reference counts cannot distinguish these cases. Snapshots preserve
event-derived state and do not resolve that upstream ambiguity. Similarly, they
preserve the existing interpretation of sparse-attention store events rather
than changing that event schema.
