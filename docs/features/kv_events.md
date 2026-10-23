# KV event snapshots

A cache-aware router builds its index from the ZMQ KV event stream. When it
starts late, restarts or misses events, it can load a snapshot of a publisher's
current cache state and then follow the live stream from where the snapshot
ends. Snapshots do not depend on the replay buffer.

## Enable snapshots

```bash
vllm serve Qwen/Qwen3-0.6B --enable-prefix-caching \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","snapshot_endpoint":"tcp://*:5559","topic":"kv-events"}'
```

Give each service its own endpoint. TCP ports are offset by the data-parallel
rank, and `tcp://*:0` picks a free port. The `/kv_event_sources` route and the
Rust frontend's `GetKvEventSources` gRPC call report each rank's addresses,
including `snapshot_endpoint`. If the snapshot socket cannot bind, the publisher
fails to start, so the server never advertises a snapshot address that does not
work.

To try it, run the example subscriber:

```bash
.venv/bin/python examples/features/kv_events/kv_events_snapshot_subscriber.py \
  --endpoint tcp://localhost:5557 \
  --snapshot-endpoint tcp://localhost:5559 --topic kv-events
```

It loads a snapshot, follows live events, and starts over after a sequence gap,
a publisher restart or a missed heartbeat. It prints the events; building a
routing index is the router's job. Run one subscriber per publisher and
data-parallel rank. These sockets follow the existing trusted-network ZMQ model;
see [security guidance](../usage/security.md).

## Wire format

Without `snapshot_endpoint`, nothing changes. With it:

| Message | Frames |
| --- | --- |
| Live or replay batch | `topic`, `sequence`, `IdentifiedKVEventBatch` |
| Snapshot request, from a REQ socket | `snapshot` |
| Snapshot reply | `sequence`, `publisher_id`, zero or more `KVEventBatch` chunks |

- `IdentifiedKVEventBatch` is a `KVEventBatch` with one more trailing element,
  the 16-byte `publisher_id`, which changes only when the publisher restarts.
  Decoders that skip extra trailing elements, such as msgspec's `KVEventBatch`,
  read it unchanged. Decoders that require the exact element count reject it,
  among them llm-d-router builds without
  [llm-d/llm-d-router#2946](https://github.com/llm-d/llm-d-router/pull/2946).
  Upgrade those consumers before you set `snapshot_endpoint`.
- The live sequence starts at 0 and is an unsigned 8-byte big-endian integer.
  Replay requests and the replay end marker are unchanged.
- An idle publisher sends an empty batch every second, so a new subscriber can
  confirm delivery and notice a lost final batch.
- The reply sequence is a signed 8-byte big-endian integer: the last batch the
  snapshot covers. `-1` means nothing has been recorded yet. `-2` means
  snapshots are unavailable, and the reply has no chunks. An empty cache can
  reply with a sequence of 0 or more and no chunks.
- The whole snapshot comes back as one multipart reply. The publisher ignores
  any request other than `snapshot`.
- Chunks use the live encoding and data-parallel rank. Their stores carry each
  block's tokens, extra keys, LoRA, group, locality and ownership, and no
  `session_id`.

## Load a snapshot and follow live events

1. Subscribe to live events and wait for one message before you request a
   snapshot. A SUB socket drops messages until its subscription reaches the
   publisher.
2. Keep buffering live messages while the request is outstanding.
3. Apply the snapshot chunks in order, as ordinary events, to an index that
   holds nothing for this publisher. Chunks can store evicted ancestors and
   token-less offload entries and remove them again; the end state is the live
   state.
4. Drop buffered messages at or below the snapshot sequence. Apply the rest
   only if their `publisher_id` matches the reply and their sequences run on
   from the snapshot sequence plus one with no gaps.
5. Keep applying consecutive live messages. Empty heartbeat batches only
   advance the sequence.
6. On a gap, an identity change, a timeout, a bad reply or a full buffer, clear
   this publisher's state and start again. Never apply a partial snapshot or a
   `-2` reply; after `-2`, try again later.

`SnapshotClient` in the example does the transport part of these steps.
`bootstrap()` returns the snapshot chunks followed by the checked buffered
messages. Its `ready` flag only reports that the transport is in sync; the
caller decides when its own index is ready.

## How a snapshot is built

The recorder keeps one record per block hash: parent, tokens and hash inputs,
plus a reference count per scope (medium, KV cache group, locality,
ownership). It keeps a record while the block is resident anywhere, while a
kept record names it as parent, and for 64 event-carrying batches after the
block was last resident, because an offload store can land after the GPU copy
is gone.

A snapshot replays the records as events:

1. Store every block it can rebuild, parents first, with its tokens: a live
   block in one of its live scopes, a dead block in its group's GPU scope.
2. Remove the dead blocks.
3. Store each live block again by hash, without tokens, until every scope has
   its exact count.
4. Remove the first store of each live block.

A consumer that counts references per scope and hash ends with the exact live
state, also when it forgets a hash once nothing holds it. `AllBlocksCleared`
from the GPU block pool clears GPU residency; offloaded blocks and their
metadata stay. Consumers must treat the live stream the same way.

KV cache groups with the same block size share one record per hash. Groups
that skip null or masked blocks, such as sliding-window or Mamba groups, list
fewer hashes than their token span. Their stores only add residency to blocks
another store taught, and a snapshot restores that residency with token-less
stores. Offload stores can omit `extra_keys`; the recorder compares keys once a
store states them.

## Limits, cost and failures

- The recorder runs on its own thread in the engine process. Before each reply
  it folds every batch queued before the request, and it answers one request at
  a time. A requester that stops reading cannot block it.
- Its input queue holds 4,096 batches. When the queue is full, the publisher
  waits up to one second for room.
- Records and live references are each capped at `snapshot_max_blocks` (one
  million by default), recently dead records at 65,536 (the oldest leave
  first), and replies at `snapshot_max_response_bytes` (256 MiB by default).
  The example caps buffered bootstrap data at 64 MiB.
- The recorder shares the engine's Python GIL, so the engine slows down while a
  snapshot is built, for about as long as the build takes. A snapshot of 375K
  blocks takes about one second under load. Request a snapshot only when a
  consumer needs to catch up.
- Snapshots are best effort. A block cannot be rebuilt when no store taught its
  tokens, for example a store whose parent or own record is already gone, or a
  store that skips blocks no other store taught. A hash restated with other
  inputs cannot be rebuilt either; KV cache groups with different block sizes
  do this. A snapshot leaves such blocks and their descendants out, and
  consumers learn them when they are stored again.
- A queue that stays full for one second, an unsupported event, or an exceeded
  limit stops the recorder. It logs, frees its state and answers `-2` until the
  engine restarts. Live publishing continues. A consumer that falls behind a
  working recorder can request another snapshot.
- Use incremental KV event reporting. Per-request `full` reporting re-announces
  existing blocks as new stores, and reference counts cannot tell the two
  apart.
