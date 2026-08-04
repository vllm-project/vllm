# P2P L2 Adapter

`lmcache/v1/distributed/l2_adapters/p2p_l2_adapter.py`

An L2 adapter that treats a **single peer cache server** as a read-only L2
tier. Instead of a storage backend, it looks up objects that are resident in
the peer's L1 (CPU RAM) and pulls them directly over the transfer channel
(RDMA). One adapter instance is created per connected peer; the peer discovery
and the dynamic add/remove that wire these into the storage manager are owned
by separate PRs (the P2P controller and the runtime `add_l2_adapter`
interface).

See the Confluence design *LMCache MP P2P design → P2P L2 adapter design* for
the full system context.

## What it talks to

| Dependency | Used for |
|---|---|
| `MessageQueueClient` → peer's P2P controller | lookup-and-lock, unlock RPCs |
| `TransferChannelContext` / `TransferChannelClient` | translating local L1 addresses + reading the peer's L1 |
| `PeriodicEventNotifier` | pulsing the lookup / load event fds (see below) |

## Lifecycle

```
lookup-and-lock  → query (addresses)  → load (RDMA read)  → unlock
```

1. **lookup-and-lock** — send `P2P_LOOKUP_AND_LOCK([keys, layout_desc])` to the
   peer; it read-locks every L1-resident key (sparse; gaps allowed) and
   returns a task id.
2. **query** — poll `P2P_QUERY_LOOKUP_RESULTS(task_id)`; once ready the peer
   returns one `TransferChannelAddress` per key (invalid offset for keys it did
   not lock). The adapter stashes the valid addresses keyed by `ObjectKey` and
   reports a found/not-found `Bitmap` to the prefetch controller.
3. **load** — for the found keys, translate the local destination objects'
   `(shm_offset, shm_byte_length)` into transfer-channel addresses, pair them
   with the stashed remote addresses, and issue `submit_read`. `query_load_result`
   maps the read's `succeeded_mask` to a `Bitmap`.
4. **unlock** — `P2P_UNLOCK_OBJECTS([keys])` releases the peer's read locks and
   drops the stashed addresses. Fire-and-forget: the result future is not awaited.

`layout_desc` is the advisory hint added to the L2 lookup interface; the P2P
adapter forwards the real descriptor verbatim so the peer can size objects.

## Why the periodic notifier

Neither the lookup RPC nor the transfer-channel read exposes a completion fd:
the MQ response resolves on the shared client polling thread, and the RDMA read
completes asynchronously inside the transfer engine. So the adapter registers
its lookup and load event fds with the `PeriodicEventNotifier` singleton, which
pulses them every few milliseconds. Each pulse drives the prefetch controller
to re-poll `query_lookup_and_lock_result` / `query_load_result`, which in turn
check the in-flight RPC / transfer state. The store fd is not pulsed by the
notifier; it is signaled directly on submit (see *No store* below).

Both RPCs are non-blocking handlers, so the adapter uses **submit-then-wait**:
`submit_lookup_and_lock_task` submits the lookup and waits (hard-coded ~3 s) for
the task id; each `query_lookup_and_lock_result` pulse submits
`P2P_QUERY_LOOKUP_RESULTS` and waits — `None` means not ready (re-queried on the
next pulse), addresses mean ready.

## Timeouts

P2P is error-prone, so both queries are bounded by a per-task deadline
(`lookup_timeout_s` / `load_timeout_s`). A lookup past its deadline returns an
all-zero `Bitmap` (treated as a miss); a load past its deadline returns an
all-zero `Bitmap` (treated as a failure). The prefetch controller then trims
those keys as if the peer never had them.

## No store / no eviction

Writing to a peer's L1 is intentionally unsupported (it would require remote
allocation and existence/status checks that can corrupt both nodes on failure).
But the store controller still tracks every task it submits — and the L1 read
locks it reserved for that task — until the result is popped. So rather than
drop store tasks, `submit_store_task` records a 0-byte success and signals the
store fd immediately, and `pop_completed_store_tasks` returns it; the controller
then finalizes its bookkeeping and releases the read locks instead of leaking
them. No data is moved and no bytes are accounted.

The adapter is constructed with `max_capacity_bytes=0`, so
`supports_global_eviction` is `False`, `delete` is the inherited no-op, and
`get_usage` reports the (empty) base counters.

## Configuration

The adapter is built through the standard factory (`create_l2_adapter`), so the
storage manager owns its lifecycle. The config carries the peer's two URLs:

```json
{
  "type": "p2p",
  "peer_mq_server_url": "tcp://peer-host:5555",
  "peer_transfer_channel_server_url": "peer-host:7600",
  "lookup_timeout_s": 10.0,
  "load_timeout_s": 10.0
}
```

## close()

Sets a closed flag (later submits are inert), unregisters the lookup/load fds
from the periodic notifier, removes the transfer-channel client from the
`TransferChannelContext` via `remove_transfer_channel_client(peer_url)` (which
closes it and releases its transport handles), closes the MQ client, and closes
the three event notifiers. `close()` is idempotent: the closed flag guards
against a second teardown of these shared resources.
