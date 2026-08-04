# Transfer Channel Design

This document describes the **transfer channel** abstraction
(`lmcache/v1/distributed/transfer_channel/`): its public API, how a read flows
end to end, the nixl-backed implementation, and how to add a new implementation.

It is intended for developers using the transfer channel to move L1 memory
objects between peers, or implementing a new transport backend.

## Purpose

The transfer channel provides **peer-to-peer reads** of registered L1 memory.
A peer registers its L1 buffer once, exchanges metadata with another peer during
a handshake, and can then read arbitrary `(offset, size)` regions out of the
remote peer's L1 buffer into its own. Only **reads** are supported today (a peer
pulls data from a remote; it never pushes).

The abstraction is transport-agnostic; the only implementation today is
[`nixl`](#nixl-implementation) (RDMA/UCX via the NIXL bindings).

## Architecture Overview

```
            ┌──────────────────────────────────────────────┐
            │            TransferChannelContext              │
            │  (one per peer; owns the transport + L1 reg.)  │
            │                                                │
            │  get_transfer_channel_server()  ── singleton ──┼──► TransferChannelServer
            │  get_transfer_channel_client(peer_url) ────────┼──► TransferChannelClient (per peer)
            │  remove_transfer_channel_client(peer_url)      │
            │  get_transfer_channel_address([(off,size)…])   │
            │  get_num_connected_clients()                   │
            │  close()                                       │
            └───────────────┬──────────────┬─────────────────┘
                            │              │
              one per ctx   │              │  one per remote peer
                            ▼              ▼
                ┌────────────────┐   ┌────────────────────────┐
                │ ...Server      │   │ ...Client              │
                │ handshake REP  │   │ submit_read(local,     │
                │ (metadata only)│   │             remote)    │
                │                │   │ query_read_status(id)  │
                │                │   │ close()                │
                └────────────────┘   └────────────────────────┘
```

Each peer constructs **one** `TransferChannelContext`. The context eagerly owns
exactly one `TransferChannelServer` (bound at construction) and a set of
`TransferChannelClient`s keyed by peer url, created lazily on first use.

The context is also available as a **process-global singleton** through the
package-level lifecycle functions (see [Global lifecycle](#global-lifecycle)),
which is the normal entry point for application code.

## Public API

All names below are exported from
`lmcache.v1.distributed.transfer_channel` (`__init__.py`).

### Data types (`api.py`)

| Type | Fields | Notes |
|---|---|---|
| `TransferChannelAddress` | `offset: int`, `size: int` | A transfer-channel address (frozen). `offset` is relative to the L1 base. Wrapped in a class for future extensibility. |
| `TransferChannelReadResult` | `finished: bool`, `succeeded_mask: list[bool]` | Result of `query_read_status`. `is_finished()` accessor. `succeeded_mask` holds a per-object success flag aligned with the submitted addresses; empty while in flight. |

### Abstract interfaces (`abstract.py`)

- **`TransferChannelContext`** — owns the server and clients; translates L1
  `(offset, size)` objects into transfer-channel addresses.
  - `get_transfer_channel_server() -> TransferChannelServer`
  - `get_transfer_channel_client(peer_advertise_url: str) -> TransferChannelClient`
    — get or create a client to the peer. Note: for bidirectional transports
    (like NIXL) the client may instead be created *passively* when the peer
    dials this context's server; this is transparent to the caller.
  - `remove_transfer_channel_client(peer_advertise_url: str) -> None`
    — discard the peer's client when reads from it are no longer needed (e.g.
    its owning L2 adapter is being removed) and free its resources. Outstanding
    task ids for that client become invalid; a later
    `get_transfer_channel_client` returns a fresh one. A no-op for an unknown
    peer.
  - `get_transfer_channel_address(lmcache_addresses: list[tuple[int, int]]) -> list[TransferChannelAddress]`
    — validate `(offset, size)` pairs against the registered region and convert.
  - `get_num_connected_clients() -> int`
  - `close() -> None`
- **`TransferChannelServer`** — listens for peer connections and exchanges
  transfer metadata during the handshake. `close()` frees its resources.
- **`TransferChannelClient`** — reads from one remote peer's memory.
  - `submit_read(local_addresses, remote_addresses) -> int` — returns a task id.
    The two address lists must have equal length; element *i* of `remote` is read
    into element *i* of `local`.
  - `query_read_status(task_id: int) -> TransferChannelReadResult`
  - `close() -> None`

### Global lifecycle (`__init__.py`)

The normal way application code obtains a context:

| Function | Purpose |
|---|---|
| `initialize_transfer_channel_context(transfer_channel_type, l1_memory_desc, listen_url, advertise_url, **kwargs)` | Create the process-global context via the registered factory. Raises if one already exists, or if the type is unknown. `**kwargs` are forwarded to the factory (e.g. `backends=["UCX"]` for nixl). |
| `get_transfer_channel_context()` | Return the global context (raises if not initialized). |
| `delete_transfer_channel_context()` | Close and clear the global context. |

`listen_url` is the `host:port` this peer's singleton server binds to;
`advertise_url` is the `host:port` this peer announces as its identity (the key
peers store its reverse client under). `l1_memory_desc` is an
`L1MemoryDesc(ptr, size, align_bytes)` (`distributed/internal_api.py`),
typically obtained from `L1MemoryManager.get_l1_memory_desc()`.

## How a read flows

1. **Register.** Each peer builds a context, which registers its whole L1 buffer
   with the transport and binds its handshake server.
2. **Connect / handshake.** The reader calls
   `get_transfer_channel_client(peer_url)`. If no client exists, the context
   dials the peer's server and exchanges transport metadata (agent identity and
   the serialized transfer-descriptor list covering the whole buffer).
3. **Translate addresses.** Both sides express objects as `(offset, size)` pairs
   relative to their own L1 base. The reader converts its **local**
   (destination) addresses via `get_transfer_channel_address(...)`. **Remote**
   (source) addresses are constructed directly as `TransferChannelAddress`
   instances (they refer to the peer's region, so they are not validated against
   the local region).
4. **Submit + poll.** `submit_read(local, remote)` returns a task id; the caller
   polls `query_read_status(task_id)` until `is_finished()`. The result's
   `succeeded_mask` holds a per-object success flag aligned with the submitted
   addresses (all `True` on success, all `False` on error).

> **Alignment contract.** Object offsets must be aligned to the registered
> `align_bytes`, and the reader and the peer must use the **same** `align_bytes`,
> because the page-index math used to address the remote buffer is computed
> against the reader's alignment but indexes the peer's page-granular descriptor
> list.

## nixl implementation

`impl/nixl_impl.py` implements the abstraction over the NIXL bindings (UCX by
default). It self-registers under the type name `"nixl"` at import time. Notable
points:

- **One agent per context.** The context creates a single `nixl_agent`, registers
  the whole L1 buffer once, and builds a page-granular transfer-descriptor list
  over `[base, base+size)` using `align_bytes` as the page size.
- **Handshake over ZMQ.** Metadata is exchanged over a ZMQ `REP`/`REQ` socket
  (not the LMCache MQ — that would be overkill and would leak transfer-channel
  functions into the global MQ). The messages are a small `msgspec` tagged union
  (`InitReq`/`InitResp`, `MemRegReq`/`MemRegResp`).
- **Passive client creation.** When a peer dials this context's server and sends
  its descriptor list, the server registers a reverse `NixlTransferChannelClient`
  for that peer, so reads can flow in either direction after a single handshake.
- **Address → page indices.** `addresses_to_indices` turns `(offset, size)` into
  the list of page indices `[offset/align, …]`, expanding multi-page objects.
- **Handshake timeout.** The dialing side sets a `RCVTIMEO` of
  `_HANDSHAKE_TIMEOUT_MS` (60 s) on the handshake socket. A wrong/unreachable
  url or a blocked port therefore raises a clear `TimeoutError` instead of
  hanging forever. The server's `_serve_loop` polls with a 1 s timeout so it can
  observe shutdown.

## Extending: add a new transfer channel implementation

Implementations live under `impl/` and **self-register a factory** with the
registry in `factory.py`. There is no central switch statement to edit.

1. **Implement the three interfaces** from `abstract.py`
   (`TransferChannelContext`, `TransferChannelServer`, `TransferChannelClient`)
   and use the `api.py` data types. Put the module under `impl/`. If it wraps a
   third-party library, name the module after the *implementation*, not the
   library it wraps (e.g. `nixl_impl.py`, not `nixl.py`, to avoid colliding with
   the third-party `nixl` package).

2. **Provide a factory and register it** at module import time:

   ```python
   from lmcache.v1.distributed.transfer_channel.factory import (
       register_transfer_channel_factory,
   )

   def create_my_transfer_channel_context(
       l1_memory_desc, listen_url, advertise_url, **kwargs
   ):
       return MyTransferChannelContext(...)

   register_transfer_channel_factory("mytype", create_my_transfer_channel_context)
   ```

   The factory is always called with the keyword arguments `l1_memory_desc`,
   `listen_url`, `advertise_url`, plus any implementation-specific keyword
   arguments forwarded by the caller. Registering a duplicate type raises
   `ValueError`.

3. **Make the module import-loaded** so the registration runs. Add it to
   `impl/__init__.py` (which the package `__init__.py` imports eagerly). Keep the
   heavy/optional third-party import lazy (inside the context constructor), so
   importing the module to register the factory does not require the dependency
   to be installed.

4. **Use it** via `initialize_transfer_channel_context("mytype", …)`. The
   `transfer_channel_type` is a plain string identifier (not a module path); it
   is resolved through the registry, and unknown types raise a `ValueError`
   listing the registered types.

`register_transfer_channel_factory` and the internal
`create_transfer_channel_context` live in `factory.py` and are deliberately
**not** part of the package's public `__all__` — implementations import the
register function directly from `factory`, and `initialize_transfer_channel_context`
is the public creation entry point.

## Testing

- Unit tests for the data types, factory registry, and global lifecycle:
  `tests/v1/distributed/transfer_channel/test_api.py`,
  `test_factory.py`, `test_context_lifecycle.py`.
- Integration tests for the nixl implementation (guarded by
  `pytest.importorskip("nixl")`, public interface only):
  `tests/v1/distributed/transfer_channel/test_nixl_impl.py`.

## Related

- Throughput benchmark tool: `lmcache/tools/transfer_channel_benchmark/`
  (design/usage symlinked at `docs/design/tools/transfer_channel_benchmark/`).
