# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Supporting utilities for the ECCPUConnector scheduler."""

from dataclasses import dataclass

import msgspec
import zmq

from vllm.logger import init_logger

# (host, port) pair identifying a remote peer's ZMQ side-channel endpoint.
PeerAddr = tuple[str, int]

logger = init_logger(__name__)


@dataclass
class PinnedEncoding:
    """Producer-side record of an encoding pinned for in-flight remote reads.

    Keyed by `mm_hash` in the producer's `_pinned_encodings` dict, so it
    stores neither the hash (the key) nor the block list (looked up live in
    `_local_encodings`, which stays valid for the pin's lifetime because
    producer eviction skips pinned blocks).

    `deadlines` holds one monotonic deadline per outstanding read — its
    length is the region pin refcount. Each `XferReq` grant pins the blocks
    once and appends a deadline; a completion notif unpins once and drops one
    deadline; the force-unpin sweep drops every expired deadline (unpinning
    one block-set per drop). Per-read deadlines mean a stalled reader times
    out on its own schedule instead of being kept alive by a later reader's
    grant. The backstop only fires when a consumer dies mid-read and its
    completion notif never arrives.

    Accessed only from the producer's router thread; needs no synchronization.
    """

    deadlines: list[float]


@dataclass
class ConsumerPeer:
    """Consumer-side per-peer connection + NIXL-registration state.

    Cached in the scheduler's `(peer_host, peer_port) → ConsumerPeer` dict.
    The DEALER is created lazily so the consumer can send an `XferReq` before
    any metadata exists; the NIXL fields stay `None` until the first OK
    `XferAck` registers the producer. `remote_read_handle is None` therefore
    discriminates "DEALER-only" from "registered" — no separate state enum.

    `nixl_metadata_bytes` is compared against each incoming `XferAck` so a
    producer restart at the same address (fresh, different metadata) triggers
    re-registration rather than reuse of a stale agent.
    """

    zmq_dealer: zmq.Socket
    zmq_monitor: zmq.Socket | None = None
    nixl_agent_name: str | None = None
    nixl_metadata_bytes: bytes | None = None
    remote_read_handle: int | None = None


@dataclass
class PendingRead:
    """Consumer-side state for one read it has initiated for an `mm_hash`.

    Held as the value in the consumer's `mm_hash → PendingRead | None` dict,
    where a `None` value is the NACK tombstone consumed by
    `ensure_cache_available` to fall through to local encode.

    `read_handle is None` discriminates the two live phases — awaiting the
    `XferAck` vs. the NIXL READ in flight — so no separate state enum is
    needed. `deadline` (monotonic) bounds whichever phase is current: the
    `XferAck` wait while `read_handle is None`, then the read budget once the
    READ is posted (it is reset at that transition).

    Block indices for this read are stored in the consumer's `_blocks` map
    (keyed by `mm_hash`) rather than here, so there is a single source of
    truth for block ownership across all in-flight phases.
    """

    addr: PeerAddr
    deadline: float
    read_handle: int | None = None


@dataclass
class QuarantinedRead:
    """Consumer-side blocks abandoned by a read but not yet safe to reuse.

    NIXL cannot abort an in-flight transfer, so when the consumer gives up on
    a read (deadline passed, or its peer went down) while the READ may still be
    DMA-ing into `dst_indices`, the blocks and handle are parked here instead
    of freed. They are polled each step and released + freed only once
    `check_xfer_state` reports a terminal state (DONE/ERR), at which point no
    transfer can still touch the memory.
    """

    dst_indices: list[int]
    read_handle: int


# Msgpack (de)serialization for the list-of-(addr, size, device_id)
# descriptor form that NIXL's `get_xfer_descs` expects.
_MemDescList = list[tuple[int, int, int]]

_mem_desc_encoder = msgspec.msgpack.Encoder()
_mem_desc_decoder = msgspec.msgpack.Decoder(_MemDescList)


def serialize_mem_descriptor(descs: _MemDescList) -> bytes:
    return _mem_desc_encoder.encode(descs)


def deserialize_mem_descriptor(payload: bytes) -> _MemDescList:
    return _mem_desc_decoder.decode(payload)


def build_block_descs(
    base_ptr: int,
    num_blocks: int,
    block_size_bytes: int,
    device_id: int = 0,
) -> _MemDescList:
    """Per-block `(addr, size, device_id)` tuples for an `ECSharedRegion`.

    Handed to `nixl.get_xfer_descs` to build a dlist where descriptor `i`
    addresses block `i` in the mmap. This makes a WRITE to block indices
    `[i1, i2, ...]` a simple `make_prepped_xfer(..., [i1, i2])` on both
    ends — no per-request address math.
    """
    return [
        (base_ptr + i * block_size_bytes, block_size_bytes, device_id)
        for i in range(num_blocks)
    ]
