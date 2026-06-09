# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Supporting utilities for the ECCPUConnector scheduler."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import torch
import zmq

from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    ECSharedRegion,
)
from vllm.logger import init_logger

# (host, port) pair identifying a remote peer's ZMQ side-channel endpoint.
PeerAddr = tuple[str, int]

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class ECRegionLayout:
    """Mmap region plus the derived layout fields both delegates need.

    Built once via `setup_ec_region(vllm_config)` and unpacked into the
    scheduler / worker; centralizes the dtype + hidden_dim + element_size
    arithmetic that previously lived in two places.
    """

    region: ECSharedRegion
    dtype: torch.dtype
    hidden_dim: int
    element_size: int
    block_size_bytes: int
    num_blocks: int


def _get_encoder_cache_hidden_dim(vllm_config: "VllmConfig") -> int:
    """Return the per-token hidden dimension for encoder cache entries.

    For most models this equals the LLM's hidden size.  Qwen3-VL (and any
    future model with deepstack visual encoding) is an exception: the ViT
    concatenates its own output with features from N decoder layers before
    storing in encoder_cache, producing a tensor of width
    ``out_hidden_size * (1 + N)`` per visual token.  Using the plain LLM
    hidden size would under-allocate EC blocks and silently truncate the
    transferred data, leading to a shape mismatch on the consumer.
    """
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    vision_config = (
        getattr(hf_config, "vision_config", None) if hf_config is not None else None
    )
    if vision_config is not None:
        out_hidden_size = getattr(vision_config, "out_hidden_size", None)
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", None)
        if out_hidden_size is not None and deepstack_indexes:
            # Each visual token carries base features + one feature vector per
            # deepstack level, all concatenated by the ViT.
            return out_hidden_size * (1 + len(deepstack_indexes))
    return model_config.get_inputs_embeds_size()


def setup_ec_region(vllm_config: "VllmConfig") -> ECRegionLayout:
    """Build the EC mmap region and derive its layout from `vllm_config`.

    Both `ECCPUScheduler` and `ECCPUWorker` need the same region (same
    `instance_id`, same `block_size_bytes`) and a subset of the same
    derived shape fields (dtype, hidden_dim, etc.). This helper performs
    that derivation in one place; each delegate picks out the fields it
    uses.
    """
    ec_config = vllm_config.ec_transfer_config
    assert ec_config is not None, "ec_transfer_config required to build region"
    assert ec_config.engine_id is not None, "engine_id is set by __post_init__"

    dtype = vllm_config.model_config.dtype
    hidden_dim = _get_encoder_cache_hidden_dim(vllm_config)
    element_size = torch.empty(0, dtype=dtype).element_size()
    block_size_bytes = hidden_dim * element_size
    num_blocks = int(ec_config.get_from_extra_config("num_ec_blocks", 100000))

    region = ECSharedRegion(
        instance_id=ec_config.engine_id,
        num_blocks=num_blocks,
        block_size_bytes=block_size_bytes,
    )
    return ECRegionLayout(
        region=region,
        dtype=dtype,
        hidden_dim=hidden_dim,
        element_size=element_size,
        block_size_bytes=block_size_bytes,
        num_blocks=num_blocks,
    )


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
    """

    addr: PeerAddr
    dst_indices: list[int]
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
