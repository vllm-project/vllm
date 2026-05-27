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
    hidden_dim = vllm_config.model_config.get_inputs_embeds_size()
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
class ProducerPeer:
    """Producer-side cache of per-consumer NIXL state.

    Populated on first `XferReq` from a given consumer. The remote xfer
    dlist must be prepared on the same NIXL agent that will issue the
    WRITE, so the producer scheduler owns this state. Accessed only
    from the router thread; needs no synchronization.
    """

    nixl_agent_name: str
    nixl_metadata_bytes: bytes
    nixl_xfer_handle: int


@dataclass
class ConsumerPeer:
    """Consumer-side per-peer connection state.

    Cached in the scheduler's `(peer_host, peer_port) → ConsumerPeer` dict;
    invalidated transparently when a producer restarts and its NIXL
    metadata bytes change underneath the same address.
    """

    zmq_dealer: zmq.Socket
    nixl_agent_name: str
    nixl_metadata_bytes: bytes
    zmq_monitor: zmq.Socket | None = None


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
