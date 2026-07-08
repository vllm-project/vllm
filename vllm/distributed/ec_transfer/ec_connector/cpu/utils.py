# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Supporting utilities for the ECCPUConnector scheduler."""

import msgspec

from vllm.logger import init_logger

# (host, port) pair identifying a remote peer's ZMQ side-channel endpoint.
PeerAddr = tuple[str, int]

logger = init_logger(__name__)


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
