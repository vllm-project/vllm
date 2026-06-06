# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire types and connector metadata for the ECCPUConnector.

Three distinct kinds of payload live here:

1. Peer-to-peer wire types exchanged between consumer and producer over
   ZMQ (`XferReq`, `XferAck`) — msgspec Structs with explicit tags.
2. A scheduler → worker per-step payload (`ECCPUConnectorMetadata`) that
   lists which mm_hashes the worker should save (GPU → mmap) or load
   (mmap → GPU) this step.
3. A peer-compatibility fingerprint (`compute_ec_compatibility_hash`) sent
   on every `XferReq` so a producer refuses to serve a mismatched peer.
"""

import hashlib
import json
from dataclasses import dataclass, field

import msgspec

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata

# Bump when the on-wire shape (XferReq/XferAck) changes in a
# backward-incompatible way. Peers reject mismatched versions.
EC_CONNECTOR_VERSION: int = 1


class XferReq(  # type: ignore[call-arg]
    msgspec.Struct,
    tag="req",
    omit_defaults=True,
):
    """Consumer → Producer: please WRITE `mm_hash` into my block indices.

    `consumer_nixl_metadata` is the opaque blob returned by
    `NixlWrapper.get_agent_metadata()`; the producer feeds it to
    `add_remote_agent` on first contact.

    `consumer_mem_descriptor` is msgpack-encoded list of
    `(base_addr, size, device_id)` tuples describing the consumer's
    pre-registered mmap region; the producer reconstructs NIXL xfer descs
    from it and prepares a remote dlist for WRITE targets.
    """

    mm_hash: str
    dst_block_indices: list[int]
    consumer_agent_name: str
    consumer_nixl_metadata: bytes
    consumer_mem_descriptor: bytes
    compatibility_hash: str
    connector_version: int = EC_CONNECTOR_VERSION


class XferAck(  # type: ignore[call-arg]
    msgspec.Struct,
    tag="ack",
    omit_defaults=True,
):
    """Producer → Consumer: xfer for `mm_hash` completed (or failed)."""

    mm_hash: str
    ok: bool


@dataclass
class ECCPUConnectorMetadata(ECConnectorMetadata):
    """Per-step scheduler → worker payload for the ECCPUConnector.

    Populated by `ECCPUScheduler.build_connector_meta`; consumed by
    `ECCPUWorker` via the mixin's `bind_connector_metadata`.
    """

    # Producer role: mm_hashes the scheduler has just allocated CPU
    # blocks for this step; the worker's save_caches copies
    # encoder_cache[mm_hash] → mmap at these indices.
    saves: dict[str, list[int]] = field(default_factory=dict)

    # Consumer role: mm_hashes whose bytes have already landed in the
    # local mmap (NIXL WRITE completed, XferAck received); the worker's
    # start_load_caches copies mmap[block_indices] → GPU encoder_cache.
    loads: dict[str, list[int]] = field(default_factory=dict)


def compute_ec_compatibility_hash(
    vllm_version: str,
    model: str,
    dtype: str,
    block_size_bytes: int,
) -> str:
    """Peer-compatibility fingerprint.

    Two peers that mismatch on any of these factors cannot interpret each
    other's bytes — mismatched dtype/hidden_dim silently corrupts the
    encoding. The producer compares the hash it computed locally against
    the one in `XferReq` and NACKs on mismatch.

    Factors are JSON-encoded so a value containing the separator (e.g. a
    model name with a `|` in it) cannot collide with a different
    assignment that joins to the same string.
    """
    factors = json.dumps(
        [vllm_version, model, dtype, block_size_bytes],
        separators=(",", ":"),
    )
    return hashlib.sha256(factors.encode("utf-8")).hexdigest()
