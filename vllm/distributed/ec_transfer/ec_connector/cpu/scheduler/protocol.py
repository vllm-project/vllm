# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side wire types for the ECCPUConnector.

Two peer-to-peer message types exchanged between consumer and producer over
ZMQ (XferReq, XferAck) plus a peer-compatibility fingerprint
(compute_ec_compatibility_hash) sent on every XferReq so a producer refuses
to serve a mismatched peer.
"""

import enum
import hashlib
import json

import msgspec

# Bump when the on-wire shape (XferReq/XferAck) changes in a
# backward-incompatible way. Peers reject mismatched versions.
EC_CONNECTOR_VERSION: int = 1


class XferStatus(enum.IntEnum):
    """Outcome the producer reports for an XferReq."""

    OK = 0
    NACK_MISSING = 1  # producer no longer holds the encoding
    NACK_INCOMPAT = 2  # peer-compatibility hash mismatch
    NACK_VERSION = 3  # connector wire-version mismatch
    NACK_INTERNAL = 4  # producer hit an unexpected error


class XferReq(  # type: ignore[call-arg]
    msgspec.Struct,
    tag="req",
    omit_defaults=True,
):
    """Consumer → Producer: I want to READ mm_hash; pin it and tell me where."""

    mm_hash: str
    compatibility_hash: str
    # Consumer's session identity. Together with mm_hash it uniquely identifies
    # this transfer on both sides: the producer keys its active xfers by
    # (session_id, mm_hash), and the NIXL notif_msg carries the same pair.
    session_id: str = ""
    connector_version: int = EC_CONNECTOR_VERSION


class XferAck(  # type: ignore[call-arg]
    msgspec.Struct,
    tag="ack",
    omit_defaults=True,
):
    """Producer → Consumer: response to an XferReq.

    On status == OK: producer has pinned source blocks and returns their
    indices plus fresh NIXL metadata for the consumer to register and READ.
    On any NACK_*: optional fields are empty; consumer falls back to local encode.
    """

    mm_hash: str
    status: XferStatus
    # Consumer's session_id echoed back from XferReq. Together with mm_hash it
    # identifies the exact transfer; the consumer uses both to build notif_msg.
    session_id: str = ""
    src_block_indices: list[int] = []
    agent_metadata: bytes = b""
    mem_descriptor: bytes = b""


def compute_ec_compatibility_hash(
    vllm_version: str,
    model: str,
    dtype: str,
    block_size_bytes: int,
) -> str:
    """Peer-compatibility fingerprint over the four factors that determine
    byte-layout compatibility. Producer NACKs any XferReq whose hash differs.
    """
    factors = json.dumps(
        [vllm_version, model, dtype, block_size_bytes],
        separators=(",", ":"),
    )
    return hashlib.sha256(factors.encode("utf-8")).hexdigest()
