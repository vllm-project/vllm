# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side wire types for the ECCPUConnector.

Two peer-to-peer message types exchanged between consumer and producer over
ZMQ (`XferReq`, `XferAck`) — msgspec Structs with explicit tags — plus a
peer-compatibility fingerprint (`compute_ec_compatibility_hash`) sent on
every `XferReq` so a producer refuses to serve a mismatched peer.
"""

import enum
import hashlib
import json

import msgspec

# Bump when the on-wire shape (XferReq/XferAck) changes in a
# backward-incompatible way. Peers reject mismatched versions.
EC_CONNECTOR_VERSION: int = 1


class XferStatus(enum.IntEnum):
    """Outcome the producer reports for an `XferReq`.

    Every non-OK code makes the consumer free its destination blocks and fall
    back to local encode; the distinction exists only so each path logs an
    accurate, operator-readable reason.
    """

    OK = 0
    NACK_MISSING = 1  # producer no longer holds the encoding (evicted/restart)
    NACK_INCOMPAT = 2  # peer-compatibility hash mismatch
    NACK_VERSION = 3  # connector wire-version mismatch
    NACK_INTERNAL = 4  # producer hit an unexpected error building the grant


class XferReq(  # type: ignore[call-arg]
    msgspec.Struct,
    tag="req",
    omit_defaults=True,
):
    """Consumer → Producer: I want to READ `mm_hash`; pin it and tell me where.

    The consumer initiates the NIXL READ, so the request carries no
    consumer-side NIXL state — only the identity of the wanted object plus
    the compatibility / version fingerprints the producer checks before
    granting.
    """

    mm_hash: str
    compatibility_hash: str
    connector_version: int = EC_CONNECTOR_VERSION


class XferAck(  # type: ignore[call-arg]
    msgspec.Struct,
    tag="ack",
    omit_defaults=True,
):
    """Producer → Consumer: response to an `XferReq`.

    On `status == OK` this is a read grant: the producer has pinned the
    source blocks and returns where they are (`src_block_indices`) together
    with the *fresh* NIXL state the consumer needs to register it and prep a
    READ-source dlist (`agent_metadata` from `NixlWrapper.get_agent_metadata`,
    `mem_descriptor` = msgpack-encoded `(addr, size, device_id)` block descs).
    The metadata travels on every grant so a same-address producer restart is
    detected by comparison on the consumer.

    On any `NACK_*` status the optional fields are empty and the consumer
    falls back to local encode. Completion is signalled out-of-band by a NIXL
    notification (`notif_msg == mm_hash`) delivered to the producer when the
    READ lands — not by a second `XferAck`.
    """

    mm_hash: str
    status: XferStatus
    src_block_indices: list[int] = []
    agent_metadata: bytes = b""
    mem_descriptor: bytes = b""


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
