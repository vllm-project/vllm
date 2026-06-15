# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2P KV cache sharing protocol constants and documentation.

Protocol Overview
=================

Two session types communicate over a bidirectional message channel:
- P2PClientSession (requests blocks from a server)
- P2PServerSession (serves blocks to a client)

Connection Lifecycle
--------------------

1. Client opens a connection and sends ConnectMsg with its identity,
   RDMA metadata, memory layout, and config fingerprint.
2. Server validates block_len and config_fingerprint, registers the
   RDMA peer, and replies with ConnectAckMsg.
3. Client receives ConnectAckMsg and transitions to ready state
   (flushes any queued messages).
4. Either side may send DisconnectMsg to gracefully close.

Block Transfer Flow (happy path)
---------------------------------

1. Client sends FetchMsg with a kv_request_id and lists of
   block keys + remote indexes where it wants the data written.
2. Server matches requested blocks against locally stored blocks:
   - Blocks already available are transferred immediately via RDMA.
   - Blocks not yet available are recorded as "demanded" and
     transferred when the server later stores them.
3. When all blocks for a kv_request_id are transferred, the server
   sends TransferDoneMsg (success=True) to the client.
4. Client reports the load job as complete.

Abort Flow (timeout path)
--------------------------

1. If the client times out waiting for TransferDoneMsg, it sends
   AbortFetchMsg to cancel the request.
2. Server cancels inflight transfers for that kv_request_id and
   replies with AbortAckMsg.
3. Client receives AbortAckMsg and reports the load job as failed.
4. If AbortAckMsg itself times out, the client fails the job anyway.

Message Format
--------------

All messages are dicts serialized with msgpack. Every message has a
TYPE_KEY key identifying its type. Additional fields depend on the
message type (see per-message class docstrings below).

Security
--------

Sessions wrap all incoming message handling in try/except to guard
against malformed messages from adversarial or buggy peers. Invalid
messages are logged and dropped without crashing the session.

The config_fingerprint field in ConnectMsg ensures peers have
compatible model configurations (model, dtype, block sizes). Mismatches
are rejected during the handshake.
"""

TYPE_KEY = "type"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require(msg: dict, key: str, typ: type, *, name: str = "") -> None:
    """Raise ValueError if msg[key] is missing or not isinstance(typ)."""
    val = msg.get(key)
    if not isinstance(val, typ):
        label = name or key
        raise ValueError(f"{label}: expected {typ.__name__}, got {type(val).__name__}")


def _require_pos_int(msg: dict, key: str, *, name: str = "") -> None:
    """Raise ValueError if msg[key] is not a positive int."""
    val = msg.get(key)
    if not isinstance(val, int) or val <= 0:
        label = name or key
        raise ValueError(f"{label}: expected positive int, got {val!r}")


def _require_non_neg_int(msg: dict, key: str, *, name: str = "") -> None:
    """Raise ValueError if msg[key] is not a non-negative int."""
    val = msg.get(key)
    if not isinstance(val, int) or val < 0:
        label = name or key
        raise ValueError(f"{label}: expected non-negative int, got {val!r}")


def _require_list(msg: dict, key: str, *, name: str = "") -> None:
    """Raise ValueError if msg[key] is not a list."""
    val = msg.get(key)
    if not isinstance(val, list):
        label = name or key
        raise ValueError(f"{label}: expected list, got {type(val).__name__}")


# ---------------------------------------------------------------------------
# Message classes
# ---------------------------------------------------------------------------


class ConnectMsg:
    """Client → Server: initial handshake request.

    Fields:
        PEER_ID: Local peer identity string.
        AGENT_METADATA: RDMA agent metadata (opaque bytes).
        BASE_ADDR: Base memory address of the KV block region.
        NUM_BLOCKS: Number of blocks in the KV block region.
        BLOCK_LEN: Size in bytes of each block (must match between peers).
        CONFIG_FINGERPRINT: SHA-256 prefix of the model configuration.
            Peers with different fingerprints are incompatible.
    """

    TYPE = "connect"
    PEER_ID = "peer_id"
    AGENT_METADATA = "agent_metadata"
    BASE_ADDR = "base_addr"
    NUM_BLOCKS = "num_blocks"
    BLOCK_LEN = "block_len"
    CONFIG_FINGERPRINT = "config_fingerprint"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, ConnectMsg.PEER_ID, str)
        _require(msg, ConnectMsg.AGENT_METADATA, bytes)
        _require_non_neg_int(msg, ConnectMsg.BASE_ADDR)
        _require_pos_int(msg, ConnectMsg.NUM_BLOCKS)
        _require_pos_int(msg, ConnectMsg.BLOCK_LEN)


class ConnectAckMsg:
    """Server → Client: handshake acknowledgement.

    Fields:
        PEER_ID: Server's peer identity string.
    """

    TYPE = "connect_ack"
    PEER_ID = "peer_id"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, ConnectAckMsg.PEER_ID, str)


class DisconnectMsg:
    """Either → Either: graceful connection close.

    No additional fields beyond TYPE_KEY.
    """

    TYPE = "disconnect"


class FetchMsg:
    """Client → Server: request blocks by key.

    Fields:
        KV_REQUEST_ID: Identifies this block transfer request.
        BLOCK_HASHES: List of block keys (OffloadKey bytes).
        BLOCK_INDEXES: List of remote block indexes (same length as BLOCK_HASHES).
    """

    TYPE = "fetch"
    KV_REQUEST_ID = "kv_request_id"
    BLOCK_HASHES = "block_hashes"
    BLOCK_INDEXES = "block_indexes"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, FetchMsg.KV_REQUEST_ID, str)
        _require_list(msg, FetchMsg.BLOCK_HASHES)
        _require_list(msg, FetchMsg.BLOCK_INDEXES)
        hashes = msg[FetchMsg.BLOCK_HASHES]
        indexes = msg[FetchMsg.BLOCK_INDEXES]
        if len(hashes) != len(indexes):
            raise ValueError(
                f"block_hashes/block_indexes length mismatch: "
                f"{len(hashes)} vs {len(indexes)}"
            )
        for idx in indexes:
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(f"block_indexes: invalid index {idx!r}")


class TransferDoneMsg:
    """Server → Client: all blocks transferred for a request.

    Fields:
        KV_REQUEST_ID: The request that completed.
        SUCCESS: Whether the transfer completed successfully.
    """

    TYPE = "transfer_done"
    KV_REQUEST_ID = "kv_request_id"
    SUCCESS = "success"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, TransferDoneMsg.KV_REQUEST_ID, str)
        _require(msg, TransferDoneMsg.SUCCESS, bool)


class AbortFetchMsg:
    """Client → Server: cancel a pending request.

    Fields:
        KV_REQUEST_ID: The request to cancel.
    """

    TYPE = "abort_fetch"
    KV_REQUEST_ID = "kv_request_id"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, AbortFetchMsg.KV_REQUEST_ID, str)


class AbortAckMsg:
    """Server → Client: acknowledge cancellation.

    Fields:
        KV_REQUEST_ID: The request that was cancelled.
    """

    TYPE = "abort_ack"
    KV_REQUEST_ID = "kv_request_id"

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        _require(msg, AbortAckMsg.KV_REQUEST_ID, str)
