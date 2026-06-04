# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PClientSession — client side, requests blocks from a server peer.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.p2p.control.base import ControlConnection
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortAckMsg,
    AbortLookupFetchMsg,
    ConnectAckMsg,
    ConnectMsg,
    DisconnectMsg,
    LookupFetchMsg,
    TransferDoneMsg,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import JobId
    from vllm.v1.kv_offload.tiering.p2p.data import DataTransport

logger = init_logger(__name__)

_LOAD_TIMEOUT_S = 30.0
_ABORT_ACK_TIMEOUT_S = 10.0


class _LoadPhase(Enum):
    """Client-side load request lifecycle phases."""

    ACTIVE = "active"
    ABORTING = "aborting"


@dataclass
class _InboundRequestState:
    """Client-side state for a single load request."""

    job_id: int
    kv_request_id: str
    submitted_at: float
    phase: _LoadPhase = _LoadPhase.ACTIVE
    abort_at: float | None = None


class LoadResult(NamedTuple):
    """Result from a client session poll."""

    job_id: int
    kv_request_id: str
    success: bool


class P2PClientSession:
    """Requests KV blocks from a single server peer.

    Handles transfer_done, abort_ack, and connect_ack messages internally.
    The manager calls request_blocks() and poll().

    The constructor sends the connect message. Messages are queued until
    connect_ack arrives (handled in _on_message).
    """

    def __init__(
        self,
        peer_id: str,
        conn: ControlConnection,
        local_id: str,
        transport: DataTransport,
    ) -> None:
        self.peer_id = peer_id
        self._conn = conn
        self._inbound: dict[str, _InboundRequestState] = {}
        self._completed: list[LoadResult] = []
        self._ready = False
        self._queued: list[dict] = []

        # Send connect message (ack arrives later via _on_message)
        conn.send(
            {
                TYPE_KEY: ConnectMsg.TYPE,
                ConnectMsg.PEER_ID: local_id,
                ConnectMsg.AGENT_METADATA: transport.get_agent_metadata(),
                ConnectMsg.BASE_ADDR: transport.base_addr,
                ConnectMsg.NUM_BLOCKS: transport.num_blocks,
                ConnectMsg.BLOCK_LEN: transport.block_len,
                ConnectMsg.CONFIG_FINGERPRINT: transport.config_fingerprint,
            }
        )

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def alive(self) -> bool:
        return self._conn.alive

    def request_blocks(
        self,
        job_id: JobId,
        kv_request_id: str,
        keys: Sequence[bytes],
        block_ids: Sequence[int],
    ) -> None:
        """Send lookup_fetch to the server."""
        logger.debug(
            "P2PClientSession %s: request_blocks job_id=%d kv_request_id=%s "
            "blocks=%d ready=%s",
            self.peer_id,
            job_id,
            kv_request_id,
            len(block_ids),
            self._ready,
        )
        self._inbound[kv_request_id] = _InboundRequestState(
            job_id=job_id,
            kv_request_id=kv_request_id,
            submitted_at=time.monotonic(),
        )
        self._send(
            {
                TYPE_KEY: LookupFetchMsg.TYPE,
                LookupFetchMsg.KV_REQUEST_ID: kv_request_id,
                LookupFetchMsg.BLOCK_HASHES: list(keys),
                LookupFetchMsg.BLOCK_INDEXES: [int(idx) for idx in block_ids],
            }
        )

    def poll(self) -> list[LoadResult]:
        """Process incoming messages, check timeouts."""
        # Process incoming messages
        for msg in self._conn.recv():
            self._on_message(msg)

        results: list[LoadResult] = []
        now = time.monotonic()

        for req_id, req in list(self._inbound.items()):
            if req.phase == _LoadPhase.ACTIVE:
                if now - req.submitted_at >= _LOAD_TIMEOUT_S:
                    req.phase = _LoadPhase.ABORTING
                    req.abort_at = now
                    logger.warning(
                        "P2PClientSession %s: %s timed out, sending abort",
                        self.peer_id,
                        req_id,
                    )
                    self._send(
                        {
                            TYPE_KEY: AbortLookupFetchMsg.TYPE,
                            AbortLookupFetchMsg.KV_REQUEST_ID: req_id,
                        }
                    )
            elif req.phase == _LoadPhase.ABORTING:
                assert req.abort_at is not None
                if now - req.abort_at >= _ABORT_ACK_TIMEOUT_S:
                    self._inbound.pop(req_id)
                    results.append(
                        LoadResult(
                            job_id=req.job_id, kv_request_id=req_id, success=False
                        )
                    )
                    logger.warning(
                        "P2PClientSession %s: abort_ack timed out for kv_request_id=%s",
                        self.peer_id,
                        req_id,
                    )

        results.extend(self._completed)
        self._completed = []

        return results

    def cancel_request(self, kv_request_id: str) -> None:
        """Cancel a pending load request. Sends abort if active."""
        req = self._inbound.pop(kv_request_id, None)
        if req is not None and req.phase == _LoadPhase.ACTIVE:
            self._send(
                {
                    TYPE_KEY: AbortLookupFetchMsg.TYPE,
                    AbortLookupFetchMsg.KV_REQUEST_ID: kv_request_id,
                }
            )

    def close(self) -> Iterable[tuple[int, str]]:
        """Shut down. Returns failed (job_id, kv_request_id) pairs."""
        failed = [(req.job_id, req.kv_request_id) for req in self._inbound.values()]
        self._inbound.clear()
        self._send({TYPE_KEY: DisconnectMsg.TYPE})
        self._conn.close()
        return failed

    # ------------------------------------------------------------------
    # Internal message handling
    # ------------------------------------------------------------------

    def _on_message(self, msg: dict) -> None:
        try:
            self._dispatch_message(msg)
        except Exception as exc:
            logger.warning(
                "P2PClientSession %s: error handling message %r: %s",
                self.peer_id,
                msg.get(TYPE_KEY) if isinstance(msg, dict) else msg,
                exc,
            )

    def _dispatch_message(self, msg: dict) -> None:
        msg_type = msg.get(TYPE_KEY)
        if msg_type == ConnectAckMsg.TYPE:
            ConnectAckMsg.validate(msg)
            logger.debug(
                "P2PClientSession %s: connect_ack received, flushing %d queued msg(s)",
                self.peer_id,
                len(self._queued),
            )
            self._ready = True
            for queued in self._queued:
                self._do_send(queued)
            self._queued.clear()
        elif msg_type == TransferDoneMsg.TYPE:
            TransferDoneMsg.validate(msg)
            self._on_transfer_done(msg)
        elif msg_type == AbortAckMsg.TYPE:
            AbortAckMsg.validate(msg)
            self._on_abort_ack(msg)
        elif msg_type == DisconnectMsg.TYPE:
            self._conn.mark_dead()
        else:
            logger.warning(
                "P2PClientSession %s: unknown message type %r",
                self.peer_id,
                msg_type,
            )

    def _on_transfer_done(self, msg: dict) -> None:
        kv_request_id = msg[TransferDoneMsg.KV_REQUEST_ID]
        success = msg[TransferDoneMsg.SUCCESS]
        req = self._inbound.pop(kv_request_id, None)
        if req is not None:
            self._completed.append(
                LoadResult(
                    job_id=req.job_id, kv_request_id=kv_request_id, success=success
                )
            )
        else:
            logger.warning(
                "P2PClientSession %s: transfer_done for unknown kv_request_id=%s",
                self.peer_id,
                kv_request_id,
            )

    def _on_abort_ack(self, msg: dict) -> None:
        kv_request_id = msg[AbortAckMsg.KV_REQUEST_ID]
        req = self._inbound.pop(kv_request_id, None)
        if req is not None:
            self._completed.append(
                LoadResult(
                    job_id=req.job_id, kv_request_id=kv_request_id, success=False
                )
            )
        else:
            logger.warning(
                "P2PClientSession %s: abort_ack for unknown kv_request_id=%s",
                self.peer_id,
                kv_request_id,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send(self, msg: dict) -> None:
        if not self._ready:
            logger.debug(
                "P2PClientSession %s: queueing %s (not ready, queue_depth=%d)",
                self.peer_id,
                msg.get(TYPE_KEY),
                len(self._queued) + 1,
            )
            self._queued.append(msg)
            return
        self._do_send(msg)

    def _do_send(self, msg: dict) -> None:
        try:
            self._conn.send(msg)
            logger.debug(
                "P2PClientSession %s: sent %s",
                self.peer_id,
                msg.get(TYPE_KEY),
            )
        except Exception:
            logger.warning(
                "P2PClientSession %s: failed to send %s",
                self.peer_id,
                msg.get(TYPE_KEY),
            )
