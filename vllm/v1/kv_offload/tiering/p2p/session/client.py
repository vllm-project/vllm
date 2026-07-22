# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client-role state machine for a single peer session.

Handles outgoing fetch requests, abort-on-timeout, abort-ack timeout,
and produces ``LoadResult`` for completed loads. The session coordinator
parses wire messages and dispatches typed arguments here; this module
never touches ``ControlConnection`` directly — it emits via the ``send``
callback injected by the coordinator (which gates on ConnectAck).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortFetchMsg,
    FetchMsg,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import JobId

logger = init_logger(__name__)

_LOAD_TIMEOUT_S = 30.0
_ABORT_ACK_TIMEOUT_S = 10.0


@dataclass
class _InboundRequestState:
    """Client-role state for a single load request."""

    job_id: int  # opaque ID assigned by the manager to this load request
    kv_request_id: str
    submitted_at: float
    aborted_at: float | None = None


class LoadResult(NamedTuple):
    """Result from a session poll, client side."""

    job_id: int
    kv_request_id: str
    success: bool


class ClientRole:
    """Client-side load state machine for one peer session.

    The coordinator owns the connection and the send-gating; this role
    is given a ``send`` callback and a ``peer_id`` for log messages and
    is otherwise self-contained.
    """

    def __init__(self, peer_id: str, send: Callable[[dict], None]) -> None:
        self._peer_id = peer_id
        self._send = send
        self._inbound: dict[str, _InboundRequestState] = {}
        self._completed_loads: list[LoadResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_blocks(
        self,
        job_id: JobId,
        kv_request_id: str,
        keys: Sequence[bytes],
        block_ids: Sequence[int],
        send_ready: bool,
    ) -> None:
        """Register a load request and send the FetchMsg."""
        logger.debug(
            "P2PSession %s: request_blocks job_id=%d kv_request_id=%s "
            "blocks=%d ready=%s",
            self._peer_id,
            job_id,
            kv_request_id,
            len(block_ids),
            send_ready,
        )
        self._inbound[kv_request_id] = _InboundRequestState(
            job_id=job_id,
            kv_request_id=kv_request_id,
            submitted_at=time.monotonic(),
        )
        self._send(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: kv_request_id,
                FetchMsg.BLOCK_HASHES: list(keys),
                FetchMsg.BLOCK_INDEXES: [int(idx) for idx in block_ids],
            }
        )

    def cancel(self, kv_request_id: str) -> None:
        """Cancel a pending load. Sends AbortFetchMsg if still active."""
        req = self._inbound.pop(kv_request_id, None)
        if req is not None and req.aborted_at is None:
            self._send(
                {
                    TYPE_KEY: AbortFetchMsg.TYPE,
                    AbortFetchMsg.KV_REQUEST_ID: kv_request_id,
                }
            )

    def on_transfer_done(self, kv_request_id: str, success: bool) -> None:
        """Handle a TransferDoneMsg from the peer."""
        req = self._inbound.pop(kv_request_id, None)
        if req is not None:
            self._completed_loads.append(
                LoadResult(
                    job_id=req.job_id,
                    kv_request_id=kv_request_id,
                    success=success,
                )
            )
        else:
            # No matching _inbound entry: either a duplicate
            # transfer_done from the peer (protocol violation) or a
            # benign race with a local cancel/abort/timeout that
            # already popped the entry. We don't track terminated ids,
            # so we can't tell — log so it's findable.
            logger.warning(
                "P2PSession %s: transfer_done for unknown kv_request_id=%s "
                "(duplicate from peer, or raced with local cancel/timeout)",
                self._peer_id,
                kv_request_id,
            )

    def on_abort_ack(self, kv_request_id: str) -> None:
        """Handle an AbortAckMsg from the peer."""
        req = self._inbound.pop(kv_request_id, None)
        if req is not None:
            logger.warning(
                "P2PSession %s: load request %s (job_id=%d) timed out; "
                "load job completed with failure. If this recurs, ensure "
                "PYTHONHASHSEED is set to the same value on all nodes.",
                self._peer_id,
                kv_request_id,
                req.job_id,
            )
            self._completed_loads.append(
                LoadResult(
                    job_id=req.job_id,
                    kv_request_id=kv_request_id,
                    success=False,
                )
            )
        else:
            # See on_transfer_done: same ambiguity (duplicate ack
            # vs. raced with local cancel/timeout that already popped).
            logger.warning(
                "P2PSession %s: abort_ack for unknown kv_request_id=%s "
                "(duplicate from peer, or raced with local cancel/timeout)",
                self._peer_id,
                kv_request_id,
            )

    def collect_results(self) -> list[LoadResult]:
        """Walk timeouts and drain completed loads.

        Active requests past ``_LOAD_TIMEOUT_S`` get an AbortFetchMsg
        sent and enter the aborting phase. Aborting requests past
        ``_ABORT_ACK_TIMEOUT_S`` are surfaced as failed loads.
        """
        now = time.monotonic()
        to_remove: list[str] = []
        for req_id, req in self._inbound.items():
            if req.aborted_at is None:
                if now - req.submitted_at >= _LOAD_TIMEOUT_S:
                    req.aborted_at = now
                    logger.warning(
                        "P2PSession %s: %s timed out, sending abort",
                        self._peer_id,
                        req_id,
                    )
                    self._send(
                        {
                            TYPE_KEY: AbortFetchMsg.TYPE,
                            AbortFetchMsg.KV_REQUEST_ID: req_id,
                        }
                    )
            else:
                if now - req.aborted_at >= _ABORT_ACK_TIMEOUT_S:
                    to_remove.append(req_id)
                    self._completed_loads.append(
                        LoadResult(
                            job_id=req.job_id,
                            kv_request_id=req_id,
                            success=False,
                        )
                    )
                    logger.warning(
                        "P2PSession %s: abort_ack timed out for kv_request_id=%s",
                        self._peer_id,
                        req_id,
                    )
        for req_id in to_remove:
            self._inbound.pop(req_id)

        results = self._completed_loads
        self._completed_loads = []
        return results

    def close(self) -> list[tuple[int, str]]:
        """Tear down. Returns ``(job_id, kv_request_id)`` for pending loads."""
        failed = [(req.job_id, req.kv_request_id) for req in self._inbound.values()]
        self._inbound.clear()
        self._completed_loads.clear()
        return failed
