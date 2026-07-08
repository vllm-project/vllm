# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PSession — bidirectional session combining client + server roles.

A single P2PSession per remote peer handles BOTH directions of the P2P
protocol on one ControlConnection: it can request blocks from the peer
("client" role, in :mod:`.client`) AND serve blocks to the peer
("server" role, in :mod:`.server`). This module is the thin coordinator
that owns the connection, the handshake, send-gating, and the message
dispatch — each parsed message is forwarded to the corresponding role.

Wire protocol is unchanged. Both sides advertise their NIXL metadata
via ConnectMsg when their session is connected; the peer's ConnectMsg
triggers transport.add_remote_peer; ConnectAckMsg confirms the peer
received our ConnectMsg, after which queued outgoing messages are flushed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.p2p.control.base import ControlConnection
from vllm.v1.kv_offload.tiering.p2p.session.client import ClientRole, LoadResult
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortAckMsg,
    AbortFetchMsg,
    ConnectAckMsg,
    ConnectMsg,
    DisconnectMsg,
    FetchMsg,
    LookupMsg,
    LookupRespMsg,
    TransferDoneMsg,
)
from vllm.v1.kv_offload.tiering.p2p.session.server import (
    ServerRole,
    StoreResult,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.tiering.base import JobId, ParentManager
    from vllm.v1.kv_offload.tiering.p2p.data import DataTransport

logger = init_logger(__name__)

# Cap on consecutive non-protocol dispatch exceptions before we tear
# down the session. Protocol violations (ValueError) disconnect on the
# first occurrence; this threshold protects against repeated internal
# bugs that may indicate a peer-induced bad state. Reset on any
# successful dispatch.
_MAX_CONSECUTIVE_DISPATCH_ERRORS = 5


class SessionPollResult(NamedTuple):
    """Result of one P2PSession.poll() tick.

    `loads`/`stores` are the same per-role results the manager has always
    consumed. `new_fetch_ids` reports kv_request_ids whose FetchMsg
    arrived this tick — the manager uses them to bind kv_request_id →
    session and replay any submit_store batches parked while no peer had
    asked yet. Reporting (rather than calling back into the manager
    mid-dispatch) keeps the dependency strictly top-down.
    """

    loads: list[LoadResult]
    stores: list[StoreResult]
    new_fetch_ids: list[str]


class P2PSession:
    """Bidirectional session — coordinator over ClientRole + ServerRole.

    Lifecycle:
      - Constructor with conn=None  ⇒ pending. Accepts add_stored_blocks
        but cannot send (used by the prefiller to buffer blocks before
        the decoder connects).
      - Constructor with conn != None ⇒ connected. Sends our own ConnectMsg
        immediately; the peer's ConnectMsg arrives in poll() and is
        dispatched to _on_connect (which calls transport.add_remote_peer
        and replies with ConnectAckMsg). Outgoing sends are queued until
        ConnectAckMsg confirms our metadata reached the peer.
      - attach_connection(conn) on a pending session ⇒ same as above,
        starting from pending.
    """

    def __init__(
        self,
        peer_id: str,
        local_id: str,
        transport: DataTransport,
        local_block_len: int,
        conn: ControlConnection | None = None,
    ) -> None:
        self.peer_id = peer_id
        self._local_id = local_id
        self._transport = transport
        self._local_block_len = local_block_len
        self._conn: ControlConnection | None = None

        self._send_ready = False  # True after the peer acked our ConnectMsg
        # Msgs waiting to be sent on connection establishment
        self._queued: list[dict] = []

        # Consecutive non-protocol dispatch errors. Reset on success.
        self._dispatch_error_count: int = 0

        # kv_request_ids whose FetchMsg arrived during the current poll
        # tick. Drained and returned in the next poll() result so the
        # manager can bind kv_request_id → session and replay any
        # submit_store batches parked before the binding existed.
        self._new_fetch_ids: list[str] = []

        self._client = ClientRole(peer_id=peer_id, send=self._send)
        self._server = ServerRole(
            peer_id=peer_id,
            transport=transport,
            send=self._send,
        )

        if conn is not None:
            self.attach_connection(conn)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        # Pending sessions (awaiting connection) are alive — only a
        # closed real connection counts as dead.
        return self._conn is None or self._conn.alive

    @property
    def connected(self) -> bool:
        return self._conn is not None

    @property
    def ready(self) -> bool:
        """True after the peer acked our ConnectMsg (we may send freely)."""
        return self._send_ready

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def attach_connection(self, conn: ControlConnection) -> None:
        """Attach a connection to a pending session and announce ourselves.

        Symmetric: every side advertises its NIXL metadata on connect, so
        whichever peer receives a session first can register the other.
        """
        if self._conn is not None:
            raise ValueError(f"P2PSession {self.peer_id}: already connected")
        self._conn = conn
        self._send_connect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_blocks(
        self,
        job_id: JobId,
        kv_request_id: str,
        keys: Sequence[bytes],
        block_ids: Sequence[int],
    ) -> None:
        """Send fetch to the peer."""
        self._client.request_blocks(
            job_id, kv_request_id, keys, block_ids, send_ready=self._send_ready
        )

    def add_stored_blocks(
        self,
        kv_request_id: str,
        keys: Sequence[OffloadKey],
        block_ids: Sequence[int],
        job_id: JobId,
    ) -> None:
        """New blocks stored locally — match against pending fetch demand."""
        self._server.add_stored_blocks(kv_request_id, keys, block_ids, job_id)

    def finish_request(self, kv_request_id: str) -> None:
        """Called when the request is finishing locally.

        Cancels any inbound load (client role), drops any pending
        symmetric-P2P lookup state (client role), and finalizes any
        outbound serving (server role) for this id. Roles that aren't
        active for this id are silent no-ops.
        """
        self._client.cancel(kv_request_id)
        self._client.cancel_lookups(kv_request_id)
        self._server.finish(kv_request_id)

    def register_lookup(self, kv_request_id: str, block_hash: bytes) -> bool | None:
        """Register or resolve one (kv_request_id, block_hash) probe.

        Called from the manager's lookup() for symmetric-P2P consumers
        (``p2p`` sub-dict in kv_transfer_params). See
        ``ClientRole.register_lookup`` for the state-machine contract.
        """
        return self._client.register_lookup(kv_request_id, block_hash)

    def flush_pending_lookups(self) -> None:
        """Flush any aggregated symmetric-P2P lookups for this peer.

        Called once per scheduler step from the manager's
        ``on_schedule_end()``. Send-gating is handled inside the
        client's ``_send`` callback (queues until ConnectAckMsg).
        """
        self._client.flush_pending_lookups()

    def serve_external_requests(self, parent: ParentManager) -> None:
        """Resolve inbound peer lookups against the tiering manager.

        Delegates to the server role; the ``parent`` handle is valid
        only for the duration of this call.
        """
        self._server.serve_external_requests(parent)

    def poll(self) -> SessionPollResult:
        """Process incoming messages, drive transfers, apply timeouts."""
        if self._conn is None:
            # Pending session — store-job timeouts still apply so buffered
            # jobs that never get picked up are surfaced as failures.
            return SessionPollResult(
                loads=[],
                stores=self._server.collect_idle_timeouts(),
                new_fetch_ids=[],
            )

        for msg in self._conn.recv():
            self._on_message(msg)

        loads = self._client.collect_results()
        stores = self._server.collect_results()
        self._server.drain_pending_aborts()

        new_fetch_ids = self._new_fetch_ids
        self._new_fetch_ids = []
        return SessionPollResult(
            loads=loads, stores=stores, new_fetch_ids=new_fetch_ids
        )

    def close(
        self,
    ) -> tuple[list[tuple[int, str]], list[int], list[ReqContext]]:
        """Shut down. Returns (failed_loads, failed_stores, orphan_ctxs).

        failed_loads: list of (job_id, kv_request_id) pairs.
        failed_stores: list of job_ids.
        orphan_ctxs: synthetic lookup ctxs still owing
            ``parent.on_request_finished`` (the manager flushes these on
            its next ``serve_external_requests``).
        """
        failed_loads = self._client.close()
        failed_stores, orphan_ctxs = self._server.close()

        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.send({TYPE_KEY: DisconnectMsg.TYPE})
            self._conn.close()
            self._conn = None

        return failed_loads, failed_stores, orphan_ctxs

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _on_message(self, msg: dict) -> None:
        msg_type = msg.get(TYPE_KEY) if isinstance(msg, dict) else msg
        try:
            self._dispatch_message(msg)
        except ValueError as exc:
            # Protocol contract violation from the peer — *Msg.validate()
            # and handler-level checks raise ValueError. Retrying won't
            # help and may corrupt session state, so disconnect now.
            self._protocol_error(f"malformed {msg_type!r}: {exc}")
            return
        except Exception as exc:
            # Anything else is most likely an internal bug rather than a
            # peer fault. Log loudly with a traceback so it doesn't
            # disappear, but don't kill the session on a single hiccup.
            # Disconnect only if errors keep arriving — that pattern is
            # consistent with a peer wedging us into a broken state.
            self._dispatch_error_count += 1
            logger.exception(
                "P2PSession %s: error handling message %r (count=%d): %s",
                self.peer_id,
                msg_type,
                self._dispatch_error_count,
                exc,
            )
            if self._dispatch_error_count >= _MAX_CONSECUTIVE_DISPATCH_ERRORS:
                self._protocol_error(
                    f"too many consecutive dispatch errors "
                    f"({self._dispatch_error_count})"
                )
            return
        self._dispatch_error_count = 0

    def _protocol_error(self, reason: str) -> None:
        """Log a protocol violation and disconnect.

        Best-effort sends ``DisconnectMsg`` so the peer learns why we're
        going away, then marks the connection dead. The manager reaps
        the session on the next poll via ``alive``.
        """
        logger.error(
            "P2PSession %s: protocol error: %s — disconnecting",
            self.peer_id,
            reason,
        )
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.send({TYPE_KEY: DisconnectMsg.TYPE})
            self._conn.mark_dead()

    def _dispatch_message(self, msg: dict) -> None:
        # Drop messages buffered before disconnect: a poll batch can
        # contain msg-after-DisconnectMsg, and dispatching them would
        # mutate state on a dead session.
        if self._conn is not None and not self._conn.alive:
            return
        msg_type = msg.get(TYPE_KEY) if isinstance(msg, dict) else None
        if msg_type == ConnectMsg.TYPE:
            self._on_connect(msg)
        elif msg_type == ConnectAckMsg.TYPE:
            ConnectAckMsg.validate(msg)
            self._on_connect_ack()
        elif msg_type == FetchMsg.TYPE:
            FetchMsg.validate(msg)
            kv_request_id = msg[FetchMsg.KV_REQUEST_ID]
            block_hashes = [
                OffloadKey(bh if isinstance(bh, bytes) else bytes(bh))
                for bh in msg[FetchMsg.BLOCK_HASHES]
            ]
            block_indexes = msg[FetchMsg.BLOCK_INDEXES]
            # Run the server-role state machine inline as today —
            # add_fetch_demand records demand against any blocks we've
            # already seen in `available`. Report the kv_request_id so
            # the manager (after poll() returns) can replay any parked
            # submit_store batches; their add_stored_blocks calls hit
            # the demand recorded here and submit transfers immediately.
            self._server.on_fetch(kv_request_id, block_hashes, block_indexes)
            self._new_fetch_ids.append(kv_request_id)
        elif msg_type == AbortFetchMsg.TYPE:
            AbortFetchMsg.validate(msg)
            self._server.on_abort_fetch(msg[AbortFetchMsg.KV_REQUEST_ID])
        elif msg_type == TransferDoneMsg.TYPE:
            TransferDoneMsg.validate(msg)
            self._client.on_transfer_done(
                msg[TransferDoneMsg.KV_REQUEST_ID],
                msg[TransferDoneMsg.SUCCESS],
            )
        elif msg_type == AbortAckMsg.TYPE:
            AbortAckMsg.validate(msg)
            self._client.on_abort_ack(msg[AbortAckMsg.KV_REQUEST_ID])
        elif msg_type == LookupMsg.TYPE:
            LookupMsg.validate(msg)
            kv_request_id = msg[LookupMsg.KV_REQUEST_ID]
            block_hashes = [
                OffloadKey(bh if isinstance(bh, bytes) else bytes(bh))
                for bh in msg[LookupMsg.BLOCK_HASHES]
            ]
            self._server.on_lookup(kv_request_id, block_hashes)
        elif msg_type == LookupRespMsg.TYPE:
            LookupRespMsg.validate(msg)
            kv_request_id = msg[LookupRespMsg.KV_REQUEST_ID]
            block_hashes = [
                OffloadKey(bh if isinstance(bh, bytes) else bytes(bh))
                for bh in msg[LookupRespMsg.BLOCK_HASHES]
            ]
            hits = msg[LookupRespMsg.HITS]
            self._client.on_lookup_resp(kv_request_id, block_hashes, hits)
        elif msg_type == DisconnectMsg.TYPE:
            if self._conn is not None:
                self._conn.mark_dead()
        else:
            logger.warning(
                "P2PSession %s: unknown message type %r", self.peer_id, msg_type
            )

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    def _on_connect(self, msg: dict) -> None:
        # Validation failures here mean an incompatible or malicious peer.
        # Mark the connection dead so the manager reaps the session;
        # don't call add_remote_peer or send connect_ack.
        if self._send_ready:
            # We've already received connect_ack, so the handshake is
            # complete. A second connect from the peer is a protocol
            # violation — re-registering would corrupt transport state.
            self._protocol_error("duplicate connect after handshake")
            return
        try:
            ConnectMsg.validate(msg)
            if msg[ConnectMsg.BLOCK_LEN] != self._local_block_len:
                raise ValueError(
                    f"block_len mismatch from {self.peer_id}: "
                    f"remote={msg[ConnectMsg.BLOCK_LEN]}, "
                    f"local={self._local_block_len}"
                )
            remote_fp = msg.get(ConnectMsg.CONFIG_FINGERPRINT, "")
            local_fp = self._transport.config_fingerprint
            if local_fp and remote_fp and remote_fp != local_fp:
                raise ValueError(
                    f"config fingerprint mismatch from {self.peer_id}: "
                    f"remote={remote_fp!r}, local={local_fp!r}"
                )
            self._transport.add_remote_peer(
                self.peer_id,
                agent_metadata=msg[ConnectMsg.AGENT_METADATA],
                base_addr=msg[ConnectMsg.BASE_ADDR],
                num_blocks=msg[ConnectMsg.NUM_BLOCKS],
                block_len=msg[ConnectMsg.BLOCK_LEN],
            )
        except ValueError as exc:
            logger.error("P2PSession %s: rejecting peer connect: %s", self.peer_id, exc)
            if self._conn is not None:
                self._conn.mark_dead()
            return

        if self._conn is not None:
            self._conn.send(
                {
                    TYPE_KEY: ConnectAckMsg.TYPE,
                    ConnectAckMsg.PEER_ID: self._local_id,
                }
            )

    def _on_connect_ack(self) -> None:
        if self._queued:
            logger.debug(
                "P2PSession %s: connect_ack received, flushing %d queued msg(s)",
                self.peer_id,
                len(self._queued),
            )
        self._send_ready = True
        for queued in self._queued:
            self._do_send(queued)
        self._queued.clear()

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    def _send_connect(self) -> None:
        """Send our ConnectMsg announcing local NIXL metadata."""
        assert self._conn is not None
        self._conn.send(
            {
                TYPE_KEY: ConnectMsg.TYPE,
                ConnectMsg.PEER_ID: self._local_id,
                ConnectMsg.AGENT_METADATA: self._transport.get_agent_metadata(),
                ConnectMsg.BASE_ADDR: self._transport.base_addr,
                ConnectMsg.NUM_BLOCKS: self._transport.num_blocks,
                ConnectMsg.BLOCK_LEN: self._transport.block_len,
                ConnectMsg.CONFIG_FINGERPRINT: self._transport.config_fingerprint,
            }
        )

    def _send(self, msg: dict) -> None:
        if self._conn is None or not self._send_ready:
            logger.debug(
                "P2PSession %s: queueing %s (ready=%s queue_depth=%d)",
                self.peer_id,
                msg.get(TYPE_KEY),
                self._send_ready,
                len(self._queued) + 1,
            )
            self._queued.append(msg)
            return
        self._do_send(msg)

    def _do_send(self, msg: dict) -> None:
        if self._conn is None:
            return
        try:
            self._conn.send(msg)
            logger.debug("P2PSession %s: sent %s", self.peer_id, msg.get(TYPE_KEY))
        except Exception:
            logger.warning(
                "P2PSession %s: failed to send %s",
                self.peer_id,
                msg.get(TYPE_KEY),
            )
