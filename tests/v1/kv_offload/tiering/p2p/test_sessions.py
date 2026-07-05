# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the unified bidirectional P2PSession.

A P2PSession owns a single ControlConnection and dispatches every message
type — both client-role (FetchMsg / TransferDoneMsg / AbortAck) and
server-role (FetchMsg / TransferDoneMsg / AbortAck from the peer's
perspective). These tests exercise both flows independently and the
bidirectional case where one session simultaneously serves a fetch and
completes its own load.
"""

from __future__ import annotations

import time

import pytest

from vllm.v1.kv_offload.tiering.p2p.session import (
    LoadResult,
    P2PSession,
    StoreResult,
)
from vllm.v1.kv_offload.tiering.p2p.session.client import (
    _ABORT_ACK_TIMEOUT_S,
    _LOAD_TIMEOUT_S,
)
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortAckMsg,
    AbortFetchMsg,
    ConnectAckMsg,
    ConnectMsg,
    DisconnectMsg,
    FetchMsg,
    TransferDoneMsg,
)
from vllm.v1.kv_offload.tiering.p2p.session.server import (
    _CANCEL_DRAIN_TIMEOUT_S,
    _InflightXfer,
)
from vllm.v1.kv_offload.tiering.p2p.session.session import (
    _MAX_CONSECUTIVE_DISPATCH_ERRORS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeDataTransport:
    """Minimal fake DataTransport for testing sessions."""

    def __init__(
        self,
        base_addr: int = 0x1000,
        num_blocks: int = 16,
        block_len: int = 4096,
        config_fingerprint: str = "",
    ) -> None:
        self._base_addr = base_addr
        self._num_blocks = num_blocks
        self._block_len = block_len
        self._config_fingerprint = config_fingerprint
        self._remote_peers: dict[str, dict] = {}
        self._transfers: dict[int, tuple] = {}
        self._next_id = 0
        self._poll_done: list[int] = []
        self._poll_failed: list[int] = []
        self._cancel_still_inflight: set[int] = set()
        self._cancel_calls: list[tuple[list[int], str]] = []

    @property
    def base_addr(self) -> int:
        return self._base_addr

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def block_len(self) -> int:
        return self._block_len

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    def get_agent_metadata(self) -> bytes:
        return b"fake-metadata"

    def add_remote_peer(
        self, peer_id, agent_metadata, base_addr, num_blocks, block_len
    ) -> None:
        self._remote_peers[peer_id] = {
            ConnectMsg.AGENT_METADATA: agent_metadata,
            ConnectMsg.BASE_ADDR: base_addr,
            "num_blocks": num_blocks,
            "block_len": block_len,
        }

    def remove_remote_peer(self, peer_id: str) -> None:
        self._remote_peers.pop(peer_id, None)

    def write_blocks(self, peer_id, local_idxs, remote_idxs) -> int | None:
        if peer_id not in self._remote_peers:
            return None
        tid = self._next_id
        self._next_id += 1
        self._transfers[tid] = (peer_id, local_idxs, remote_idxs)
        return tid

    def poll(self, owner=None):
        from vllm.v1.kv_offload.tiering.p2p.data.base import PollResult

        result = PollResult(done=list(self._poll_done), failed=list(self._poll_failed))
        self._poll_done.clear()
        self._poll_failed.clear()
        return result

    def cancel(self, transfer_ids, mode: str = "immediate") -> list[int]:
        ids = list(transfer_ids)
        self._cancel_calls.append((ids, mode))
        if mode == "wait":
            still: list[int] = []
            for tid in ids:
                if tid in self._cancel_still_inflight:
                    still.append(tid)
                else:
                    self._transfers.pop(tid, None)
            return still
        for tid in ids:
            self._transfers.pop(tid, None)
            self._cancel_still_inflight.discard(tid)
        return []

    def close(self) -> None:
        pass


class FakeConnection:
    """Fake ControlConnection that captures sent messages."""

    def __init__(self, peer_id: str = "peer:8000") -> None:
        self.peer_id = peer_id
        self._inbox: list[dict] = []
        self._sent: list[dict] = []
        self._closed = False

    @property
    def alive(self) -> bool:
        return not self._closed

    def send(self, msg: dict) -> None:
        self._sent.append(msg)

    def recv(self) -> list[dict]:
        msgs = self._inbox
        self._inbox = []
        return msgs

    def enqueue(self, msg: dict) -> None:
        self._inbox.append(msg)

    def mark_dead(self) -> None:
        self._closed = True

    def close(self) -> None:
        self._closed = True


def _peer_connect_msg(
    peer_id: str = "peer:8000",
    block_len: int = 4096,
    fingerprint: str | None = None,
) -> dict:
    """Build a ConnectMsg as if the peer sent it."""
    msg = {
        TYPE_KEY: ConnectMsg.TYPE,
        ConnectMsg.PEER_ID: peer_id,
        ConnectMsg.AGENT_METADATA: b"peer-metadata",
        ConnectMsg.BASE_ADDR: 0x2000,
        ConnectMsg.NUM_BLOCKS: 16,
        ConnectMsg.BLOCK_LEN: block_len,
    }
    if fingerprint is not None:
        msg[ConnectMsg.CONFIG_FINGERPRINT] = fingerprint
    return msg


def _make_session(
    conn: FakeConnection | None = None,
    transport: FakeDataTransport | None = None,
    peer_id: str = "peer:8000",
    local_id: str = "local:9000",
) -> tuple[P2PSession, FakeConnection, FakeDataTransport]:
    if conn is None:
        conn = FakeConnection(peer_id=peer_id)
    if transport is None:
        transport = FakeDataTransport()
    session = P2PSession(
        peer_id=peer_id,
        local_id=local_id,
        transport=transport,  # type: ignore[arg-type]
        local_block_len=transport.block_len,
        conn=conn,  # type: ignore[arg-type]
    )
    return session, conn, transport


def _activate(
    session: P2PSession, conn: FakeConnection, peer_id: str = "peer:8000"
) -> None:
    """Drive the handshake: peer sends ConnectMsg + ConnectAckMsg."""
    conn.enqueue(_peer_connect_msg(peer_id=peer_id))
    conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: peer_id})
    session.poll()


# ---------------------------------------------------------------------------
# Connect / handshake
# ---------------------------------------------------------------------------


class TestConnectHandshake:
    def test_connect_msg_sent_on_creation(self):
        """Session sends its own ConnectMsg on connection."""
        session, conn, _ = _make_session()
        assert len(conn._sent) == 1
        msg = conn._sent[0]
        assert msg[TYPE_KEY] == ConnectMsg.TYPE
        assert msg[ConnectMsg.PEER_ID] == "local:9000"
        assert msg[ConnectMsg.NUM_BLOCKS] == 16
        assert msg[ConnectMsg.BLOCK_LEN] == 4096
        assert ConnectMsg.AGENT_METADATA in msg

    def test_peer_connect_triggers_add_remote_and_ack(self):
        """Receiving ConnectMsg registers peer and replies with ConnectAck."""
        session, conn, transport = _make_session()
        conn.enqueue(_peer_connect_msg())
        session.poll()
        assert "peer:8000" in transport._remote_peers
        ack = next(m for m in conn._sent if m[TYPE_KEY] == ConnectAckMsg.TYPE)
        assert ack[ConnectAckMsg.PEER_ID] == "local:9000"

    def test_connect_ack_makes_session_ready(self):
        """Session.ready becomes True after ConnectAckMsg."""
        session, conn, _ = _make_session()
        assert not session.ready
        conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "peer:8000"})
        session.poll()
        assert session.ready

    def test_messages_queued_before_ack_flush_on_ack(self):
        """Outgoing messages sent before ConnectAck are flushed after."""
        session, conn, _ = _make_session()
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        # Before ack: only our ConnectMsg was sent.
        assert len(conn._sent) == 1
        assert conn._sent[0][TYPE_KEY] == ConnectMsg.TYPE
        # Ack arrives.
        conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "peer:8000"})
        session.poll()
        # Queued fetch is now sent.
        assert any(m[TYPE_KEY] == FetchMsg.TYPE for m in conn._sent)

    def test_block_len_mismatch_marks_dead(self):
        """Mismatched block_len rejects peer and marks connection dead."""
        session, conn, transport = _make_session()
        conn.enqueue(_peer_connect_msg(block_len=8192))  # mismatch
        session.poll()
        assert "peer:8000" not in transport._remote_peers
        assert not session.alive

    def test_config_fingerprint_mismatch_marks_dead(self):
        """Mismatched config fingerprint rejects peer."""
        transport = FakeDataTransport(config_fingerprint="abc123")
        session, conn, _ = _make_session(transport=transport)
        conn.enqueue(_peer_connect_msg(fingerprint="different"))
        session.poll()
        assert "peer:8000" not in transport._remote_peers
        assert not session.alive

    def test_config_fingerprint_match_succeeds(self):
        """Matching fingerprints register the peer."""
        transport = FakeDataTransport(config_fingerprint="same_fp")
        session, conn, _ = _make_session(transport=transport)
        conn.enqueue(_peer_connect_msg(fingerprint="same_fp"))
        session.poll()
        assert "peer:8000" in transport._remote_peers
        assert session.alive

    def test_missing_fingerprint_allowed(self):
        """Missing fingerprint on either side is allowed."""
        transport = FakeDataTransport(config_fingerprint="abc123")
        session, conn, _ = _make_session(transport=transport)
        conn.enqueue(_peer_connect_msg())  # no fingerprint
        session.poll()
        assert "peer:8000" in transport._remote_peers


# ---------------------------------------------------------------------------
# Client-role flows
# ---------------------------------------------------------------------------


class TestClientFlows:
    def test_request_blocks_sends_fetch(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k1", b"k2"], block_ids=[0, 1]
        )
        lookup = conn._sent[-1]
        assert lookup[TYPE_KEY] == FetchMsg.TYPE
        assert lookup[FetchMsg.KV_REQUEST_ID] == "req-1"
        assert lookup[FetchMsg.BLOCK_HASHES] == [b"k1", b"k2"]
        assert lookup[FetchMsg.BLOCK_INDEXES] == [0, 1]

    def test_transfer_done_success(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.KV_REQUEST_ID: "req-1",
                TransferDoneMsg.SUCCESS: True,
            }
        )
        loads = session.poll().loads
        assert loads == [LoadResult(job_id=1, kv_request_id="req-1", success=True)]

    def test_transfer_done_failure(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.KV_REQUEST_ID: "req-1",
                TransferDoneMsg.SUCCESS: False,
            }
        )
        loads = session.poll().loads
        assert loads == [LoadResult(job_id=1, kv_request_id="req-1", success=False)]

    def test_finish_request_sends_abort(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        session.finish_request("req-1")
        abort = conn._sent[-1]
        assert abort[TYPE_KEY] == AbortFetchMsg.TYPE
        assert abort[AbortFetchMsg.KV_REQUEST_ID] == "req-1"

    def test_load_timeout_sends_abort(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        session._client._inbound["req-1"].submitted_at = time.monotonic() - 60.0
        session.poll()
        abort = conn._sent[-1]
        assert abort[TYPE_KEY] == AbortFetchMsg.TYPE

    def test_load_abort_ack_timeout_surfaces_failure(self):
        """After load timeout sends AbortFetch, if no AbortAck arrives within
        _ABORT_ACK_TIMEOUT_S the request is surfaced as failed and removed
        from _inbound — the engine cannot wait forever on a peer that won't
        ack.
        """
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=7, kv_request_id="req-7", keys=[b"k"], block_ids=[0]
        )
        # 1) Trip the load timeout to send AbortFetch and stamp aborted_at.
        session._client._inbound["req-7"].submitted_at = (
            time.monotonic() - _LOAD_TIMEOUT_S - 1.0
        )
        loads = session.poll().loads
        assert loads == []
        assert any(
            m.get(TYPE_KEY) == AbortFetchMsg.TYPE
            and m[AbortFetchMsg.KV_REQUEST_ID] == "req-7"
            for m in conn._sent
        )
        assert session._client._inbound["req-7"].aborted_at is not None

        # 2) Now backdate aborted_at past the abort-ack timeout. No ack ever
        # arrived from the peer.
        session._client._inbound["req-7"].aborted_at = (
            time.monotonic() - _ABORT_ACK_TIMEOUT_S - 1.0
        )
        loads = session.poll().loads
        assert loads == [LoadResult(job_id=7, kv_request_id="req-7", success=False)]
        assert "req-7" not in session._client._inbound

    def test_load_abort_ack_clears_request(self):
        """After load timeout sends AbortFetch, an arriving AbortAckMsg from
        the peer surfaces the failure cleanly and removes the request from
        _inbound — covers the on_abort_ack arrival path."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=8, kv_request_id="req-8", keys=[b"k"], block_ids=[0]
        )
        session._client._inbound["req-8"].submitted_at = (
            time.monotonic() - _LOAD_TIMEOUT_S - 1.0
        )
        # First poll: AbortFetch goes out.
        session.poll()
        assert session._client._inbound["req-8"].aborted_at is not None

        # Peer acks the abort.
        conn.enqueue(
            {
                TYPE_KEY: AbortAckMsg.TYPE,
                AbortAckMsg.KV_REQUEST_ID: "req-8",
            }
        )
        loads = session.poll().loads
        assert loads == [LoadResult(job_id=8, kv_request_id="req-8", success=False)]
        assert "req-8" not in session._client._inbound


# ---------------------------------------------------------------------------
# Server-role flows
# ---------------------------------------------------------------------------


class TestServerFlows:
    def test_store_then_fetch_matches(self):
        """Blocks stored before fetch demand are matched on demand arrival."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1", b"k2"], [0, 1], job_id=1)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1", b"k2"],
                FetchMsg.BLOCK_INDEXES: [10, 11],
            }
        )
        session.poll()
        assert len(transport._transfers) == 1
        _, (peer, local, remote) = next(iter(transport._transfers.items()))
        assert local == [0, 1]
        assert remote == [10, 11]

    def test_fetch_then_store_matches(self):
        """Fetch demand registered before store; store fulfills it."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        assert len(transport._transfers) == 0
        session.add_stored_blocks("req-1", [b"k1"], [3], job_id=1)
        assert len(transport._transfers) == 1

    def test_transfer_completion_emits_store_result_and_done(self):
        """Completed transfer reports StoreResult and sends TransferDoneMsg."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        stores = session.poll().stores
        assert StoreResult(job_id=1, success=True) in stores
        assert any(m[TYPE_KEY] == TransferDoneMsg.TYPE for m in conn._sent)

    def test_abort_fetch_replies_with_ack(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        ack = next(m for m in conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE)
        assert ack[AbortAckMsg.KV_REQUEST_ID] == "req-1"
        assert "req-1" not in session._server._pending_aborts

    def test_abort_fetch_defers_ack_when_cancel_pending(self):
        """If cancel(mode='wait') reports still-inflight tids, the ack is
        deferred and the abort is parked in _pending_aborts."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        # Seed an inflight transfer for req-1 that the transport pretends
        # cannot be canceled yet.
        tid = 42
        session._server._inflight_add(
            tid,
            _InflightXfer(kv_request_id="req-1", block_count=1, job_ids={1}),
        )
        transport._cancel_still_inflight.add(tid)

        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()

        assert not any(m[TYPE_KEY] == AbortAckMsg.TYPE for m in conn._sent)
        assert "req-1" in session._server._pending_aborts
        # First attempt happens inside _on_abort_fetch; the per-tick
        # drain runs again at the end of poll() — both are wait-mode.
        assert all(mode == "wait" for _, mode in transport._cancel_calls)
        assert tid in session._server._inflight  # still tracked

    def test_abort_fetch_acks_after_drain(self):
        """Once the transport reports the tid as DONE the parked abort
        completes and AbortAckMsg is sent."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        tid = 42
        session._server._inflight_add(
            tid,
            _InflightXfer(kv_request_id="req-1", block_count=1, job_ids={1}),
        )
        transport._cancel_still_inflight.add(tid)

        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        assert "req-1" in session._server._pending_aborts

        # Backend finishes draining: transport.poll() will return tid as
        # DONE, and the next cancel(mode='wait') call sees it's gone.
        transport._cancel_still_inflight.discard(tid)
        transport._poll_done.append(tid)

        session.poll()

        ack = next(m for m in conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE)
        assert ack[AbortAckMsg.KV_REQUEST_ID] == "req-1"
        assert "req-1" not in session._server._pending_aborts
        assert tid not in session._server._inflight

    def test_abort_fetch_force_cancels_after_timeout(self):
        """If wait-mode never drains, the deadline forces immediate
        cancel and an ack is still sent."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        tid = 42
        session._server._inflight_add(
            tid,
            _InflightXfer(kv_request_id="req-1", block_count=1, job_ids={1}),
        )
        transport._cancel_still_inflight.add(tid)

        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        assert "req-1" in session._server._pending_aborts
        # Backdate past the drain deadline.
        session._server._pending_aborts["req-1"] = (
            time.monotonic() - _CANCEL_DRAIN_TIMEOUT_S - 1.0
        )
        # Even if the transport still claims it can't cancel, the
        # session must force-pop and ack.
        transport._cancel_calls.clear()

        session.poll()

        ack = next(m for m in conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE)
        assert ack[AbortAckMsg.KV_REQUEST_ID] == "req-1"
        assert "req-1" not in session._server._pending_aborts
        assert tid not in session._server._inflight
        assert ([tid], "immediate") in transport._cancel_calls

    def test_abort_fetch_idempotent_while_draining(self):
        """Receiving AbortFetchMsg twice for the same kv_request_id
        keeps a single pending entry and produces a single ack."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        tid = 42
        session._server._inflight_add(
            tid,
            _InflightXfer(kv_request_id="req-1", block_count=1, job_ids={1}),
        )
        transport._cancel_still_inflight.add(tid)

        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        first_started_at = session._server._pending_aborts["req-1"]

        # Second AbortFetchMsg for the same kv_request_id while still
        # draining must not reset the deadline.
        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        assert session._server._pending_aborts["req-1"] == first_started_at

        # Now let the drain succeed and confirm exactly one ack ever.
        transport._cancel_still_inflight.discard(tid)
        transport._poll_done.append(tid)
        session.poll()

        acks = [m for m in conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE]
        assert len(acks) == 1
        assert acks[0][AbortAckMsg.KV_REQUEST_ID] == "req-1"

    def test_store_timeout(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        # Backdate.
        session._server._store_jobs[1] = time.monotonic() - 60.0
        stores = session.poll().stores
        assert StoreResult(job_id=1, success=False) in stores

    def test_store_timeout_then_late_completion_no_duplicate(self):
        """A job timed out by _timeout_pending_store_jobs must not also
        emit a contradictory StoreResult(success=True) when the transport
        later reports the same transfer as done."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        tid = next(iter(transport._transfers))

        # Backdate the store job so the next poll times it out.
        session._server._store_jobs[1] = time.monotonic() - 60.0
        stores = session.poll().stores
        assert StoreResult(job_id=1, success=False) in stores
        assert StoreResult(job_id=1, success=True) not in stores

        # Transport later reports the same transfer as done — must not
        # emit a second (contradictory) StoreResult for job_id=1.
        transport._poll_done.append(tid)
        stores = session.poll().stores
        assert all(s.job_id != 1 for s in stores), (
            f"unexpected duplicate StoreResult after timeout: {stores}"
        )

    def test_store_timeout_then_late_failure_no_duplicate(self):
        """Symmetric guard: a timed-out job must not also emit a second
        StoreResult(success=False) when the transport later reports the
        same transfer as failed."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        tid = next(iter(transport._transfers))

        session._server._store_jobs[1] = time.monotonic() - 60.0
        stores = session.poll().stores
        assert [s for s in stores if s.job_id == 1] == [
            StoreResult(job_id=1, success=False)
        ]

        transport._poll_failed.append(tid)
        stores = session.poll().stores
        assert all(s.job_id != 1 for s in stores), (
            f"unexpected duplicate StoreResult after timeout: {stores}"
        )


# ---------------------------------------------------------------------------
# finish_request server-role early-fail flow
# ---------------------------------------------------------------------------


class TestFinishRequestServerSide:
    def _last_transfer_done(self, conn: FakeConnection) -> dict | None:
        for msg in reversed(conn._sent):
            if msg[TYPE_KEY] == TransferDoneMsg.TYPE:
                return msg
        return None

    def test_no_inflight_unmatched_demand_sends_failure(self):
        """finish_request with unmatched demand and no inflight ->
        immediate TransferDoneMsg(success=False); _outbound cleared."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        # Decoder demanded a block we never stored.
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        assert "req-1" in session._server._outbound

        session.finish_request("req-1")

        msg = self._last_transfer_done(conn)
        assert msg is not None
        assert msg[TransferDoneMsg.KV_REQUEST_ID] == "req-1"
        assert msg[TransferDoneMsg.SUCCESS] is False
        assert "req-1" not in session._server._outbound

    def test_with_inflight_defers_then_fires_on_last_transfer(self):
        """finish_request with inflight defers; last transfer fires the
        early-fail message and clears _outbound."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        # Demand 2 blocks; we store 1 (kicks one inflight transfer).
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1", b"k2"],
                FetchMsg.BLOCK_INDEXES: [10, 11],
            }
        )
        session.poll()
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        assert len(transport._transfers) == 1

        # finish_request while inflight: no early-fail yet.
        before = len(conn._sent)
        session.finish_request("req-1")
        assert len(conn._sent) == before
        assert "req-1" in session._server._outbound
        assert session._server._outbound["req-1"].finishing

        # Last inflight settles -> early-fail fires.
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        session.poll()

        msg = self._last_transfer_done(conn)
        assert msg is not None
        assert msg[TransferDoneMsg.KV_REQUEST_ID] == "req-1"
        assert msg[TransferDoneMsg.SUCCESS] is False
        assert "req-1" not in session._server._outbound

    def test_full_demand_satisfied_still_sends_success(self):
        """finish_request must not override a fully-satisfied transfer:
        when remaining hits 0, success=True still fires."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [10],
            }
        )
        session.poll()
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        # Mark finishing (e.g., on_request_finished racing with the last
        # store) — but all demand is satisfied.
        session.finish_request("req-1")

        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        session.poll()

        msg = next(m for m in conn._sent if m[TYPE_KEY] == TransferDoneMsg.TYPE)
        assert msg[TransferDoneMsg.SUCCESS] is True

    def test_prefiller_first_finish_before_fetch(self):
        """Prefiller-first: finish_request runs before the decoder's
        fetch arrives. State is held until fetch, then
        finalized — success=True if all demand was matched against
        available blocks, else success=False."""
        # Case A: all demand satisfied by available blocks.
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        # finish_request first — no demand received yet -> defer.
        session.finish_request("req-1")
        assert "req-1" in session._server._outbound
        # Fetch arrives now: demand fully satisfied by available.
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [10],
            }
        )
        session.poll()
        # The transfer was inflight; finalize it.
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        session.poll()
        msg = next(m for m in conn._sent if m[TYPE_KEY] == TransferDoneMsg.TYPE)
        assert msg[TransferDoneMsg.SUCCESS] is True

        # Case B: demand exceeds available -> early-fail fires from fetch.
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-2", [b"k1"], [0], job_id=2)
        session.finish_request("req-2")
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-2",
                FetchMsg.BLOCK_HASHES: [b"k1", b"k2"],
                FetchMsg.BLOCK_INDEXES: [10, 11],
            }
        )
        session.poll()
        # Inflight for k1 still in flight; nothing yet for the early-fail
        # — the same code path will fire from _collect_store_results.
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        session.poll()
        msg = next(m for m in conn._sent if m[TYPE_KEY] == TransferDoneMsg.TYPE)
        assert msg[TransferDoneMsg.SUCCESS] is False
        assert "req-2" not in session._server._outbound

    def test_unknown_request_is_noop(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        before = len(conn._sent)
        session.finish_request("never-existed")
        assert len(conn._sent) == before

    def test_finish_request_no_inflight_emits_store_failure(self):
        """finish_request with a stored-but-unmatched job and no inflight ->
        TransferDoneMsg(success=False) AND deferred
        StoreResult(success=False) for the submit_store'd job, instead of
        the 30s _STORE_TIMEOUT_S path."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        # Decoder demanded b"demand"; we never stored it.
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"demand"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        # We did submit_store a different block — goes to available, never
        # matches demand. Without the shortcut, job 42 sits in _store_jobs
        # for _STORE_TIMEOUT_S.
        session.add_stored_blocks("req-1", [b"unrelated"], [0], job_id=42)
        assert session._server._outbound["req-1"].pending_job_ids == {42}

        session.finish_request("req-1")

        # Peer notified immediately with success=False (remaining > 0).
        msg = self._last_transfer_done(conn)
        assert msg is not None
        assert msg[TransferDoneMsg.SUCCESS] is False
        assert "req-1" not in session._server._outbound

        # Local store job surfaces on the next poll, success=False.
        stores = session.poll().stores
        assert StoreResult(job_id=42, success=False) in stores
        assert 42 not in session._server._store_jobs

    def test_finish_request_remaining_zero_emits_success_via_inflight(self):
        """Deferred-via-inflight path: finish_request with inflight, last
        transfer drains remaining to 0 -> TransferDoneMsg(success=True)
        AND StoreResult(success=True)."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [10],
            }
        )
        session.poll()
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=7)
        # finish_request races with the inflight transfer.
        session.finish_request("req-1")
        assert "req-1" in session._server._outbound  # deferred

        # Last inflight completes -> _finalize_outbound(success=True) fires.
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        stores = session.poll().stores

        msg = self._last_transfer_done(conn)
        assert msg is not None
        assert msg[TransferDoneMsg.SUCCESS] is True
        assert StoreResult(job_id=7, success=True) in stores
        assert "req-1" not in session._server._outbound

    def test_write_blocks_failure_finalizes_with_failure(self):
        """write_blocks returning None must not leave the request hanging.

        The matched blocks are gone from req.demanded but no inflight
        will satisfy them, so remaining > 0 forever. Setting finishing
        and calling _finalize_outbound(success=False) immediately (no
        other inflight) tells the peer + emits StoreResult(success=False)
        without waiting on _STORE_TIMEOUT_S or _LOAD_TIMEOUT_S.
        """
        session, conn, transport = _make_session()
        _activate(session, conn)
        # Decoder demands b"k1".
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [10],
            }
        )
        session.poll()
        # Force write_blocks to fail on the next call.
        transport.write_blocks = lambda *a, **kw: None  # type: ignore[assignment]

        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=42)

        # Outbound was finalized immediately (no other inflight).
        assert "req-1" not in session._server._outbound
        # Peer notified with success=False.
        msg = next(m for m in conn._sent if m[TYPE_KEY] == TransferDoneMsg.TYPE)
        assert msg[TransferDoneMsg.KV_REQUEST_ID] == "req-1"
        assert msg[TransferDoneMsg.SUCCESS] is False
        # Local store job surfaces on the next poll.
        stores = session.poll().stores
        assert StoreResult(job_id=42, success=False) in stores
        assert 42 not in session._server._store_jobs

    def test_partial_match_completes_in_two_rounds(self):
        """Peer demand for [k1, k2, k3]; first round only k1 is available,
        second round adds k2 and k3. Each round transfers what's matched
        and the request finalizes with success once remaining hits zero.
        """
        session, conn, transport = _make_session()
        _activate(session, conn)

        # Peer fetches three blocks.
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1", b"k2", b"k3"],
                FetchMsg.BLOCK_INDEXES: [10, 11, 12],
            }
        )
        session.poll()
        # Demand registered, no matches yet.
        assert session._server._inflight == {}
        outbound = session._server._outbound["req-1"]
        assert outbound.remaining == 3
        assert set(outbound.demanded.keys()) == {b"k1", b"k2", b"k3"}

        # Round 1: only k1 is stored locally.
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=100)
        # One inflight transfer for k1.
        assert len(session._server._inflight) == 1
        tid_1 = next(iter(session._server._inflight))
        assert transport._transfers[tid_1][1] == [0]
        assert transport._transfers[tid_1][2] == [10]
        # k2 and k3 still demanded.
        assert set(outbound.demanded.keys()) == {b"k2", b"k3"}

        # Transfer 1 completes.
        transport._poll_done.append(tid_1)
        stores = session.poll().stores
        assert StoreResult(job_id=100, success=True) in stores
        assert session._server._inflight == {}
        assert outbound.remaining == 2
        # Not yet finalized — still 2 blocks demanded.
        assert "req-1" in session._server._outbound

        # Round 2: k2 and k3 arrive together.
        session.add_stored_blocks("req-1", [b"k2", b"k3"], [1, 2], job_id=200)
        assert len(session._server._inflight) == 1
        tid_2 = next(iter(session._server._inflight))
        assert tid_2 != tid_1
        assert sorted(transport._transfers[tid_2][1]) == [1, 2]

        # Transfer 2 completes — request now fully satisfied.
        transport._poll_done.append(tid_2)
        stores = session.poll().stores
        assert StoreResult(job_id=200, success=True) in stores
        # _finalize_outbound fired — request gone, peer notified with success.
        assert "req-1" not in session._server._outbound
        done = next(m for m in conn._sent if m.get(TYPE_KEY) == TransferDoneMsg.TYPE)
        assert done[TransferDoneMsg.KV_REQUEST_ID] == "req-1"
        assert done[TransferDoneMsg.SUCCESS] is True

    def test_write_blocks_failure_finalizes_after_last_inflight_completes(self):
        """write_blocks returns None on a SECOND match while a first transfer
        is still inflight. The request should NOT finalize until the inflight
        completes, then the elif branch in collect_results
        (``finishing and not _has_inflight_for(...)``) finalizes it as failure.
        """
        session, conn, transport = _make_session()
        _activate(session, conn)

        # Peer demands two blocks.
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1", b"k2"],
                FetchMsg.BLOCK_INDEXES: [10, 11],
            }
        )
        session.poll()

        # Round 1: k1 transfers cleanly.
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=100)
        assert len(session._server._inflight) == 1
        tid_1 = next(iter(session._server._inflight))
        outbound = session._server._outbound["req-1"]
        assert outbound.remaining == 2  # decrement happens on completion
        assert outbound.finishing is False

        # Round 2: write_blocks fails for k2 while transfer_1 is still inflight.
        transport.write_blocks = lambda *a, **kw: None  # type: ignore[assignment]
        session.add_stored_blocks("req-1", [b"k2"], [1], job_id=200)
        # No new transfer was registered.
        assert list(session._server._inflight.keys()) == [tid_1]
        # Marked finishing, but NOT finalized yet (transfer_1 still inflight).
        assert outbound.finishing is True
        assert "req-1" in session._server._outbound
        done_msgs = [m for m in conn._sent if m.get(TYPE_KEY) == TransferDoneMsg.TYPE]
        assert done_msgs == []

        # Transfer 1 completes — now ``_has_inflight_for("req-1")`` is False
        # and the elif branch in collect_results fires _finalize(success=False).
        # The k1 success result is direct; the k2 failure result is queued
        # in _pending_store_results and surfaces on the NEXT poll.
        transport._poll_done.append(tid_1)
        stores_first = session.poll().stores
        assert StoreResult(job_id=100, success=True) in stores_first
        # Outbound state cleaned up; peer notified with success=False.
        assert "req-1" not in session._server._outbound
        done = next(m for m in conn._sent if m.get(TYPE_KEY) == TransferDoneMsg.TYPE)
        assert done[TransferDoneMsg.KV_REQUEST_ID] == "req-1"
        assert done[TransferDoneMsg.SUCCESS] is False

        # Next poll drains the queued failure.
        stores_second = session.poll().stores
        assert StoreResult(job_id=200, success=False) in stores_second


# ---------------------------------------------------------------------------
# Bidirectional — the case the unification is meant to fix
# ---------------------------------------------------------------------------


class TestBidirectional:
    def test_session_handles_both_roles_concurrently(self):
        """Single session simultaneously serves a fetch and completes a load.

        This is the regression test for the unification: with the old
        split design the inbound FetchMsg would be dispatched to a
        client-only session and dropped (or a server-only session would
        miss the TransferDoneMsg). One unified session handles both.
        """
        session, conn, transport = _make_session()
        _activate(session, conn)

        # Server role: the peer fetches a block from us.
        session.add_stored_blocks("req-srv", [b"served"], [0], job_id=100)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-srv",
                FetchMsg.BLOCK_HASHES: [b"served"],
                FetchMsg.BLOCK_INDEXES: [7],
            }
        )

        # Client role: we ask the peer for a different block.
        session.request_blocks(
            job_id=200, kv_request_id="req-cli", keys=[b"loaded"], block_ids=[3]
        )

        # Both flows progress in the same poll.
        session.poll()

        # Server side: write_blocks was submitted.
        assert len(transport._transfers) == 1
        # Client side: the lookup was sent.
        assert any(
            m[TYPE_KEY] == FetchMsg.TYPE and m[FetchMsg.KV_REQUEST_ID] == "req-cli"
            for m in conn._sent
        )

        # Peer now signals: server-side transfer completes AND a
        # TransferDoneMsg arrives for our client-side request, all in
        # one batch on the same connection.
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.KV_REQUEST_ID: "req-cli",
                TransferDoneMsg.SUCCESS: True,
            }
        )
        result_ = session.poll()
        loads = result_.loads
        stores = result_.stores

        assert LoadResult(job_id=200, kv_request_id="req-cli", success=True) in loads
        assert StoreResult(job_id=100, success=True) in stores


# ---------------------------------------------------------------------------
# Pending sessions (no connection yet)
# ---------------------------------------------------------------------------


class TestPendingSession:
    def test_pending_session_buffers_stored_blocks(self):
        """Pending session accepts add_stored_blocks but cannot send."""
        transport = FakeDataTransport()
        session = P2PSession(
            peer_id="peer:8000",
            local_id="local:9000",
            transport=transport,  # type: ignore[arg-type]
            local_block_len=4096,
            conn=None,
        )
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        result_ = session.poll()
        loads = result_.loads
        stores = result_.stores
        assert loads == []
        assert stores == []
        assert not session.connected
        assert session.alive

    def test_attach_connection_sends_connect(self):
        """attach_connection triggers our ConnectMsg send."""
        transport = FakeDataTransport()
        session = P2PSession(
            peer_id="peer:8000",
            local_id="local:9000",
            transport=transport,  # type: ignore[arg-type]
            local_block_len=4096,
            conn=None,
        )
        conn = FakeConnection(peer_id="peer:8000")
        session.attach_connection(conn)  # type: ignore[arg-type]
        assert conn._sent
        assert conn._sent[0][TYPE_KEY] == ConnectMsg.TYPE

    def test_attach_connection_twice_raises(self):
        """attach_connection on an already-connected session raises."""
        session, conn, _ = _make_session()
        with pytest.raises(ValueError, match="already connected"):
            session.attach_connection(FakeConnection())  # type: ignore[arg-type]

    def test_pending_close_returns_pending_stores(self):
        """Closing a pending session reports buffered stores as failed."""
        transport = FakeDataTransport()
        session = P2PSession(
            peer_id="peer:8000",
            local_id="local:9000",
            transport=transport,  # type: ignore[arg-type]
            local_block_len=4096,
            conn=None,
        )
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        session.add_stored_blocks("req-2", [b"k2"], [1], job_id=2)
        failed_loads, failed_stores = session.close()
        assert failed_loads == []
        assert set(failed_stores) == {1, 2}


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------


class TestDisconnect:
    def test_disconnect_marks_session_dead(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({TYPE_KEY: DisconnectMsg.TYPE})
        session.poll()
        assert not session.alive

    def test_close_returns_pending_loads_and_stores(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(1, "req-1", [b"k"], [0])
        session.request_blocks(2, "req-2", [b"k"], [0])
        session.add_stored_blocks("req-srv", [b"k"], [0], job_id=10)
        failed_loads, failed_stores = session.close()
        assert set(failed_loads) == {(1, "req-1"), (2, "req-2")}
        assert set(failed_stores) == {10}


# ---------------------------------------------------------------------------
# Adversarial / malformed messages
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_unknown_message_type_logged(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({TYPE_KEY: "evil_command"})
        result_ = session.poll()
        loads = result_.loads
        stores = result_.stores
        assert loads == []
        assert stores == []

    def test_empty_message(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({})
        result_ = session.poll()
        loads = result_.loads
        stores = result_.stores
        assert loads == []
        assert stores == []

    def test_non_dict_message(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn._inbox.append(42)  # type: ignore[arg-type]
        result_ = session.poll()
        loads = result_.loads
        stores = result_.stores
        assert loads == []
        assert stores == []

    def test_fetch_mismatched_lengths(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-bad",
                FetchMsg.BLOCK_HASHES: [b"k1", b"k2"],
                FetchMsg.BLOCK_INDEXES: [1],
            }
        )
        session.poll()
        assert len(transport._transfers) == 0
        # Protocol violation: session disconnects immediately so the peer
        # can't keep wedging us with malformed traffic.
        assert not session.alive
        assert any(m[TYPE_KEY] == DisconnectMsg.TYPE for m in conn._sent)

    def test_transfer_done_missing_kv_request_id(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({TYPE_KEY: TransferDoneMsg.TYPE, TransferDoneMsg.SUCCESS: True})
        loads = session.poll().loads
        assert loads == []
        assert not session.alive
        assert any(m[TYPE_KEY] == DisconnectMsg.TYPE for m in conn._sent)

    def test_duplicate_connect_ack(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "peer:8000"})
        session.poll()
        assert session.ready


class TestDispatchErrorHandling:
    """Errors raised by message handlers split into two classes:

    - Protocol-contract violations from the peer (ValueError) → disconnect
      on the first occurrence; retrying won't help and may corrupt state.
    - Anything else is treated as an internal bug: log loudly, count, and
      only disconnect once errors arrive in a tight burst. A successful
      dispatch in between resets the counter.
    """

    def test_value_error_disconnects_on_first_occurrence(self):
        """A FetchMsg that fails validate() raises ValueError and must
        terminate the session immediately, with a DisconnectMsg sent."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        # length mismatch → FetchMsg.validate raises ValueError
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-bad",
                FetchMsg.BLOCK_HASHES: [b"k1", b"k2"],
                FetchMsg.BLOCK_INDEXES: [1],
            }
        )
        session.poll()
        assert not session.alive
        assert any(m[TYPE_KEY] == DisconnectMsg.TYPE for m in conn._sent)

    def test_transfer_done_missing_field_disconnects(self):
        """Same contract for a malformed TransferDoneMsg from the peer."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({TYPE_KEY: TransferDoneMsg.TYPE, TransferDoneMsg.SUCCESS: True})
        session.poll()
        assert not session.alive

    def test_internal_error_does_not_disconnect_once(self):
        """A non-ValueError raised by a handler is treated as an internal
        bug: counter increments, session stays alive on a single hit."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated internal bug")

        session._server.on_fetch = _boom  # type: ignore[assignment]
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.BLOCK_HASHES: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [0],
            }
        )
        session.poll()
        assert session.alive
        assert session._dispatch_error_count == 1

    def test_internal_error_threshold_disconnects(self):
        """Once consecutive non-protocol errors hit the threshold, the
        session tears down via _protocol_error."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated internal bug")

        session._server.on_fetch = _boom  # type: ignore[assignment]
        for _ in range(_MAX_CONSECUTIVE_DISPATCH_ERRORS):
            conn.enqueue(
                {
                    TYPE_KEY: FetchMsg.TYPE,
                    FetchMsg.KV_REQUEST_ID: "req-1",
                    FetchMsg.BLOCK_HASHES: [b"k1"],
                    FetchMsg.BLOCK_INDEXES: [0],
                }
            )
        session.poll()
        assert not session.alive
        assert any(m[TYPE_KEY] == DisconnectMsg.TYPE for m in conn._sent)

    def test_internal_error_counter_resets_on_success(self):
        """A successful dispatch between errors prevents the threshold
        from being reached."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        original_on_fetch = session._server.on_fetch

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated internal bug")

        # Alternate (boom, success) (_MAX-1) times: counter rises to 1
        # then resets to 0 each cycle, never reaching the threshold.
        for _ in range(_MAX_CONSECUTIVE_DISPATCH_ERRORS - 1):
            session._server.on_fetch = _boom  # type: ignore[assignment]
            conn.enqueue(
                {
                    TYPE_KEY: FetchMsg.TYPE,
                    FetchMsg.KV_REQUEST_ID: "req-1",
                    FetchMsg.BLOCK_HASHES: [b"k1"],
                    FetchMsg.BLOCK_INDEXES: [0],
                }
            )
            session.poll()
            session._server.on_fetch = original_on_fetch  # type: ignore[assignment]
            # A benign no-op message (unknown type) dispatches cleanly
            # and resets the consecutive-error counter.
            conn.enqueue({TYPE_KEY: "unknown_for_test"})
            session.poll()

        assert session.alive
        assert session._dispatch_error_count == 0


class TestInflightPerReqInvariant:
    """`_inflight_per_req` is the O(1) replacement for the previous
    O(N) scan in `_has_inflight_for`. These tests check that every
    mutation site keeps the counter in sync with `_inflight` and that
    the lookup is correct under high fan-out.
    """

    def test_invariant_holds_through_lifecycle(self):
        """Run a full submit→complete sequence for two concurrent
        kv_request_ids and assert the counter matches `_inflight` at
        every observable step, including the empty-after-finish case."""
        session, conn, transport = _make_session()
        _activate(session, conn)

        def _invariant_holds() -> bool:
            counted = sum(session._server._inflight_per_req.values())
            return counted == len(session._server._inflight) and all(
                v > 0 for v in session._server._inflight_per_req.values()
            )

        assert _invariant_holds()

        # Two requests, two blocks each, all dispatched in one batch.
        session.add_stored_blocks("req-A", [b"a1", b"a2"], [0, 1], job_id=10)
        session.add_stored_blocks("req-B", [b"b1", b"b2"], [2, 3], job_id=11)
        for kv_id, hashes, indexes in (
            ("req-A", [b"a1", b"a2"], [100, 101]),
            ("req-B", [b"b1", b"b2"], [102, 103]),
        ):
            conn.enqueue(
                {
                    TYPE_KEY: FetchMsg.TYPE,
                    FetchMsg.KV_REQUEST_ID: kv_id,
                    FetchMsg.BLOCK_HASHES: hashes,
                    FetchMsg.BLOCK_INDEXES: indexes,
                }
            )
        session.poll()

        assert _invariant_holds()
        assert session._server._has_inflight_for("req-A")
        assert session._server._has_inflight_for("req-B")
        assert not session._server._has_inflight_for("req-C")

        # Complete req-A's transfer first; req-B should still be inflight.
        a_tids = [
            tid
            for tid, x in session._server._inflight.items()
            if x.kv_request_id == "req-A"
        ]
        for tid in a_tids:
            transport._poll_done.append(tid)
        session.poll()

        assert _invariant_holds()
        assert not session._server._has_inflight_for("req-A")
        assert "req-A" not in session._server._inflight_per_req  # entry was removed
        assert session._server._has_inflight_for("req-B")

        # Complete req-B; counter must drain to empty.
        b_tids = [
            tid
            for tid, x in session._server._inflight.items()
            if x.kv_request_id == "req-B"
        ]
        for tid in b_tids:
            transport._poll_done.append(tid)
        session.poll()

        assert _invariant_holds()
        assert session._server._inflight == {}
        assert session._server._inflight_per_req == {}

    def test_has_inflight_for_correct_with_many_requests(self):
        """Populate many inflight xfers across many ids; lookup must
        match the actual presence in `_inflight` for both hits and
        misses. The whole point of the counter is that this lookup is
        constant-time, but we assert correctness, not timing."""
        session, _, _ = _make_session()
        for kv_id_idx in range(100):
            kv_id = f"req-{kv_id_idx}"
            for j in range(10):
                tid = kv_id_idx * 10 + j
                session._server._inflight_add(
                    tid,
                    _InflightXfer(kv_request_id=kv_id, block_count=1, job_ids={tid}),
                )
        assert sum(session._server._inflight_per_req.values()) == len(
            session._server._inflight
        )
        assert session._server._has_inflight_for("req-0")
        assert session._server._has_inflight_for("req-99")
        assert not session._server._has_inflight_for("req-missing")

        # Drain all entries for req-50 and confirm the entry disappears.
        tids_50 = [
            tid
            for tid, x in session._server._inflight.items()
            if x.kv_request_id == "req-50"
        ]
        for tid in tids_50:
            session._server._inflight_pop(tid)
        assert "req-50" not in session._server._inflight_per_req
        assert not session._server._has_inflight_for("req-50")
        # Other ids unaffected.
        assert session._server._has_inflight_for("req-49")


# ---------------------------------------------------------------------------
# Protocol validation tests (unchanged from the prior file)
# ---------------------------------------------------------------------------


class TestConnectMsgValidation:
    def _valid_msg(self) -> dict:
        return {
            TYPE_KEY: ConnectMsg.TYPE,
            ConnectMsg.PEER_ID: "peer:1",
            ConnectMsg.AGENT_METADATA: b"meta",
            ConnectMsg.BASE_ADDR: 0x1000,
            ConnectMsg.NUM_BLOCKS: 8,
            ConnectMsg.BLOCK_LEN: 4096,
        }

    def test_valid_message_passes(self):
        ConnectMsg.validate(self._valid_msg())

    def test_missing_peer_id(self):
        msg = self._valid_msg()
        del msg[ConnectMsg.PEER_ID]
        with pytest.raises(ValueError, match="peer_id"):
            ConnectMsg.validate(msg)

    def test_missing_agent_metadata(self):
        msg = self._valid_msg()
        del msg[ConnectMsg.AGENT_METADATA]
        with pytest.raises(ValueError, match="agent_metadata"):
            ConnectMsg.validate(msg)

    def test_agent_metadata_wrong_type(self):
        msg = self._valid_msg()
        msg[ConnectMsg.AGENT_METADATA] = "not bytes"
        with pytest.raises(ValueError, match="agent_metadata"):
            ConnectMsg.validate(msg)

    def test_base_addr_negative(self):
        msg = self._valid_msg()
        msg[ConnectMsg.BASE_ADDR] = -1
        with pytest.raises(ValueError, match="base_addr"):
            ConnectMsg.validate(msg)

    def test_num_blocks_zero(self):
        msg = self._valid_msg()
        msg[ConnectMsg.NUM_BLOCKS] = 0
        with pytest.raises(ValueError, match="num_blocks"):
            ConnectMsg.validate(msg)

    def test_block_len_zero(self):
        msg = self._valid_msg()
        msg[ConnectMsg.BLOCK_LEN] = 0
        with pytest.raises(ValueError, match="block_len"):
            ConnectMsg.validate(msg)


class TestFetchMsgValidation:
    def _valid_msg(self) -> dict:
        return {
            TYPE_KEY: FetchMsg.TYPE,
            FetchMsg.KV_REQUEST_ID: "req-1",
            FetchMsg.BLOCK_HASHES: [b"k1", b"k2"],
            FetchMsg.BLOCK_INDEXES: [0, 1],
        }

    def test_valid_message_passes(self):
        FetchMsg.validate(self._valid_msg())

    def test_length_mismatch(self):
        msg = self._valid_msg()
        msg[FetchMsg.BLOCK_INDEXES] = [0]
        with pytest.raises(ValueError, match="length mismatch"):
            FetchMsg.validate(msg)

    def test_negative_index(self):
        msg = self._valid_msg()
        msg[FetchMsg.BLOCK_INDEXES] = [0, -1]
        with pytest.raises(ValueError, match="invalid index"):
            FetchMsg.validate(msg)


class TestTransferDoneMsgValidation:
    def test_valid_message_passes(self):
        msg = {
            TYPE_KEY: TransferDoneMsg.TYPE,
            TransferDoneMsg.KV_REQUEST_ID: "req-1",
            TransferDoneMsg.SUCCESS: True,
        }
        TransferDoneMsg.validate(msg)

    def test_success_wrong_type(self):
        msg = {
            TYPE_KEY: TransferDoneMsg.TYPE,
            TransferDoneMsg.KV_REQUEST_ID: "req-1",
            TransferDoneMsg.SUCCESS: 1,
        }
        with pytest.raises(ValueError, match="success"):
            TransferDoneMsg.validate(msg)
