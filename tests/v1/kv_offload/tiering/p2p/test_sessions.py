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
from vllm.v1.kv_offload.tiering.p2p.session.session import (
    _CANCEL_DRAIN_TIMEOUT_S,
    _InflightXfer,
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

    def poll(self):
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
        loads, _ = session.poll()
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
        loads, _ = session.poll()
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
        session._inbound["req-1"].submitted_at = time.monotonic() - 60.0
        session.poll()
        abort = conn._sent[-1]
        assert abort[TYPE_KEY] == AbortFetchMsg.TYPE


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
        _, stores = session.poll()
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
        assert "req-1" not in session._pending_aborts

    def test_abort_fetch_defers_ack_when_cancel_pending(self):
        """If cancel(mode='wait') reports still-inflight tids, the ack is
        deferred and the abort is parked in _pending_aborts."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        # Seed an inflight transfer for req-1 that the transport pretends
        # cannot be canceled yet.
        tid = 42
        session._inflight[tid] = _InflightXfer(
            kv_request_id="req-1", block_count=1, job_ids={1}
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
        assert "req-1" in session._pending_aborts
        # First attempt happens inside _on_abort_fetch; the per-tick
        # drain runs again at the end of poll() — both are wait-mode.
        assert all(mode == "wait" for _, mode in transport._cancel_calls)
        assert tid in session._inflight  # still tracked

    def test_abort_fetch_acks_after_drain(self):
        """Once the transport reports the tid as DONE the parked abort
        completes and AbortAckMsg is sent."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        tid = 42
        session._inflight[tid] = _InflightXfer(
            kv_request_id="req-1", block_count=1, job_ids={1}
        )
        transport._cancel_still_inflight.add(tid)

        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        assert "req-1" in session._pending_aborts

        # Backend finishes draining: transport.poll() will return tid as
        # DONE, and the next cancel(mode='wait') call sees it's gone.
        transport._cancel_still_inflight.discard(tid)
        transport._poll_done.append(tid)

        session.poll()

        ack = next(m for m in conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE)
        assert ack[AbortAckMsg.KV_REQUEST_ID] == "req-1"
        assert "req-1" not in session._pending_aborts
        assert tid not in session._inflight

    def test_abort_fetch_force_cancels_after_timeout(self):
        """If wait-mode never drains, the deadline forces immediate
        cancel and an ack is still sent."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        tid = 42
        session._inflight[tid] = _InflightXfer(
            kv_request_id="req-1", block_count=1, job_ids={1}
        )
        transport._cancel_still_inflight.add(tid)

        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        assert "req-1" in session._pending_aborts
        # Backdate past the drain deadline.
        session._pending_aborts["req-1"] = (
            time.monotonic() - _CANCEL_DRAIN_TIMEOUT_S - 1.0
        )
        # Even if the transport still claims it can't cancel, the
        # session must force-pop and ack.
        transport._cancel_calls.clear()

        session.poll()

        ack = next(m for m in conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE)
        assert ack[AbortAckMsg.KV_REQUEST_ID] == "req-1"
        assert "req-1" not in session._pending_aborts
        assert tid not in session._inflight
        assert ([tid], "immediate") in transport._cancel_calls

    def test_abort_fetch_idempotent_while_draining(self):
        """Receiving AbortFetchMsg twice for the same kv_request_id
        keeps a single pending entry and produces a single ack."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        tid = 42
        session._inflight[tid] = _InflightXfer(
            kv_request_id="req-1", block_count=1, job_ids={1}
        )
        transport._cancel_still_inflight.add(tid)

        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        first_started_at = session._pending_aborts["req-1"]

        # Second AbortFetchMsg for the same kv_request_id while still
        # draining must not reset the deadline.
        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        assert session._pending_aborts["req-1"] == first_started_at

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
        session._store_jobs[1] = time.monotonic() - 60.0
        _, stores = session.poll()
        assert StoreResult(job_id=1, success=False) in stores


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
        assert "req-1" in session._outbound

        session.finish_request("req-1")

        msg = self._last_transfer_done(conn)
        assert msg is not None
        assert msg[TransferDoneMsg.KV_REQUEST_ID] == "req-1"
        assert msg[TransferDoneMsg.SUCCESS] is False
        assert "req-1" not in session._outbound

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
        assert "req-1" in session._outbound
        assert session._outbound["req-1"].finishing

        # Last inflight settles -> early-fail fires.
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        session.poll()

        msg = self._last_transfer_done(conn)
        assert msg is not None
        assert msg[TransferDoneMsg.KV_REQUEST_ID] == "req-1"
        assert msg[TransferDoneMsg.SUCCESS] is False
        assert "req-1" not in session._outbound

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
        assert "req-1" in session._outbound
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
        assert "req-2" not in session._outbound

    def test_unknown_request_is_noop(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        before = len(conn._sent)
        session.finish_request("never-existed")
        assert len(conn._sent) == before


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
        loads, stores = session.poll()

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
        loads, stores = session.poll()
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
        loads, stores = session.poll()
        assert loads == []
        assert stores == []

    def test_empty_message(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({})
        loads, stores = session.poll()
        assert loads == []
        assert stores == []

    def test_non_dict_message(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn._inbox.append(42)  # type: ignore[arg-type]
        loads, stores = session.poll()
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

    def test_transfer_done_missing_kv_request_id(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({TYPE_KEY: TransferDoneMsg.TYPE, TransferDoneMsg.SUCCESS: True})
        loads, _ = session.poll()
        assert loads == []

    def test_duplicate_connect_ack(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "peer:8000"})
        session.poll()
        assert session.ready


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
