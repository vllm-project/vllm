# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for P2PSecondaryTierManager.

Tests the manager's job routing, session lifecycle, and result collection
using fake transport and session objects.
"""

from __future__ import annotations

import time

import numpy as np

from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult
from vllm.v1.kv_offload.tiering.p2p import manager as manager_module
from vllm.v1.kv_offload.tiering.p2p.manager import (
    _PENDING_SESSION_TIMEOUT_S,
    P2PSecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.p2p.session import LoadResult, StoreResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kv_params(
    remote_host: str = "10.0.0.1",
    remote_port: int = 8000,
    kv_request_id: str = "req-1",
    do_remote_decode: bool = True,
    do_remote_prefill: bool = True,
) -> dict:
    return {
        "remote_host": remote_host,
        "remote_port": remote_port,
        "kv_request_id": kv_request_id,
        "do_remote_decode": do_remote_decode,
        "do_remote_prefill": do_remote_prefill,
    }


def _req_context(kv_params: dict | None = None) -> ReqContext:
    return ReqContext(req_id="test", kv_transfer_params=kv_params)


def _job_metadata(
    job_id: int,
    keys: list[bytes] | None = None,
    block_ids: list[int] | None = None,
    kv_params: dict | None = None,
) -> JobMetadata:
    if keys is None:
        keys = [b"key1"]
    if block_ids is None:
        block_ids = list(range(len(keys)))
    return JobMetadata(
        job_id=job_id,
        keys=keys,
        block_ids=np.array(block_ids),
        is_promotion=False,
        req_context=_req_context(kv_params),
    )


def _make_manager() -> P2PSecondaryTierManager:
    """Create a manager with stubbed __init__."""
    mgr = P2PSecondaryTierManager.__new__(P2PSecondaryTierManager)
    mgr._local_id = "127.0.0.1:7777"
    mgr._finished_jobs = []
    mgr._failed_req_ids = set()
    mgr._sessions = {}
    mgr._pending_session_created_at = {}
    return mgr


# ---------------------------------------------------------------------------
# Tests for _remote_id_from_params
# ---------------------------------------------------------------------------


class TestRemoteIdFromParams:
    def test_valid_params(self):
        result = P2PSecondaryTierManager._remote_id_from_params(
            {"remote_host": "10.0.0.1", "remote_port": 8000}
        )
        assert result == "10.0.0.1:8000"

    def test_missing_host(self):
        result = P2PSecondaryTierManager._remote_id_from_params({"remote_port": 8000})
        assert result is None

    def test_missing_port(self):
        result = P2PSecondaryTierManager._remote_id_from_params(
            {"remote_host": "10.0.0.1"}
        )
        assert result is None

    def test_empty_dict(self):
        result = P2PSecondaryTierManager._remote_id_from_params({})
        assert result is None


# ---------------------------------------------------------------------------
# Tests for lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_lookup_returns_false_without_kv_params(self):
        mgr = _make_manager()
        ctx = _req_context(kv_params=None)
        assert mgr.lookup(b"key", ctx) is False

    def test_lookup_returns_false_without_required_fields(self):
        mgr = _make_manager()
        ctx = _req_context(kv_params={"remote_host": "x"})
        assert mgr.lookup(b"key", ctx) is False

    def test_lookup_returns_true_for_valid_request(self):
        mgr = _make_manager()
        ctx = _req_context(kv_params=_kv_params())
        assert mgr.lookup(b"key", ctx) is True

    def test_lookup_returns_false_for_failed_request(self):
        mgr = _make_manager()
        mgr._failed_req_ids.add("req-1")
        ctx = _req_context(kv_params=_kv_params(kv_request_id="req-1"))
        assert mgr.lookup(b"key", ctx) is False

    def test_lookup_returns_true_for_different_request_id(self):
        mgr = _make_manager()
        mgr._failed_req_ids.add("req-1")
        ctx = _req_context(kv_params=_kv_params(kv_request_id="req-2"))
        assert mgr.lookup(b"key", ctx) is True


# ---------------------------------------------------------------------------
# Tests for submit_store
# ---------------------------------------------------------------------------


class TestSubmitStore:
    def test_no_do_remote_decode_succeeds_immediately(self):
        """Without do_remote_decode, job succeeds immediately."""
        mgr = _make_manager()
        params = _kv_params()
        del params["do_remote_decode"]
        job = _job_metadata(job_id=1, kv_params=params)
        mgr.submit_store(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=True)]

    def test_missing_kv_request_id_fails(self):
        """Missing kv_request_id fails the job."""
        mgr = _make_manager()
        params = _kv_params()
        del params["kv_request_id"]
        job = _job_metadata(job_id=1, kv_params=params)
        mgr.submit_store(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=False)]

    def test_no_session_creates_pending_session(self):
        """Without an existing session, submit_store lazily creates a pending
        one so blocks can be buffered until the decoder connects.
        """

        class FakeData:
            block_len = 4096

        mgr = _make_manager()
        mgr._data = FakeData()  # type: ignore[assignment]
        job = _job_metadata(job_id=1, kv_params=_kv_params())
        mgr.submit_store(job)

        peer_id = "10.0.0.1:8000"
        assert peer_id in mgr._sessions
        # Pending: no connection attached yet.
        assert not mgr._sessions[peer_id].connected
        # The job is buffered, not completed yet.
        assert mgr._finished_jobs == []


# ---------------------------------------------------------------------------
# Tests for submit_load
# ---------------------------------------------------------------------------


class TestSubmitLoad:
    def test_missing_params_fails(self):
        """Missing required kv_params fields fails the job."""
        mgr = _make_manager()
        job = _job_metadata(job_id=1, kv_params={})
        mgr.submit_load(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=False)]

    def test_empty_keys_succeeds_immediately(self):
        """Empty key list succeeds immediately."""
        mgr = _make_manager()
        job = _job_metadata(job_id=1, keys=[], block_ids=[], kv_params=_kv_params())
        mgr.submit_load(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=True)]

    def test_no_session_fails(self):
        """No session for peer fails and marks request failed."""
        mgr = _make_manager()
        job = _job_metadata(job_id=1, kv_params=_kv_params())
        mgr.submit_load(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=False)]
        assert "req-1" in mgr._failed_req_ids


# ---------------------------------------------------------------------------
# Tests for on_request_finished
# ---------------------------------------------------------------------------


class TestOnRequestFinished:
    def _make_with_failed(self) -> P2PSecondaryTierManager:
        mgr = _make_manager()
        mgr._failed_req_ids = {"req-1"}
        return mgr

    def test_prunes_failed_req_ids(self):
        mgr = self._make_with_failed()
        ctx = _req_context(kv_params=_kv_params(kv_request_id="req-1"))
        mgr.on_request_finished(ctx)
        assert "req-1" not in mgr._failed_req_ids

    def test_no_kv_params_does_nothing(self):
        mgr = self._make_with_failed()
        ctx = _req_context(kv_params=None)
        mgr.on_request_finished(ctx)
        assert "req-1" in mgr._failed_req_ids

    def test_no_kv_request_id_does_nothing(self):
        mgr = self._make_with_failed()
        ctx = _req_context(kv_params={"remote_host": "x", "remote_port": 1})
        mgr.on_request_finished(ctx)
        assert "req-1" in mgr._failed_req_ids

    def test_calls_session_finish_request(self):
        """on_request_finished forwards to session.finish_request,
        which handles both client-role cancel and server-role early-fail."""
        mgr = _make_manager()
        peer_id = "10.0.0.1:8000"
        session = _FakeSession(peer_id=peer_id)
        mgr._sessions[peer_id] = session
        ctx = _req_context(kv_params=_kv_params(kv_request_id="req-1"))
        mgr.on_request_finished(ctx)
        assert session.finishes == ["req-1"]


# ---------------------------------------------------------------------------
# Tests for get_finished_jobs
# ---------------------------------------------------------------------------


class _FakeServerHalf:
    def __init__(self) -> None:
        self._inflight: dict[int, object] = {}


class _FakeClientHalf:
    def __init__(self) -> None:
        self._inbound: dict[int, object] = {}


class _FakeSession:
    """Fake bidirectional session that returns canned poll() results."""

    def __init__(
        self,
        peer_id: str = "fake:1",
        alive: bool = True,
        connected: bool = True,
        loads: list[LoadResult] | None = None,
        stores: list[StoreResult] | None = None,
        close_loads: list[tuple[int, str]] | None = None,
        close_stores: list[int] | None = None,
    ) -> None:
        self.peer_id = peer_id
        self.alive = alive
        self.connected = connected
        self.ready = True
        self._loads = loads or []
        self._stores = stores or []
        self._close_loads = close_loads or []
        self._close_stores = close_stores or []
        self.requests: list[tuple[int, str]] = []
        self.finishes: list[str] = []
        # Mirror P2PSession._server._inflight (transfer_id → handle) and
        # P2PSession._client._inbound for the shutdown-drain and drain_jobs
        # paths. Tests populate _server._inflight when needed.
        self._server = _FakeServerHalf()
        self._client = _FakeClientHalf()

    def poll(self):
        loads = self._loads
        stores = self._stores
        self._loads = []
        self._stores = []
        return loads, stores

    def request_blocks(self, job_id, kv_request_id, keys, block_ids):
        self.requests.append((job_id, kv_request_id))

    def finish_request(self, kv_request_id):
        self.finishes.append(kv_request_id)

    def close(self):
        return self._close_loads, self._close_stores


class TestGetFinished:
    def _make(self) -> P2PSecondaryTierManager:
        mgr = _make_manager()
        mgr._finished_jobs = [
            JobResult(job_id=1, success=True),
            JobResult(job_id=2, success=False),
        ]

        class FakeControl:
            def poll(self):
                return []

        mgr._control = FakeControl()  # type: ignore[assignment]
        mgr._data = None  # type: ignore[assignment]
        return mgr

    def test_drains_finished_jobs(self):
        """get_finished_jobs returns and clears accumulated results."""
        mgr = self._make()
        results = list(mgr.get_finished_jobs())
        assert len(results) == 2
        assert JobResult(job_id=1, success=True) in results
        assert JobResult(job_id=2, success=False) in results
        # Second call returns empty.
        assert list(mgr.get_finished_jobs()) == []

    def test_reaps_dead_sessions(self):
        """Dead connected sessions are removed and their pending jobs failed."""

        class FakeData:
            def remove_remote_peer(self, pid):
                pass

        mgr = self._make()
        mgr._data = FakeData()  # type: ignore[assignment]
        dead = _FakeSession(
            peer_id="dead:1234",
            alive=False,
            connected=True,
            close_loads=[(20, "req-load")],
            close_stores=[10, 11],
        )
        mgr._sessions["dead:1234"] = dead  # type: ignore[assignment]

        results = list(mgr.get_finished_jobs())
        # 2 baseline + 1 failed load + 2 failed stores
        assert len(results) == 5
        assert JobResult(job_id=10, success=False) in results
        assert JobResult(job_id=11, success=False) in results
        assert JobResult(job_id=20, success=False) in results
        assert "dead:1234" not in mgr._sessions
        assert "req-load" in mgr._failed_req_ids

    def test_pending_sessions_are_not_reaped(self):
        """Fresh pending (unconnected) sessions stay across a poll."""
        mgr = self._make()
        pending = _FakeSession(peer_id="pending:1", alive=True, connected=False)
        mgr._sessions["pending:1"] = pending  # type: ignore[assignment]
        mgr._pending_session_created_at["pending:1"] = time.monotonic()
        list(mgr.get_finished_jobs())
        assert "pending:1" in mgr._sessions
        assert "pending:1" in mgr._pending_session_created_at

    def test_pending_session_reaped_after_timeout(self):
        """Pending session whose stamp is older than _PENDING_SESSION_TIMEOUT_S
        is reaped, surfacing its buffered store/load jobs as failures."""

        class FakeData:
            def remove_remote_peer(self, pid):
                pass

        mgr = self._make()
        mgr._data = FakeData()  # type: ignore[assignment]
        stale = _FakeSession(
            peer_id="pending:stale",
            alive=True,
            connected=False,
            close_loads=[(20, "req-load")],
            close_stores=[10, 11],
        )
        mgr._sessions["pending:stale"] = stale  # type: ignore[assignment]
        # Backdate so the stamp is past the deadline.
        mgr._pending_session_created_at["pending:stale"] = (
            time.monotonic() - _PENDING_SESSION_TIMEOUT_S - 1.0
        )

        results = list(mgr.get_finished_jobs())

        assert "pending:stale" not in mgr._sessions
        assert "pending:stale" not in mgr._pending_session_created_at
        # 2 baseline + 1 buffered load + 2 buffered stores
        assert JobResult(job_id=10, success=False) in results
        assert JobResult(job_id=11, success=False) in results
        assert JobResult(job_id=20, success=False) in results
        assert "req-load" in mgr._failed_req_ids

    def test_submit_store_records_pending_stamp(self):
        """submit_store stamps the creation time so the pending sweep can
        find sessions that have been stranded long enough to reap."""

        class FakeData:
            block_len = 4096

        mgr = _make_manager()
        mgr._data = FakeData()  # type: ignore[assignment]
        job = _job_metadata(job_id=1, kv_params=_kv_params())
        mgr.submit_store(job)
        assert "10.0.0.1:8000" in mgr._pending_session_created_at


# ---------------------------------------------------------------------------
# has_pending_work
# ---------------------------------------------------------------------------


class TestHasPendingWork:
    """has_pending_work() must always return True so the engine keeps
    ticking the offload pipeline — that's the only thread driving
    _control.poll() (incoming peer connects) and session.poll()
    (incoming fetch messages on existing sessions)."""

    def test_returns_true_with_no_state(self):
        mgr = _make_manager()
        assert mgr.has_pending_work() is True

    def test_returns_true_with_active_sessions(self):
        mgr = _make_manager()
        mgr._sessions["peer:1"] = _FakeSession(peer_id="peer:1")  # type: ignore[assignment]
        assert mgr.has_pending_work() is True


# ---------------------------------------------------------------------------
# Shutdown drain
# ---------------------------------------------------------------------------


class _ShutdownFakeData:
    """Fake DataTransport that records cancel/poll/close calls and
    drives the wait-cancel loop with a scriptable `still` queue."""

    def __init__(self, still_queue: list[list[int]] | None = None) -> None:
        # Each list in still_queue is the set of ids the next
        # cancel(mode="wait") should report as still inflight. The last
        # entry repeats once exhausted.
        self._still_queue = list(still_queue) if still_queue else [[]]
        self.cancel_calls: list[tuple[list[int], str]] = []
        self.poll_calls: int = 0
        self.close_calls: int = 0

    def cancel(self, transfer_ids, mode: str = "immediate") -> list[int]:
        ids = list(transfer_ids)
        self.cancel_calls.append((ids, mode))
        if mode == "wait":
            if len(self._still_queue) > 1:
                return list(self._still_queue.pop(0))
            return list(self._still_queue[0])
        return []

    def poll(self):
        self.poll_calls += 1

        class _Empty:
            done: list[int] = []
            failed: list[int] = []

        return _Empty()

    def close(self) -> None:
        self.close_calls += 1


class _ShutdownFakeControl:
    def __init__(self) -> None:
        self.close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


class TestShutdownDrain:
    """shutdown() drains inflight transfers via cancel(mode='wait')
    before calling _data.close(), with a 3s deadline fallback to
    cancel(mode='immediate')."""

    def _prep(
        self,
        still_queue: list[list[int]] | None = None,
        inflight_ids: list[int] | None = None,
    ) -> tuple[P2PSecondaryTierManager, _ShutdownFakeData, _ShutdownFakeControl]:
        mgr = _make_manager()
        data = _ShutdownFakeData(still_queue=still_queue)
        control = _ShutdownFakeControl()
        mgr._data = data  # type: ignore[assignment]
        mgr._control = control  # type: ignore[assignment]
        if inflight_ids:
            session = _FakeSession(peer_id="peer:1", connected=True)
            session._server._inflight = {tid: object() for tid in inflight_ids}
            mgr._sessions["peer:1"] = session  # type: ignore[assignment]
        return mgr, data, control

    def test_shutdown_drains_inflight_via_wait_cancel(self):
        # First cancel(wait) returns the input still inflight; second returns [].
        mgr, data, control = self._prep(
            still_queue=[[42, 43], []],
            inflight_ids=[42, 43],
        )
        mgr.shutdown()

        wait_calls = [c for c in data.cancel_calls if c[1] == "wait"]
        immediate_calls = [c for c in data.cancel_calls if c[1] == "immediate"]
        assert len(wait_calls) >= 1
        assert wait_calls[0][0] == [42, 43]
        assert immediate_calls == []
        # poll() was driven between cancel attempts.
        assert data.poll_calls >= 1
        # _data and _control were closed exactly once each, after the drain.
        assert data.close_calls == 1
        assert control.close_calls == 1

    def test_shutdown_force_cancels_after_timeout(self, monkeypatch):
        # Drain never completes — wait-cancel keeps returning the inflight set.
        monkeypatch.setattr(manager_module, "_SHUTDOWN_DRAIN_TIMEOUT_S", 0.05)
        mgr, data, control = self._prep(
            still_queue=[[42]],
            inflight_ids=[42],
        )
        mgr.shutdown()

        wait_calls = [c for c in data.cancel_calls if c[1] == "wait"]
        immediate_calls = [c for c in data.cancel_calls if c[1] == "immediate"]
        assert len(wait_calls) >= 1
        assert immediate_calls == [([42], "immediate")]
        assert data.close_calls == 1
        assert control.close_calls == 1

    def test_shutdown_no_inflight_skips_drain(self):
        mgr, data, control = self._prep()
        mgr.shutdown()

        assert data.cancel_calls == []
        assert data.poll_calls == 0
        assert data.close_calls == 1
        assert control.close_calls == 1


# ---------------------------------------------------------------------------
# Bidirectional regression test — both managers act as client AND server
# toward each other on the same peer_id. This is the case the unification
# is meant to fix.
# ---------------------------------------------------------------------------


class _LoopbackControl:
    """In-memory control transport that pairs two managers head-to-head.

    Each manager hands its outbound message buffer to the other's inbound
    queue on poll(). connect() returns a connection whose send() writes
    into the peer's inbound side; recv() reads what the peer's connect-or-
    poll path delivered for us.
    """

    def __init__(self, local_id: str) -> None:
        self._local_id = local_id
        self._peer: _LoopbackControl | None = None
        self._inbound_outbound: dict[str, _LoopbackConnection] = {}
        # Pending inbound for a peer that has not yet been registered.
        self._pending: list[tuple[str, dict]] = []

    def pair(self, peer: _LoopbackControl) -> None:
        self._peer = peer
        peer._peer = self

    def connect(self, peer_id: str):
        if peer_id in self._inbound_outbound:
            raise AssertionError(f"already connected to {peer_id}")
        conn = _LoopbackConnection(self, peer_id)
        self._inbound_outbound[peer_id] = conn
        return conn

    def poll(self):
        # Drain whatever the peer has sent toward us.
        new = []
        if self._peer is not None:
            for pid, msg in self._peer._drain_outbound_to(self._local_id):
                conn = self._inbound_outbound.get(pid)
                if conn is None:
                    conn = _LoopbackConnection(self, pid)
                    self._inbound_outbound[pid] = conn
                    new.append(conn)
                conn._inbox.append(msg)
        return new

    def _drain_outbound_to(self, peer_local_id: str):
        # Peer is calling our poll → return all messages we've sent toward
        # peer_local_id (which is the peer's own local_id).
        out: list[tuple[str, dict]] = []
        for pid, conn in list(self._inbound_outbound.items()):
            # Each outgoing message goes to peer_local_id and is tagged
            # with the sender's local_id (i.e., self._local_id).
            for msg in conn._outbox:
                out.append((self._local_id, msg))
            conn._outbox.clear()
        return out

    def close(self):
        for conn in self._inbound_outbound.values():
            conn.close()
        self._inbound_outbound.clear()


class _LoopbackConnection:
    def __init__(self, transport: _LoopbackControl, peer_id: str) -> None:
        self._transport = transport
        self.peer_id = peer_id
        self._inbox: list[dict] = []
        self._outbox: list[dict] = []
        self._closed = False

    @property
    def alive(self) -> bool:
        return not self._closed

    def send(self, msg: dict) -> None:
        if self._closed:
            raise RuntimeError("send on closed conn")
        self._outbox.append(msg)

    def recv(self) -> list[dict]:
        msgs = self._inbox
        self._inbox = []
        return msgs

    def mark_dead(self) -> None:
        self._closed = True

    def close(self) -> None:
        self._closed = True


class _FakeData:
    """Minimal NIXL fake that lets matched transfers complete on the next poll."""

    def __init__(self, local_id: str) -> None:
        self._local_id = local_id
        self.block_len = 4096
        self.base_addr = 0x1000
        self.num_blocks = 16
        self.config_fingerprint = ""
        self._remote_peers: dict[str, dict] = {}
        self._inflight_done: list[int] = []
        self._next_id = 0

    def get_agent_metadata(self) -> bytes:
        return f"meta-{self._local_id}".encode()

    def add_remote_peer(
        self, peer_id, agent_metadata, base_addr, num_blocks, block_len
    ) -> None:
        self._remote_peers[peer_id] = {
            "agent_metadata": agent_metadata,
            "base_addr": base_addr,
            "num_blocks": num_blocks,
            "block_len": block_len,
        }

    def remove_remote_peer(self, peer_id: str) -> None:
        self._remote_peers.pop(peer_id, None)

    def write_blocks(self, peer_id, local_idxs, remote_idxs):
        if peer_id not in self._remote_peers:
            return None
        tid = self._next_id
        self._next_id += 1
        self._inflight_done.append(tid)
        return tid

    def poll(self):
        from vllm.v1.kv_offload.tiering.p2p.data.base import PollResult

        done = self._inflight_done[:]
        self._inflight_done.clear()
        return PollResult(done=done, failed=[])

    def cancel(self, transfer_ids) -> None:
        pass

    def close(self) -> None:
        pass


def _build_paired_managers() -> tuple[P2PSecondaryTierManager, P2PSecondaryTierManager]:
    """Two managers each acting as both client and server toward the other.

    Wires _LoopbackControl pair + per-side _FakeData so transfers complete
    on the next poll. The test drives polling by calling get_finished_jobs(),
    which invokes _poll_once synchronously on the calling thread.
    """
    mgr_a = _make_manager()
    mgr_b = _make_manager()
    mgr_a._local_id = "A:1"
    mgr_b._local_id = "B:2"

    ctrl_a = _LoopbackControl(mgr_a._local_id)
    ctrl_b = _LoopbackControl(mgr_b._local_id)
    ctrl_a.pair(ctrl_b)

    mgr_a._control = ctrl_a  # type: ignore[assignment]
    mgr_b._control = ctrl_b  # type: ignore[assignment]
    mgr_a._data = _FakeData(mgr_a._local_id)  # type: ignore[assignment]
    mgr_b._data = _FakeData(mgr_b._local_id)  # type: ignore[assignment]

    return mgr_a, mgr_b


class TestBidirectionalManager:
    """Two managers each load FROM and serve TO the other over a single peer."""

    def test_both_loads_succeed(self):
        mgr_a, mgr_b = _build_paired_managers()

        a_loads_kv = "req-AtoB-load"  # A loads, B serves
        b_loads_kv = "req-BtoA-load"  # B loads, A serves

        a_decoder_params = {
            "remote_host": "B",
            "remote_port": 2,
            "kv_request_id": a_loads_kv,
            "do_remote_decode": False,
            "do_remote_prefill": True,
        }
        b_decoder_params = {
            "remote_host": "A",
            "remote_port": 1,
            "kv_request_id": b_loads_kv,
            "do_remote_decode": False,
            "do_remote_prefill": True,
        }
        a_prefiller_params = {
            "remote_host": "B",
            "remote_port": 2,
            "kv_request_id": b_loads_kv,
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }
        b_prefiller_params = {
            "remote_host": "A",
            "remote_port": 1,
            "kv_request_id": a_loads_kv,
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }

        # 1. Both sides open client-role sessions toward the peer.
        mgr_a.on_new_request(_req_context(a_decoder_params))
        mgr_b.on_new_request(_req_context(b_decoder_params))

        # 2. Both sides store the blocks the peer will fetch.
        mgr_a.submit_store(
            _job_metadata(
                job_id=100,
                keys=[b"a-block"],
                block_ids=[0],
                kv_params=a_prefiller_params,
            )
        )
        mgr_b.submit_store(
            _job_metadata(
                job_id=200,
                keys=[b"b-block"],
                block_ids=[0],
                kv_params=b_prefiller_params,
            )
        )

        # 3. Both sides submit loads.
        mgr_a.submit_load(
            _job_metadata(
                job_id=101,
                keys=[b"b-block"],
                block_ids=[0],
                kv_params=a_decoder_params,
            )
        )
        mgr_b.submit_load(
            _job_metadata(
                job_id=201,
                keys=[b"a-block"],
                block_ids=[0],
                kv_params=b_decoder_params,
            )
        )

        # 4. Drive several poll iterations on each side. Each
        # get_finished_jobs() call invokes _poll_once synchronously.
        all_a: list[JobResult] = []
        all_b: list[JobResult] = []
        for _ in range(8):
            all_a.extend(list(mgr_a.get_finished_jobs()))
            all_b.extend(list(mgr_b.get_finished_jobs()))

        # Both load jobs and both store jobs must complete successfully.
        a_ok = {r.job_id for r in all_a if r.success}
        b_ok = {r.job_id for r in all_b if r.success}
        # A: load job 101 + store job 100
        assert 101 in a_ok, f"A loads succeeded: {a_ok}"
        assert 100 in a_ok, f"A stores succeeded: {a_ok}"
        # B: load job 201 + store job 200
        assert 201 in b_ok, f"B loads succeeded: {b_ok}"
        assert 200 in b_ok, f"B stores succeeded: {b_ok}"
