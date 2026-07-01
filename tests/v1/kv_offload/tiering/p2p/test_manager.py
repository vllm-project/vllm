# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for P2PSecondaryTierManager.

Tests the manager's job routing, session lifecycle, and result collection
using fake transport and session objects.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from vllm.v1.kv_offload.base import LookupResult, ReqContext, ScheduleEndContext
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult
from vllm.v1.kv_offload.tiering.p2p import manager as manager_module
from vllm.v1.kv_offload.tiering.p2p.manager import (
    _UNBOUND_STORE_TIMEOUT_S,
    P2PSecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.p2p.session import (
    LoadResult,
    SessionPollResult,
    StoreResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prefill_kv_params(
    remote_host: str = "10.0.0.1",
    remote_port: int = 8000,
    kv_request_id: str = "req-1",
) -> dict:
    """Decoder-side kv_transfer_params: ``prefill`` sub-dict carries
    kv_request_id + remote_host + remote_port."""
    return {
        "prefill": {
            "kv_request_id": kv_request_id,
            "remote_host": remote_host,
            "remote_port": remote_port,
        },
    }


def _p2p_kv_params(
    remote_host: str = "10.0.0.1",
    remote_port: int = 8000,
    kv_request_id: str = "req-1",
) -> dict:
    """Symmetric-P2P consumer kv_transfer_params: ``p2p`` sub-dict has
    the same shape as ``prefill`` (kv_request_id + remote_host + port)."""
    return {
        "p2p": {
            "kv_request_id": kv_request_id,
            "remote_host": remote_host,
            "remote_port": remote_port,
        },
    }


def _decode_kv_params(kv_request_id: str = "req-1") -> dict:
    """Prefiller-side kv_transfer_params: ``decode`` sub-dict carries
    kv_request_id only."""
    return {"decode": {"kv_request_id": kv_request_id}}


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
    from vllm.v1.kv_offload.tiering.p2p.tiering_callbacks import _AllMissCallbacks

    mgr = P2PSecondaryTierManager.__new__(P2PSecondaryTierManager)
    mgr._local_id = "127.0.0.1:7777"
    mgr._finished_jobs = []
    mgr._failed_req_ids = set()
    mgr._sessions = {}
    mgr._kv_to_session = {}
    mgr._unbound_stores = {}
    mgr._tiering_callbacks = _AllMissCallbacks()
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
    def test_lookup_returns_miss_without_kv_params(self):
        mgr = _make_manager()
        ctx = _req_context(kv_params=None)
        assert mgr.lookup(b"key", ctx) is LookupResult.MISS

    def test_lookup_returns_miss_without_required_fields(self):
        mgr = _make_manager()
        ctx = _req_context(kv_params={"prefill": {"remote_host": "x"}})
        assert mgr.lookup(b"key", ctx) is LookupResult.MISS

    def test_lookup_returns_hit_for_valid_request(self):
        mgr = _make_manager()
        ctx = _req_context(kv_params=_prefill_kv_params())
        assert mgr.lookup(b"key", ctx) is LookupResult.HIT

    def test_lookup_returns_miss_for_failed_request(self):
        mgr = _make_manager()
        mgr._failed_req_ids.add("req-1")
        ctx = _req_context(kv_params=_prefill_kv_params(kv_request_id="req-1"))
        assert mgr.lookup(b"key", ctx) is LookupResult.MISS

    def test_lookup_returns_hit_for_different_request_id(self):
        mgr = _make_manager()
        mgr._failed_req_ids.add("req-1")
        ctx = _req_context(kv_params=_prefill_kv_params(kv_request_id="req-2"))
        assert mgr.lookup(b"key", ctx) is LookupResult.HIT

    def test_lookup_returns_miss_without_prefill_key(self):
        """No ``prefill`` sub-dict means the request was not routed for
        remote prefill — local prefill should run instead, so lookup()
        returns MISS even when a stale ``decode`` block is present."""
        mgr = _make_manager()
        ctx = _req_context(kv_params=_decode_kv_params())
        assert mgr.lookup(b"key", ctx) is LookupResult.MISS


# ---------------------------------------------------------------------------
# Tests for TieringCallbacks plumbing
# ---------------------------------------------------------------------------


class TestTieringCallbacksPlumbing:
    def test_default_uses_all_miss_callbacks(self):
        """When no callbacks are passed, the manager defaults to
        ``_AllMissCallbacks`` and that instance is forwarded to every
        session's server role."""
        from vllm.v1.kv_offload.tiering.p2p.tiering_callbacks import (
            _AllMissCallbacks,
        )

        mgr = _make_manager()
        # _make_manager already sets _AllMissCallbacks; sanity check.
        assert isinstance(mgr._tiering_callbacks, _AllMissCallbacks)

    def test_custom_callbacks_forwarded_to_accepted_sessions(self):
        """Inbound connections create sessions that share the manager's
        ``tiering_callbacks`` instance."""

        class _Stub:
            def lookup(self, key, ctx):
                from vllm.v1.kv_offload.base import LookupResult

                return LookupResult.MISS

            def create_store_job(self, keys, ctx):
                raise AssertionError("unreachable for MISS-only stub")

            def finish_request(self, ctx):
                pass

        class _FakeData:
            block_len = 4096
            base_addr = 0x1000
            num_blocks = 16
            config_fingerprint = ""

            def get_agent_metadata(self):
                return b"meta"

            def add_remote_peer(self, *args, **kwargs):
                pass

        class _Conn:
            def __init__(self, pid: str) -> None:
                self.peer_id = pid
                self.alive = True

            def send(self, msg: dict) -> None:
                pass

            def close(self) -> None:
                self.alive = False

        mgr = _make_manager()
        stub = _Stub()
        mgr._tiering_callbacks = stub  # type: ignore[assignment]
        mgr._data = _FakeData()  # type: ignore[assignment]
        peer_id = "10.0.0.5:9000"

        mgr._accept_new_peers([_Conn(peer_id)])  # type: ignore[arg-type]

        session = mgr._sessions[peer_id]
        assert session._server._cb is stub


# ---------------------------------------------------------------------------
# Tests for submit_store
# ---------------------------------------------------------------------------


class TestSubmitStore:
    def test_no_decode_succeeds_immediately(self):
        """Without a ``decode`` block, job succeeds immediately."""
        mgr = _make_manager()
        job = _job_metadata(job_id=1, kv_params={})
        mgr.submit_store(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=True)]

    def test_missing_kv_request_id_fails(self):
        """Missing kv_request_id inside ``decode`` fails the job."""
        mgr = _make_manager()
        params: dict = {"decode": {}}
        job = _job_metadata(job_id=1, kv_params=params)
        mgr.submit_store(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=False)]

    def test_no_binding_yet_parks_in_unbound_stores(self):
        """submit_store without a bound session buffers the batch keyed
        by kv_request_id; no session is pre-created (peer_id is unknown
        to the producer at store time)."""
        mgr = _make_manager()
        job = _job_metadata(
            job_id=1,
            keys=[b"k1", b"k2"],
            block_ids=[3, 4],
            kv_params=_decode_kv_params(kv_request_id="req-1"),
        )
        mgr.submit_store(job)

        assert mgr._sessions == {}
        assert mgr._finished_jobs == []
        batches = mgr._unbound_stores["req-1"]
        assert len(batches) == 1
        assert (
            batches[0].job_id,
            list(batches[0].keys),
            list(batches[0].block_ids),
        ) == (
            1,
            [b"k1", b"k2"],
            [3, 4],
        )

    def test_routes_to_bound_session(self):
        """If a session has already received FetchMsg for this
        kv_request_id (so _kv_to_session is populated), submit_store
        forwards directly to that session rather than re-buffering."""
        mgr = _make_manager()
        bound = _FakeSession(peer_id="10.0.0.1:8000", connected=True)
        mgr._kv_to_session["req-1"] = bound  # type: ignore[assignment]
        # Note: _sessions is intentionally untouched — this test isolates
        # the kv_request_id → session fast path.
        job = _job_metadata(
            job_id=7,
            keys=[b"k1", b"k2"],
            block_ids=[3, 4],
            kv_params=_decode_kv_params(kv_request_id="req-1"),
        )
        mgr.submit_store(job)

        assert len(bound.stores_added) == 1
        kv_req_id, keys, _, job_id = bound.stores_added[0]
        assert (kv_req_id, keys, job_id) == ("req-1", [b"k1", b"k2"], 7)
        assert mgr._unbound_stores == {}
        assert mgr._finished_jobs == []

    def test_extra_top_level_keys_are_ignored(self):
        """Producer-side kv_transfer_params should not pre-create a
        session even when a stale caller still passes a top-level
        ``remote_host``/``remote_port`` next to ``decode``."""
        mgr = _make_manager()
        params = _decode_kv_params()
        params["remote_host"] = "stale"
        params["remote_port"] = 12345
        job = _job_metadata(job_id=1, kv_params=params)
        mgr.submit_store(job)
        # No session pre-created, no peer-keyed state.
        assert mgr._sessions == {}
        assert "req-1" in mgr._unbound_stores


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
        job = _job_metadata(
            job_id=1, keys=[], block_ids=[], kv_params=_prefill_kv_params()
        )
        mgr.submit_load(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=True)]

    def test_no_session_fails(self):
        """No session for peer fails and marks request failed."""
        mgr = _make_manager()
        job = _job_metadata(job_id=1, kv_params=_prefill_kv_params())
        mgr.submit_load(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=False)]
        assert "req-1" in mgr._failed_req_ids

    def test_happy_path_with_active_session(self):
        """When the peer's session exists, submit_load forwards to
        session.request_blocks and does NOT add a finished result yet."""
        mgr = _make_manager()
        peer_id = "10.0.0.1:8000"
        existing = _FakeSession(peer_id=peer_id, connected=True)
        mgr._sessions[peer_id] = existing  # type: ignore[assignment]
        job = _job_metadata(
            job_id=42,
            keys=[b"k1", b"k2"],
            block_ids=[5, 6],
            kv_params=_prefill_kv_params(kv_request_id="req-42"),
        )
        mgr.submit_load(job)

        assert existing.requests == [(42, "req-42")]
        assert mgr._finished_jobs == []
        assert "req-42" not in mgr._failed_req_ids

    def test_missing_consumer_flag_fails(self):
        """Peer fields present but neither do_remote_prefill nor
        do_p2p_fetch is set — submit_load fails the job rather than
        emit a stray FetchMsg."""
        mgr = _make_manager()
        params = {
            "remote_host": "10.0.0.1",
            "remote_port": 8000,
            "kv_request_id": "req-1",
        }
        job = _job_metadata(job_id=1, kv_params=params)
        mgr.submit_load(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=False)]


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
        ctx = _req_context(kv_params=_prefill_kv_params(kv_request_id="req-1"))
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

    def test_decoder_side_calls_session_finish_request(self):
        """Decoder-side finish (``prefill`` set) still routes via peer_id
        because the consumer addresses the producer it loaded from. The
        session's finish_request cancels the client-role load."""
        mgr = _make_manager()
        peer_id = "10.0.0.1:8000"
        session = _FakeSession(peer_id=peer_id)
        mgr._sessions[peer_id] = session
        ctx = _req_context(kv_params=_prefill_kv_params(kv_request_id="req-1"))
        mgr.on_request_finished(ctx)
        assert session.finishes == ["req-1"]

    def test_p2p_consumer_side_calls_session_finish_request(self):
        """Symmetric-P2P consumer finish (``p2p`` set) routes via peer_id
        so the session drops any pending lookups (cancel_lookups) and
        cancels any inbound load."""
        mgr = _make_manager()
        peer_id = "10.0.0.1:8000"
        session = _FakeSession(peer_id=peer_id)
        mgr._sessions[peer_id] = session
        ctx = _req_context(kv_params=_p2p_kv_params(kv_request_id="req-1"))
        mgr.on_request_finished(ctx)
        assert session.finishes == ["req-1"]

    def test_prefiller_bound_id_routes_via_kv_to_session(self):
        """Prefiller-side finish for an id whose session is already bound
        (FetchMsg received) routes via _kv_to_session and pops the entry."""
        mgr = _make_manager()
        bound = _FakeSession(peer_id="some-peer:1", connected=True)
        mgr._kv_to_session["req-1"] = bound  # type: ignore[assignment]
        ctx = _req_context(kv_params=_decode_kv_params(kv_request_id="req-1"))
        mgr.on_request_finished(ctx)
        assert bound.finishes == ["req-1"]
        assert "req-1" not in mgr._kv_to_session

    def test_prefiller_unbound_id_leaves_batches_parked(self):
        """Prefiller-side finish for an id with parked unbound batches
        and no session binding is a no-op on `_unbound_stores`. The
        parked batches survive until a peer fetches them or the
        `_reap_unbound_stores` timeout fires — `on_request_finished`
        must not evict them."""
        from vllm.v1.kv_offload.tiering.p2p.manager import _UnboundStoreBatch

        mgr = _make_manager()
        mgr._unbound_stores["req-1"] = [
            _UnboundStoreBatch(job_id=10, keys=[b"k"], block_ids=[0]),
            _UnboundStoreBatch(job_id=11, keys=[b"k2"], block_ids=[1]),
        ]
        ctx = _req_context(kv_params=_decode_kv_params(kv_request_id="req-1"))
        mgr.on_request_finished(ctx)
        assert "req-1" in mgr._unbound_stores
        assert [b.job_id for b in mgr._unbound_stores["req-1"]] == [10, 11]
        outcomes = {(r.job_id, r.success) for r in mgr._finished_jobs}
        assert (10, False) not in outcomes
        assert (11, False) not in outcomes


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
        new_fetch_ids: list[str] | None = None,
        close_loads: list[tuple[int, str]] | None = None,
        close_stores: list[int] | None = None,
    ) -> None:
        self.peer_id = peer_id
        self.alive = alive
        self.connected = connected
        self.ready = True
        self._loads = loads or []
        self._stores = stores or []
        self._new_fetch_ids = new_fetch_ids or []
        self._close_loads = close_loads or []
        self._close_stores = close_stores or []
        self.requests: list[tuple[int, str]] = []
        self.stores_added: list[tuple[str, list, object, int]] = []
        self.attached: list[object] = []
        self.finishes: list[str] = []
        # Mirror P2PSession._server._inflight (transfer_id → handle) and
        # P2PSession._client._inbound for the shutdown-drain and drain_jobs
        # paths. Tests populate _server._inflight when needed.
        self._server = _FakeServerHalf()
        self._client = _FakeClientHalf()

    def poll(self):
        result = SessionPollResult(
            loads=self._loads,
            stores=self._stores,
            new_fetch_ids=self._new_fetch_ids,
        )
        self._loads = []
        self._stores = []
        self._new_fetch_ids = []
        return result

    def request_blocks(self, job_id, kv_request_id, keys, block_ids):
        self.requests.append((job_id, kv_request_id))

    def add_stored_blocks(self, kv_request_id, keys, block_ids, job_id):
        self.stores_added.append((kv_request_id, list(keys), block_ids, job_id))

    def attach_connection(self, conn):
        self.attached.append(conn)
        self.connected = True

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

    def test_unbound_store_kept_within_timeout(self):
        """Recently-parked unbound stores stay across a poll."""
        mgr = self._make()
        from vllm.v1.kv_offload.tiering.p2p.manager import _UnboundStoreBatch

        mgr._unbound_stores["req-fresh"] = [
            _UnboundStoreBatch(job_id=99, keys=[b"k"], block_ids=[0])
        ]
        list(mgr.get_finished_jobs())
        assert "req-fresh" in mgr._unbound_stores

    def test_unbound_store_reaped_after_timeout(self):
        """Unbound stores past _UNBOUND_STORE_TIMEOUT_S surface as failed
        and their kv_request_id lands in _failed_req_ids so a late
        FetchMsg/lookup doesn't try to satisfy them."""
        from vllm.v1.kv_offload.tiering.p2p.manager import _UnboundStoreBatch

        mgr = self._make()
        stale = _UnboundStoreBatch(job_id=10, keys=[b"k"], block_ids=[0])
        # Backdate the submission so the head batch is past the deadline.
        stale.submitted_at = time.monotonic() - _UNBOUND_STORE_TIMEOUT_S - 1.0
        mgr._unbound_stores["req-stale"] = [
            stale,
            _UnboundStoreBatch(job_id=11, keys=[b"k2"], block_ids=[1]),
        ]

        results = list(mgr.get_finished_jobs())

        assert "req-stale" not in mgr._unbound_stores
        # 2 baseline + 2 buffered stores
        assert JobResult(job_id=10, success=False) in results
        assert JobResult(job_id=11, success=False) in results
        assert "req-stale" in mgr._failed_req_ids

    def test_submit_store_parks_unbound_batch(self):
        """submit_store on an unbound id appends a batch with a fresh
        submitted_at stamp so the unbound-store sweep can age it out."""
        mgr = _make_manager()
        job = _job_metadata(
            job_id=1, kv_params=_decode_kv_params(kv_request_id="req-1")
        )
        before = time.monotonic()
        mgr.submit_store(job)
        after = time.monotonic()
        batches = mgr._unbound_stores["req-1"]
        assert len(batches) == 1
        assert before <= batches[0].submitted_at <= after


# ---------------------------------------------------------------------------
# has_pending_work
# ---------------------------------------------------------------------------


class TestHasPendingWork:
    """has_pending_work() must always return True so the engine keeps
    ticking the offload pipeline — that's the only thread driving
    _control.poll() (incoming peer connects) and session.poll()
    (incoming fetch messages on existing sessions)."""

    def test_returns_true_unconditionally_to_keep_engine_ticking(self):
        """Even with no sessions and no jobs, has_pending_work() returns
        True so the engine keeps calling get_finished_jobs(), which is
        what drives _control.poll() for inbound peer connects."""
        mgr = _make_manager()
        assert mgr.has_pending_work() is True

    def test_returns_true_even_when_sessions_present(self):
        """The result is the same regardless of session state — there is
        no 'idle' branch."""
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
        # Use a synthetic clock so the test does not depend on real wallclock
        # being able to advance in <50ms on a loaded CI node:
        #   call 1 (deadline calc): 100.0  -> deadline = 100.05
        #   call 2 (loop predicate): 100.0 -> enters loop, one wait-cancel
        #   call 3 (loop predicate): 100.06 -> exits, force-cancel runs
        monkeypatch.setattr(manager_module, "_SHUTDOWN_DRAIN_TIMEOUT_S", 0.05)
        monkeypatch.setattr(manager_module, "_DRAIN_SLEEP_S", 0.0)
        times = iter([100.0, 100.0, 100.06])
        # Patch via a fake module on `manager_module.time` so we do not mutate
        # the global `time` module — other code in the process (e.g. the
        # buildkite test collector's pytest_runtest_logreport hook) calls
        # time.monotonic() before monkeypatch teardown.
        fake_time = SimpleNamespace(monotonic=lambda: next(times), sleep=time.sleep)
        monkeypatch.setattr(manager_module, "time", fake_time)

        mgr, data, control = self._prep(
            still_queue=[[42]],
            inflight_ids=[42],
        )
        mgr.shutdown()

        wait_calls = [c for c in data.cancel_calls if c[1] == "wait"]
        immediate_calls = [c for c in data.cancel_calls if c[1] == "immediate"]
        assert len(wait_calls) == 1
        assert wait_calls[0][0] == [42]
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
            "prefill": {
                "kv_request_id": a_loads_kv,
                "remote_host": "B",
                "remote_port": 2,
            },
        }
        b_decoder_params = {
            "prefill": {
                "kv_request_id": b_loads_kv,
                "remote_host": "A",
                "remote_port": 1,
            },
        }
        a_prefiller_params = {"decode": {"kv_request_id": b_loads_kv}}
        b_prefiller_params = {"decode": {"kv_request_id": a_loads_kv}}

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


# ---------------------------------------------------------------------------
# _accept_new_peers — duplicate connection rejection
# ---------------------------------------------------------------------------


class _RecordingConn:
    """Inbound connection stub that records close()/peer_id only."""

    def __init__(self, peer_id: str) -> None:
        self.peer_id = peer_id
        self.close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


class TestAcceptNewPeers:
    """A second inbound from an already-connected peer is rejected and the
    new conn is closed; the existing session is left untouched."""

    def test_duplicate_connection_is_closed_and_existing_session_untouched(self):
        mgr = _make_manager()
        peer_id = "10.0.0.1:8000"
        existing = _FakeSession(peer_id=peer_id, connected=True)
        mgr._sessions[peer_id] = existing  # type: ignore[assignment]

        new_conn = _RecordingConn(peer_id)
        mgr._accept_new_peers([new_conn])

        # Manager swallowed the ValueError and closed the duplicate conn.
        assert new_conn.close_calls == 1
        # Existing session was NOT re-attached.
        assert existing.attached == []
        # Session map unchanged.
        assert mgr._sessions[peer_id] is existing

    def test_creates_session_for_new_peer(self):
        """An inbound conn from a peer with no existing session creates
        a fresh connected session and registers it under conn.peer_id.
        The prefiller has no pre-created pending session anymore — the
        first signal of a peer's existence is its inbound connection."""

        class FakeData:
            block_len = 4096
            base_addr = 0x1000
            num_blocks = 16
            config_fingerprint = ""

            def get_agent_metadata(self):
                return b"meta"

            def add_remote_peer(self, *args, **kwargs):
                pass

        mgr = _make_manager()
        mgr._data = FakeData()  # type: ignore[assignment]
        peer_id = "10.0.0.1:8000"

        # Real ControlConnection-shaped fake: send/close/peer_id only.
        sent: list[dict] = []

        class _Conn:
            def __init__(self, pid: str) -> None:
                self.peer_id = pid
                self.alive = True

            def send(self, msg: dict) -> None:
                sent.append(msg)

            def close(self) -> None:
                self.alive = False

        mgr._accept_new_peers([_Conn(peer_id)])  # type: ignore[arg-type]

        assert peer_id in mgr._sessions
        assert mgr._sessions[peer_id].connected is True
        # Session sent its ConnectMsg on the new connection.
        assert any(m for m in sent)


# ---------------------------------------------------------------------------
# _poll_once orchestration
# ---------------------------------------------------------------------------


class TestPollOnce:
    """_poll_once must drain control, accept new peers, poll every session,
    surface results, and reap dead sessions — in that order."""

    def test_orchestrates_accept_poll_and_reap(self):
        mgr = _make_manager()

        # Alive session whose poll() returns one load + one store.
        peer_alive = "10.0.0.2:9000"
        alive = _FakeSession(
            peer_id=peer_alive,
            alive=True,
            connected=True,
            loads=[LoadResult(job_id=11, kv_request_id="req-11", success=True)],
            stores=[StoreResult(job_id=22, success=True)],
        )
        mgr._sessions[peer_alive] = alive  # type: ignore[assignment]

        # Dead session whose pending close() jobs surface as failures.
        peer_dead = "10.0.0.3:9999"
        dead = _FakeSession(
            peer_id=peer_dead,
            alive=False,
            connected=True,
            close_loads=[(33, "req-33")],
            close_stores=[44],
        )
        mgr._sessions[peer_dead] = dead  # type: ignore[assignment]

        class _Ctrl:
            def poll(self_inner):
                return []

        class _Data:
            def remove_remote_peer(self_inner, pid):
                pass

        mgr._control = _Ctrl()  # type: ignore[assignment]
        mgr._data = _Data()  # type: ignore[assignment]

        mgr._poll_once()

        # Every session was polled — alive's results landed.
        # Dead session was reaped — its close() failures landed.
        finished = mgr._finished_jobs
        ok = {(r.job_id, r.success) for r in finished}
        assert (11, True) in ok  # alive load result
        assert (22, True) in ok  # alive store result
        assert (33, False) in ok  # dead session's pending load
        assert (44, False) in ok  # dead session's pending store
        assert "req-33" in mgr._failed_req_ids
        assert peer_dead not in mgr._sessions
        assert peer_alive in mgr._sessions

    def test_new_fetch_id_binds_and_replays_unbound_batches(self):
        """When session.poll() reports a kv_request_id whose FetchMsg
        arrived this tick, the manager binds it to that session and
        replays every parked submit_store batch via add_stored_blocks."""
        from vllm.v1.kv_offload.tiering.p2p.manager import _UnboundStoreBatch

        mgr = _make_manager()
        peer = "10.0.0.1:8000"
        sess = _FakeSession(
            peer_id=peer,
            alive=True,
            connected=True,
            new_fetch_ids=["req-1"],
        )
        mgr._sessions[peer] = sess  # type: ignore[assignment]
        mgr._unbound_stores["req-1"] = [
            _UnboundStoreBatch(job_id=5, keys=[b"k1"], block_ids=[0]),
            _UnboundStoreBatch(job_id=6, keys=[b"k2"], block_ids=[1]),
        ]

        class _Ctrl:
            def poll(self_inner):
                return []

        mgr._control = _Ctrl()  # type: ignore[assignment]
        mgr._poll_once()

        assert mgr._kv_to_session["req-1"] is sess
        assert "req-1" not in mgr._unbound_stores
        replayed = [
            (kv_req_id, list(keys), job_id)
            for kv_req_id, keys, _, job_id in sess.stores_added
        ]
        assert replayed == [("req-1", [b"k1"], 5), ("req-1", [b"k2"], 6)]

    def test_new_fetch_id_with_no_unbound_still_binds(self):
        """A FetchMsg for a kv_request_id with no parked batches still
        records the binding so subsequent submit_stores route fast."""
        mgr = _make_manager()
        peer = "10.0.0.1:8000"
        sess = _FakeSession(
            peer_id=peer,
            alive=True,
            connected=True,
            new_fetch_ids=["req-fast"],
        )
        mgr._sessions[peer] = sess  # type: ignore[assignment]

        class _Ctrl:
            def poll(self_inner):
                return []

        mgr._control = _Ctrl()  # type: ignore[assignment]
        mgr._poll_once()

        assert mgr._kv_to_session["req-fast"] is sess
        assert sess.stores_added == []

    def test_failed_load_records_kv_request_id(self):
        """A LoadResult(success=False) from session.poll() must add its
        kv_request_id to _failed_req_ids so future lookups return MISS."""
        mgr = _make_manager()
        peer = "10.0.0.1:8000"
        sess = _FakeSession(
            peer_id=peer,
            alive=True,
            connected=True,
            loads=[LoadResult(job_id=5, kv_request_id="req-5", success=False)],
        )
        mgr._sessions[peer] = sess  # type: ignore[assignment]

        class _Ctrl:
            def poll(self_inner):
                return []

        mgr._control = _Ctrl()  # type: ignore[assignment]

        mgr._poll_once()

        assert mgr._finished_jobs == [JobResult(job_id=5, success=False)]
        assert "req-5" in mgr._failed_req_ids


# ---------------------------------------------------------------------------
# drain_jobs
# ---------------------------------------------------------------------------


class _DrainCtrl:
    """Trivial control fake whose poll() returns an empty list."""

    def poll(self):
        return []


class TestDrainJobs:
    def test_returns_immediately_when_quiescent(self):
        """No sessions and no inflight: drain_jobs returns without sleeping."""
        mgr = _make_manager()
        mgr._control = _DrainCtrl()  # type: ignore[assignment]

        sleeps: list[float] = []
        # If drain_jobs sleeps when nothing is pending, that's a regression.
        import vllm.v1.kv_offload.tiering.p2p.manager as m

        original_sleep = m.time.sleep
        m.time.sleep = lambda s: sleeps.append(s)  # type: ignore[assignment]
        try:
            mgr.drain_jobs()
        finally:
            m.time.sleep = original_sleep  # type: ignore[assignment]

        assert sleeps == []

    def test_returns_when_session_has_no_inflight_or_inbound(self):
        """A session with empty _inbound and _inflight does not block drain."""
        mgr = _make_manager()
        mgr._control = _DrainCtrl()  # type: ignore[assignment]
        mgr._sessions["peer:1"] = _FakeSession(peer_id="peer:1")  # type: ignore[assignment]
        # Should return on the first iteration.
        mgr.drain_jobs()

    def test_logs_warning_after_5s_then_completes(self, monkeypatch):
        """A session that stays inflight past 5s triggers the warning, and
        once it clears the loop returns."""
        mgr = _make_manager()
        mgr._control = _DrainCtrl()  # type: ignore[assignment]
        sess = _FakeSession(peer_id="peer:1")
        sess._server._inflight = {1: object()}  # non-empty
        mgr._sessions["peer:1"] = sess  # type: ignore[assignment]

        # Synthetic monotonic clock: 100.0 for the start stamp, then 106.0
        # for the first elapsed-check (past the 5s warning threshold), then
        # steady at 106.0 for any later checks.
        clock = iter([100.0, 106.0])

        def fake_monotonic() -> float:
            try:
                return next(clock)
            except StopIteration:
                return 106.0

        monkeypatch.setattr(manager_module, "_DRAIN_SLEEP_S", 0.0)

        # Spy on the warning logger directly — vllm's logger does not
        # propagate to root, so caplog can't see it.
        warnings: list[str] = []

        def record_warning(msg, *args, **_kwargs):
            warnings.append(msg % args if args else msg)

        monkeypatch.setattr(manager_module.logger, "warning", record_warning)

        # Clear inflight after the first sleep so drain can exit on the
        # next iteration's `pending` check.
        n_sleeps = 0

        def clearing_sleep(_s):
            nonlocal n_sleeps
            n_sleeps += 1
            sess._server._inflight = {}

        # Patch via a fake module on `manager_module.time` so we do not mutate
        # the global `time` module — other code in the process (e.g. the
        # buildkite test collector's pytest_runtest_logreport hook) calls
        # time.monotonic() before monkeypatch teardown.
        fake_time = SimpleNamespace(monotonic=fake_monotonic, sleep=clearing_sleep)
        monkeypatch.setattr(manager_module, "time", fake_time)

        mgr.drain_jobs()

        assert any("still draining after 5s" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# on_schedule_end
# ---------------------------------------------------------------------------


class TestOnScheduleEnd:
    def test_is_noop(self):
        """on_schedule_end is a documented no-op; just confirm it doesn't
        raise and doesn't mutate state."""
        mgr = _make_manager()
        before_sessions = dict(mgr._sessions)
        before_jobs = list(mgr._finished_jobs)
        assert (
            mgr.on_schedule_end(
                ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
            )
            is None
        )
        assert mgr._sessions == before_sessions
        assert mgr._finished_jobs == before_jobs


# ---------------------------------------------------------------------------
# Connection death mid-transfer (real P2PSession via paired managers)
# ---------------------------------------------------------------------------


class TestConnectionDeathMidTransfer:
    """When a peer's control connection dies while a load is in flight,
    the load surfaces as failed and its kv_request_id lands in
    _failed_req_ids so future lookups route to local prefill. The
    prefiller-side store no longer travels through the session at store
    time (it's parked in _unbound_stores keyed by kv_request_id), so its
    cleanup on connection death is via on_request_finished or the
    unbound-store timeout — covered separately below."""

    def test_dead_connection_with_pending_work_surfaces_failures(self):
        mgr_a, mgr_b = _build_paired_managers()

        a_decoder_params = {
            "prefill": {
                "kv_request_id": "req-load",
                "remote_host": "B",
                "remote_port": 2,
            },
        }
        a_prefiller_params = {"decode": {"kv_request_id": "req-store"}}

        # Open the outbound session A->B and submit one load + one store.
        mgr_a.on_new_request(_req_context(a_decoder_params))
        mgr_a.submit_store(
            _job_metadata(
                job_id=900,
                keys=[b"a-block"],
                block_ids=[0],
                kv_params=a_prefiller_params,
            )
        )
        mgr_a.submit_load(
            _job_metadata(
                job_id=901,
                keys=[b"b-block"],
                block_ids=[0],
                kv_params=a_decoder_params,
            )
        )

        # Drain anything the loopback can deliver synchronously, but stop
        # before the remote side has had time to complete the transfers.
        list(mgr_a.get_finished_jobs())

        # Sanity: store 900 is parked in unbound_stores, not in any
        # session — the producer no longer learns the peer at store time.
        assert "req-store" in mgr_a._unbound_stores

        # Kill the control connection out from under the session.
        peer_id = "B:2"
        sess = mgr_a._sessions[peer_id]
        assert sess._conn is not None
        sess._conn.mark_dead()

        # Reap surfaces the load (which lived inside the session) as
        # failed. The store 900 is not session-scoped — it survives the
        # session reap and waits for the unbound-store timeout.
        results: list[JobResult] = []
        for _ in range(3):
            results.extend(list(mgr_a.get_finished_jobs()))

        outcomes = {(r.job_id, r.success) for r in results}
        assert (901, False) in outcomes, f"load should fail: {outcomes}"
        assert (900, False) not in outcomes, f"store should still be parked: {outcomes}"
        assert "req-load" in mgr_a._failed_req_ids
        # Session removed.
        assert peer_id not in mgr_a._sessions
        # Store batch is still parked.
        assert "req-store" in mgr_a._unbound_stores

        # The engine signals the producer's request is done. That is a
        # no-op for the parked batch — `on_request_finished` does not
        # evict unbound stores; only `_reap_unbound_stores` does, after
        # the unbound-store timeout. Job 900 stays unfinished here.
        mgr_a.on_request_finished(_req_context(a_prefiller_params))
        finishes = {(r.job_id, r.success) for r in mgr_a._finished_jobs}
        assert (900, False) not in finishes
        assert (900, True) not in finishes
        assert "req-store" in mgr_a._unbound_stores
