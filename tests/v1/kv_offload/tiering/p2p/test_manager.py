# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for P2PSecondaryTierManager.

Tests the manager's job routing, session lifecycle, and result collection
using fake transport and session objects.
"""

from __future__ import annotations

import numpy as np

from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult
from vllm.v1.kv_offload.tiering.p2p.manager import P2PSecondaryTierManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kv_params(
    remote_host: str = "10.0.0.1",
    remote_port: int = 8000,
    kv_request_id: str = "req-1",
    do_remote_decode: bool = True,
) -> dict:
    return {
        "remote_host": remote_host,
        "remote_port": remote_port,
        "kv_request_id": kv_request_id,
        "do_remote_decode": do_remote_decode,
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
    """Test the lookup method using a manager with mocked internals."""

    def _make_manager(self) -> P2PSecondaryTierManager:
        """Create a manager with stubbed __init__."""
        mgr = P2PSecondaryTierManager.__new__(P2PSecondaryTierManager)
        mgr._local_id = "127.0.0.1:7777"
        mgr._finished_jobs = []
        mgr._failed_req_ids = set()
        mgr._server_sessions = {}
        mgr._client_sessions = {}
        return mgr

    def test_lookup_returns_false_without_kv_params(self):
        mgr = self._make_manager()
        ctx = _req_context(kv_params=None)
        assert mgr.lookup(b"key", ctx) is False

    def test_lookup_returns_false_without_required_fields(self):
        mgr = self._make_manager()
        ctx = _req_context(kv_params={"remote_host": "x"})
        assert mgr.lookup(b"key", ctx) is False

    def test_lookup_returns_true_for_valid_request(self):
        mgr = self._make_manager()
        ctx = _req_context(kv_params=_kv_params())
        assert mgr.lookup(b"key", ctx) is True

    def test_lookup_returns_false_for_failed_request(self):
        mgr = self._make_manager()
        mgr._failed_req_ids.add("req-1")
        ctx = _req_context(kv_params=_kv_params(kv_request_id="req-1"))
        assert mgr.lookup(b"key", ctx) is False

    def test_lookup_returns_true_for_different_request_id(self):
        mgr = self._make_manager()
        mgr._failed_req_ids.add("req-1")
        ctx = _req_context(kv_params=_kv_params(kv_request_id="req-2"))
        assert mgr.lookup(b"key", ctx) is True


# ---------------------------------------------------------------------------
# Tests for submit_store
# ---------------------------------------------------------------------------


class TestSubmitStore:
    def _make_manager(self) -> P2PSecondaryTierManager:
        mgr = P2PSecondaryTierManager.__new__(P2PSecondaryTierManager)
        mgr._local_id = "127.0.0.1:7777"
        mgr._finished_jobs = []
        mgr._failed_req_ids = set()
        mgr._server_sessions = {}
        mgr._client_sessions = {}
        return mgr

    def test_no_do_remote_decode_succeeds_immediately(self):
        """Without do_remote_decode, job succeeds immediately."""
        mgr = self._make_manager()
        params = _kv_params()
        del params["do_remote_decode"]
        job = _job_metadata(job_id=1, kv_params=params)
        mgr.submit_store(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=True)]

    def test_missing_kv_request_id_fails(self):
        """Missing kv_request_id fails the job."""
        mgr = self._make_manager()
        params = _kv_params()
        del params["kv_request_id"]
        job = _job_metadata(job_id=1, kv_params=params)
        mgr.submit_store(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=False)]

    def test_no_server_session_fails(self):
        """No server session for peer fails the job."""
        mgr = self._make_manager()
        job = _job_metadata(job_id=1, kv_params=_kv_params())
        mgr.submit_store(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=False)]


# ---------------------------------------------------------------------------
# Tests for submit_load
# ---------------------------------------------------------------------------


class TestSubmitLoad:
    def _make_manager(self) -> P2PSecondaryTierManager:
        mgr = P2PSecondaryTierManager.__new__(P2PSecondaryTierManager)
        mgr._local_id = "127.0.0.1:7777"
        mgr._finished_jobs = []
        mgr._failed_req_ids = set()
        mgr._server_sessions = {}
        mgr._client_sessions = {}
        return mgr

    def test_missing_params_fails(self):
        """Missing required kv_params fields fails the job."""
        mgr = self._make_manager()
        job = _job_metadata(job_id=1, kv_params={})
        mgr.submit_load(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=False)]

    def test_empty_keys_succeeds_immediately(self):
        """Empty key list succeeds immediately."""
        mgr = self._make_manager()
        job = _job_metadata(job_id=1, keys=[], block_ids=[], kv_params=_kv_params())
        mgr.submit_load(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=True)]

    def test_no_client_session_fails(self):
        """No client session for peer fails and marks request failed."""
        mgr = self._make_manager()
        job = _job_metadata(job_id=1, kv_params=_kv_params())
        mgr.submit_load(job)
        assert mgr._finished_jobs == [JobResult(job_id=1, success=False)]
        assert "req-1" in mgr._failed_req_ids


# ---------------------------------------------------------------------------
# Tests for on_request_finished
# ---------------------------------------------------------------------------


class TestOnRequestFinished:
    def _make_manager(self) -> P2PSecondaryTierManager:
        mgr = P2PSecondaryTierManager.__new__(P2PSecondaryTierManager)
        mgr._local_id = "127.0.0.1:7777"
        mgr._finished_jobs = []
        mgr._failed_req_ids = {"req-1"}
        mgr._server_sessions = {}
        mgr._client_sessions = {}
        return mgr

    def test_prunes_failed_req_ids(self):
        """on_request_finished removes the kv_request_id from failed set."""
        mgr = self._make_manager()
        ctx = _req_context(kv_params=_kv_params(kv_request_id="req-1"))
        mgr.on_request_finished(ctx)
        assert "req-1" not in mgr._failed_req_ids

    def test_no_kv_params_does_nothing(self):
        """No kv_params is a no-op."""
        mgr = self._make_manager()
        ctx = _req_context(kv_params=None)
        mgr.on_request_finished(ctx)
        assert "req-1" in mgr._failed_req_ids

    def test_no_kv_request_id_does_nothing(self):
        """Missing kv_request_id is a no-op."""
        mgr = self._make_manager()
        ctx = _req_context(kv_params={"remote_host": "x", "remote_port": 1})
        mgr.on_request_finished(ctx)
        assert "req-1" in mgr._failed_req_ids


# ---------------------------------------------------------------------------
# Tests for get_finished_jobs
# ---------------------------------------------------------------------------


class TestGetFinished:
    def _make_manager(self) -> P2PSecondaryTierManager:
        mgr = P2PSecondaryTierManager.__new__(P2PSecondaryTierManager)
        mgr._local_id = "127.0.0.1:7777"
        mgr._finished_jobs = [
            JobResult(job_id=1, success=True),
            JobResult(job_id=2, success=False),
        ]
        mgr._failed_req_ids = set()
        mgr._server_sessions = {}
        mgr._client_sessions = {}

        # Mock control transport that returns no new connections
        class FakeControl:
            def poll(self):
                return []

        mgr._control = FakeControl()  # type: ignore[assignment]
        mgr._data = None  # type: ignore[assignment]
        return mgr

    def test_drains_finished_jobs(self):
        """get_finished_jobs returns and clears accumulated results."""
        mgr = self._make_manager()
        results = list(mgr.get_finished_jobs())
        assert len(results) == 2
        assert JobResult(job_id=1, success=True) in results
        assert JobResult(job_id=2, success=False) in results

        # Second call returns empty
        assert list(mgr.get_finished_jobs()) == []

    def test_reaps_dead_server_sessions(self):
        """Dead server sessions are removed and jobs failed."""

        class FakeServerSession:
            peer_id = "dead:1234"
            alive = False

            def poll(self):
                return []

            def close(self):
                return [10, 11]

        class FakeData:
            def remove_remote_peer(self, pid):
                pass

        mgr = self._make_manager()
        mgr._data = FakeData()  # type: ignore[assignment]
        mgr._server_sessions["dead:1234"] = FakeServerSession()  # type: ignore[assignment]

        results = list(mgr.get_finished_jobs())
        # Original 2 + 2 failed from dead session
        assert len(results) == 4
        assert JobResult(job_id=10, success=False) in results
        assert JobResult(job_id=11, success=False) in results
        assert "dead:1234" not in mgr._server_sessions
