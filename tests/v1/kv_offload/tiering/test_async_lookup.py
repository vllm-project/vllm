# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AsyncLookupManager."""

import threading
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager


def _key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


def _ctx(req_id: str = "r1") -> ReqContext:
    return ReqContext(req_id=req_id)


class InMemoryLookupManager(AsyncLookupManager):
    """Test subclass backed by an in-memory set."""

    def __init__(self, existing_keys: set[OffloadKey] | None = None):
        super().__init__(tier_type="test")
        self._existing = existing_keys or set()
        self._results_ready = threading.Event()

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        results = [k in self._existing for k in keys]
        self._results_ready.set()
        return results


class TestAsyncLookupManager:
    def test_new_key_returns_none(self):
        mgr = InMemoryLookupManager()
        assert mgr.lookup(_key(1), _ctx()) is None
        mgr.shutdown()

    def test_found_key_returns_true(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        assert mgr.lookup(_key(1), _ctx()) is None
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), _ctx()) is True
        mgr.shutdown()

    def test_not_found_key_returns_false(self):
        mgr = InMemoryLookupManager(existing_keys=set())
        assert mgr.lookup(_key(1), _ctx()) is None
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), _ctx()) is False
        mgr.shutdown()

    def test_multiple_keys_single_step(self):
        existing = {_key(1), _key(3)}
        mgr = InMemoryLookupManager(existing_keys=existing)
        ctx = _ctx()
        for i in range(1, 5):
            assert mgr.lookup(_key(i), ctx) is None
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), ctx) is True
        assert mgr.lookup(_key(2), ctx) is False
        assert mgr.lookup(_key(3), ctx) is True
        assert mgr.lookup(_key(4), ctx) is False
        mgr.shutdown()

    def test_cleanup_removes_entries(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx = _ctx("req_a")
        mgr.lookup(_key(1), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), ctx) is True
        mgr.cleanup("req_a")
        assert _key(1) not in mgr._lookup_state
        mgr.shutdown()

    def test_cleanup_preserves_shared_entries(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx_a = _ctx("req_a")
        ctx_b = _ctx("req_b")
        mgr.lookup(_key(1), ctx_a)
        mgr.lookup(_key(1), ctx_b)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        # Drain so result is applied
        mgr.lookup(_key(1), ctx_a)
        mgr.cleanup("req_a")
        # Key still present because req_b still references it
        assert _key(1) in mgr._lookup_state
        mgr.cleanup("req_b")
        assert _key(1) not in mgr._lookup_state
        mgr.shutdown()

    def test_flush_no_queue_post_when_empty(self):
        mgr = InMemoryLookupManager()
        mgr.flush()
        assert mgr._lookup_queue.empty()
        mgr.shutdown()

    def test_repeated_lookup_same_key_no_duplicate_batch(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx = _ctx()
        mgr.lookup(_key(1), ctx)
        mgr.lookup(_key(1), ctx)
        assert len(mgr._lookup_batch) == 1
        mgr.shutdown()

    def test_cleanup_unknown_req_id_is_noop(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx = _ctx("req_a")
        mgr.lookup(_key(1), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        mgr.lookup(_key(1), ctx)
        mgr.cleanup("nonexistent")
        assert _key(1) in mgr._lookup_state
        mgr.shutdown()

    def test_multiple_flushes_across_steps(self):
        existing = {_key(1), _key(2), _key(3)}
        mgr = InMemoryLookupManager(existing_keys=existing)
        ctx = _ctx()

        # Step 1: lookup key 1, flush
        mgr.lookup(_key(1), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()

        # Step 2: lookup keys 2 and 3, flush
        mgr.lookup(_key(2), ctx)
        mgr.lookup(_key(3), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()

        # All results should be available
        assert mgr.lookup(_key(1), ctx) is True
        assert mgr.lookup(_key(2), ctx) is True
        assert mgr.lookup(_key(3), ctx) is True
        mgr.shutdown()

    def test_shutdown_unblocks_worker(self):
        mgr = InMemoryLookupManager()
        mgr.shutdown()
        assert not mgr._thread.is_alive()

    def test_invalidate_drops_cached_verdict(self):
        """Regression for the failed-load livelock (#49176): after
        invalidate(), the cached True is gone and the next lookup re-runs a
        fresh existence check instead of serving the stale verdict."""
        existing = {_key(1)}
        mgr = InMemoryLookupManager(existing_keys=existing)
        ctx = _ctx("livelock")
        assert mgr.lookup(_key(1), ctx) is None
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), ctx) is True

        # The backing data goes away and the load fails; the tier calls
        # invalidate. The same request must now get a fresh (negative)
        # answer, not the cached True.
        existing.discard(_key(1))
        mgr.invalidate([_key(1)])
        assert mgr.lookup(_key(1), ctx) is None  # re-enqueued, not cached
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), ctx) is False
        mgr.shutdown()

    def test_invalidate_keeps_reverse_index_consistent(self):
        """cleanup() direct-indexes _lookup_state for every key in its
        reverse index; invalidate() must maintain both structures or a
        later cleanup() raises KeyError."""
        mgr = InMemoryLookupManager(existing_keys={_key(1), _key(2)})
        ctx = _ctx("reqA")
        mgr.lookup(_key(1), ctx)
        mgr.lookup(_key(2), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()

        mgr.invalidate([_key(1)])
        mgr.cleanup("reqA")  # must not raise
        assert _key(1) not in mgr._lookup_state
        assert _key(2) not in mgr._lookup_state
        assert "reqA" not in mgr._req_keys
        mgr.shutdown()

    def test_invalidate_unknown_key_is_noop(self):
        mgr = InMemoryLookupManager()
        mgr.invalidate([_key(99)])  # must not raise
        mgr.shutdown()

    def test_late_result_cannot_resurrect_invalidated_entry(self):
        """A worker result that arrives after invalidate() must be
        discarded, not recreate the entry (drain_results' .get() guard)."""
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx = _ctx("late")
        mgr.lookup(_key(1), ctx)
        mgr.invalidate([_key(1)])
        # Simulate the in-flight batch resolving late.
        mgr._pending_results.put([(_key(1), True)])
        mgr.drain_results()
        assert _key(1) not in mgr._lookup_state
        mgr.shutdown()
