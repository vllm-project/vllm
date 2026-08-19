# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AsyncLookupManager."""

import threading
from collections.abc import Iterable

import pytest

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

    def test_mark_miss_flips_cached_verdict_without_reprobing(self):
        """Failed-load livelock regression (#49176). After a failed load the
        tier calls mark_miss(), flipping the cached True to False; every
        subsequent lookup then returns False (MISS) directly, WITHOUT enqueuing
        a fresh batch_lookup — that is what makes the request unable to loop.
        The entry is retained (as False) not dropped, so cleanup()'s reverse
        index stays consistent; an unknown key is a no-op."""
        mgr = InMemoryLookupManager(existing_keys={_key(1), _key(2)})
        ctx = _ctx("reqA")
        assert mgr.lookup(_key(1), ctx) is None
        mgr.lookup(_key(2), ctx)
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), ctx) is True

        # An unknown key must not raise or plant an entry.
        mgr.mark_miss([_key(99)])
        assert _key(99) not in mgr._lookup_state

        # The block is still "present" per the backing set, so a re-probe would
        # wrongly return True again — the verdict must be served from cache as
        # False, with no re-probe enqueued, and stay False across steps.
        mgr.mark_miss([_key(1)])
        assert mgr.lookup(_key(1), ctx) is False
        assert mgr._lookup_batch == []  # no fresh probe enqueued
        mgr.flush()
        assert mgr._lookup_queue.empty()  # nothing posted to the worker
        assert mgr.lookup(_key(1), ctx) is False

        # Entry retained (now False) with reverse index intact, so cleanup()
        # (which direct-indexes _lookup_state per reverse-index key) tears down
        # both structures without raising.
        assert mgr._lookup_state[_key(1)].result is False
        assert _key(1) in mgr._req_keys["reqA"]
        mgr.cleanup("reqA")
        assert _key(1) not in mgr._lookup_state
        assert _key(2) not in mgr._lookup_state
        assert "reqA" not in mgr._req_keys
        mgr.shutdown()

    def test_enqueue_once_invariant_enforced(self):
        """A key is enqueued for probing exactly once, so drain_results() may
        receive at most one result per key. Normal operation resolves a key a
        single time; a second result for an already-decided key trips the assert
        that guards the invariant (a silent overwrite could flip a corrected
        miss back to True and reopen the failed-load livelock)."""
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx = _ctx("reqA")

        # (a) Normal operation: the key is enqueued once and resolved once, with
        # no second result left pending.
        assert mgr.lookup(_key(1), ctx) is None
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), ctx) is True  # decided exactly once
        assert mgr._pending_results.empty()

        # (b) A stray/duplicate result for the now-decided key violates the
        # enqueue-once invariant and must trip the assert.
        mgr._pending_results.put([(_key(1), True)])
        with pytest.raises(AssertionError):
            mgr.drain_results()
        mgr.shutdown()
