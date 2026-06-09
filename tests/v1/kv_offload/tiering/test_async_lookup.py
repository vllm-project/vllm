# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AsyncLookupManager."""

import time
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

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        return (k in self._existing for k in keys)


class TestAsyncLookupManager:
    def _wait_for_drain(self, mgr: InMemoryLookupManager, timeout: float = 1.0):
        """Spin until drain_results has something to apply."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not mgr._pending_results.empty():
                return
            time.sleep(0.01)
        raise TimeoutError("Worker did not post results in time")

    def test_new_key_returns_none(self):
        mgr = InMemoryLookupManager()
        assert mgr.lookup(_key(1), _ctx()) is None
        mgr.shutdown()

    def test_found_key_returns_true(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        assert mgr.lookup(_key(1), _ctx()) is None
        mgr.flush()
        self._wait_for_drain(mgr)
        assert mgr.lookup(_key(1), _ctx()) is True
        mgr.shutdown()

    def test_not_found_key_returns_false(self):
        mgr = InMemoryLookupManager(existing_keys=set())
        assert mgr.lookup(_key(1), _ctx()) is None
        mgr.flush()
        self._wait_for_drain(mgr)
        assert mgr.lookup(_key(1), _ctx()) is False
        mgr.shutdown()

    def test_multiple_keys_single_step(self):
        existing = {_key(1), _key(3)}
        mgr = InMemoryLookupManager(existing_keys=existing)
        ctx = _ctx()
        for i in range(1, 5):
            assert mgr.lookup(_key(i), ctx) is None
        mgr.flush()
        self._wait_for_drain(mgr)
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
        self._wait_for_drain(mgr)
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
        self._wait_for_drain(mgr)
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
        self._wait_for_drain(mgr)
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
        self._wait_for_drain(mgr)

        # Step 2: lookup keys 2 and 3, flush
        mgr.lookup(_key(2), ctx)
        mgr.lookup(_key(3), ctx)
        mgr.flush()
        self._wait_for_drain(mgr)

        # All results should be available
        assert mgr.lookup(_key(1), ctx) is True
        assert mgr.lookup(_key(2), ctx) is True
        assert mgr.lookup(_key(3), ctx) is True
        mgr.shutdown()

    def test_shutdown_unblocks_worker(self):
        mgr = InMemoryLookupManager()
        mgr.shutdown()
        assert not mgr._thread.is_alive()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
