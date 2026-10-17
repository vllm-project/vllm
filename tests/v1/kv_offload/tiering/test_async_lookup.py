# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AsyncLookupManager."""

import threading
from collections.abc import Iterable

import pytest

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.async_lookup import (
    AsyncLookupManager,
    LookupResultStatus,
    LookupSubmissionStatus,
    LookupToken,
)


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
        self.batch_lookup_calls = 0

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        self.batch_lookup_calls += 1
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
        assert (
            mgr._lookup_state[_key(1)].submission_status
            is LookupSubmissionStatus.PENDING
        )
        mgr.flush()
        assert (
            mgr._lookup_state[_key(1)].submission_status
            is LookupSubmissionStatus.IN_FLIGHT
        )
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), _ctx()) is True
        assert (
            mgr._lookup_state[_key(1)].request_statuses["r1"] is LookupResultStatus.HIT
        )
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

    def test_cleanup_reuses_in_flight_probe(self, monkeypatch: pytest.MonkeyPatch):
        """A replacement request shares the in-flight probe and its verdict."""
        key = _key(1)
        mgr = InMemoryLookupManager(existing_keys={key})
        ctx_b = _ctx("req_b")
        probe_started = threading.Event()
        release_probe = threading.Event()
        batch_lookup = mgr.batch_lookup

        def blocking_lookup(keys, req_context):
            probe_started.set()
            if not release_probe.wait(timeout=5):
                raise TimeoutError("Test did not release the backend probe")
            return batch_lookup(keys, req_context)

        monkeypatch.setattr(mgr, "batch_lookup", blocking_lookup)
        try:
            assert mgr.lookup(key, _ctx("req_a")) is None
            mgr.flush()
            assert probe_started.wait(timeout=5)
            assert (
                mgr._lookup_state[key].submission_status
                is LookupSubmissionStatus.IN_FLIGHT
            )
            mgr.cleanup("req_a")
            assert mgr.lookup(key, ctx_b) is None
            assert (
                mgr._lookup_state[key].submission_status
                is LookupSubmissionStatus.IN_FLIGHT
            )
            mgr.flush()

            release_probe.set()
            batch = mgr._pending_results.get(timeout=5)
            mgr._pending_results.put(batch)
            replacement_result = mgr.lookup(key, ctx_b)
        finally:
            release_probe.set()
            mgr.shutdown()

        assert mgr.batch_lookup_calls == 1
        assert replacement_result is True

    @pytest.mark.parametrize(
        "reclaim_at_shutdown", [False, True], ids=["flush", "shutdown"]
    )
    def test_unclaimed_probe_reclaimed_without_lookup(self, reclaim_at_shutdown: bool):
        """A completed orphan is released by flush or shutdown without a lookup."""
        key = _key(1)
        mgr = InMemoryLookupManager(existing_keys={key})
        try:
            mgr.lookup(key, _ctx("req_a"))
            mgr.flush()
            mgr.cleanup("req_a")
            assert key in mgr._lookup_state
            assert not mgr._req_keys

            if not reclaim_at_shutdown:
                batch = mgr._pending_results.get(timeout=5)
                mgr._pending_results.put(batch)
                mgr.flush()
                assert key not in mgr._lookup_state
        finally:
            mgr.shutdown()

        assert not mgr._lookup_state
        assert mgr._pending_results.empty()

    def test_stale_result_ignored_after_cleanup_and_key_reuse(self):
        key = _key(1)
        mgr = InMemoryLookupManager()
        ctx_a = _ctx("req_a")
        ctx_b = _ctx("req_b")
        assert mgr.lookup(key, ctx_a) is None
        stale_generation = mgr._lookup_state[key].generation
        mgr.cleanup("req_a")

        assert mgr.lookup(key, ctx_b) is None
        generation = mgr._lookup_state[key].generation
        assert generation != stale_generation

        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        current_result = mgr._pending_results.get(timeout=5)
        mgr._pending_results.put([(key, stale_generation, True)])
        mgr._pending_results.put(current_result)
        mgr.drain_results()
        assert mgr.lookup(key, ctx_b) is False
        mgr.shutdown()

    def test_flush_no_queue_post_when_empty(self):
        mgr = InMemoryLookupManager()
        mgr.flush()
        assert mgr._lookup_queue.empty()
        mgr.shutdown()

    def test_flush_skips_lookup_cleaned_up_before_submit(self):
        mgr = InMemoryLookupManager()
        mgr.lookup(_key(1), _ctx("req_a"))
        mgr.cleanup("req_a")
        mgr.flush()
        mgr.shutdown()

        assert mgr.batch_lookup_calls == 0

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

    def test_mark_miss_is_request_scoped(self):
        """A failed promotion changes only the originating request verdict."""
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        ctx_a = _ctx("reqA")
        ctx_b = _ctx("reqB")
        assert mgr.lookup(_key(1), ctx_a) is None
        assert mgr.lookup(_key(1), ctx_b) is None
        mgr.flush()
        mgr._results_ready.wait()
        mgr._results_ready.clear()
        assert mgr.lookup(_key(1), ctx_a) is True
        assert mgr.lookup(_key(1), ctx_b) is True

        mgr.mark_miss([mgr.lookup_token(_key(1), ctx_a.req_id)])
        assert mgr.lookup(_key(1), ctx_a) is False
        assert mgr.lookup(_key(1), ctx_b) is True
        assert (
            mgr._lookup_state[_key(1)].submission_status
            is LookupSubmissionStatus.RESOLVED
        )
        assert (
            mgr._lookup_state[_key(1)].request_statuses[ctx_a.req_id]
            is LookupResultStatus.MARKED_MISS
        )
        mgr.flush()
        assert mgr._lookup_queue.empty()
        assert mgr.lookup(_key(1), ctx_a) is False
        assert mgr.lookup(_key(1), ctx_b) is True

        mgr.cleanup(ctx_a.req_id)
        mgr.cleanup(ctx_b.req_id)
        assert _key(1) not in mgr._lookup_state
        mgr.shutdown()

    def test_mark_miss_unknown_token_is_noop(self):
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        try:
            mgr.mark_miss([LookupToken(key=_key(99), req_id="missing", generation=0)])
            assert _key(99) not in mgr._lookup_state
        finally:
            mgr.shutdown()

    def test_lookup_token_requires_resolved_verdict(self):
        """A promotion is only initiated by a resolved verdict, so minting a
        token for anything else is a contract violation rather than an
        unclaimed token: an unresolved probe, an unknown key, or a request
        that does not hold the key must all fail loudly."""
        key = _key(1)
        mgr = InMemoryLookupManager(existing_keys={key})
        ctx = _ctx("reqA")
        other = _ctx("reqB")
        try:
            assert mgr.lookup(key, ctx) is None
            with pytest.raises(AssertionError):  # PENDING
                mgr.lookup_token(key, ctx.req_id)
            mgr.flush()
            assert (
                mgr._lookup_state[key].submission_status
                is LookupSubmissionStatus.IN_FLIGHT
            )
            with pytest.raises(AssertionError):  # IN_FLIGHT, not yet drained
                mgr.lookup_token(key, ctx.req_id)
            with pytest.raises(AssertionError):  # unknown key
                mgr.lookup_token(_key(99), ctx.req_id)
            with pytest.raises(AssertionError):  # must hold the key
                mgr.lookup_token(key, other.req_id)

            mgr._results_ready.wait()
            mgr._results_ready.clear()
            assert mgr.lookup(key, ctx) is True
            token = mgr.lookup_token(key, ctx.req_id)
            assert token.generation == mgr._lookup_state[key].generation
        finally:
            mgr.shutdown()

    def test_mark_miss_ignores_stale_generation(self):
        """A failed load from an old probe cannot change a newer probe."""
        key = _key(1)
        mgr = InMemoryLookupManager(existing_keys={key})
        ctx_a = _ctx("reqA")
        ctx_b = _ctx("reqB")
        try:
            assert mgr.lookup(key, ctx_a) is None
            mgr.flush()
            mgr._results_ready.wait()
            mgr._results_ready.clear()
            assert mgr.lookup(key, ctx_a) is True
            old_token = mgr.lookup_token(key, ctx_a.req_id)
            mgr.cleanup(ctx_a.req_id)

            assert mgr.lookup(key, ctx_b) is None
            assert mgr._lookup_state[key].generation != old_token.generation
            mgr.mark_miss([old_token])
            assert (
                mgr._lookup_state[key].submission_status
                is LookupSubmissionStatus.PENDING
            )

            mgr.flush()
            mgr._results_ready.wait()
            mgr._results_ready.clear()
            assert mgr.lookup(key, ctx_b) is True
        finally:
            mgr.shutdown()

    def test_mark_miss_ignores_newer_resolved_probe(self):
        """A stale correction must not overwrite a newer probe that has already
        resolved. A resolved verdict is served from cache without re-probing, so
        flipping it to a miss strands the newer request on a wrong MISS for a
        key the tier still holds -- the gap a phase-only guard leaves open."""
        key = _key(1)
        mgr = InMemoryLookupManager(existing_keys={key})
        ctx_a = _ctx("reqA")
        ctx_b = _ctx("reqB")
        try:
            assert mgr.lookup(key, ctx_a) is None
            mgr.flush()
            mgr._results_ready.wait()
            mgr._results_ready.clear()
            assert mgr.lookup(key, ctx_a) is True
            old_token = mgr.lookup_token(key, ctx_a.req_id)
            mgr.cleanup(ctx_a.req_id)

            # B re-probes and its own probe already resolved to HIT.
            assert mgr.lookup(key, ctx_b) is None
            mgr.flush()
            mgr._results_ready.wait()
            mgr._results_ready.clear()
            assert mgr.lookup(key, ctx_b) is True

            # A's promotion fails only now.
            mgr.mark_miss([old_token])
            assert mgr.lookup(key, ctx_b) is True
            assert mgr._lookup_batch == []  # served from cache, no re-probe
        finally:
            mgr.shutdown()

    @pytest.mark.parametrize("flushed", [True, False], ids=["in_flight", "pending"])
    def test_mark_miss_skips_newer_unresolved_probe(self, flushed: bool):
        """A failed load reported for an older generation of a key must not
        touch a newer probe of that key. Sequence: request A resolves K and
        promotes it, A finishes (cleanup deletes the entry), request B
        re-probes K, then A's load fails and the tier corrects the miss using
        A's token. Forcing B's PENDING / IN_FLIGHT entry to a verdict made
        flush() or drain_results() assert on the scheduler thread."""
        mgr = InMemoryLookupManager(existing_keys={_key(1)})
        try:
            assert mgr.lookup(_key(1), _ctx("A")) is None
            mgr.flush()
            mgr._results_ready.wait()
            mgr._results_ready.clear()
            assert mgr.lookup(_key(1), _ctx("A")) is True
            token_a = mgr.lookup_token(_key(1), "A")
            mgr.cleanup("A")
            assert _key(1) not in mgr._lookup_state

            assert mgr.lookup(_key(1), _ctx("B")) is None
            if flushed:
                mgr.flush()
                mgr._results_ready.wait()
                mgr._results_ready.clear()
            expected = (
                LookupSubmissionStatus.IN_FLIGHT
                if flushed
                else LookupSubmissionStatus.PENDING
            )
            assert mgr._lookup_state[_key(1)].submission_status is expected

            mgr.mark_miss([token_a])
            if not flushed:
                mgr.flush()
                mgr._results_ready.wait()
                mgr._results_ready.clear()
            # B's own probe result is delivered.
            assert mgr.lookup(_key(1), _ctx("B")) is True
        finally:
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
        generation = mgr._lookup_state[_key(1)].generation
        mgr._pending_results.put([(_key(1), generation, True)])
        with pytest.raises(AssertionError):
            mgr.drain_results()
        mgr.shutdown()
