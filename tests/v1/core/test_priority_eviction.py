# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict

import pytest

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import KVCacheBlock, init_none_hash
from vllm.v1.core.priority_eviction_queue import (
    PriorityEvictionQueue,
    RetentionMeta,
)
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_hints.retain import RetainDirective

from .test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)


def _d(**kwargs) -> RetainDirective:
    return RetainDirective(**kwargs)


def make_retain_envelope(directives, scope=None) -> KvHintsEnvelope:
    """Accepts dicts or RetainDirective values (the port converts literals)."""
    payload_directives = [
        asdict(d) if isinstance(d, RetainDirective) else d for d in directives
    ]
    return KvHintsEnvelope(
        protocol_version="0.1",
        message_id="m",
        actions=[
            KvHintAction(
                action_id="a",
                action_type="kv.retain",
                action_version="1.0",
                payload={"scope": scope, "directives": payload_directives},
            )
        ],
    )


class _HintedRequest:
    """Minimal stand-in for the parts of Request the retention hook reads."""

    def __init__(self, directives, scope, session_id=None, num_prompt_tokens=16):
        self.kv_hints = (
            make_retain_envelope(directives, scope) if directives is not None else None
        )
        self.session_id = session_id
        self.num_prompt_tokens = num_prompt_tokens


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _make_block(block_id: int) -> KVCacheBlock:
    return KVCacheBlock(block_id=block_id)


def _set_meta(
    queue: PriorityEvictionQueue,
    block: KVCacheBlock,
    *,
    priority: int,
    expiry: float | None = None,
    scope: str | None = None,
    last_freed: float = 0.0,
) -> None:
    """Test helper: install a sidecar entry directly without going through
    apply_directives. Reaches into the queue's private dict — acceptable
    in tests of the queue itself."""
    queue._meta[block.block_id] = RetentionMeta(
        priority=priority,
        expiry=expiry,
        scope=scope,
        last_freed_time=last_freed,
    )
    # Every writer of _meta indexes the expiry; release_expired walks that
    # index instead of the queue.
    queue._track_expiry(block.block_id)


class TestPriorityEvictionQueue:
    def test_empty_queue(self):
        queue = PriorityEvictionQueue()
        assert queue.num_blocks == 0
        assert queue.pop_lowest() is None

    def test_insert_and_pop_single(self):
        queue = PriorityEvictionQueue()
        block = _make_block(1)
        _set_meta(queue, block, priority=50)
        assert queue.try_insert(block) is True
        assert queue.num_blocks == 1
        popped = queue.pop_lowest()
        assert popped is block
        assert queue.num_blocks == 0

    def test_try_insert_returns_false_for_unprioritized_block(self):
        queue = PriorityEvictionQueue()
        block = _make_block(1)
        # No _set_meta call — sidecar entry absent.
        assert queue.try_insert(block) is False
        assert queue.num_blocks == 0

    def test_eviction_order_by_priority(self):
        queue = PriorityEvictionQueue()
        blocks = [_make_block(i) for i in range(3)]
        # Insert in non-ascending order to verify the heap reorders.
        for blk, p in zip(blocks, [80, 20, 50]):
            _set_meta(queue, blk, priority=p)
            queue.try_insert(blk)
        # Lowest priority must come out first.
        assert queue.pop_lowest() is blocks[1]  # priority 20
        assert queue.pop_lowest() is blocks[2]  # priority 50
        assert queue.pop_lowest() is blocks[0]  # priority 80

    def test_eviction_order_tiebreak_by_time(self):
        queue = PriorityEvictionQueue()
        blocks = [_make_block(i) for i in range(3)]
        # Same priority; differ only in last_freed_time.
        for blk, t in zip(blocks, [300.0, 100.0, 200.0]):
            _set_meta(queue, blk, priority=50, last_freed=t)
            queue.try_insert(blk)
        # Oldest-freed evicted first.
        assert queue.pop_lowest() is blocks[1]  # t=100
        assert queue.pop_lowest() is blocks[2]  # t=200
        assert queue.pop_lowest() is blocks[0]  # t=300

    def test_suspend_keeps_sidecar(self):
        queue = PriorityEvictionQueue()
        block = _make_block(1)
        _set_meta(queue, block, priority=50)
        queue.try_insert(block)
        queue.suspend(block)
        assert queue.num_blocks == 0
        # Sidecar entry survives so that priority returns if the block
        # is freed again later.
        assert block.block_id in queue._meta
        assert queue.try_insert(block) is True
        assert queue.num_blocks == 1

    def test_suspend_nonexistent_is_noop(self):
        queue = PriorityEvictionQueue()
        block = _make_block(1)
        # No insert; remove must not raise.
        queue.suspend(block)
        assert queue.num_blocks == 0

    def test_stale_heap_entries_are_skipped_in_pop_lowest(self):
        queue = PriorityEvictionQueue()
        # Insert two blocks; remove one (leaving a stale heap entry).
        block_a = _make_block(1)
        block_b = _make_block(2)
        _set_meta(queue, block_a, priority=10)
        _set_meta(queue, block_b, priority=50)
        queue.try_insert(block_a)
        queue.try_insert(block_b)
        queue.suspend(block_a)  # block_a now stale in heap
        # pop_lowest must skip the stale entry and return block_b.
        assert queue.pop_lowest() is block_b
        assert queue.pop_lowest() is None

    def test_reinsert_after_suspend_orders_by_current_priority(self):
        """Regression (priority inversion): a block removed (touch/reuse) and
        re-inserted with an ESCALATED priority must be ordered by the current
        priority, not the stale heap tuple left behind by the lazy remove().

        Sequence: A enters at 50 -> remove() (lazy delete leaves a (50,A)
        tuple in the heap) -> A escalated to 90 -> re-inserted (pushes a
        (90,A) tuple; both A tuples now satisfy the block_id-in-_in_queue
        test). A live priority-70 block B must evict BEFORE A. The buggy
        code pops the stale (50,A) tuple first and evicts the
        escalated-to-90 block ahead of the 70 block."""
        queue = PriorityEvictionQueue()
        a = _make_block(1)
        b = _make_block(2)
        _set_meta(queue, a, priority=50)
        queue.try_insert(a)
        queue.suspend(a)  # touch: lazy delete, stale (50,A) tuple stays
        _set_meta(queue, a, priority=90)  # escalation via apply_directives
        queue.try_insert(a)  # re-freed: pushes (90,A); (50,A) still in heap
        _set_meta(queue, b, priority=70)
        queue.try_insert(b)
        assert queue.pop_lowest() is b, (
            "priority-70 block must evict before the escalated-to-90 block; "
            "a stale heap tuple must not order eviction by the old priority."
        )
        assert queue.pop_lowest() is a
        assert queue.pop_lowest() is None

    def test_reinsert_after_suspend_orders_by_current_time(self):
        """Regression (recency tiebreak): same priority, but a block removed
        and re-freed with a NEWER last_freed_time must be ordered by the new
        time. The stale tuple carries the OLD (smaller) time and would
        otherwise make the block look older-freed than it is, evicting it
        before a genuinely older block."""
        queue = PriorityEvictionQueue()
        a = _make_block(1)
        b = _make_block(2)
        _set_meta(queue, a, priority=50, last_freed=100.0)
        queue.try_insert(a, last_freed_time=100.0)
        queue.suspend(a)  # stale (t=100,A) left in heap
        queue.try_insert(a, last_freed_time=300.0)  # re-freed later; A now newest
        _set_meta(queue, b, priority=50, last_freed=200.0)
        queue.try_insert(b, last_freed_time=200.0)
        # A was last freed at 300 (newer than B's 200) -> B evicts first.
        assert queue.pop_lowest() is b, (
            "block re-freed at t=300 must evict after the t=200 block; "
            "a stale t=100 tuple must not defeat the recency tiebreak."
        )
        assert queue.pop_lowest() is a
        assert queue.pop_lowest() is None

    def test_ttl_not_expired(self, monkeypatch):
        import time as time_mod

        queue = PriorityEvictionQueue()
        block = _make_block(1)
        # Set "now" to 100; expiry is at 200 (not yet reached).
        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(queue, block, priority=50, expiry=200.0)
        queue.try_insert(block)
        # Even when "now" advances to 150, expiry (200) is in the future.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 150.0)
        assert queue.pop_lowest() is block

    def test_release_expired_then_pop_lowest_workflow(self, monkeypatch):
        """The full eviction workflow with TTL: release_expired first
        removes expired entries from the queue (caller demotes them to
        LRU). pop_lowest then sees only live entries.

        Expired entries no longer leak into pop_lowest — that was the
        limbo-fix-era band-aid and is superseded by release_expired.
        """
        import time as time_mod

        queue = PriorityEvictionQueue()
        block = _make_block(1)
        # Insert with expiry=200.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(queue, block, priority=50, expiry=200.0)
        queue.try_insert(block)

        # Advance past expiry.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 250.0)

        # New semantic: release_expired returns the block_id, queue is
        # empty afterwards. The caller (BlockPool.get_new_blocks) is
        # responsible for routing the corresponding block into the LRU.
        drained = queue.release_expired()
        assert drained == [block.block_id]
        assert queue.num_blocks == 0
        # pop_lowest now sees an empty queue.
        assert queue.pop_lowest() is None

    def test_release_expired_returns_only_expired_block_ids(self, monkeypatch):
        """release_expired returns block_ids whose sidecar.expiry has passed,
        and removes those entries from _in_queue + _meta. Non-expired
        entries stay in the queue."""
        import time as time_mod

        queue = PriorityEvictionQueue()
        b_exp = _make_block(1)
        b_live = _make_block(2)
        b_no_exp = _make_block(3)

        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(queue, b_exp, priority=50, expiry=150.0)  # will expire
        _set_meta(queue, b_live, priority=70, expiry=250.0)  # still alive
        _set_meta(queue, b_no_exp, priority=90, expiry=None)  # no TTL
        queue.try_insert(b_exp)
        queue.try_insert(b_live)
        queue.try_insert(b_no_exp)

        # Advance time past b_exp's expiry only.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 200.0)
        drained = queue.release_expired()

        assert drained == [b_exp.block_id]
        assert b_exp.block_id not in queue._in_queue
        assert b_exp.block_id not in queue._meta
        assert b_live.block_id in queue._in_queue
        assert b_no_exp.block_id in queue._in_queue
        assert queue.num_blocks == 2

    def test_release_expired_honours_refreshed_expiry(self, monkeypatch):
        """A directive that extends a block's expiry supersedes the earlier one:
        the block is not released at the old expiry and is at the new one. The
        stale index tuple for the old expiry must be skipped, not acted on."""
        import time as time_mod

        queue = PriorityEvictionQueue()
        block = _make_block(1)
        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        directive = _d(start=0, end=None, priority=50, duration=50.0)
        queue.apply_directives([block], [directive], "s", 16)  # expiry 150
        monkeypatch.setattr(time_mod, "monotonic", lambda: 120.0)
        queue.apply_directives([block], [directive], "s", 16)  # expiry 170
        queue.try_insert(block)

        monkeypatch.setattr(time_mod, "monotonic", lambda: 160.0)
        assert queue.release_expired() == []
        assert block in queue

        monkeypatch.setattr(time_mod, "monotonic", lambda: 180.0)
        assert queue.release_expired() == [block.block_id]
        assert block not in queue
        assert block.block_id not in queue._meta

    def test_release_expired_leaves_referenced_block_to_try_insert(self, monkeypatch):
        """A lapsed entry on a block that is still referenced is not the
        pool's to route yet: release_expired leaves it and try_insert drops it
        when the block is finally freed."""
        import time as time_mod

        queue = PriorityEvictionQueue()
        block = _make_block(1)
        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(queue, block, priority=50, expiry=150.0)  # referenced: no insert

        monkeypatch.setattr(time_mod, "monotonic", lambda: 200.0)
        assert queue.release_expired() == []
        assert block.block_id in queue._meta
        assert queue.try_insert(block) is False
        assert block.block_id not in queue._meta

    def test_expiry_index_stays_bounded_under_refresh_churn(self, monkeypatch):
        """Refreshing the same blocks' expiry over and over leaves one stale
        index tuple per refresh; the index is rebuilt once stale tuples
        dominate, so it cannot grow without bound over a long run."""
        import time as time_mod

        queue = PriorityEvictionQueue()
        blocks = [_make_block(i) for i in range(64)]
        directive = _d(start=0, end=None, priority=50, duration=60.0)
        for step in range(400):
            monkeypatch.setattr(time_mod, "monotonic", lambda s=step: 100.0 + s)
            queue.apply_directives(blocks, [directive], "s", 16)
        assert len(queue._expiry_heap) <= 2 * len(queue._meta) + 4096
        for block in blocks:
            queue.try_insert(block)
        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0 + 400 + 61.0)
        assert sorted(queue.release_expired()) == list(range(64))
        assert queue.num_blocks == 0

    def test_release_expired_empty_queue_is_noop(self):
        """release_expired on an empty queue returns an empty list."""
        queue = PriorityEvictionQueue()
        assert queue.release_expired() == []
        assert queue.num_blocks == 0

    def test_try_insert_expired_meta_routes_to_lru(self, monkeypatch):
        """try_insert must return False when the sidecar entry is already
        expired so the caller (free_blocks) routes the block to the LRU
        free list instead of the priority queue. Otherwise the block would
        live in the priority queue forever, or land in limbo on the next
        pop_lowest."""
        import time as time_mod

        queue = PriorityEvictionQueue()
        block = _make_block(1)
        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(queue, block, priority=50, expiry=150.0)
        # Advance past expiry BEFORE try_insert.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 200.0)
        assert queue.try_insert(block, last_freed_time=200.0) is False
        # The expired sidecar must be cleaned up — otherwise a later
        # apply_directives could re-prime the same block back into the
        # priority queue.
        assert block.block_id not in queue._meta
        assert queue.num_blocks == 0


class TestApplyDirectives:
    def _peek_meta(self, queue: PriorityEvictionQueue, block_id: int):
        return queue._meta.get(block_id)

    def test_apply_retention_to_block_single_match(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)  # tokens 0..15 (block_size=16)
        directives = [_d(start=0, end=16, priority=50)]
        queue.apply_directives([block], directives, scope=None, block_size=16)
        meta = self._peek_meta(queue, 0)
        assert meta is not None
        assert meta.priority == 50

    def test_apply_retention_no_match(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)  # tokens 0..15
        # Directive covers tokens 100..200 — no overlap.
        directives = [_d(start=100, end=200, priority=50)]
        queue.apply_directives([block], directives, scope=None, block_size=16)
        assert self._peek_meta(queue, 0) is None

    def test_apply_retention_highest_priority_wins(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)  # tokens 0..15
        directives = [
            _d(start=0, end=16, priority=30),
            _d(start=0, end=16, priority=80),  # higher wins
            _d(start=0, end=16, priority=50),
        ]
        queue.apply_directives([block], directives, scope=None, block_size=16)
        assert self._peek_meta(queue, 0).priority == 80

    def test_apply_retention_open_ended_range(self):
        queue = PriorityEvictionQueue()
        blocks = [_make_block(i) for i in range(3)]  # tokens 0..15, 16..31, 32..47
        # end=None means "from start to end of sequence".
        directives = [_d(start=16, end=None, priority=70)]
        queue.apply_directives(blocks, directives, scope=None, block_size=16)
        assert self._peek_meta(queue, 0) is None  # tokens 0..15 not covered
        assert self._peek_meta(queue, 1).priority == 70
        assert self._peek_meta(queue, 2).priority == 70

    def test_apply_retention_with_duration(self, monkeypatch):
        import time as time_mod

        monkeypatch.setattr(time_mod, "monotonic", lambda: 1000.0)
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        directives = [_d(start=0, end=16, priority=50, duration=60.0)]
        queue.apply_directives([block], directives, scope=None, block_size=16)
        meta = self._peek_meta(queue, 0)
        assert meta.expiry == 1060.0

    def test_escalation_keeps_the_longer_hold(self, monkeypatch):
        """Raising priority must not cut the hold short: a block held until
        t+600 that someone escalates with a 60s duration stays until t+600,
        otherwise a higher-priority claim would make the block expire sooner
        than the lower-priority one it replaced."""
        import time as time_mod

        monkeypatch.setattr(time_mod, "monotonic", lambda: 1000.0)
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=30, scope="alice", expiry=1600.0)
        queue.apply_directives(
            [block],
            [_d(start=0, end=16, priority=80, duration=60.0)],
            scope="bob",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta.priority == 80
        assert meta.expiry == 1600.0

    def test_escalation_extends_a_shorter_hold(self, monkeypatch):
        import time as time_mod

        monkeypatch.setattr(time_mod, "monotonic", lambda: 1000.0)
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=30, scope="alice", expiry=1100.0)
        queue.apply_directives(
            [block],
            [_d(start=0, end=16, priority=80, duration=600.0)],
            scope="bob",
            block_size=16,
        )
        assert self._peek_meta(queue, 0).expiry == 1600.0

    def test_escalation_keeps_an_unlimited_hold(self):
        """An entry with no expiry never expires, so escalating it with a
        duration must not give it one."""
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=30, scope="alice", expiry=None)
        queue.apply_directives(
            [block],
            [_d(start=0, end=16, priority=80, duration=60.0)],
            scope="bob",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta.priority == 80
        assert meta.expiry is None

    def test_escalation_from_different_scope(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=30, scope="alice")
        # bob escalates to 80 — allowed regardless of scope.
        queue.apply_directives(
            [block],
            [_d(start=0, end=16, priority=80)],
            scope="bob",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta.priority == 80
        assert meta.scope == "bob"

    def test_downgrade_blocked_from_different_scope(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=80, scope="alice")
        # bob tries to downgrade to 20 — denied (alice owns the block).
        queue.apply_directives(
            [block],
            [_d(start=0, end=16, priority=20)],
            scope="bob",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta.priority == 80
        assert meta.scope == "alice"

    def test_owner_can_downgrade(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=80, scope="alice")
        queue.apply_directives(
            [block],
            [_d(start=0, end=16, priority=20)],
            scope="alice",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta.priority == 20
        assert meta.scope == "alice"

    def test_owner_saying_nothing_does_not_clear(self):
        """Silence is not a release. An owner whose directives skip a block it
        still holds must not drop that block's protection: its own next turn --
        or another turn reusing the same content -- may be relying on it. A
        caller that really is done names the block at a low priority with a
        short duration instead."""
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50, scope="alice")
        queue.apply_directives(
            [block],
            [_d(start=100, end=200, priority=90)],
            scope="alice",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta is not None, "an uncovered block must keep its protection"
        assert meta.priority == 50
        assert meta.scope == "alice"

    def test_owner_releases_with_priority_zero(self):
        """Priority 0 is the explicit release: the owner names the range and the
        entry is dropped. This is the only directive-driven way to unprotect a
        block, so a caller can hand a block back without an omission doing it by
        accident."""
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50, scope="alice")
        queue.apply_directives(
            [block],
            [_d(start=0, end=16, priority=0)],
            scope="alice",
            block_size=16,
        )
        assert self._peek_meta(queue, 0) is None
        assert block not in queue

    def test_priority_zero_from_non_owner_is_ignored(self):
        """A release is a downgrade, so the same restriction applies: bob must
        not be able to drop protection alice is relying on."""
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50, scope="alice")
        queue.apply_directives(
            [block],
            [_d(start=0, end=16, priority=0)],
            scope="bob",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta is not None
        assert meta.priority == 50
        assert meta.scope == "alice"

    def test_priority_zero_on_unprotected_block_is_noop(self):
        """Releasing something already unprotected must not create an entry."""
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        queue.apply_directives(
            [block],
            [_d(start=0, end=16, priority=0)],
            scope="alice",
            block_size=16,
        )
        assert self._peek_meta(queue, 0) is None

    def test_protection_wins_over_release_on_the_same_block(self):
        """When two directives cover one block, the highest priority decides, so
        a release cannot cancel a protection claim in the same request."""
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50, scope="alice")
        queue.apply_directives(
            [block],
            [
                _d(start=0, end=16, priority=0),
                _d(start=0, end=16, priority=90, duration=30.0),
            ],
            scope="alice",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta is not None
        assert meta.priority == 90

    def test_non_owner_no_clear_on_no_match(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50, scope="alice")
        # bob's directives don't cover this block — alice's entry stays.
        queue.apply_directives(
            [block],
            [_d(start=100, end=200, priority=90)],
            scope="bob",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta is not None
        assert meta.scope == "alice"

    def test_no_scope_no_clear(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50, scope="alice")
        # scope=None caller is anonymous — must not clear anyone's entry.
        queue.apply_directives(
            [block],
            [_d(start=100, end=200, priority=90)],
            scope=None,
            block_size=16,
        )
        assert self._peek_meta(queue, 0) is not None


class TestSidecarLifecycle:
    def test_sidecar_entry_cleared_on_pop_lowest(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50)
        queue.try_insert(block)
        queue.pop_lowest()
        assert 0 not in queue._meta

    def test_sidecar_entry_cleared_on_unprotect(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50)
        queue.try_insert(block)
        queue.unprotect(0)
        assert 0 not in queue._meta
        assert 0 not in queue._in_queue

    def test_sidecar_persists_through_reuse_cycle(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50)
        queue.try_insert(block)
        queue.suspend(block)  # block reused via touch
        # Sidecar entry is preserved so try_insert succeeds again on next free.
        assert queue.try_insert(block) is True
        # And the heap entry reflects the same priority.
        popped = queue.pop_lowest()
        assert popped is block

    def test_sidecar_cleared_on_clear_all(self):
        queue = PriorityEvictionQueue()
        for i in range(3):
            block = _make_block(i)
            _set_meta(queue, block, priority=50)
            queue.try_insert(block)
        queue.clear()
        assert len(queue._meta) == 0
        assert len(queue._in_queue) == 0
        assert queue.num_blocks == 0

    def test_unprotect_for_unknown_block_is_noop(self):
        queue = PriorityEvictionQueue()
        queue.unprotect(999)  # not present — must not raise

    def test_contains_reflects_heap_membership(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50)
        assert block not in queue
        queue.try_insert(block)
        assert block in queue
        queue.suspend(block)
        assert block not in queue


class TestBlockPoolPriorityEviction:
    def _make_pool(self, num_blocks=8, block_size=16):
        from vllm.v1.core.block_pool import BlockPool

        return BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=True,
            hash_block_size=block_size,
            enable_kv_cache_events=False,
        )

    def test_pool_has_priority_eviction_queue(self):
        pool = self._make_pool()
        assert isinstance(pool.priority_eviction_queue, PriorityEvictionQueue)
        assert pool.priority_eviction_queue.num_blocks == 0

    def test_reset_prefix_cache_clears_priority_queue(self):
        pool = self._make_pool()
        # Seed the queue with one prioritized free block. Remove from the
        # LRU first so the block lives in exactly one queue, matching the
        # invariant that free_blocks enforces.
        block = pool.blocks[1]
        pool.free_block_queue.remove(block)
        _set_meta(pool.priority_eviction_queue, block, priority=50)
        pool.priority_eviction_queue.try_insert(block)
        assert pool.priority_eviction_queue.num_blocks == 1
        pool.reset_prefix_cache()
        assert pool.priority_eviction_queue.num_blocks == 0

    def test_reset_prefix_cache_returns_priority_blocks_to_lru(self):
        """reset_prefix_cache must hand every block held by the priority
        eviction queue back to the LRU free list. Clearing the queue's
        bookkeeping alone leaks the blocks out of BOTH queues — no
        allocator path can ever reach them again, permanently shrinking
        the pool until engine restart: usage stays pinned at the directive
        size, any larger request waits forever, and follow-up resets fail
        with "blocks not freed"."""
        pool = self._make_pool(num_blocks=8, block_size=16)
        num_free_before = pool.get_num_free_blocks()
        for block_id in (1, 2, 3):
            block = pool.blocks[block_id]
            pool.free_block_queue.remove(block)
            _set_meta(pool.priority_eviction_queue, block, priority=50)
            pool.priority_eviction_queue.try_insert(block)
        # Soft-pin invariant: prioritized blocks still count as allocatable.
        assert pool.get_num_free_blocks() == num_free_before

        assert pool.reset_prefix_cache() is True
        assert pool.priority_eviction_queue.num_blocks == 0
        assert pool.free_block_queue.num_free_blocks == num_free_before, (
            "blocks held by the priority queue must return to the LRU free "
            "list on reset; leaking them shrinks the pool permanently"
        )
        # And every block must be allocatable again.
        blocks = pool.get_new_blocks(num_free_before)
        assert len(blocks) == num_free_before

    def test_cache_full_blocks_routes_directives_to_queue(self):
        pool = self._make_pool(num_blocks=8, block_size=16)
        request = _HintedRequest([{"start": 0, "end": 16, "priority": 80}], "alice")

        blocks = [pool.blocks[1]]
        # Drive the hook directly to isolate its behavior from the rest of
        # cache_full_blocks.
        pool._apply_retention_hook(request, blocks, num_full_blocks=1, block_size=16)

        meta = pool.priority_eviction_queue._meta.get(blocks[0].block_id)
        assert meta is not None
        assert meta.priority == 80
        assert meta.scope == "alice"

    def test_no_hints_zero_overhead_path(self):
        pool = self._make_pool()
        request = _HintedRequest(None, None)
        blocks = [pool.blocks[1]]
        pool._apply_retention_hook(request, blocks, num_full_blocks=1, block_size=16)
        assert pool.priority_eviction_queue.num_blocks == 0
        assert blocks[0].block_id not in pool.priority_eviction_queue._meta

    def test_eviction_drains_lru_before_priority(self):
        pool = self._make_pool(num_blocks=8, block_size=16)
        # Mark 3 blocks as prioritized, remove them from LRU first so they
        # live exclusively in the priority queue (avoids double-allocation).
        prioritized_ids = [1, 2, 3]
        for bid in prioritized_ids:
            block = pool.blocks[bid]
            pool.free_block_queue.remove(block)
            _set_meta(pool.priority_eviction_queue, block, priority=50)
            pool.priority_eviction_queue.try_insert(block)
        # After removal from LRU, count the remaining LRU-only free blocks.
        free_lru_before = pool.free_block_queue.num_free_blocks
        # Allocate up to free_lru_before + 1 blocks — the +1 must come from
        # the priority queue.
        ret = pool.get_new_blocks(free_lru_before + 1)
        assert len(ret) == free_lru_before + 1
        # The last block returned should be the one we marked prioritized
        # (since the LRU drained first).
        assert ret[-1].block_id in prioritized_ids

    def test_get_num_free_blocks_sums_both(self):
        pool = self._make_pool(num_blocks=8)
        free_before = pool.get_num_free_blocks()
        block = pool.blocks[1]
        _set_meta(pool.priority_eviction_queue, block, priority=50)
        pool.priority_eviction_queue.try_insert(block)
        # The block is "in" both the LRU and the priority queue at this
        # point — but the LRU count remains the same; the priority count
        # adds.
        assert pool.get_num_free_blocks() == free_before + 1

    def test_touch_removes_from_priority_queue(self):
        pool = self._make_pool()
        block = pool.blocks[1]
        block.ref_cnt = 0  # free state
        # Remove from LRU first so the block lives in exactly one queue,
        # consistent with the Task-13 invariant.
        pool.free_block_queue.remove(block)
        _set_meta(pool.priority_eviction_queue, block, priority=50)
        pool.priority_eviction_queue.try_insert(block)
        assert pool.priority_eviction_queue.num_blocks == 1
        pool.touch([block])
        # touch() reuses the block: it must leave the priority queue.
        assert pool.priority_eviction_queue.num_blocks == 0
        assert block.ref_cnt == 1

    def test_touch_removes_from_lru(self):
        pool = self._make_pool()
        # Use a block already in LRU (not prioritized).
        block = pool.blocks[2]
        block.ref_cnt = 0
        # Block is in free_block_queue by default after init.
        pool.touch([block])
        assert block.ref_cnt == 1

    def test_free_unprioritized_goes_to_lru(self):
        pool = self._make_pool()
        block = pool.blocks[1]
        block.ref_cnt = 1
        # Block must not be in LRU while ref_cnt > 0 — remove it first to
        # mirror the production state of an active block.
        pool.free_block_queue.remove(block)
        free_before = pool.free_block_queue.num_free_blocks
        pool.free_blocks([block])
        assert pool.free_block_queue.num_free_blocks == free_before + 1
        assert pool.priority_eviction_queue.num_blocks == 0

    def test_free_prioritized_goes_to_priority_queue(self, monkeypatch):
        import time as time_mod

        monkeypatch.setattr(time_mod, "monotonic", lambda: 12345.0)
        pool = self._make_pool()
        block = pool.blocks[1]
        block.ref_cnt = 1
        # Block must not be in LRU while ref_cnt > 0 — remove it first.
        pool.free_block_queue.remove(block)
        # Install a sidecar entry so try_insert recognizes the block as
        # prioritized.
        _set_meta(pool.priority_eviction_queue, block, priority=50)
        free_before = pool.free_block_queue.num_free_blocks
        pool.free_blocks([block])
        # Did NOT land in the LRU queue.
        assert pool.free_block_queue.num_free_blocks == free_before
        # Did land in the priority queue with updated last_freed_time.
        assert pool.priority_eviction_queue.num_blocks == 1
        assert (
            pool.priority_eviction_queue._meta[block.block_id].last_freed_time
            == 12345.0
        )

    def test_touch_on_limbo_block_does_not_raise(self):
        """A block in neither the priority queue nor the LRU free list
        must not crash touch(). This guards against the pre-fix scenario
        where pop_lowest silently dropped an expired entry, leaving the
        block in limbo and crashing the next prefix-cache hit."""
        pool = self._make_pool()
        block = pool.blocks[1]
        # Take it out of LRU by hand to simulate the post-pop_lowest
        # limbo: ref_cnt=0, not in priority queue, not in free list.
        pool.free_block_queue.remove(block)
        assert block.prev_free_block is None
        assert block.next_free_block is None
        assert block not in pool.priority_eviction_queue
        assert block.ref_cnt == 0
        # touch() must not raise.
        pool.touch([block])
        assert block.ref_cnt == 1

    def test_get_new_blocks_drains_all_expired_to_lru(self, monkeypatch):
        """release_expired must move ALL expired entries to the LRU, not
        just enough to satisfy the current allocation. Otherwise the next
        get_new_blocks call would re-evict the entries left behind in the
        priority queue.

        Pre-fix behavior: get_new_blocks(1) pops 1 entry from the
        priority queue via pop_lowest, leaves the other 2 expired
        entries in the queue. Each subsequent get_new_blocks would
        evict another cached block from the map.

        Post-fix behavior: release_expired moves all 3 expired blocks to
        the LRU tail BEFORE any pop happens. get_new_blocks(1) then
        pops 1 from the LRU and the other 2 expired blocks sit in the
        LRU with their cached hashes intact until normal LRU order
        reaches them.
        """
        import time as time_mod

        pool = self._make_pool()

        # Stash 3 blocks into the priority queue with expiring sidecars.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        target_ids = []
        for bid in (1, 2, 3):
            block = pool.blocks[bid]
            pool.free_block_queue.remove(block)  # take out of LRU
            _set_meta(
                pool.priority_eviction_queue,
                block,
                priority=50,
                expiry=150.0,
                last_freed=100.0,
            )
            pool.priority_eviction_queue.try_insert(block, last_freed_time=100.0)
            target_ids.append(bid)
        assert pool.priority_eviction_queue.num_blocks == 3
        lru_before = pool.free_block_queue.num_free_blocks

        # Advance past expiry, then ask for ONE block.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 200.0)
        allocated = pool.get_new_blocks(1)
        assert len(allocated) == 1

        # Post-fix invariant: the priority queue is fully drained (0
        # entries), AND the LRU has gained 2 entries (we drained 3 and
        # consumed 1).
        # Pre-fix would leave 2 entries in the priority queue and the
        # LRU would have lost 0 entries net (started empty for our 3
        # blocks, ended empty too).
        assert pool.priority_eviction_queue.num_blocks == 0, (
            f"priority queue should be drained, has "
            f"{pool.priority_eviction_queue.num_blocks} entries"
        )
        assert pool.free_block_queue.num_free_blocks == lru_before + 2, (
            f"LRU should have gained 2 demoted-from-priority entries; "
            f"got {pool.free_block_queue.num_free_blocks - lru_before} delta"
        )
        # Sidecars also cleaned up for all 3.
        for bid in target_ids:
            assert bid not in pool.priority_eviction_queue._meta

    def test_evict_blocks_clears_sidecar(self):
        pool = self._make_pool()
        block = pool.blocks[1]
        _set_meta(pool.priority_eviction_queue, block, priority=50)
        # Don't insert into heap — just install sidecar (simulating a
        # block whose ref_cnt > 0 but had a prior priority).
        pool.evict_blocks({block.block_id})
        assert block.block_id not in pool.priority_eviction_queue._meta

    def test_priority_queue_pop_clears_cache_map(self, monkeypatch):
        """EAGER variant: a priority-queue pop is treated as a real
        eviction — it resets the block hash and drops the cache map
        entry, exactly like an LRU pop. (The lazy variant on
        `retention-minimal` preserves them instead; this assertion marks
        the eager/lazy difference.)
        """
        import time as time_mod

        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            make_block_hash_with_group_id,
        )

        pool = self._make_pool()
        block = pool.blocks[1]
        # Promote to in-use + cached, then free into the priority queue.
        pool.free_block_queue.remove(block)
        block.ref_cnt = 1
        raw_hash = BlockHash((42).to_bytes(32, "little"))
        h = make_block_hash_with_group_id(raw_hash, 0)
        block.set_block_hash(h)
        pool.cached_block_hash_to_block.insert(h, block)

        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(
            pool.priority_eviction_queue,
            block,
            priority=50,
            expiry=None,
            last_freed=100.0,
        )
        pool.free_blocks([block])
        assert block in pool.priority_eviction_queue
        assert pool.get_cached_block(raw_hash, [0]) is not None

        # Drain the LRU so the next get_new_blocks must dip into the
        # priority queue.
        for b in list(pool.free_block_queue.get_all_free_blocks()):
            if b is not block and b is not pool.null_block:
                pool.free_block_queue.remove(b)
                b.ref_cnt = 1
        assert pool.free_block_queue.num_free_blocks == 0
        assert pool.priority_eviction_queue.num_blocks == 1

        # Now pop via get_new_blocks. The block must come back with its
        # hash intact and the cache map entry preserved.
        allocated = pool.get_new_blocks(1)
        assert len(allocated) == 1
        assert allocated[0] is block
        assert block.ref_cnt == 1
        # Eager guarantee: PQ pop is a real eviction.
        assert block.block_hash is None, (
            "Eager: block hash must be reset on priority-queue pop "
            "(same as an LRU pop)."
        )
        cached = pool.get_cached_block(raw_hash, [0])
        assert cached is None, (
            "Eager: cached_block_hash_to_block entry must be evicted on "
            "priority-queue pop. (Lazy variant preserves it for later "
            "prefix hits — eager gives that up.)"
        )

    def test_lru_popleft_still_clears_cache_map(self):
        """LRU eviction's semantics are unchanged: popleft on a cached
        block must clear the cache map entry and reset the hash. Guards
        against accidentally decoupling the LRU path along with the
        priority-queue path."""
        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            make_block_hash_with_group_id,
        )

        pool = self._make_pool()
        block = pool.blocks[1]
        # Cache the block but leave it in LRU (no retention meta).
        pool.free_block_queue.remove(block)
        block.ref_cnt = 1
        raw_hash = BlockHash((77).to_bytes(32, "little"))
        h = make_block_hash_with_group_id(raw_hash, 0)
        block.set_block_hash(h)
        pool.cached_block_hash_to_block.insert(h, block)
        pool.free_blocks([block])
        # Block is now in LRU with cache map entry intact.
        assert block not in pool.priority_eviction_queue
        cached = pool.get_cached_block(raw_hash, [0])
        assert cached is not None and cached[0] is block

        # Force LRU drain: ask for everything in LRU.
        n_free = pool.free_block_queue.num_free_blocks
        pool.get_new_blocks(n_free)

        # LRU semantics: cache map entry is gone, hash is cleared.
        assert pool.get_cached_block(raw_hash, [0]) is None, (
            "LRU popleft must still clear cached_block_hash_to_block; "
            "the fix targets PQ pop only."
        )
        assert block.block_hash is None, (
            "LRU popleft must still reset block.block_hash; the fix "
            "targets PQ pop only."
        )

    def test_eager_pop_clears_then_cache_full_blocks_assigns_new(self, monkeypatch):
        """EAGER variant: the priority-queue pop already evicts the old
        hash at allocation time, so the block arrives at cache_full_blocks
        already cleared. cache_full_blocks then just registers the new
        hash (no lazy-cleanup branch needed). Pairs with
        test_priority_queue_pop_clears_cache_map.
        """
        import time as time_mod

        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            make_block_hash_with_group_id,
        )

        pool = self._make_pool()
        block = pool.blocks[1]
        pool.free_block_queue.remove(block)
        block.ref_cnt = 1
        raw_old = BlockHash((123).to_bytes(32, "little"))
        h_old = make_block_hash_with_group_id(raw_old, 0)
        block.set_block_hash(h_old)
        pool.cached_block_hash_to_block.insert(h_old, block)

        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(
            pool.priority_eviction_queue,
            block,
            priority=50,
            expiry=None,
            last_freed=100.0,
        )
        pool.free_blocks([block])

        # Drain LRU + pop the priority-queue entry. Block now carries
        # h_old; cache map still has h_old → block.
        for b in list(pool.free_block_queue.get_all_free_blocks()):
            if b is not block and b is not pool.null_block:
                pool.free_block_queue.remove(b)
                b.ref_cnt = 1
        pool.get_new_blocks(1)
        assert block.block_hash is None, (
            "Eager: PQ pop already evicts the old hash at allocation."
        )
        assert pool.get_cached_block(raw_old, [0]) is None, (
            "Eager: old cache map entry already removed at PQ pop."
        )

        # Now drive cache_full_blocks with a NEW raw hash. The block is
        # already clear, so cache_full_blocks just registers h_new.
        raw_new = BlockHash((456).to_bytes(32, "little"))
        h_new = make_block_hash_with_group_id(raw_new, 0)

        # Minimal stub request — same pattern as
        # test_cache_full_blocks_routes_directives_to_queue. Without
        # kv_hints the retention hook is a no-op, and with
        # enable_kv_cache_events=False the events branch is skipped, so
        # only block_hashes is load-bearing.
        class _Req:
            kv_hints: None
            block_hashes: list

        request = _Req()
        request.kv_hints = None
        request.block_hashes = [raw_new]

        pool.cache_full_blocks(
            request=request,
            blocks=[block],
            num_cached_blocks=0,
            num_full_blocks=1,
            block_size=pool.hash_block_size,
            kv_cache_group_id=0,
        )

        assert pool.get_cached_block(raw_old, [0]) is None, (
            "cache_full_blocks must lazy-clean the stale hash before "
            "assigning the new one."
        )
        assert block.block_hash == h_new, (
            "cache_full_blocks must register the new hash on the block."
        )
        cached_new = pool.get_cached_block(raw_new, [0])
        assert cached_new is not None and cached_new[0] is block

    def test_no_prefix_hit_after_priority_queue_pop(self, monkeypatch):
        """EAGER variant end-to-end: after a block is popped via the
        priority queue (ref_cnt becomes 1), the cache map entry is gone,
        so a subsequent get_cached_block for the same hash MISSES.
        (The lazy variant would still hit — this is the behavioral cost
        the eager variant trades away.)
        """
        import time as time_mod

        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            make_block_hash_with_group_id,
        )

        pool = self._make_pool()
        block = pool.blocks[1]
        pool.free_block_queue.remove(block)
        block.ref_cnt = 1
        raw_hash = BlockHash((321).to_bytes(32, "little"))
        h = make_block_hash_with_group_id(raw_hash, 0)
        block.set_block_hash(h)
        pool.cached_block_hash_to_block.insert(h, block)

        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(
            pool.priority_eviction_queue,
            block,
            priority=50,
            expiry=None,
            last_freed=100.0,
        )
        pool.free_blocks([block])

        # Drain LRU + pop priority queue → block.ref_cnt = 1.
        for b in list(pool.free_block_queue.get_all_free_blocks()):
            if b is not block and b is not pool.null_block:
                pool.free_block_queue.remove(b)
                b.ref_cnt = 1
        pool.get_new_blocks(1)
        assert block.ref_cnt == 1

        # Eager: the cache map entry was evicted at PQ pop, so the
        # lookup must MISS.
        hit = pool.get_cached_block(raw_hash, [0])
        assert hit is None, (
            "Eager: prefix-cache entry is evicted on PQ pop, so the "
            "lookup misses. (Lazy variant would still hit — eager gives "
            "that up.)"
        )

    def test_scope_falls_back_to_session_id(self):
        pool = self._make_pool(num_blocks=8, block_size=16)
        request = _HintedRequest(
            [{"start": 0, "end": 16, "priority": 80}], scope=None, session_id="sess-7"
        )
        blocks = [pool.blocks[1]]
        pool._apply_retention_hook(request, blocks, num_full_blocks=1, block_size=16)
        meta = pool.priority_eviction_queue._meta[blocks[0].block_id]
        assert meta.scope == "sess-7"

    def test_covers_output_resolves_to_prompt_tail(self):
        pool = self._make_pool(num_blocks=8, block_size=16)
        request = _HintedRequest(
            [{"covers_output": True, "priority": 70}], "s", num_prompt_tokens=32
        )
        blocks = pool.blocks[1:5]  # tokens 0..63; output starts at 32
        pool._apply_retention_hook(request, blocks, num_full_blocks=4, block_size=16)
        meta = pool.priority_eviction_queue._meta
        assert blocks[0].block_id not in meta and blocks[1].block_id not in meta
        assert meta[blocks[2].block_id].priority == 70
        assert meta[blocks[3].block_id].priority == 70

    def test_evict_blocks_routes_protected_free_block_to_lru(self):
        pool = self._make_pool(num_blocks=8, block_size=16)
        block = pool.blocks[1]
        pool.priority_eviction_queue.apply_directives(
            [block], [_d(start=0, end=16, priority=50)], "s", 16
        )
        pool.free_block_queue.remove(block)
        assert pool.priority_eviction_queue.try_insert(block)
        free_before = pool.get_num_free_blocks()
        pool.evict_blocks({block.block_id})
        assert block not in pool.priority_eviction_queue
        assert pool.get_num_free_blocks() == free_before
        assert pool.free_block_queue.popleft() is block

    def test_lru_pop_drops_stale_sidecar_entry(self):
        pool = self._make_pool(num_blocks=8, block_size=16)
        block = pool.free_block_queue.popleft()
        pool.priority_eviction_queue.apply_directives(
            [block], [_d(start=0, end=16, priority=50)], "s", 16
        )
        pool.free_block_queue.prepend_n([block])  # LRU list, entry still on it
        (got,) = pool.get_new_blocks(1)
        assert got is block
        assert block.block_id not in pool.priority_eviction_queue._meta
        pool.free_blocks([block])
        assert block not in pool.priority_eviction_queue


class TestRetentionOrdering:
    def test_downgrade_of_a_queued_block_reorders_eviction(self):
        queue = PriorityEvictionQueue()
        high, mid = _make_block(1), _make_block(2)
        queue.apply_directives([high], [_d(start=0, end=16, priority=80)], "a", 16)
        queue.apply_directives([mid], [_d(start=0, end=16, priority=50)], "b", 16)
        assert queue.try_insert(high, last_freed_time=1.0)
        assert queue.try_insert(mid, last_freed_time=2.0)

        queue.apply_directives([high], [_d(start=0, end=16, priority=10)], "a", 16)

        assert queue.pop_lowest() is high


class TestReleaseRouting:
    """A release must re-list only blocks it took out of the priority queue.

    A free block can carry an entry while it is still linked on the LRU list
    (an unpinned block, or any other path that leaves an entry on a block in
    the LRU list). Appending it to the LRU list again corrupts the list: its
    count drifts from its links, and allocation hands out the tail sentinel.
    """

    @staticmethod
    def _manager_with_free_prefix():
        block_size = 16
        manager = make_kv_cache_manager(
            make_kv_cache_config(block_size, 11),
            max_model_len=8192,
            enable_caching=True,
            hash_block_size=block_size,
        )
        common = [i for i in range(3) for _ in range(block_size)]
        request = make_request("p", common + [3] * 7, block_size, sha256)
        computed, num_computed, _ = manager.get_computed_blocks(request)
        manager.allocate_slots(request, 55, num_computed, computed)
        manager.free(request)
        return manager, common

    @staticmethod
    def _protect_then_release(manager, common):
        pool = manager.block_pool
        request = make_request("a", common + [3] * 5, 16, sha256)
        computed, _, _ = manager.get_computed_blocks(request)
        (hit,) = computed.blocks
        for priority in (75, 0):
            pool._apply_retention_hook(
                _HintedRequest([_d(start=0, end=48, priority=priority)], "s"),
                list(hit),
                len(hit),
                16,
            )

    @staticmethod
    def _walk(free_queue):
        count, block = 0, free_queue.fake_free_list_head.next_free_block
        seen: set[int] = set()
        while block is not free_queue.fake_free_list_tail and block is not None:
            assert block.block_id not in seen, "cycle in the free list"
            seen.add(block.block_id)
            count += 1
            block = block.next_free_block
        return count

    def test_release_of_an_lru_listed_entry_keeps_the_list_consistent(self):
        manager, common = self._manager_with_free_prefix()
        self._protect_then_release(manager, common)

        free_queue = manager.block_pool.free_block_queue
        assert free_queue.num_free_blocks == self._walk(free_queue)

    def test_allocation_after_such_a_release_returns_distinct_real_blocks(self):
        manager, common = self._manager_with_free_prefix()
        self._protect_then_release(manager, common)

        ids = [block.block_id for block in manager.block_pool.get_new_blocks(9)]

        assert len(set(ids)) == len(ids) and all(i > 0 for i in ids), ids

    def test_pool_release_leaves_an_lru_listed_block_in_place(self):
        from vllm.v1.core.block_pool import BlockPool

        pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
        block = pool.blocks[1]
        pool.priority_eviction_queue.apply_directives(
            [block], [_d(start=0, end=16, priority=50)], "s", 16
        )
        free_before = pool.free_block_queue.num_free_blocks

        pool._apply_retention_hook(
            _HintedRequest([_d(start=0, end=16, priority=0)], "s"), [block], 1, 16
        )

        assert block.block_id not in pool.priority_eviction_queue._meta
        assert pool.free_block_queue.num_free_blocks == free_before
        assert self._walk(pool.free_block_queue) == free_before


class TestStructuralInvariants:
    """Lock in the spec's 'sidecar pattern' contract:
    - KVCacheBlock must not gain feature-specific fields for retention.
    - Request must not gain retention attributes.

    If these tests fail, you are about to break the additive-only feel
    of this PR. Move the new state into PriorityEvictionQueue's sidecar
    instead.
    """

    def test_kv_cache_block_has_no_priority_fields(self):
        from dataclasses import fields

        from vllm.v1.core.kv_cache_utils import KVCacheBlock

        names = {f.name for f in fields(KVCacheBlock)}
        forbidden = {
            "priority",
            "priority_expiry",
            "priority_scope",
            "last_freed_time",
        }
        leaks = names & forbidden
        assert not leaks, (
            f"KVCacheBlock has retention-specific fields {leaks!r}. "
            "Move them to PriorityEvictionQueue's sidecar."
        )

    def test_request_has_no_retention_attributes(self):
        import inspect

        from vllm.v1.request import Request

        src = inspect.getsource(Request.__init__)
        forbidden = ("retention_directives", "retention_scope")
        leaks = [name for name in forbidden if f"self.{name}" in src]
        assert not leaks, (
            f"Request.__init__ assigns to {leaks!r}. Read retention from "
            "request.kv_hints at the use site instead."
        )


def test_apply_directives_reports_blocks_it_released():
    """apply_directives must name the blocks whose protection it dropped.

    unprotect() discards the block from the priority queue, but it cannot put it
    back on the LRU list -- that list belongs to the pool. So the pool has to be
    told which blocks were released, the same way release_expired() reports
    lapsed ones.
    """
    pq = PriorityEvictionQueue()
    block = KVCacheBlock(block_id=1)
    claim = [_d(start=0, end=16, priority=50, duration=600.0)]
    release = [_d(start=0, end=16, priority=0)]

    assert pq.apply_directives([block], claim, "sess", 16) == []
    assert pq.try_insert(block)
    assert pq.apply_directives([block], release, "sess", 16) == [1]
    # already unprotected -> nothing to report
    assert pq.apply_directives([block], release, "sess", 16) == []


def test_a_non_owner_release_reports_nothing():
    pq = PriorityEvictionQueue()
    block = KVCacheBlock(block_id=2)
    pq.apply_directives(
        [block],
        [_d(start=0, end=16, priority=50, duration=600.0)],
        "owner",
        16,
    )
    assert pq.try_insert(block)
    assert (
        pq.apply_directives(
            [block], [_d(start=0, end=16, priority=0)], "someone-else", 16
        )
        == []
    )
    # still protected: the owner's own release now has something to drop
    assert pq.apply_directives(
        [block], [_d(start=0, end=16, priority=0)], "owner", 16
    ) == [2]


def test_pool_returns_a_released_free_block_to_the_lru_list():
    """The pool must re-list a block released while it was free.

    A release can reach a block that is free and still in the priority queue.
    Dropping its entry there used to leave it in neither list, shrinking the
    pool for the rest of the run.
    """
    from vllm.v1.core.block_pool import BlockPool

    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    pq = pool.priority_eviction_queue
    block = pool.blocks[1]
    claim = [_d(start=0, end=16, priority=50, duration=600.0)]

    pq.apply_directives([block], claim, "sess", 16)
    pool.free_block_queue.remove(block)  # free + protected, as after a free
    assert pq.try_insert(block) is True
    reachable_before = pool.get_num_free_blocks()

    pool._apply_retention_hook(
        _HintedRequest([_d(start=0, end=16, priority=0)], "sess"),
        [block],
        1,
        16,
    )

    assert block not in pq
    assert pool.get_num_free_blocks() == reachable_before, "block left both lists"


def test_heap_stays_bounded_under_free_touch_churn():
    """A block that is hit and freed over and over pushes a heap tuple per free;
    only pop_lowest ever discards the stale ones. With allocations served from
    the LRU list for long stretches (expiring protections), nothing pops, and a
    two-hour replay piled up millions of stale tuples that the next pop had to
    skip through. The queue must compact on its own, and popping afterwards
    must still return the lowest live entry."""
    queue = PriorityEvictionQueue()
    blocks = [_make_block(i) for i in range(64)]
    for b in blocks:
        _set_meta(queue, b, priority=50, scope="s")
    for round_ in range(2000):  # 128k inserts, 64 live
        for b in blocks:
            queue.suspend(b)  # touch(): referenced again
            assert queue.try_insert(b, last_freed_time=float(round_))
    assert len(queue._heap) <= 2 * queue.num_blocks + queue._COMPACT_SLACK
    low = _make_block(999)
    _set_meta(queue, low, priority=1, scope="s")
    assert queue.try_insert(low, last_freed_time=0.0)
    assert queue.pop_lowest() is low
    popped = queue.pop_lowest()
    assert popped is not None and popped.block_id < 64
    assert queue.num_blocks == 63


def _small_pool(enable_caching=True):
    from vllm.v1.core.block_pool import BlockPool

    return BlockPool(
        num_gpu_blocks=8,
        enable_caching=enable_caching,
        hash_block_size=16,
        enable_kv_cache_events=False,
    )


def test_reading_a_claimed_block_keeps_the_claim():
    """Reading a queued block suspends it but keeps the client's claim: the
    block is coming back to the queue when it is freed again."""
    pool = _small_pool()
    pq = pool.priority_eviction_queue
    block = pool.blocks[1]
    pool.free_block_queue.remove(block)
    _set_meta(pq, block, priority=90, expiry=None, scope="alice")
    pq.try_insert(block)

    pool.touch([block])

    meta = pq._meta.get(block.block_id)
    assert meta is not None and meta.priority == 90 and meta.scope == "alice"
    assert block not in pq  # suspended while referenced


def test_retention_hook_skips_another_pools_blocks():
    """A HiSparse host group's blocks come from a separate pool; this pool must
    neither protect them nor route a device block that shares their id."""
    pool, other = _small_pool(), _small_pool()
    foreign, own = other.blocks[1], pool.blocks[2]
    request = _HintedRequest([_d(start=0, end=32, priority=50)], "s")

    pool._apply_retention_hook(request, [foreign, own], 2, 16)

    assert foreign.block_id not in pool.priority_eviction_queue._meta
    assert own.block_id in pool.priority_eviction_queue._meta


class TestSlidingWindowGroupNarrowing:
    """A sliding-window group serves a hit only from the window preceding the
    reuse point, so a directive over a whole prompt is narrowed to that tail
    for such a group and the rest of the range is released. Full-attention
    groups keep the range as given."""

    def _pool_and_request(self, window):
        from vllm.v1.core.block_pool import BlockPool

        pool = BlockPool(
            num_gpu_blocks=64,
            enable_caching=True,
            hash_block_size=16,
            enable_kv_cache_events=False,
        )
        pool.retention_group_windows = (None, window)
        blocks = pool.blocks[1:21]  # 20 full blocks = 320 tokens
        request = _HintedRequest(
            [_d(start=0, end=None, priority=60, duration=600.0)], "s"
        )
        return pool, blocks, request

    def test_full_attention_group_keeps_whole_range(self):
        pool, blocks, request = self._pool_and_request(window=128)
        pool._apply_retention_hook(request, blocks, 20, 16, kv_cache_group_id=0)
        meta = pool.priority_eviction_queue._meta
        assert all(meta.get(b.block_id) is not None for b in blocks)

    def test_sliding_window_group_protects_only_the_tail(self):
        pool, blocks, request = self._pool_and_request(window=128)
        pool._apply_retention_hook(request, blocks, 20, 16, kv_cache_group_id=1)
        meta = pool.priority_eviction_queue._meta
        # window 128 -> cdiv(127, 16) + 1 = 9 blocks kept: indices 11..19
        kept = [b for b in blocks[11:]]
        dropped = [b for b in blocks[:11]]
        assert all(meta.get(b.block_id) is not None for b in kept)
        assert all(meta.get(b.block_id) is None for b in dropped)
        assert meta[kept[0].block_id].priority == 60

    def test_narrowing_releases_a_previously_protected_head(self):
        pool, blocks, request = self._pool_and_request(window=128)
        # Protect everything first as if it were a full-attention group, then
        # apply as the sliding-window group: the head must be released.
        pool._apply_retention_hook(request, blocks, 20, 16, kv_cache_group_id=0)
        for b in blocks:
            pool.free_block_queue.remove(b)
        pool._apply_retention_hook(request, blocks, 20, 16, kv_cache_group_id=1)
        meta = pool.priority_eviction_queue._meta
        assert all(meta.get(b.block_id) is None for b in blocks[:11])
        assert all(meta.get(b.block_id) is not None for b in blocks[11:])
