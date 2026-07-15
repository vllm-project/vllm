# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for StepTracker."""

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.step_tracker import (
    StepTracker,
)


def test_basic_step_count_expiry_depth_1():
    """With max_concurrent_batches=1, entry expires after 1 step."""
    t = StepTracker(max_concurrent_batches=1)
    t.add("h1", "req_a")

    # Step 1: entry was added, committed to slot. Deque fills to maxlen=1,
    # but no expiry yet (the entry is in the slot just committed).
    result = t.step(set())
    assert result == []

    # Step 2: deque is full, oldest slot (containing h1) expires.
    result = t.step(set())
    assert result == ["h1"]


def test_basic_step_count_expiry_depth_2():
    """With max_concurrent_batches=2, entry expires after 2 steps."""
    t = StepTracker(max_concurrent_batches=2)
    t.add("h1", "req_a")

    assert t.step(set()) == []  # step 1: committed
    assert t.step(set()) == []  # step 2: deque fills but h1 in newer slot
    assert t.step(set()) == ["h1"]  # step 3: h1's slot is oldest, expires


def test_first_finish_fast_path():
    """Entry is returned immediately when request_id appears in finished."""
    t = StepTracker(max_concurrent_batches=3)
    t.add("h1", "req_a")

    result = t.step({"req_a"})
    assert result == ["h1"]


def test_no_double_processing_fast_path_then_expiry():
    """Entry fast-pathed should NOT fire again on deque expiry."""
    t = StepTracker(max_concurrent_batches=1)
    t.add("h1", "req_a")

    # Fast-path fires h1.
    result = t.step({"req_a"})
    assert result == ["h1"]

    # Normal expiry would fire now, but h1 is already processed.
    result = t.step(set())
    assert result == []


def test_no_double_processing_expiry_then_fast_path():
    """Entry expired via step-count should NOT fire again via fast-path."""
    t = StepTracker(max_concurrent_batches=1)
    t.add("h1", "req_a")

    t.step(set())  # commit
    result = t.step(set())  # expiry fires h1
    assert result == ["h1"]

    # req_a finishes later — must not re-fire.
    result = t.step({"req_a"})
    assert result == []


def test_multiplicity_same_hash_different_requests():
    """Two adds of the same mm_hash (different requests) → returned twice."""
    t = StepTracker(max_concurrent_batches=3)
    t.add("h1", "req_a")
    t.add("h1", "req_b")

    # Both requests finish in the same step.
    result = t.step({"req_a", "req_b"})
    assert sorted(result) == ["h1", "h1"]


def test_mixed_expiry_and_fast_path():
    """Some entries expire, others fast-path, in the same step."""
    t = StepTracker(max_concurrent_batches=1)
    t.add("h1", "req_a")

    t.step(set())  # commit h1's slot

    # Add h2 in the next step.
    t.add("h2", "req_b")

    # This step: h1's slot expires AND req_b finishes.
    result = t.step({"req_b"})
    assert set(result) == {"h1", "h2"}


def test_drain_all_returns_unprocessed():
    """drain_all returns everything pending and clears state."""
    t = StepTracker(max_concurrent_batches=3)
    t.add("h1", "req_a")
    t.add("h2", "req_b")
    t.step(set())  # commit both
    t.add("h3", "req_c")  # in _current, not yet committed

    result = t.drain_all()
    assert set(result) == {"h1", "h2", "h3"}

    # After drain, everything is empty.
    assert t.step(set()) == []
    assert t.drain_all() == []


def test_drain_all_skips_already_processed():
    """drain_all should not re-return entries already fast-pathed."""
    t = StepTracker(max_concurrent_batches=3)
    t.add("h1", "req_a")
    t.add("h2", "req_b")

    # Fast-path h1.
    result = t.step({"req_a"})
    assert result == ["h1"]

    # drain should only return h2.
    result = t.drain_all()
    assert result == ["h2"]


def test_empty_step_returns_nothing():
    """step() with nothing pending returns empty list."""
    t = StepTracker(max_concurrent_batches=2)
    assert t.step(set()) == []
    assert t.step({"req_x"}) == []


def test_add_after_step_goes_to_next_slot():
    """Entries added after a step go into the next slot, not the current."""
    t = StepTracker(max_concurrent_batches=1)

    t.step(set())  # empty step
    t.add("h1", "req_a")

    # h1 is in _current. Next step commits it.
    t.step(set())

    # Now the deque is full (1 slot with h1). Next step expires it.
    result = t.step(set())
    assert result == ["h1"]


def test_fast_path_on_current_entries():
    """Entries still in _current (not yet committed) can be fast-pathed."""
    t = StepTracker(max_concurrent_batches=3)
    t.add("h1", "req_a")

    # req_a finishes in the same step h1 was added.
    result = t.step({"req_a"})
    assert result == ["h1"]

    # Entry is committed to slot but marked processed — won't re-fire.
    for _ in range(5):
        assert t.step(set()) == []
