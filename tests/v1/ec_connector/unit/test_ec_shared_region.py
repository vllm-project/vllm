# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECSharedRegion."""

import os
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    AllocationError,
    ECSharedRegion,
    _wait_for_file_size,
)


def _make_region(num_blocks: int = 8, block_size_bytes: int = 64) -> ECSharedRegion:
    return ECSharedRegion(
        instance_id=str(uuid.uuid4()),
        num_blocks=num_blocks,
        block_size_bytes=block_size_bytes,
    )


@pytest.fixture
def region() -> ECSharedRegion:
    r = _make_region()
    yield r
    r.cleanup()


# ── alloc ────────────────────────────────────────────────────────────────────


def test_alloc_returns_unique_indices(region):
    partial = region.alloc(3)
    assert len(partial) == 3 and len(set(partial)) == 3
    region.free(partial)
    full = region.alloc(8)
    assert len(full) == 8 and len(set(full)) == 8


def test_alloc_raises_when_empty(region):
    region.alloc(8)
    with pytest.raises(AllocationError):
        region.alloc(1)


def test_alloc_raises_when_insufficient(region):
    region.alloc(6)  # 2 free remain
    with pytest.raises(AllocationError):
        region.alloc(3)


def test_alloc_zero_returns_empty_list(region):
    """Boundary: requesting zero blocks must succeed and not consume any."""
    assert region.alloc(0) == []
    # Pool is untouched.
    assert region.alloc(8)


# ── free ─────────────────────────────────────────────────────────────────────


def test_free_returns_blocks_to_pool(region):
    indices = region.alloc(4)
    region.free(indices)
    indices2 = region.alloc(4)
    assert len(indices2) == 4


def test_free_empty_is_noop(region):
    region.alloc(8)
    region.free([])  # must not raise; pool unchanged
    with pytest.raises(AllocationError):
        region.alloc(1)


def test_free_asserts_on_pinned(region):
    indices = region.alloc(2)
    region.pin(indices)
    with pytest.raises(AssertionError):
        region.free(indices)
    region.unpin(indices)
    region.free(indices)


def test_free_asserts_on_partially_pinned(region):
    indices = region.alloc(3)
    region.pin([indices[1]])
    with pytest.raises(AssertionError):
        region.free(indices)
    region.unpin([indices[1]])
    region.free(indices)


# ── try_free ─────────────────────────────────────────────────────────────────


def test_try_free_unpinned_returns_true(region):
    indices = region.alloc(3)
    assert region.try_free(indices) is True
    # Blocks returned to pool — should be allocatable again.
    region.alloc(3)


def test_try_free_pinned_returns_false(region):
    indices = region.alloc(3)
    region.pin(indices)
    assert region.try_free(indices) is False
    # Region unchanged — still only 5 free (8 - 3).
    with pytest.raises(AllocationError):
        region.alloc(6)
    region.unpin(indices)
    region.free(indices)


def test_try_free_partially_pinned_returns_false(region):
    indices = region.alloc(3)
    region.pin([indices[0]])
    assert region.try_free(indices) is False
    region.unpin([indices[0]])
    assert region.try_free(indices) is True


def test_try_free_after_unpin_returns_true(region):
    indices = region.alloc(2)
    region.pin(indices)
    region.unpin(indices)
    assert region.try_free(indices) is True


# ── pin / unpin ref counting ──────────────────────────────────────────────────


def test_nested_pin_requires_matching_unpin(region):
    indices = region.alloc(1)
    region.pin(indices)
    region.pin(indices)  # ref_count == 2
    region.unpin(indices)  # ref_count == 1 — still pinned
    assert region.try_free(indices) is False
    region.unpin(indices)  # ref_count == 0
    assert region.try_free(indices) is True


def test_unpin_of_unpinned_block_asserts(region):
    indices = region.alloc(1)
    with pytest.raises(AssertionError):
        region.unpin(indices)
    region.free(indices)


def test_pin_does_not_affect_free_pool_size(region):
    indices = region.alloc(3)
    region.pin(indices)
    # Remaining free count: 8 - 3 = 5.
    region.alloc(5)
    with pytest.raises(AllocationError):
        region.alloc(1)
    region.unpin(indices)
    region.free(indices)


def test_pin_empty_is_noop(region):
    region.pin([])  # must not raise
    indices = region.alloc(1)
    # Region state intact: nothing pinned yet, free / unconditional must work.
    region.free(indices)


def test_unpin_empty_is_noop(region):
    region.unpin([])  # must not raise even with empty ref_count


# ── mmap sharing between two instances ───────────────────────────────────────


def test_second_instance_opens_existing_file_and_shares_memory():
    instance_id = str(uuid.uuid4())
    r1 = ECSharedRegion(instance_id=instance_id, num_blocks=4, block_size_bytes=64)
    try:
        r2 = ECSharedRegion(instance_id=instance_id, num_blocks=4, block_size_bytes=64)
        try:
            assert r1._is_creator
            assert not r2._is_creator
            # Both map the same physical pages; writes via r1 are visible via r2.
            r1.blocks[0, :4] = torch.tensor([10, 20, 30, 40], dtype=torch.int8)
            assert r2.blocks[0, :4].tolist() == [10, 20, 30, 40]
        finally:
            r2.cleanup()
    finally:
        r1.cleanup()


def test_only_creator_unlinks_file_on_cleanup():
    """Critical contract: if the non-creator unlinks, the creator's mmap path
    becomes a dangling backing file and a third opener would create a new one
    out from under the creator."""
    instance_id = str(uuid.uuid4())
    r1 = ECSharedRegion(instance_id=instance_id, num_blocks=4, block_size_bytes=64)
    path = r1.mmap_path
    r2 = ECSharedRegion(instance_id=instance_id, num_blocks=4, block_size_bytes=64)

    # Non-creator goes away first — file must still be on disk for r1.
    r2.cleanup()
    assert os.path.exists(path), "non-creator cleanup must not unlink the file"

    # Creator goes away — file is removed.
    r1.cleanup()
    assert not os.path.exists(path), "creator cleanup must unlink the file"


# ── _wait_for_file_size ──────────────────────────────────────────────────────


def test_wait_for_file_size_returns_when_already_big_enough(tmp_path):
    """The fast path: file already at expected size — return immediately."""
    p = tmp_path / "f.bin"
    p.write_bytes(b"\x00" * 128)
    fd = os.open(str(p), os.O_RDONLY)
    try:
        _wait_for_file_size(fd, expected_size=128, timeout=1.0)  # must not raise
    finally:
        os.close(fd)


def test_wait_for_file_size_times_out_when_file_stays_empty(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"")
    fd = os.open(str(p), os.O_RDONLY)
    try:
        with pytest.raises(TimeoutError):
            _wait_for_file_size(fd, expected_size=4096, timeout=0.05)
    finally:
        os.close(fd)


# ── pin_memory ───────────────────────────────────────────────────────────────


def test_pin_memory_success_sets_flag(region):
    """When cudaHostRegister returns 0, is_pinned flips to True
    and cleanup will correspondingly call cudaHostUnregister."""
    fake_cudart = MagicMock()
    success = MagicMock()
    success.value = 0
    fake_cudart.cudaHostRegister.return_value = success
    fake_cudart.cudaHostUnregister.return_value = success

    with patch("torch.cuda.cudart", return_value=fake_cudart):
        region.pin_memory()
        assert region.is_pinned is True
        # cleanup must pair with cudaHostUnregister exactly once.
        region.cleanup()
        fake_cudart.cudaHostUnregister.assert_called_once()


def test_pin_memory_failure_leaves_flag_false():
    """If cudaHostRegister fails (non-zero), don't pretend it succeeded —
    cleanup must NOT call cudaHostUnregister on memory we never registered."""
    r = _make_region()
    try:
        fake_cudart = MagicMock()
        fail = MagicMock()
        fail.value = 1  # non-zero == error
        fake_cudart.cudaHostRegister.return_value = fail

        with patch("torch.cuda.cudart", return_value=fake_cudart):
            r.pin_memory()
            assert r.is_pinned is False
            # Now run cleanup with a fresh cudart spy and verify
            # cudaHostUnregister was NOT called.
            unregister_spy = MagicMock()
            fake_cudart.cudaHostUnregister = unregister_spy
            r.cleanup()
            unregister_spy.assert_not_called()
    finally:
        # cleanup may have already run; idempotent.
        r.cleanup()


# ── cleanup idempotency ───────────────────────────────────────────────────────


def test_cleanup_is_idempotent(region):
    region.cleanup()
    region.cleanup()  # fixture calls a third time — must not raise


# ── thread safety ─────────────────────────────────────────────────────────────


def test_concurrent_alloc_free_yields_no_duplicate_indices():
    """Tighter than a smoke test: every thread tracks the indices it currently
    holds; if the lock is broken, two threads will hold the same index at the
    same time, which we check on every alloc."""
    r = _make_region(num_blocks=4, block_size_bytes=64)
    errors: list[Exception] = []
    seen_lock = threading.Lock()
    currently_held: set[int] = set()
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        my: list[int] = []
        try:
            for _ in range(500):
                try:
                    idx = r.alloc(1)
                except AllocationError:
                    continue
                with seen_lock:
                    if idx[0] in currently_held:
                        raise AssertionError(
                            f"index {idx[0]} held by two threads simultaneously"
                        )
                    currently_held.add(idx[0])
                my.extend(idx)
                # Tight loop: free immediately so contention stays high.
                with seen_lock:
                    currently_held.discard(idx[0])
                r.free(idx)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    r.cleanup()
    assert not errors, errors
    assert not currently_held, "thread leaked an index"


def test_concurrent_pin_unpin_balances_to_zero():
    r = _make_region(num_blocks=4, block_size_bytes=64)
    indices = r.alloc(4)
    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        try:
            for _ in range(500):
                r.pin(indices)
                r.unpin(indices)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All pins matched: ref counts must net to zero, otherwise some increment
    # was lost to a race.
    assert not r._ref_count, f"leaked ref counts: {r._ref_count}"
    r.free(indices)
    r.cleanup()
    assert not errors, errors
