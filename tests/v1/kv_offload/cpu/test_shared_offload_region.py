# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SharedOffloadRegion."""

import contextlib
import mmap
import os
import threading
import time
import uuid

import pytest

from vllm.utils.system_utils import get_mp_context
from vllm.v1.kv_offload.cpu.shared_offload_region import (
    SharedOffloadRegion,
    _wait_for_file_size,
)

PAGE_SIZE = mmap.PAGESIZE


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_spawn_method(monkeypatch):
    # On WSL, NVML is not compatible with fork so vLLM auto-overrides the
    # multiprocessing start method to 'spawn' with a warning. Set it explicitly
    # here so the override is a no-op and the warning is suppressed.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _make_region(
    instance_id: str,
    num_blocks: int = 4,
    cpu_page_size: int = PAGE_SIZE,
    num_workers: int = 1,
    rank: int = 0,
) -> SharedOffloadRegion:
    total_size_bytes = num_blocks * num_workers * cpu_page_size
    assert total_size_bytes % PAGE_SIZE == 0
    return SharedOffloadRegion(
        instance_id=instance_id,
        total_size_bytes=total_size_bytes,
        num_blocks=num_blocks,
        rank=rank,
        num_workers=num_workers,
        cpu_page_size=cpu_page_size,
    )


def _cleanup_file(path: str) -> None:
    """Best-effort file removal for test teardown."""
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


@contextlib.contextmanager
def _region(instance_id: str, **kwargs):
    """Context manager: create one region, clean up on exit."""
    r = _make_region(instance_id, **kwargs)
    try:
        yield r
    finally:
        r.cleanup()
        _cleanup_file(r.mmap_path)


@contextlib.contextmanager
def _multi_region(
    instance_id: str,
    num_workers: int,
    num_blocks: int = 4,
    cpu_page_size: int = PAGE_SIZE,
):
    """Context manager: create one SharedOffloadRegion per rank, clean up on exit."""
    total = num_blocks * num_workers * cpu_page_size
    regions = [
        SharedOffloadRegion(
            instance_id=instance_id,
            total_size_bytes=total,
            num_blocks=num_blocks,
            rank=rank,
            num_workers=num_workers,
            cpu_page_size=cpu_page_size,
        )
        for rank in range(num_workers)
    ]
    try:
        yield regions
    finally:
        for r in regions:
            r.cleanup()
        _cleanup_file(regions[0].mmap_path)


def _race_construct(
    instance_id: str,
    num_workers: int,
    num_blocks: int = 4,
    cpu_page_size: int = PAGE_SIZE,
) -> tuple[list[SharedOffloadRegion], list[Exception]]:
    """Spawn num_workers threads that all race to construct SharedOffloadRegion."""
    total = num_blocks * num_workers * cpu_page_size
    regions: list[SharedOffloadRegion | None] = [None] * num_workers
    errors: list[Exception] = []
    barrier = threading.Barrier(num_workers)

    def worker(rank: int) -> None:
        barrier.wait()  # all threads start at the same instant
        try:
            regions[rank] = SharedOffloadRegion(
                instance_id=instance_id,
                total_size_bytes=total,
                num_blocks=num_blocks,
                rank=rank,
                num_workers=num_workers,
                cpu_page_size=cpu_page_size,
            )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return [r for r in regions if r is not None], errors


def _mp_race_construct_and_write(
    instance_id: str,
    total_bytes: int,
    num_blocks: int,
    rank: int,
    num_workers: int,
    cpu_page_size: int,
    fill_value: int,
    done_queue,
    cleanup_queue,
) -> None:
    """Race to construct a SharedOffloadRegion, write fill_value, then wait
    for the parent's cleanup signal before tearing down.  The wait gives the
    parent a window to read the raw mmap before the creator removes the file."""
    try:
        region = SharedOffloadRegion(
            instance_id=instance_id,
            total_size_bytes=total_bytes,
            num_blocks=num_blocks,
            rank=rank,
            num_workers=num_workers,
            cpu_page_size=cpu_page_size,
        )
        t = region.create_next_view(cpu_page_size)
        t[:, :] = fill_value
        done_queue.put({"rank": rank, "error": None})
        cleanup_queue.get()  # wait for parent's verification to finish
        del t  # release view before cleanup to avoid BufferError
        region.cleanup()
    except Exception as e:
        done_queue.put({"rank": rank, "error": repr(e)})


@pytest.fixture
def iid():
    """Fresh instance ID for each test."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# create_next_view — shape, stride and storage offset
# ---------------------------------------------------------------------------


def test_create_next_view_shape_and_stride(iid):
    """Returned tensor must have shape (num_blocks, tensor_page_size) and
    stride (row_stride, 1) where row_stride = cpu_page_size * num_workers."""
    with _region(iid, num_blocks=4, cpu_page_size=2 * PAGE_SIZE) as r:
        t = r.create_next_view(PAGE_SIZE)
        assert t.shape == (4, PAGE_SIZE)
        # num_workers=1 → row_stride = cpu_page_size
        assert t.stride() == (2 * PAGE_SIZE, 1)
        del t


def test_create_next_view_storage_offset_rank0(iid):
    """rank=0 worker's first tensor must start at byte 0 of the mmap."""
    with _region(iid, cpu_page_size=PAGE_SIZE, num_workers=2, rank=0) as r:
        t = r.create_next_view(PAGE_SIZE)
        assert t.data_ptr() == r._base.data_ptr()  # storage_offset == 0
        del t


def test_create_next_view_storage_offset_rank1(iid):
    """rank=1 worker's first tensor must start cpu_page_size bytes into the mmap."""
    with _multi_region(iid, num_workers=2, num_blocks=4) as (r0, r1):
        t1 = r1.create_next_view(PAGE_SIZE)
        assert t1.data_ptr() == r1._base.data_ptr() + PAGE_SIZE
        del t1


def test_create_next_view_row_stride_with_multiple_workers(iid):
    """With num_workers=4, row_stride must be 4 * cpu_page_size."""
    with _region(iid, num_blocks=2, num_workers=4) as r:
        t = r.create_next_view(PAGE_SIZE)
        assert t.stride(0) == 4 * PAGE_SIZE
        del t


# ---------------------------------------------------------------------------
# create_next_view — cursor advancement
# ---------------------------------------------------------------------------


def test_create_next_view_cursor_advances(iid):
    """Each call to create_next_view must advance _worker_offset by tensor_page_size."""
    with _region(iid, cpu_page_size=3 * PAGE_SIZE) as r:
        assert r._worker_offset == 0
        r.create_next_view(PAGE_SIZE)
        assert r._worker_offset == PAGE_SIZE
        r.create_next_view(PAGE_SIZE)
        assert r._worker_offset == 2 * PAGE_SIZE
        r.create_next_view(PAGE_SIZE)
        assert r._worker_offset == 3 * PAGE_SIZE  # exactly at area end


def test_create_next_view_exact_fill_succeeds(iid):
    """Allocations whose total exactly equals cpu_page_size must all succeed."""
    with _region(iid, cpu_page_size=2 * PAGE_SIZE) as r:
        r.create_next_view(PAGE_SIZE)  # first half
        r.create_next_view(PAGE_SIZE)  # fills to area end — must not raise


# ---------------------------------------------------------------------------
# create_next_view — overflow guard
# ---------------------------------------------------------------------------


def test_create_next_view_single_overflow_raises(iid):
    """A single allocation larger than cpu_page_size must raise AssertionError."""
    with (
        _region(iid) as r,
        pytest.raises(AssertionError, match="exceeds worker area end"),
    ):
        r.create_next_view(PAGE_SIZE + 1)


def test_create_next_view_cumulative_overflow_raises(iid):
    """Successive allocations that cumulatively exceed cpu_page_size must raise."""
    with _region(iid, cpu_page_size=2 * PAGE_SIZE) as r:
        r.create_next_view(PAGE_SIZE)  # ok — half used
        r.create_next_view(PAGE_SIZE)  # ok — full
        with pytest.raises(AssertionError, match="exceeds worker area end"):
            r.create_next_view(1)  # one byte too many


def test_create_next_view_overflow_does_not_mutate_cursor(iid):
    """A failed create_next_view must leave _worker_offset unchanged."""
    with _region(iid) as r:
        offset_before = r._worker_offset
        with pytest.raises(AssertionError):
            r.create_next_view(PAGE_SIZE + 1)
        assert r._worker_offset == offset_before


# ---------------------------------------------------------------------------
# create_next_view — data correctness and layout
# ---------------------------------------------------------------------------


def test_create_next_view_write_visible_in_raw_mmap(iid):
    """Writes into a create_next_view view must appear at the correct raw mmap offset"""
    with _region(iid, num_blocks=4) as r:
        t = r.create_next_view(PAGE_SIZE)
        t[2, :] = 42  # write to block row 2

        raw = memoryview(r.mmap_obj)
        # num_workers=1 → row_stride = PAGE_SIZE; block 2 starts at byte 2*PAGE_SIZE
        chunk = bytes(raw[2 * PAGE_SIZE : 3 * PAGE_SIZE])
        assert all(b == 42 for b in chunk)
        del raw, t


def test_create_next_view_multi_tensor_layout(iid):
    """Two tensors from the same worker land at consecutive byte offsets per row."""
    with _region(iid, num_blocks=2, cpu_page_size=2 * PAGE_SIZE) as r:
        ta = r.create_next_view(PAGE_SIZE)
        tb = r.create_next_view(PAGE_SIZE)

        ta[:, :] = 1
        tb[:, :] = 2

        raw = memoryview(r.mmap_obj)
        for blk in range(2):
            row_offset = blk * 2 * PAGE_SIZE  # num_workers=1
            assert all(b == 1 for b in raw[row_offset : row_offset + PAGE_SIZE])
            assert all(
                b == 2 for b in raw[row_offset + PAGE_SIZE : row_offset + 2 * PAGE_SIZE]
            )
        del raw, ta, tb


def test_create_next_view_multiprocess_slots(iid):
    """Each worker process calls create_next_view and writes distinct data;
    the parent verifies each slot lands at the correct interleaved offset."""
    num_workers = 2
    num_blocks = 4
    total_bytes = num_blocks * num_workers * PAGE_SIZE

    ctx = get_mp_context()
    done_queue = ctx.Queue()
    cleanup_queue = ctx.Queue()

    # Parent is rank 0 (creator); child is rank 1 (joiner).
    region = SharedOffloadRegion(
        instance_id=iid,
        total_size_bytes=total_bytes,
        num_blocks=num_blocks,
        rank=0,
        num_workers=num_workers,
        cpu_page_size=PAGE_SIZE,
    )
    try:
        child = ctx.Process(
            target=_mp_race_construct_and_write,
            args=(
                iid,
                total_bytes,
                num_blocks,
                1,
                num_workers,
                PAGE_SIZE,
                22,
                done_queue,
                cleanup_queue,
            ),
        )
        child.start()

        t0 = region.create_next_view(PAGE_SIZE)
        t0[:, :] = 11

        result = done_queue.get(timeout=30)
        assert result["error"] is None, result["error"]

        raw = memoryview(region.mmap_obj)
        for blk in range(num_blocks):
            row_start = blk * num_workers * PAGE_SIZE
            w0 = bytes(raw[row_start : row_start + PAGE_SIZE])
            w1 = bytes(raw[row_start + PAGE_SIZE : row_start + 2 * PAGE_SIZE])
            assert all(b == 11 for b in w0), f"block {blk}: rank0 slot wrong"
            assert all(b == 22 for b in w1), f"block {blk}: rank1 slot wrong"

        del raw, t0  # release before finally triggers cleanup
        cleanup_queue.put(True)
        child.join(timeout=10)
        assert child.exitcode == 0
    finally:
        region.cleanup()
        _cleanup_file(region.mmap_path)


def test_create_next_view_worker_isolation(iid):
    """Writes by worker 0 must not affect worker 1's slot and vice versa."""
    num_workers = 2
    num_blocks = 4
    with _multi_region(iid, num_workers=num_workers, num_blocks=num_blocks) as regions:
        t0 = regions[0].create_next_view(PAGE_SIZE)
        t1 = regions[1].create_next_view(PAGE_SIZE)

        t0[:, :] = 11
        t1[:, :] = 22

        raw = memoryview(regions[0].mmap_obj)
        for blk in range(num_blocks):
            row_start = blk * num_workers * PAGE_SIZE
            w0 = bytes(raw[row_start : row_start + PAGE_SIZE])
            w1 = bytes(raw[row_start + PAGE_SIZE : row_start + 2 * PAGE_SIZE])
            assert all(b == 11 for b in w0), f"block {blk}: worker0 slot corrupted"
            assert all(b == 22 for b in w1), f"block {blk}: worker1 slot corrupted"
        del raw, t0, t1  # release before finally triggers cleanup


# ---------------------------------------------------------------------------
# Constructor — creator vs joiner semantics
# ---------------------------------------------------------------------------


def test_creator_flag_set_on_first_open(iid):
    """The first worker to open the file must have _creator == True."""
    with _region(iid) as r:
        assert r._creator is True


def test_joiner_flag_not_set(iid):
    """A second worker opening the same file must have _creator == False."""
    with _multi_region(iid, num_workers=2) as (r0, r1):
        assert r0._creator is True
        assert r1._creator is False


def test_file_exists_after_construction(iid):
    """The mmap file must be present on disk after __init__ completes."""
    with _region(iid) as r:
        assert os.path.exists(r.mmap_path)


def test_file_has_correct_size(iid):
    """The mmap file size on disk must equal total_size_bytes."""
    with _region(iid, num_blocks=4) as r:
        assert os.path.getsize(r.mmap_path) == 4 * PAGE_SIZE


# ---------------------------------------------------------------------------
# Multi-worker race — concurrent construction
# ---------------------------------------------------------------------------


def test_multi_worker_race_exactly_one_creator(iid):
    """When N threads race to create the same region, exactly one becomes creator."""
    num_workers = 8
    regions, errors = _race_construct(iid, num_workers=num_workers)
    try:
        assert not errors, f"Workers raised: {errors}"
        assert len(regions) == num_workers, "Some workers failed to construct"

        creators = [r for r in regions if r._creator]
        assert len(creators) == 1, f"Expected 1 creator, got {len(creators)}"
        assert sum(1 for r in regions if not r._creator) == num_workers - 1, (
            f"Expected {num_workers - 1} non-creators, got "
            f"{sum(1 for r in regions if not r._creator)}"
        )

        for r in regions:
            assert not r.mmap_obj.closed
            assert r.total_size_bytes == 4 * num_workers * PAGE_SIZE
    finally:
        for r in regions:
            r.cleanup()
        _cleanup_file(regions[0].mmap_path)


def test_multi_worker_race_shared_memory_visible(iid):
    """After a concurrent construction race, MAP_SHARED is intact across all workers."""
    num_workers = 4
    regions, errors = _race_construct(iid, num_workers=num_workers)
    assert not errors
    try:
        regions[0].mmap_obj[0:1] = b"\xab"
        for r in regions[1:]:
            assert memoryview(r.mmap_obj)[0:1] == b"\xab"
    finally:
        for r in regions:
            r.cleanup()
        _cleanup_file(regions[0].mmap_path)


def test_multiprocess_race_construct_and_write(iid):
    """N processes race to construct the same SharedOffloadRegion, each writes
    fill_value = rank+1 into their slot; parent verifies interleaved layout."""
    num_workers = 4
    num_blocks = 3
    total_bytes = num_blocks * num_workers * PAGE_SIZE

    ctx = get_mp_context()
    done_queue = ctx.Queue()
    cleanup_queue = ctx.Queue()

    procs = [
        ctx.Process(
            target=_mp_race_construct_and_write,
            args=(
                iid,
                total_bytes,
                num_blocks,
                rank,
                num_workers,
                PAGE_SIZE,
                rank + 1,
                done_queue,
                cleanup_queue,
            ),
        )
        for rank in range(num_workers)
    ]
    for p in procs:
        p.start()

    results = {}
    for _ in range(num_workers):
        r = done_queue.get(timeout=30)
        results[r["rank"]] = r

    for rank, r in results.items():
        assert r["error"] is None, f"rank {rank}: {r['error']}"

    # Read the raw file while all workers still hold it open.
    mmap_path = f"/dev/shm/vllm_offload_{iid}.mmap"
    with open(mmap_path, "rb") as f:
        raw = f.read()

    for blk in range(num_blocks):
        for w in range(num_workers):
            slot_start = (blk * num_workers + w) * PAGE_SIZE
            slot = raw[slot_start : slot_start + PAGE_SIZE]
            expected = w + 1  # fill_value = rank + 1
            assert all(b == expected for b in slot), (
                f"block {blk}, worker {w}: expected {expected} but got wrong bytes"
            )

    # Unblock all workers to clean up.
    for _ in range(num_workers):
        cleanup_queue.put(True)
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_cleanup_creator_all_effects(iid):
    """cleanup() on the creator closes mmap, closes fd, and removes the file."""
    r = _make_region(iid)
    path = r.mmap_path
    fd = r.fd
    mmap_obj = r.mmap_obj

    r.cleanup()

    assert mmap_obj.closed, "mmap should be closed after cleanup"
    assert not os.path.exists(path), "creator should remove the file"
    with pytest.raises(OSError):
        os.fstat(fd)  # fd should be closed


def test_cleanup_non_creator_all_effects(iid):
    """cleanup() on a non-creator closes mmap and fd, but leaves the file on disk."""
    r0 = _make_region(iid)  # creator
    r1 = _make_region(iid)  # joiner
    path = r0.mmap_path
    fd1 = r1.fd
    mmap_obj1 = r1.mmap_obj
    try:
        r1.cleanup()

        assert mmap_obj1.closed, "mmap should be closed after cleanup"
        assert os.path.exists(path), "non-creator must not remove the file"
        with pytest.raises(OSError):
            os.fstat(fd1)  # fd should be closed
    finally:
        r0.cleanup()
        _cleanup_file(path)


def test_cleanup_idempotent(iid):
    """Calling cleanup() twice must not raise any exception."""
    r = _make_region(iid)
    r.cleanup()
    r.cleanup()  # must be a no-op


def test_cleanup_after_create_next_view_releases_mmap(iid):
    """cleanup() must close the mmap even after create_next_view was called.
    create_next_view returns a view that shares storage with _base; both must be
    released before mmap.close() can succeed."""
    r = _make_region(iid)
    mmap_obj = r.mmap_obj

    t = r.create_next_view(PAGE_SIZE)
    del t

    r.cleanup()

    assert mmap_obj.closed, "mmap should be closed after releasing the tensor"


# ---------------------------------------------------------------------------
# _wait_for_file_size
# ---------------------------------------------------------------------------


def test_wait_for_file_size_already_large_enough(tmp_path):
    """_wait_for_file_size must return immediately when file is already big enough."""
    fd = os.open(str(tmp_path / "ready.mmap"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, PAGE_SIZE)
        start = time.monotonic()
        _wait_for_file_size(fd, PAGE_SIZE, timeout=5.0)
        assert time.monotonic() - start < 0.5
    finally:
        os.close(fd)


def test_wait_for_file_size_waits_for_grow(tmp_path):
    """_wait_for_file_size must return once a background thread grows the file."""
    fd = os.open(str(tmp_path / "grow.mmap"), os.O_CREAT | os.O_RDWR, 0o600)
    try:

        def grow():
            time.sleep(0.05)
            os.ftruncate(fd, PAGE_SIZE)

        t = threading.Thread(target=grow)
        t.start()
        _wait_for_file_size(fd, PAGE_SIZE, timeout=5.0)  # must not raise
        t.join()
    finally:
        os.close(fd)


def test_wait_for_file_size_timeout(tmp_path):
    """_wait_for_file_size must raise TimeoutError when the file never grows."""
    fd = os.open(str(tmp_path / "stuck.mmap"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with pytest.raises(TimeoutError):
            _wait_for_file_size(fd, PAGE_SIZE, timeout=0.1)
    finally:
        os.close(fd)
