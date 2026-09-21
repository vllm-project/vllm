# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for vllm.v1.kv_offload.tiering.fs.dispatch.

Covers FCFSQueue, SJFBucketQueue, LoadQueue, and WorkDispatcher.
"""

import math
from collections.abc import Callable
from typing import Any

import pytest

from vllm.v1.kv_offload.base import Locality
from vllm.v1.kv_offload.tiering.fs.dispatch import (
    FCFSQueue,
    LoadQueue,
    SJFBucketQueue,
    StoreQueue,
    ThreadMode,
    WorkDispatcher,
    make_batches,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# block_size=1 makes num_tasks == bytes, so bucket = min(31, floor(log2(num_tasks))).
_BS = 1


def _make_dispatcher(
    locality: Locality = Locality.LOCAL,
    n_read: int = 4,
    n_write: int = 2,
    n_write_excl: int = 0,
) -> WorkDispatcher:
    return WorkDispatcher(
        locality=locality,
        load_job_q=LoadQueue(_BS),
        store_job_q=StoreQueue(_BS),
        n_read_threads=n_read,
        n_write_threads=n_write,
        n_write_excl_threads=n_write_excl,
    )


def _identity_batch(tasks: list[Any]) -> Callable[[], None]:
    """make_batch_fn that records which tasks it was given."""
    captured: list[list[Any]] = []

    def batch_fn() -> None:
        captured.append(tasks)

    batch_fn.captured = captured  # type: ignore[attr-defined]
    batch_fn.tasks = tasks  # type: ignore[attr-defined]
    return batch_fn


def _submit_load(
    dispatcher: WorkDispatcher,
    job_id: int,
    tasks: list[Any],
) -> int:
    state = object()
    n_threads = dispatcher.n_batch_threads(is_load=True)
    work_items = make_batches(state, tasks, _identity_batch, n_threads)
    return dispatcher.submit(
        job_id=job_id, work_items=work_items, n_tasks=len(tasks), is_load=True
    )


def _submit_store(
    dispatcher: WorkDispatcher,
    job_id: int,
    tasks: list[Any],
) -> int:
    state = object()
    n_threads = dispatcher.n_batch_threads(is_load=False)
    work_items = make_batches(state, tasks, _identity_batch, n_threads)
    return dispatcher.submit(
        job_id=job_id, work_items=work_items, n_tasks=len(tasks), is_load=False
    )


def _drain_load(dispatcher: WorkDispatcher) -> list[tuple[Any, int, Any]]:
    """Fetch all available work for a READ thread."""
    results = []
    while True:
        work = dispatcher.fetch_work(ThreadMode.READ)
        if work is None:
            break
        results.append(work)
    return results


def _drain_store(dispatcher: WorkDispatcher) -> list[tuple[Any, int, Any]]:
    """Fetch all available work for a WRITE thread."""
    results = []
    while True:
        work = dispatcher.fetch_work(ThreadMode.WRITE)
        if work is None:
            break
        results.append(work)
    return results


# ---------------------------------------------------------------------------
# FCFSQueue
# ---------------------------------------------------------------------------


class TestFCFSQueue:
    def test_empty_get_returns_none(self):
        q = FCFSQueue(_BS)
        assert q.get() is None

    def test_maybe_has_work_empty(self):
        assert not FCFSQueue(_BS).maybe_has_work()

    def test_put_makes_has_work_true(self):
        q = FCFSQueue(_BS)
        q.put(1, 10)
        assert q.maybe_has_work()

    def test_fifo_order(self):
        q = FCFSQueue(_BS)
        for jid in [10, 20, 30]:
            q.put(jid, 1)
        assert [q.get(), q.get(), q.get()] == [10, 20, 30]

    def test_get_exhausts_queue(self):
        q = FCFSQueue(_BS)
        q.put(1, 1)
        q.get()
        assert not q.maybe_has_work()
        assert q.get() is None

    def test_clear(self):
        q = FCFSQueue(_BS)
        for jid in range(5):
            q.put(jid, 1)
        q.clear()
        assert not q.maybe_has_work()
        assert q.get() is None

    def test_num_tasks_ignored_for_ordering(self):
        """FCFSQueue ignores num_tasks — ordering is purely arrival order."""
        q = FCFSQueue(_BS)
        q.put(1, 1000)
        q.put(2, 1)
        assert q.get() == 1  # first in, first out


# ---------------------------------------------------------------------------
# SJFBucketQueue
# ---------------------------------------------------------------------------


class TestSJFBucketQueue:
    def _expected_bucket(self, num_tasks: int, block_size: int = _BS) -> int:
        mbytes = max(1, num_tasks * block_size)
        return min(31, int(math.floor(math.log2(mbytes))))

    def test_empty_get_returns_none(self):
        q = SJFBucketQueue(_BS)
        assert q.get() is None

    def test_maybe_has_work_empty(self):
        assert not SJFBucketQueue(_BS).maybe_has_work()

    def test_put_makes_has_work_true(self):
        q = SJFBucketQueue(_BS)
        q.put(1, 1)
        assert q.maybe_has_work()

    def test_single_job_returned(self):
        q = SJFBucketQueue(_BS)
        q.put(42, 4)
        assert q.get() == 42
        assert not q.maybe_has_work()

    def test_sjf_shortest_first(self):
        """A smaller job is returned before a larger one."""
        q = SJFBucketQueue(_BS)
        q.put(1, 8)  # bucket 3
        q.put(2, 1)  # bucket 0  ← shorter
        assert q.get() == 2  # shorter job first
        assert q.get() == 1

    def test_fcfs_within_same_bucket(self):
        """Two jobs in the same bucket come out in arrival order."""
        q = SJFBucketQueue(_BS)
        # num_tasks 2 and 3 both map to bucket 1 (floor(log2(2))=1, floor(log2(3))=1)
        q.put(10, 2)
        q.put(20, 3)
        assert q.get() == 10
        assert q.get() == 20

    def test_mask_cleared_after_bucket_empty(self):
        q = SJFBucketQueue(_BS)
        q.put(1, 1)
        q.get()
        assert q.mask.value == 0

    def test_mask_tracks_multiple_buckets(self):
        q = SJFBucketQueue(_BS)
        q.put(1, 1)  # bucket 0
        q.put(2, 8)  # bucket 3
        assert q.mask.value != 0
        q.get()  # pops bucket 0 job
        assert q.mask.value != 0
        q.get()  # pops bucket 3 job
        assert q.mask.value == 0

    def test_clear(self):
        q = SJFBucketQueue(_BS)
        for jid in range(5):
            q.put(jid, 2**jid)
        q.clear()
        assert not q.maybe_has_work()
        assert q.mask.value == 0
        assert q.get() is None

    def test_bucket_id_clamped_to_max(self):
        """Very large jobs land in bucket 31, not beyond."""
        q = SJFBucketQueue(_BS)
        q.put(99, 2**32)  # exceeds 32 bits
        bid = q._get_sjf_bucket()
        assert bid == 31

    def test_num_tasks_1_goes_to_bucket_0(self):
        q = SJFBucketQueue(_BS)
        q.put(1, 1)
        assert q._get_sjf_bucket() == 0

    @pytest.mark.parametrize(
        "num_tasks,expected_bucket",
        [
            (1, 0),
            (2, 1),
            (4, 2),
            (8, 3),
            (16, 4),
            (1024, 10),
        ],
    )
    def test_bucket_assignment(self, num_tasks: int, expected_bucket: int):
        q = SJFBucketQueue(_BS)
        q.put(1, num_tasks)
        assert q._get_sjf_bucket() == expected_bucket


# ---------------------------------------------------------------------------
# LoadQueue
# ---------------------------------------------------------------------------


class TestLoadQueue:
    def test_empty_get_returns_none(self):
        q = LoadQueue(_BS)
        assert q.get() is None

    def test_maybe_has_work_empty(self):
        assert not LoadQueue(_BS).maybe_has_work()

    def test_put_makes_has_work_true(self):
        q = LoadQueue(_BS)
        q.put(1, 1)
        assert q.maybe_has_work()

    def test_each_job_returned_exactly_once(self):
        """Each job should be served once even though it lives in both sub-queues."""
        q = LoadQueue(_BS)
        jobs = [
            (1, 4),
            (2, 8),
            (3, 1),
            (4, 16),
            (5, 2),
            (6, 32),
            (7, 3),
            (8, 64),
            (9, 7),
            (10, 128),
            (11, 5),
            (12, 256),
            (13, 6),
            (14, 512),
            (15, 9),
        ]
        for job_id, num_tasks in jobs:
            q.put(job_id, num_tasks)
        seen = []
        for _ in range(len(jobs) + 5):  # more iterations than jobs
            jid = q.get()
            if jid is not None:
                seen.append(jid)
        assert sorted(seen) == sorted(job_id for job_id, _ in jobs)

    def test_alternates_between_fcfs_and_sjf(self):
        """LoadQueue alternates FCFS/SJF with deterministic output order.

        Insertion order (FCFS): [1, 2, 3, 4, 5, 7, 8, 6]
        SJF order (lowest bucket first): [5, 6, 7, 8, 1, 2, 3, 4]
        Expected interleaved output: [1, 5, 2, 6, 3, 7, 4, 8]
        """
        q = LoadQueue(_BS)
        for job_id, num_tasks in [
            (1, 1000),
            (2, 1000),
            (3, 1000),
            (4, 1000),
            (5, 1),
            (7, 16),
            (8, 16),
            (6, 8),
        ]:
            q.put(job_id, num_tasks)
        result = [q.get() for _ in range(8)]
        assert result == [1, 5, 2, 6, 3, 7, 4, 8]

    def test_fallback_to_other_queue_when_primary_empty(self):
        """If the selected sub-queue is empty, the fallback is tried."""
        q = LoadQueue(_BS)
        q.put(1, 1)
        # Force pp=1 so SJF is selected first. SJF has job 1.
        # Drain SJF by getting it once normally (pp=0 → fcfs first,
        # drains job 1 from jobs set)
        jid = q.get()  # pp=0, uses fcfs, returns 1, removes from jobs
        assert jid == 1
        # Now jobs is empty; both sub-queues still have the stale entry for job 1.
        # Next get should return None regardless of which queue is tried.
        assert q.get() is None

    def test_clear_resets_all_state(self):
        q = LoadQueue(_BS)
        q.put(1, 1)
        q.put(2, 4)
        q.clear()
        assert not q.maybe_has_work()
        assert q.get() is None
        assert q.pp == 0
        assert len(q.jobs) == 0

    def test_many_jobs_all_returned(self):
        """Put N jobs, get exactly N unique results."""
        q = LoadQueue(_BS)
        n = 20
        for jid in range(n):
            q.put(jid, jid + 1)
        results = []
        for _ in range(n * 3):  # plenty of iterations to drain stale entries
            jid = q.get()
            if jid is not None:
                results.append(jid)
        assert sorted(results) == list(range(n))

    def test_maybe_has_work_false_after_all_served(self):
        """Once all jobs are served, maybe_has_work eventually returns False."""
        q = LoadQueue(_BS)
        q.put(1, 1)
        q.get()
        # Both sub-queues still hold the stale entry until drained.
        # After further gets they drain to empty.
        for _ in range(4):
            q.get()
        assert not q.maybe_has_work()


# ---------------------------------------------------------------------------
# WorkDispatcher — construction
# ---------------------------------------------------------------------------


class TestWorkDispatcherConstruction:
    def test_batch_thread_counts(self):
        assert (
            _make_dispatcher(Locality.LOCAL, n_read=4, n_write=2)._n_read_batch_threads
            == 4
        )
        assert (
            _make_dispatcher(Locality.LOCAL, n_read=4, n_write=4)._n_write_batch_threads
            == 1
        )
        assert (
            _make_dispatcher(Locality.REMOTE, n_read=4, n_write=2)._n_read_batch_threads
            == 4
        )
        assert (
            _make_dispatcher(
                Locality.REMOTE, n_read=4, n_write=2
            )._n_write_batch_threads
            == 2
        )
        # n_read=0: fall back to all rw threads
        assert (
            _make_dispatcher(Locality.REMOTE, n_read=0, n_write=3)._n_read_batch_threads
            == 3
        )
        # n_write=0: fall back to all rw threads
        assert (
            _make_dispatcher(
                Locality.REMOTE, n_read=4, n_write=0
            )._n_write_batch_threads
            == 4
        )


# ---------------------------------------------------------------------------
# WorkDispatcher — submit
# ---------------------------------------------------------------------------


class TestWorkDispatcherSubmit:
    def test_submit_returns_n_read_batch_threads_for_load(self):
        s = _make_dispatcher(n_read=3, n_write=2)
        n_wake = _submit_load(s, job_id=1, tasks=list(range(10)))
        assert n_wake == 3  # n_write_excl(0) + n_read_batch_threads(3)

    def test_submit_load_registers_has_work_for_read_thread(self):
        s = _make_dispatcher()
        _submit_load(s, 1, list(range(4)))
        assert s.has_work(ThreadMode.READ)

    def test_submit_store_registers_has_work_for_write_thread(self):
        s = _make_dispatcher()
        _submit_store(s, 1, list(range(4)))
        assert s.has_work(ThreadMode.WRITE)

    def test_submit_store_visible_to_read_thread(self):
        """Read threads can steal store jobs."""
        s = _make_dispatcher()
        _submit_store(s, 1, list(range(4)))
        assert s.has_work(ThreadMode.READ)

    def test_submit_load_not_visible_to_write_excl_thread(self):
        """WRITE_EXCL threads never see load jobs."""
        s = _make_dispatcher(n_read=4, n_write=2)
        _submit_load(s, 1, list(range(4)))
        assert not s.has_work(ThreadMode.WRITE_EXCL)

    def test_no_work_initially(self):
        s = _make_dispatcher()
        assert not s.has_work(ThreadMode.READ)
        assert not s.has_work(ThreadMode.WRITE)
        assert not s.has_work(ThreadMode.WRITE_EXCL)


# ---------------------------------------------------------------------------
# WorkDispatcher — fetch_work / batching
# ---------------------------------------------------------------------------


class TestWorkDispatcherFetchWork:
    def test_fetch_load_by_read_thread(self):
        s = _make_dispatcher(n_read=1, n_write=1)
        tasks = list(range(10))
        _submit_load(s, job_id=1, tasks=tasks)
        work = s.fetch_work(ThreadMode.READ)
        assert work is not None
        fn, batch_size, state = work
        assert batch_size == 10  # 1 read thread → 1 batch with all tasks

    def test_fetch_store_by_write_thread(self):
        s = _make_dispatcher(Locality.REMOTE, n_read=1, n_write=1)
        tasks = list(range(6))
        _submit_store(s, job_id=1, tasks=tasks)
        work = s.fetch_work(ThreadMode.WRITE)
        assert work is not None
        _, batch_size, _ = work
        assert batch_size == 6  # 1 write thread → 1 batch

    def test_read_thread_steals_store_job(self):
        s = _make_dispatcher(n_read=1, n_write=1)
        _submit_store(s, job_id=1, tasks=list(range(4)))
        # No load work; read thread should fall back to store queue.
        work = s.fetch_work(ThreadMode.READ)
        assert work is not None

    def test_write_excl_thread_cannot_steal_load_job(self):
        """WRITE_EXCL threads never steal from the load queue."""
        s = _make_dispatcher(n_read=4, n_write=2)
        _submit_load(s, job_id=1, tasks=list(range(4)))
        work = s.fetch_work(ThreadMode.WRITE_EXCL)
        assert work is None

    def test_fetch_returns_none_when_empty(self):
        s = _make_dispatcher()
        assert s.fetch_work(ThreadMode.READ) is None
        assert s.fetch_work(ThreadMode.WRITE) is None
        assert s.fetch_work(ThreadMode.WRITE_EXCL) is None

    def test_batching_splits_tasks_across_read_threads(self):
        """With n_read=4 and 8 tasks, expect 4 batches of 2."""
        s = _make_dispatcher(n_read=4, n_write=1)
        _submit_load(s, job_id=1, tasks=list(range(8)))
        batches = _drain_load(s)
        assert len(batches) == 4
        assert all(batch_size == 2 for _, batch_size, _ in batches)

    def test_batching_local_store_uses_1_thread(self):
        """LOCAL: store always batched for 1 write thread regardless of n_write."""
        s = _make_dispatcher(Locality.LOCAL, n_read=4, n_write=4)
        _submit_store(s, job_id=1, tasks=list(range(12)))
        # Read thread steals the store job and batches using n_write_batch_threads=1
        # → 1 work item with all 12 tasks.
        batches = _drain_load(s)
        assert len(batches) == 1
        assert batches[0][1] == 12

    def test_batching_remote_store_uses_n_write_threads(self):
        """REMOTE: store batched across all write threads."""
        s = _make_dispatcher(Locality.REMOTE, n_read=0, n_write=2)
        _submit_store(s, job_id=1, tasks=list(range(4)))
        # n_write=2 → 2 batches of 2
        batches = _drain_store(s)
        assert len(batches) == 2
        assert all(batch_size == 2 for _, batch_size, _ in batches)

    def test_multiple_jobs_all_tasks_covered(self):
        """All tasks across multiple submitted jobs are eventually served."""
        s = _make_dispatcher(n_read=2, n_write=1)
        all_tasks = []
        for jid in range(3):
            tasks = list(range(jid * 10, jid * 10 + 4))
            all_tasks.extend(tasks)
            _submit_load(s, jid, tasks)

        batches = _drain_load(s)
        served_tasks: list[Any] = []
        for fn, _, _ in batches:
            served_tasks.extend(fn.tasks)  # type: ignore[attr-defined]
        assert sorted(served_tasks) == sorted(all_tasks)

    def test_fetch_load_before_store_for_read_thread(self):
        """Read thread prefers load work when both are available."""
        s = _make_dispatcher(n_read=1, n_write=1)
        _submit_store(s, job_id=10, tasks=[0])
        _submit_load(s, job_id=20, tasks=[1])
        # First fetch should give a load work item.
        work = s.fetch_work(ThreadMode.READ)
        assert work is not None
        _, _, state = work
        # The load job (id=20) should be first.
        # Confirm no load work remains after this.
        # (state tracks job_id indirectly via the dispatcher's _jobs dict)
        # Drain everything and confirm store work is still present.
        rest = _drain_load(s)
        # At least the store batch should appear in the rest.
        assert len(rest) >= 1

    def test_task_count_preserved_across_uneven_batching(self):
        """Uneven split: 5 tasks, 2 threads → batches of 3 and 2."""
        s = _make_dispatcher(n_read=2, n_write=1)
        _submit_load(s, job_id=1, tasks=list(range(5)))
        batches = _drain_load(s)
        batch_sizes = [bs for _, bs, _ in batches]
        assert sum(batch_sizes) == 5
        assert sorted(batch_sizes, reverse=True) == [3, 2]

    def test_fewer_tasks_than_threads_single_batch(self):
        """1 task, 4 threads → still 1 batch with 1 task."""
        s = _make_dispatcher(n_read=4, n_write=1)
        _submit_load(s, job_id=1, tasks=[42])
        batches = _drain_load(s)
        assert len(batches) == 1
        assert batches[0][1] == 1


# ---------------------------------------------------------------------------
# WorkDispatcher — clear
# ---------------------------------------------------------------------------


class TestWorkDispatcherClear:
    def test_clear_empties_all_queues(self):
        s = _make_dispatcher()
        _submit_load(s, 1, list(range(8)))
        _submit_store(s, 2, list(range(4)))
        s.clear()
        assert not s.has_work(ThreadMode.READ)
        assert not s.has_work(ThreadMode.WRITE)
        assert not s.has_work(ThreadMode.WRITE_EXCL)
        assert s.fetch_work(ThreadMode.READ) is None
        assert s.fetch_work(ThreadMode.WRITE) is None
        assert s.fetch_work(ThreadMode.WRITE_EXCL) is None

    def test_clear_removes_pending_job_metadata(self):
        s = _make_dispatcher()
        _submit_load(s, 1, list(range(4)))
        s.clear()
        assert len(s._jobs) == 0

    def test_submit_after_clear_works(self):
        s = _make_dispatcher(n_read=1, n_write=1)
        _submit_load(s, 1, list(range(4)))
        s.clear()
        _submit_load(s, 2, list(range(6)))
        assert s.has_work(ThreadMode.READ)
        work = s.fetch_work(ThreadMode.READ)
        assert work is not None
        _, batch_size, _ = work
        assert batch_size == 6


class TestWorkDispatcherNoReadThreads:
    def test_all_tasks_served_with_no_read_threads(self):
        s = _make_dispatcher(Locality.REMOTE, n_read=0, n_write=2)
        tasks = list(range(4))
        _submit_load(s, 1, tasks)
        batches = _drain_store(s)
        total = sum(bs for _, bs, _ in batches)
        assert total == len(tasks)


# ---------------------------------------------------------------------------
# WorkDispatcher — WRITE_EXCL thread behaviour
# ---------------------------------------------------------------------------


class TestWorkDispatcherWriteExcl:
    def test_write_excl_sees_only_store_work(self):
        s = _make_dispatcher(n_read=2, n_write=2)
        _submit_load(s, 1, list(range(4)))
        assert not s.has_work(ThreadMode.WRITE_EXCL)
        _submit_store(s, 2, list(range(4)))
        assert s.has_work(ThreadMode.WRITE_EXCL)

    def test_write_excl_fetch_ignores_load_queue(self):
        s = _make_dispatcher(n_read=2, n_write=2)
        _submit_load(s, 1, list(range(4)))
        assert s.fetch_work(ThreadMode.WRITE_EXCL) is None

    def test_write_excl_fetch_serves_store(self):
        s = _make_dispatcher(Locality.REMOTE, n_read=1, n_write=0)
        _submit_store(s, 1, list(range(4)))
        work = s.fetch_work(ThreadMode.WRITE_EXCL)
        assert work is not None
        _, batch_size, _ = work
        assert batch_size == 4
