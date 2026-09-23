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
) -> WorkDispatcher:
    return WorkDispatcher(
        locality=locality,
        load_job_q=LoadQueue(_BS),
        store_job_q=StoreQueue(_BS),
        n_read_threads=n_read,
        n_write_threads=n_write,
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
        work = dispatcher.fetch_work(True)
        if work is None:
            break
        results.append(work)
    return results


def _drain_store(dispatcher: WorkDispatcher) -> list[tuple[Any, int, Any]]:
    """Fetch all available work for a WRITE thread."""
    results = []
    while True:
        work = dispatcher.fetch_work(False)
        if work is None:
            break
        results.append(work)
    return results


# ---------------------------------------------------------------------------
# FCFSQueue
# ---------------------------------------------------------------------------


class TestFCFSQueue:
    def test_lifecycle(self):
        q = FCFSQueue(_BS)

        # empty state
        assert not q.maybe_has_work()
        assert q.get() is None

        # put makes it non-empty; num_tasks is ignored for ordering
        q.put(1, 1000)
        q.put(2, 1)
        q.put(3, 500)
        assert q.maybe_has_work()

        # strict FIFO regardless of task count
        assert [q.get(), q.get(), q.get()] == [1, 2, 3]

        # fully drained
        assert not q.maybe_has_work()
        assert q.get() is None

        # clear resets all state
        for jid in range(5):
            q.put(jid, 1)
        q.clear()
        assert not q.maybe_has_work()
        assert q.get() is None


# ---------------------------------------------------------------------------
# SJFBucketQueue
# ---------------------------------------------------------------------------


class TestSJFBucketQueue:
    def _expected_bucket(self, num_tasks: int, block_size: int = _BS) -> int:
        mbytes = max(1, num_tasks * block_size)
        return min(31, int(math.floor(math.log2(mbytes))))

    def test_lifecycle(self):
        q = SJFBucketQueue(_BS)

        # empty state
        assert not q.maybe_has_work()
        assert q.get() is None

        # single put/get round-trip
        q.put(42, 4)
        assert q.maybe_has_work()
        assert q.get() == 42
        assert not q.maybe_has_work()
        assert q.mask.value == 0

        # SJF: shorter job (lower bucket) comes out first regardless of insertion order
        q.put(1, 8)  # bucket 3
        q.put(2, 1)  # bucket 0 ← shorter
        assert q.get() == 2
        assert q.get() == 1
        assert q.mask.value == 0

        # FCFS within the same bucket (tasks 2 and 3 both → bucket 1)
        q.put(10, 2)
        q.put(20, 3)
        assert q.get() == 10
        assert q.get() == 20

        # mask tracks occupancy across buckets and clears as each empties
        q.put(1, 1)  # bucket 0
        q.put(2, 8)  # bucket 3
        assert q.mask.value != 0
        q.get()  # drains bucket 0
        assert q.mask.value != 0
        q.get()  # drains bucket 3
        assert q.mask.value == 0

        # clear resets everything
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
    def test_lifecycle(self):
        q = LoadQueue(_BS)

        # empty state
        assert not q.maybe_has_work()
        assert q.get() is None

        # put makes it non-empty
        q.put(1, 1)
        assert q.maybe_has_work()
        q.clear()

        # each job returned exactly once despite living in both sub-queues
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
        for _ in range(len(jobs) + 5):
            jid = q.get()
            if jid is not None:
                seen.append(jid)
        assert sorted(seen) == sorted(job_id for job_id, _ in jobs)

        # FCFS/SJF interleave is deterministic
        # Insertion order (FCFS): [1, 2, 3, 4, 5, 7, 8, 6]
        # SJF order (lowest bucket first): [5, 6, 7, 8, 1, 2, 3, 4]
        # Expected interleaved output: [1, 5, 2, 6, 3, 7, 4, 8]
        q.clear()  # reset pp to 0
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
        assert [q.get() for _ in range(8)] == [1, 5, 2, 6, 3, 7, 4, 8]

        # clear resets all state
        q.put(1, 1)
        q.put(2, 4)
        q.clear()
        assert not q.maybe_has_work()
        assert q.get() is None
        assert q.pp == 0
        assert len(q.jobs) == 0

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
# WorkDispatcher
# ---------------------------------------------------------------------------


class TestWorkDispatcher:
    def test_lifecycle(self):
        s = _make_dispatcher(n_read=1, n_write=1)

        # empty: no thread mode has work
        assert not s.has_work(True)
        assert not s.has_work(False)
        assert s.fetch_work(True) is None
        assert s.fetch_work(False) is None

        # load: visible to READ; WRITE can steal
        _submit_load(s, job_id=1, tasks=list(range(4)))
        assert s.has_work(True)
        assert s.has_work(False)

        # READ thread consumes the load work; 1 read thread -> 1 batch
        work = s.fetch_work(True)
        assert work is not None
        _, batch_size, _ = work
        assert batch_size == 4

        # store: visible to WRITE and READ (steal)
        _submit_store(s, job_id=2, tasks=list(range(6)))
        assert s.has_work(False)
        assert s.has_work(True)

        # READ thread prefers load over store when both available
        _submit_load(s, job_id=3, tasks=list(range(4)))
        _submit_store(s, job_id=4, tasks=[0])
        _submit_load(s, job_id=5, tasks=[1])
        load_work = s.fetch_work(True)
        assert load_work is not None
        assert s.fetch_work(True) is not None  # steal store

        # clear: resets all queues and job metadata
        _submit_load(s, 10, list(range(8)))
        _submit_store(s, 11, list(range(4)))
        s.clear()
        assert not s.has_work(True)
        assert not s.has_work(False)
        assert s.fetch_work(True) is None
        assert len(s._jobs) == 0

        # submit after clear works normally
        _submit_load(s, 20, list(range(6)))
        work = s.fetch_work(True)
        assert work is not None
        _, batch_size, _ = work
        assert batch_size == 6

    def test_submit_n_wake(self):
        s = _make_dispatcher(n_read=3, n_write=2)
        assert _submit_load(s, job_id=1, tasks=list(range(10))) == 3

    def test_batching_load_splits_across_read_threads(self):
        s = _make_dispatcher(n_read=4, n_write=1)
        _submit_load(s, job_id=1, tasks=list(range(8)))
        batches = _drain_load(s)
        assert len(batches) == 4
        assert all(batch_size == 2 for _, batch_size, _ in batches)

    def test_batching_local_store_uses_1_thread(self):
        s = _make_dispatcher(Locality.LOCAL, n_read=4, n_write=4)
        _submit_store(s, job_id=1, tasks=list(range(12)))
        batches = _drain_load(s)
        assert len(batches) == 1
        assert batches[0][1] == 12

    def test_batching_remote_store_uses_n_write_threads(self):
        s = _make_dispatcher(Locality.REMOTE, n_read=0, n_write=2)
        _submit_store(s, job_id=1, tasks=list(range(4)))
        batches = _drain_store(s)
        assert len(batches) == 2
        assert all(batch_size == 2 for _, batch_size, _ in batches)

    def test_uneven_split(self):
        s = _make_dispatcher(n_read=2, n_write=1)
        _submit_load(s, job_id=1, tasks=list(range(5)))
        batch_sizes = sorted(bs for _, bs, _ in _drain_load(s))
        assert sum(batch_sizes) == 5
        assert batch_sizes == [2, 3]

    def test_fewer_tasks_than_threads(self):
        s = _make_dispatcher(n_read=4, n_write=1)
        _submit_load(s, job_id=1, tasks=[42])
        batches = _drain_load(s)
        assert len(batches) == 1
        assert batches[0][1] == 1

    @pytest.mark.parametrize(
        "locality,n_read,n_write",
        [
            (Locality.LOCAL, 2, 2),
            (Locality.REMOTE, 2, 2),
            # read-only: steals stores
            (Locality.REMOTE, 2, 0),
            # write-only: steals loads
            (Locality.REMOTE, 0, 2),
        ],
    )
    def test_multiple_jobs_all_tasks_covered(
        self,
        locality: Locality,
        n_read: int,
        n_write: int,
    ):
        s = _make_dispatcher(locality, n_read=n_read, n_write=n_write)
        load_tasks: list[Any] = []
        store_tasks: list[Any] = []
        for jid in range(3):
            tasks = list(range(jid * 10, jid * 10 + 4))
            load_tasks.extend(tasks)
            _submit_load(s, jid, tasks)
        for jid in range(3, 6):
            tasks = list(range(jid * 10, jid * 10 + 4))
            store_tasks.extend(tasks)
            _submit_store(s, jid, tasks)

        served: list[Any] = []
        for fn, _, _ in _drain_load(s):
            served.extend(fn.tasks)  # type: ignore[attr-defined]
        for fn, _, _ in _drain_store(s):
            served.extend(fn.tasks)  # type: ignore[attr-defined]

        assert sorted(served) == sorted(load_tasks + store_tasks)

    def test_no_read_threads_tasks_served_via_write(self):
        s = _make_dispatcher(Locality.REMOTE, n_read=0, n_write=2)
        tasks = list(range(4))
        _submit_load(s, 1, tasks)
        assert sum(bs for _, bs, _ in _drain_store(s)) == len(tasks)
