# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.base import JobId

logger = init_logger(__name__)


@dataclass
class Task:
    """
    I/O Task inputs
    """

    key: OffloadKey
    path: str
    offset: int


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(success) when it finishes.
    """

    __slots__ = (
        "_job_id",
        "_n_tasks",
        "_completed",
        "_success",
        "_start_time",
        "_finish_time",
        "_lock",
    )

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._start_time: float | None = None
        self._finish_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def maybe_start(self) -> None:
        """Record when the first parallel task starts transferring data."""
        with self._lock:
            if self._start_time is None:
                self._start_time = time.monotonic()

    def task_done(
        self, batch_size: int, success: bool, finish_time: float
    ) -> tuple[bool, bool, float]:
        """Returns if job completed and success flag"""
        with self._lock:
            self._completed += batch_size
            if not success:
                self._success = False

            self._finish_time = max(self._finish_time, finish_time)
            assert self._start_time is not None
            transfer_time = self._finish_time - self._start_time
            return self._completed == self._n_tasks, self._success, transfer_time


class DualQueueThreadPool:
    """
    Thread pool with two task queues (load and store) and two thread groups.

    Load-priority threads drain the load queue first, then fall back to the
    store queue.  Store-priority threads do the reverse.  Both queues share
    a single condition variable.
    """

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        block_size: int | None = None,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._n_read_threads = n_read_threads
        self._n_write_threads = n_write_threads
        self._block_size = block_size
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[tuple[JobId, bool, float]] = deque()
        self._inflight_jobs = 0  # guarded by _condition

        # fanout_target_bytes is the number of bytes sufficient to saturate the
        # device bandwidth with a single read/write call. This factors into
        # how Job tasks are batched.
        self._fanout_target_bytes = 32 * (2**20)  # 32MiB

        assert self.total_threads > 0, "ThreadPool needs at least one thread"

        for i in range(self._n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(True,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(self._n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(False,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    @property
    def total_threads(self) -> int:
        return self._n_read_threads + self._n_write_threads

    def _batch_tasks(
        self,
        tasks: list[Task],
        n_threads: int,
    ) -> Iterator[list[Task]]:
        """
        Batch tasks so that the request's tasks are split evenly across the
        n_threads.
        """
        assert n_threads > 0

        n_tasks = len(tasks)
        q, r = divmod(n_tasks, n_threads)
        batch_sizes = [q + 1 if i < r else q for i in range(n_threads)]
        assert sum(batch_sizes) == n_tasks
        start = 0
        for bs in batch_sizes[: min(n_tasks, n_threads)]:
            yield tasks[start : start + bs]
            start += bs

    def _fanout_degree(self, n_tasks: int, n_threads: int) -> int:
        """How many batches to split a job of ``n_tasks`` blocks into.

        Splitting only pays while the device is not already saturated. A large
        block keeps it busy on its own, and once enough jobs are in flight the
        threads are busy regardless; in both regimes extra batches only add
        queue entries, wake-ups and GIL round-trips without moving more bytes.

        Callers must hold ``self._condition``.
        """
        if self._block_size is None:
            return min(n_tasks, n_threads)
        budget = -(-self._fanout_target_bytes // self._block_size)
        # No more reads can be outstanding than there are workers to issue them.
        budget = min(budget, self.total_threads)
        jobs = max(1, self._inflight_jobs)
        return max(1, min(-(-budget // jobs), n_tasks, n_threads))

    def _enqueue(
        self,
        queue: deque,
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
        job_id: JobId,
        tasks: Iterable[Task],
        n_tasks: int,
        n_threads: int,
    ) -> None:
        """Batch `tasks` and append (fn, state, batch_size) entries to `queue`."""
        if n_tasks == 0:
            self._finished_q.append((job_id, True, 0.0))
            return
        state = JobState(job_id, n_tasks)
        task_lst = list(tasks)  # Materialize tasks out of self._condition
        assert len(task_lst) == n_tasks, "Unaccounted tasks"
        n_batches = 0
        with self._condition:
            self._inflight_jobs += 1
            n_threads = self._fanout_degree(n_tasks, n_threads)
            for batch in self._batch_tasks(task_lst, n_threads):
                queue.append((make_batch_fn(batch), len(batch), state))
                n_batches += 1
            self._condition.notify(n_batches)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""

        self._enqueue(
            self._load_q,
            make_batch_fn,
            job_id,
            tasks,
            n_tasks=n_tasks,
            n_threads=self._n_read_threads
            if self._n_read_threads > 0
            else self.total_threads,
        )

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""

        self._enqueue(
            self._store_q,
            make_batch_fn,
            job_id,
            tasks,
            n_tasks=n_tasks,
            n_threads=self._n_write_threads
            if self._n_write_threads > 0
            else self.total_threads,
        )

    def get_finished(self) -> list[tuple[JobId, bool, float]]:
        # No lock needed: deque is thread-safe for concurrent append/popleft,
        # and the manager is the sole popper.
        jobs = []
        while self._finished_q:
            jobs.append(self._finished_q.popleft())
        return jobs

    def wait_idle(self) -> None:
        """Block until there are no in-flight jobs.

        After this returns, every submitted job has had its last task
        finish, so no worker thread is still copying data. Note:
        completed jobs may still be sitting in ``_finished_q`` waiting
        for ``get_finished()`` to drain them.
        """
        with self._condition:
            self._condition.wait_for(lambda: self._inflight_jobs == 0)

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            self._load_q.clear()
            self._store_q.clear()
            # Cancelled tasks will not decrement _inflight_jobs; reset it so a
            # subsequent wait_idle() returns instead of hanging.
            self._inflight_jobs = 0
            self._condition.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, load_priority: bool) -> None:
        # Wait for tasks, process from primary queue first, fall back to secondary.
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._load_q or self._store_q
                )
                if self._stop:
                    return
                primary = self._load_q if load_priority else self._store_q
                secondary = self._store_q if load_priority else self._load_q
                fn, batch_size, state = (
                    primary.popleft() if primary else secondary.popleft()
                )
            try:
                state.maybe_start()
                fn()
                job_finished, success, total_time = state.task_done(
                    batch_size, True, finish_time=time.monotonic()
                )
            except Exception as exc:
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success, total_time = state.task_done(
                    batch_size, False, finish_time=time.monotonic()
                )

            if job_finished:
                with self._condition:
                    self._finished_q.append((state.job_id, success, total_time))
                    self._inflight_jobs -= 1
                    self._condition.notify_all()
