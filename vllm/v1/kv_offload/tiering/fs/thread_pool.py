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
    key: OffloadKey
    offset: int


@dataclass
class PoolJobResult:
    """Outcome of a job whose tasks have all completed."""

    job_id: JobId
    success: bool
    transfer_time: float
    failed_tasks: list[Task]


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(keys, num_succeeded, success, transfer_time)
    when it finishes.
    """

    __slots__ = (
        "_job_id",
        "_n_tasks",
        "_completed",
        "_success",
        "_transfer_time",
        "_failed_tasks",
        "_lock",
    )

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._transfer_time = 0.0
        self._failed_tasks: list[Task] = []
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(
        self,
        batch: list[Task],
        num_succeeded: int,
        success: bool,
        transfer_time: float,
    ) -> PoolJobResult | None:
        """
        Records this task's outcome; returns the job's PoolJobResult once
        every task has reported in, else None.
        """
        with self._lock:
            self._completed += len(batch)
            self._transfer_time += transfer_time
            if not success:
                self._success = False
                self._failed_tasks.extend(batch[num_succeeded:])
            if self._completed != self._n_tasks:
                return None
            return PoolJobResult(
                job_id=self._job_id,
                success=self._success,
                transfer_time=self._transfer_time,
                failed_tasks=self._failed_tasks,
            )


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
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[PoolJobResult] = deque()
        self._inflight_jobs = 0  # guarded by _condition

        for i in range(n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(True,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(False,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

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

    def _enqueue(
        self,
        queue: deque,
        make_io_fn: Callable[[list[Task]], Callable[[], None]],
        job_id: JobId,
        tasks: Iterable[Task],
        n_tasks: int,
        n_threads: int,
    ) -> None:
        """Batch `tasks` and append (fn, state, batch_size) entries to `queue`."""
        if n_tasks == 0:
            self._finished_q.append(PoolJobResult(job_id, True, 0.0, []))
            return
        state = JobState(job_id, n_tasks)
        task_lst = list(tasks)  # Materialize tasks out of self._condition
        assert len(task_lst) == n_tasks, "Unaccounted tasks"
        n_batches = 0
        with self._condition:
            self._inflight_jobs += 1
            for batch in self._batch_tasks(task_lst, n_threads):
                queue.append((make_io_fn(batch), batch, state))
                n_batches += 1
            self._condition.notify(n_batches)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
        make_io_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""

        self._enqueue(
            self._load_q,
            make_io_fn,
            job_id,
            tasks,
            n_tasks=n_tasks,
            n_threads=1,
        )

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
        make_io_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""

        self._enqueue(
            self._store_q,
            make_io_fn,
            job_id,
            tasks,
            n_tasks=n_tasks,
            n_threads=1,
        )

    def get_finished(self) -> list[PoolJobResult]:
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
                fn, batch, state = primary.popleft() if primary else secondary.popleft()
            try:
                start_time = time.monotonic()
                fn()
                transfer_time = time.monotonic() - start_time
                result = state.task_done(batch, len(batch), True, transfer_time)
            except Exception as exc:
                transfer_time = time.monotonic() - start_time
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                # num_succeeded is only set on a partial failure (see
                # io.batch_load_block); other exceptions leave none of this
                # task's keys credited as successful.
                num_succeeded = getattr(exc, "num_succeeded", 0)
                result = state.task_done(batch, num_succeeded, False, transfer_time)

            if result is not None:
                with self._condition:
                    self._finished_q.append(result)
                    self._inflight_jobs -= 1
                    self._condition.notify_all()
