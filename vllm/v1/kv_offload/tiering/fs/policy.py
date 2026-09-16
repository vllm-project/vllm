# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import heapq
import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

from vllm.v1.kv_offload.base import Locality
from vllm.v1.kv_offload.tiering.base import JobId


class JobQueue(ABC):
    def __init__(self, block_size: int):
        self._block_size = block_size

    @abstractmethod
    def put(self, job_id: JobId, num_tasks: int):
        pass

    @abstractmethod
    def get(self) -> JobId | None:
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def maybe_has_work(self):
        pass


# put tuples (job_id, num_tasks) as and when they arrive.
# On get(), return job in FCFS order.
class FCFSQueue(JobQueue):
    def __init__(self, block_size: int):
        super().__init__(block_size)
        self.q: deque[JobId] = deque()

    def put(self, job_id: JobId, num_tasks: int):
        self.q.append(job_id)

    def get(self) -> JobId | None:
        return self.q.popleft() if self.q else None

    def clear(self):
        self.q.clear()

    def maybe_has_work(self):
        return bool(self.q)


class SJFHeapQueue(JobQueue):
    """Min-heap queue that returns the shortest job (fewest tasks) first."""

    def __init__(self, block_size: int):
        super().__init__(block_size)
        # heap entries: (num_tasks, job_id)
        self._heap: list[tuple[int, JobId]] = []

    def put(self, job_id: JobId, num_tasks: int):
        heapq.heappush(self._heap, (num_tasks, job_id))

    def get(self) -> JobId | None:
        if not self._heap:
            return None
        _, job_id = heapq.heappop(self._heap)
        return job_id

    def clear(self):
        self._heap.clear()

    def maybe_has_work(self):
        return bool(self._heap)


# put tuples (job_id, num_tasks) as and when they arrive.
# On get(), return the FCFS job in the minimum bucket
class SJFBucketQueue(JobQueue):
    """
    Assign incoming jobs into buckets based on the size of the job.
    When queried, determine if the current workload and multi-modal,
    i.e. could be classified in to short / long jobs, and return a
    job from the short bucket.
    - Advantages of a bucket approach:
        * Maintain FCFS within each bucket. 2 jobs with similar
        num_tasks are executed in FCFS order.
        * Allows adding / removing elements in constant time.
        * It also allows to reason about the multi-modality of
          the current workload. For example,
           - if only one bucket is filled. Then it is uni-modal
           - if contiguous buckets are filled, then it is somewhat bi-modal
           - if 2 non contiguous buckets are filled, the it definitely
             bi-modal.

    Implementation:
        - Bucket boundaries grow exponentially (log₂ of job size in bytes).
          This provides high resolution where it matters most — distinguishing
          small jobs — while coarsely grouping jobs at the large end.
    """

    def __init__(self, block_size: int):
        super().__init__(block_size)
        # bitmask indicating if a bucket has jobs
        self.mask = ctypes.c_uint32(0x00000000)

        self.num_buckets = ctypes.sizeof(self.mask) * 8
        self.q: list[FCFSQueue] = [
            FCFSQueue(block_size) for _ in range(self.num_buckets)
        ]

    def _get_bucket_id(self, job_id: JobId, num_tasks: int):
        assert num_tasks != 0
        mbytes = max(1, num_tasks * self._block_size)
        return min(self.num_buckets - 1, int(math.floor(math.log2(mbytes))))

    def _get_sjf_bucket(self):
        x = self.mask.value
        assert x != 0
        # ~v inverts the bits.
        # + 1 completes the two's complement.
        # & ones (0xFFFFFFFF) forces it to stay within 32-bit unsigned bounds.
        ones = (1 << self.num_buckets) - 1
        pow2 = x & ((~x + 1) & ones)
        return pow2.bit_length() - 1

    def put(self, job_id: JobId, num_tasks: int):
        bid = self._get_bucket_id(job_id, num_tasks)
        self.q[bid].put(job_id, num_tasks)
        self.mask.value |= 1 << bid

    def get(self) -> JobId | None:
        if not self.maybe_has_work():
            return None
        bid = self._get_sjf_bucket()
        job_id = self.q[bid].get()
        assert job_id is not None
        if not self.q[bid].maybe_has_work():
            self.mask.value &= ~(1 << bid)
        return job_id

    def clear(self):
        for x in self.q:
            x.clear()
        self.mask = ctypes.c_uint32(0x00000000)

    def maybe_has_work(self):
        return self.mask.value != 0


class LoadQueue(JobQueue):
    # For loads, we care about both,
    # 1. FCFS - For better TTFT, and
    # 2. SJF  - For clearing CPU KV cache.
    # As a result we alternate between the FCFS and SJF queues
    # for get().

    def __init__(self, block_size: int):
        super().__init__(block_size)
        self.fcfs: FCFSQueue = FCFSQueue(block_size)
        self.sjf: SJFBucketQueue = SJFBucketQueue(block_size)
        # ping-ping index
        self.pp = 0

        # set of jobs in the queues.
        self.jobs: set[JobId] = set()

    def put(self, job_id: JobId, num_tasks: int):
        self.fcfs.put(job_id, num_tasks)
        self.sjf.put(job_id, num_tasks)
        self.jobs.add(job_id)

    def get(self) -> JobId | None:
        def get_job(job_q: JobQueue):
            while (jid := job_q.get()) is not None:
                if jid in self.jobs:
                    break
            return jid

        q, fallback = (self.fcfs, self.sjf) if self.pp == 0 else (self.sjf, self.fcfs)

        if (jid := get_job(q)) is None:
            jid = get_job(fallback)

        if jid is None:
            return None

        self.pp = 1 if self.pp == 0 else 0
        self.jobs.remove(jid)
        return jid

    def maybe_has_work(self):
        return self.fcfs.maybe_has_work() or self.sjf.maybe_has_work()

    def clear(self):
        self.fcfs.clear()
        self.sjf.clear()
        self.pp = 0
        self.jobs.clear()


StoreQueue = SJFBucketQueue


class Scheduler:
    def __init__(
        self,
        locality: Locality,
        load_job_q: JobQueue,
        store_job_q: JobQueue,
        n_read_threads: int,
        n_write_threads: int,
    ):
        self._locality = locality
        self._load_job_q = load_job_q
        self._store_job_q = store_job_q
        self._n_read_threads = n_read_threads
        self._n_write_threads = n_write_threads

        if self._locality == Locality.LOCAL:
            # Assume SSD
            self._n_read_batch_threads = n_read_threads or self.total_threads
            # Limit concurrent SSD writes to 1 thread. When a NAND die is
            # busy with a write, any read to that die stalls until the write
            # completes. More write threads means more dies occupied at once,
            # directly hurting read latency.
            # NOTE: We can make this configurable in the future if needed.
            self._n_write_batch_threads = 1
        else:
            # Assume NFS
            self._n_read_batch_threads = n_read_threads or self.total_threads
            self._n_write_batch_threads = n_write_threads or self.total_threads

        # Allow load threads share store jobs.
        # - Store jobs are tied to engine running requests.
        #   They are effectively bounded by engine forward pass.
        #   i.e. they are relatively short and quick.
        #
        # Don't allow store threads to grab load jobs.
        # - Load jobs are tied to engine waiting requests.
        #   If we receive a lot of requests, we'd lock up the
        #   the store threads from draining the store queue.
        # - Load jobs maybe long. This will commit the store
        # threads to long running loads, stalling the stores
        # from draining.
        self._read_threads_can_write = True
        self._write_threads_can_read = False or self._n_read_threads == 0

        # Work queues that the threads draw work from
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._jobs: dict[JobId, Any] = {}

    @property
    def total_threads(self):
        return self._n_read_threads + self._n_write_threads

    def submit(
        self,
        job_id: JobId,
        state: Any,
        tasks: list[Any],
        make_batch_fn: Callable,
        is_load: bool,
    ) -> int:
        """
        Submit job and return number of threads to wake up.
        """
        self._jobs[job_id] = (state, tasks, make_batch_fn, is_load)
        if is_load:
            self._load_job_q.put(job_id, len(tasks))
        else:
            self._store_job_q.put(job_id, len(tasks))

        # TODO(varun): unfortunate! - wake selectively
        return self._n_read_threads + self._n_write_threads

    def has_work(self, load_priority: bool):
        is_read_thread = load_priority
        has_load_work = self._load_q or self._load_job_q.maybe_has_work()
        has_store_work = self._store_q or self._store_job_q.maybe_has_work()
        if is_read_thread:
            return (
                has_load_work or has_store_work
                if self._read_threads_can_write
                else has_load_work
            )
        else:
            return (
                has_load_work or has_store_work
                if self._write_threads_can_read
                else has_store_work
            )

    def _batch_tasks(
        self,
        tasks: list[Any],
        n_threads: int,
    ) -> Iterator[list[Any]]:
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

    def _maybe_populate_work_q(
        self, work_q: deque, job_q: JobQueue, n_batch_threads: int
    ):
        if work_q:
            return
        job_id = job_q.get()
        if job_id is None:
            return

        work_item = self._jobs.pop(job_id)
        state, tasks, make_batch_fn, _ = work_item
        for b in self._batch_tasks(tasks, n_batch_threads):
            work_q.append((make_batch_fn(b), len(b), state))
        return

    def fetch_work(self, load_priority: bool):
        is_read_thread = load_priority
        if is_read_thread:
            self._maybe_populate_work_q(
                self._load_q, self._load_job_q, self._n_read_batch_threads
            )
            if self._load_q:
                return self._load_q.popleft()
            if self._read_threads_can_write:
                self._maybe_populate_work_q(
                    self._store_q, self._store_job_q, self._n_write_batch_threads
                )
                if self._store_q:
                    return self._store_q.popleft()
            return None
        else:
            self._maybe_populate_work_q(
                self._store_q, self._store_job_q, self._n_write_batch_threads
            )
            if self._store_q:
                return self._store_q.popleft()
            if self._write_threads_can_read:
                self._maybe_populate_work_q(
                    self._load_q, self._load_job_q, self._n_read_batch_threads
                )
                if self._load_q:
                    return self._load_q.popleft()
            return None

    def clear(
        self,
    ):
        self._load_job_q.clear()
        self._store_job_q.clear()
        self._load_q.clear()
        self._store_q.clear()
        self._jobs.clear()
