# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PooledTierManager: shared job bookkeeping for secondary tiers whose
transfers are blocking calls run on a DualQueueThreadPool.

Subclasses build one task per job (a callable that moves every chunk of the
job) and an AsyncLookupManager; this base class tracks the jobs, keeps the
prefix of a partially failed load, and emits KV events for successful stores.

A load task that fails part-way raises an exception carrying
``num_succeeded``, the number of chunks loaded before the failing one, so
those chunks are kept in the primary tier.
"""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, ClassVar

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.backpressure import BackpressureDetector
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class PartialLoadError(Exception):
    """A load task failed after loading its first ``num_succeeded`` chunks."""

    def __init__(self, num_succeeded: int) -> None:
        super().__init__(f"load failed after {num_succeeded} chunks")
        self.num_succeeded = num_succeeded


class PooledTierManager(SecondaryTierManager):
    """Secondary tier running one blocking task per job on a thread pool.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    Subclasses must assign ``self._lookup_manager`` in their ``__init__``.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    _lookup_manager: AsyncLookupManager

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        n_read_threads: int,
        n_write_threads: int,
        thread_name_prefix: str,
        enable_kv_events: bool = False,
        locality: str | None = None,
        backpressure_detector: BackpressureDetector | None = None,
    ):
        """Args:
        offloading_spec: Contains normalized offloading configuration and
            blocks_per_chunk.
        primary_kv_view: Memoryview of the primary tier's CPU KV cache.
        tier_type: Tier type identifier, set by SecondaryTierFactory.
        n_read_threads: Number of read-priority I/O threads.
        n_write_threads: Number of write-priority I/O threads.
        thread_name_prefix: Name prefix for the pool's threads.
        enable_kv_events: Emit BlockStored KV events for blocks
            successfully stored to this tier. Effective only when KV
            cache events are enabled globally (kv_events_config).
        locality: Whether this tier's storage is LOCAL or REMOTE relative
            to the publishing vLLM instance.
        backpressure_detector: Optional backpressure detector.

        """
        super().__init__(
            offloading_spec, primary_kv_view, tier_type, backpressure_detector
        )
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Keys of in-flight load (promotion) jobs, so a failed load can mark
        # its own cached lookup verdicts False (see get_finished_jobs).
        self._load_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Block count per in-flight job, used to report transfer_bytes.
        self._job_block_counts: dict[JobId, int] = {}
        # Per load job: how many blocks loaded before a failure (partial keep).
        # Written by the pool worker inside the load task before it raises (so
        # before task_done publishes the job); read on the scheduler thread in
        # get_finished_jobs only for job ids the finished queue returned. Under
        # the GIL that read cannot observe the finished job without the prior
        # write, so no extra lock is needed (get_finished is itself lock-free).
        self._load_progress: dict[JobId, int] = {}

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix=thread_name_prefix,
        )

    def _chunk_offsets(self, job_metadata: TransferJob) -> list[int]:
        """Byte offset in the primary view of each chunk in the job."""
        return [int(cid) * self._block_size_bytes for cid in job_metadata.chunk_ids]

    def _enqueue_store(
        self, job_metadata: TransferJob, task: Callable[[], None]
    ) -> None:
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        self._job_block_counts[job_metadata.job_id] = len(job_metadata.keys)
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    def _enqueue_load(
        self, job_metadata: TransferJob, task: Callable[[], None]
    ) -> None:
        job_id = job_metadata.job_id
        # Track this load's keys so a failed promotion can mark only its failed
        # keys as a miss (see get_finished_jobs).
        keys = list(job_metadata.keys)
        self._load_job_keys[job_id] = keys
        self._job_block_counts[job_id] = len(keys)

        def load_task() -> None:
            try:
                task()
            except Exception as exc:
                # Runs on the pool worker thread. Record how many blocks loaded
                # before the failure so get_finished_jobs can keep them; this
                # write precedes task_done, so the scheduler reads it safely
                # under the GIL once the finished queue hands back this job.
                num_succeeded = getattr(exc, "num_succeeded", 0)
                self._load_progress[job_id] = num_succeeded
                logger.debug(
                    "Load of %d blocks for job %s failed at block %d: %s",
                    len(keys),
                    job_id,
                    num_succeeded,
                    exc,
                )
                raise

        self._pool.enqueue_load(job_id, 1, [load_task])

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Collect finished jobs; a failed promotion marks only its failed keys
        as a miss here (scheduler thread)."""
        results = []
        for job_id, success, transfer_time in self._pool.get_finished():
            block_count = self._job_block_counts.pop(job_id, 0)
            transfer_bytes = (
                block_count * self._block_size_bytes if block_count else None
            )
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            load_keys = self._load_job_keys.pop(job_id, None)
            num_succeeded = self._load_progress.pop(job_id, 0)
            if load_keys is not None and not success:
                # A batched load stops at the first bad block and reports how
                # many loaded before it. Those earlier blocks are kept in the
                # primary tier (reported via successful_keys); only this block
                # and the ones after it are marked a miss and recomputed.
                successful = load_keys[:num_succeeded]
                failed = load_keys[num_succeeded:]
                self._lookup_manager.mark_miss(failed)
                results.append(
                    JobResult(
                        job_id=job_id,
                        success=False,
                        successful_keys=tuple(successful) if successful else None,
                        transfer_time=transfer_time,
                        transfer_bytes=transfer_bytes,
                    )
                )
                continue
            results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    transfer_time=transfer_time,
                    transfer_bytes=transfer_bytes,
                )
            )
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
