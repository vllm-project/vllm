# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TieringOffloadingManager: Multi-tier KV cache offloading orchestrator.

This manager coordinates between a CPU primary tier (with direct GPU access)
and zero or more secondary tiers (Storage, Network, etc.) to provide
hierarchical KV cache offloading.

Key Design Principles:
1. Always offload to all tiers — When a chunk is stored to the primary tier,
   it is cascaded to ALL secondary tiers
2. Primary tier is the gateway — Secondary tiers cannot access GPU memory
   directly; all data flows through the CPU primary tier
3. Staged promotion — Chunks in secondary tiers must be promoted to the
   primary tier before GPU can access them
4. Transparent retry mechanism — Return None from lookup() to signal
   "data is being promoted, try later"
5. ref_cnt as eviction protection — primary.prepare_read() increments ref_cnt,
   protecting chunks from eviction until complete_read() is called
"""

import contextlib
import threading
import time
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    ParentManager,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.metrics import TieringMetricsTracker

logger = init_logger(__name__)

# Sleep between control-plane sweeps. A tier's poll is non-blocking (the p2p
# tier's ZMQ transport returns immediately when there is no traffic), so the
# thread has to pace itself. Small enough to be irrelevant next to a step,
# large enough not to fight the scheduler thread for the GIL.
_CONTROL_POLL_INTERVAL_S = 0.001

# How long a sweep waits for the step lock before looping. Bounded so the stop
# event is always observed promptly, and so a step that never reached
# on_schedule_end is reported instead of silently wedging the control plane.
_CONTROL_LOCK_TIMEOUT_S = 1.0

# Warn once the control plane has been unable to run for this long.
_CONTROL_STARVATION_WARN_S = 10.0

# How long shutdown waits for the control thread to finish its current sweep.
_CONTROL_THREAD_JOIN_TIMEOUT_S = 2.0


@dataclass
class PendingPromotion:
    """Accumulator for chunks awaiting submit_load() for one (tier, request)."""

    req_context: ReqContext
    keys: list[OffloadKey] = field(default_factory=list)
    chunk_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class RequestState:
    req_context: ReqContext
    pending_primary_stores: int = 0
    is_finished: bool = False
    request_level_tiers: set[int] | None = None
    pending_cascade_keys: list[OffloadKey] = field(default_factory=list)


class JobMetadata(NamedTuple):
    transfer_job: TransferJob
    tier_idx: int


class CPUPrimaryTierOffloadingManager(CPUOffloadingManager):
    """CPUOffloadingManager with a primary/secondary transfer interface.

    The inherited prepare_store/complete_store/prepare_load/complete_load are the
    GPU-facing OffloadingManager interface. These aliases expose the same operations
    from the secondary tier perspective, where read/write refers to secondary
    accessing primary. This avoids confusion when reading TieringOffloadingManager
    code (e.g. calling prepare_load inside a cascade/store path would be misleading).
    """

    def __init__(
        self,
        num_chunks: int,
        mmap_region: SharedOffloadRegion,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
    ):
        super().__init__(
            num_chunks=num_chunks,
            cache_policy=cache_policy,
            cache_policy_module_path=cache_policy_module_path,
            enable_events=enable_events,
        )
        self._mmap_region = mmap_region
        # read/write is for CPU<->secondary transfers,
        # load/store is for CPU<->GPU transfers.
        # These aliases avoid calling prepare_load inside a store path.
        self.complete_read = self.complete_load
        self.prepare_write = self.prepare_store
        self.complete_write = self.complete_store

        self._kv_memoryview = mmap_region.create_kv_memoryview()

    def prepare_read(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        """Pin chunks for a CPU-to-secondary transfer.

        Cascade reads are implementation details of tiering, not additional
        request accesses, so they must not alter request-scoped recency.
        """
        return self._prepare_load(keys, req_context, record_access=False)

    def get_kv_memoryview(self) -> memoryview:
        """Return the memoryview over the primary tier's KV cache buffer.

        The view has shape (num_chunks, row_stride_bytes) and is backed by the
        SharedOffloadRegion mmap.  Secondary tiers address chunk *c* as
        ``view[c]``.
        """
        return self._kv_memoryview

    @override
    def shutdown(self) -> None:
        super().shutdown()
        self._kv_memoryview.release()
        self._mmap_region.cleanup()


class _SecondaryTierFacingParent(ParentManager):
    """Wrapper that implements ParentManager by delegating to the
    TieringOffloadingManager with exclude_tier_idx set to the origin tier.

    Handed to a tier only for the duration of serve_external_requests(), whose
    caller — the control-plane thread, or on_schedule_end when that thread is
    disabled — already holds the manager lock. So every method here delegates
    to the *_unlocked variant: taking the lock again would deadlock, since it
    is not reentrant.
    """

    __slots__ = ("_m", "_origin_idx")

    def __init__(
        self,
        manager: "TieringOffloadingManager",
        tier_idx: int,
    ):
        self._m = manager
        self._origin_idx = tier_idx

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return self._m._on_new_request_unlocked(req_context, self._origin_idx)

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return self._m._lookup_unlocked(key, req_context, self._origin_idx)

    def create_store_job(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> TransferJob:
        return self._m._create_store_job_unlocked(keys, req_context, self._origin_idx)

    def on_request_finished(self, req_context: ReqContext) -> None:
        return self._m._on_request_finished_unlocked(req_context, self._origin_idx)


class TieringOffloadingManager(OffloadingManager):
    """Orchestrates multi-tier KV cache offloading.

    This manager coordinates between a CPU primary tier (with direct GPU access)
    and zero or more secondary tiers (Storage, Network, etc.) to provide
    hierarchical KV cache offloading.

    Key internal state:
      - Minimal state tracking; relies on secondary tiers to report completion
        via get_finished_jobs()
      - Secondary tiers return JobResult objects containing all necessary
        information
      - job_id_counter: monotonically increasing counter for job IDs

    Threading
    ---------
    Two threads touch this manager, serialized by ``_lock``:

    * The scheduler thread runs every public method. It takes ``_lock`` on its
      first call of a scheduler step and drops it at the end of
      ``on_schedule_end`` — step granularity, not per call. That matters: the
      connector learns a chunk is a HIT from ``lookup()`` and only reads it in
      a later ``prepare_load()`` hook, and in between the chunk is still
      evictable. A promotion started while serving a peer lookup in that window
      could evict it out from under the pending load.
    * The control-plane thread (``_control_plane_loop``) polls tiers for
      finished jobs and lets them serve inbound peer requests. It holds
      ``_lock`` for a whole sweep, so it runs only between steps — in practice
      during model execution, which is exactly the time a rank busy with a long
      prefill has to spare. Without it, a peer's control-plane round trip costs
      a full step boundary.

    The read-mostly hooks (``has_pending_work``, ``get_stats``,
    ``take_events``) are the exception: they hold the lock only for their own
    duration, via ``_observation_lock``. The engine calls them on every tick,
    so folding them into the step-wide hold would leave an idle engine with no
    gap between steps at all.

    Tiers re-enter the manager from that thread through
    ``_SecondaryTierFacingParent``, which calls the ``*_unlocked`` variants
    because its caller already holds the lock.
    """

    def __init__(
        self,
        primary_tier: CPUPrimaryTierOffloadingManager,
        secondary_tiers: list[SecondaryTierManager] | None = None,
        *,
        control_plane_thread: bool = True,
        control_poll_interval_s: float = _CONTROL_POLL_INTERVAL_S,
    ):
        """Initialize the TieringOffloadingManager.

        Args:
            primary_tier: The primary tier manager (CPU-based).
            secondary_tiers: List of secondary tier managers (e.g., Storage,
                            Network). Can be None or empty list.
            control_plane_thread: Whether to poll tiers and serve their inbound
                            peer requests from a dedicated thread rather than
                            from the per-step hooks. Pass False to keep all of
                            it on the calling thread, which is what
                            step-driven tests need.
            control_poll_interval_s: Sleep between control-plane sweeps.

        """
        self.primary_tier: CPUPrimaryTierOffloadingManager = primary_tier
        self.secondary_tiers = secondary_tiers or []

        self._job_id_counter: int = 0
        # Job tracking: maps job_id to metadata for all in-flight transfers.
        # TransferJob.is_promotion distinguishes direction:
        #   True:  secondary → primary (promotion)
        #   False: primary → secondary (cascade)
        self._jobs: dict[JobId, JobMetadata] = {}
        primary_view = self.primary_tier.get_kv_memoryview()
        assert primary_view.strides is not None
        self._metrics = TieringMetricsTracker(
            tier_types=[tier.tier_type for tier in self.secondary_tiers],
            num_primary_chunks=self.primary_tier._num_chunks,
            primary_chunk_size=primary_view.strides[0],
        )

        # Pending promotion requests accumulated during lookup() calls; flushed
        # as one batched submit_load() per (tier, request) in on_schedule_end().
        # Outer key: tier index. Inner key: req_context.req_id — the same ReqContext
        # object is reused for all chunk lookups of a given request per engine step.
        self._pending_load_submissions: dict[int, dict[str, PendingPromotion]] = {}

        # Gate for once-per-step execution of _maybe_process_finished_jobs().
        # Reset at the end of each step in on_schedule_end().
        self._processed_jobs_this_step: bool = False

        # Per-request state for prepared GPU->primary stores and finalization.
        # Secondary tiers are finalized only after pending primary stores reach
        # complete_store(), since complete_store() can still submit cascades.
        self._req_state: dict[str, RequestState] = {}

        # Cached ParentManager wrappers for each secondary tier.
        self._tier_parents: dict[SecondaryTierManager, _SecondaryTierFacingParent] = {
            tier: _SecondaryTierFacingParent(self, tier_idx)
            for tier_idx, tier in enumerate(self.secondary_tiers)
        }

        # Serializes the scheduler thread against the control-plane thread. See
        # the class docstring for why it is held for a whole step.
        self._lock = threading.Lock()
        # True while the scheduler side holds _lock for the current step. Only
        # scheduler-side callers read or write it, and that side is
        # single-threaded, so it needs no protection of its own; the
        # control-plane thread never touches it.
        self._step_locked: bool = False

        self._control_thread_enabled: bool = control_plane_thread and bool(
            self.secondary_tiers
        )
        self._control_poll_interval_s: float = control_poll_interval_s
        self._control_thread: threading.Thread | None = None
        self._control_thread_stop = threading.Event()
        # Exception raised inside the control-plane thread, re-raised on the
        # next scheduler-side call. These failures used to happen on the
        # scheduler thread and take the engine down with a real traceback;
        # swallowing them here would instead leave the control plane silently
        # dead, which looks like a hang.
        self._control_thread_exc: BaseException | None = None

    # ------------------------------------------------------------------
    # Step lock
    # ------------------------------------------------------------------

    def _enter_step(self, *, raise_control_error: bool = True) -> None:
        """Take the manager lock for the current scheduler step.

        Idempotent within a step: the first scheduler-side call acquires, later
        ones are no-ops, and on_schedule_end releases. Never called from the
        control-plane thread, which holds _lock around its whole sweep and
        re-enters the manager through the *_unlocked helpers.

        Args:
            raise_control_error: Whether to re-raise a control-plane thread
                failure. Only shutdown passes False, so a failed sweep cannot
                mask teardown.

        Raises:
            BaseException: Whatever the control-plane thread failed with.

        """
        if raise_control_error and self._control_thread_exc is not None:
            raise self._control_thread_exc
        if self._step_locked:
            return
        self._lock.acquire()
        self._step_locked = True

    def _exit_step(self) -> None:
        """Release the manager lock, letting the control plane run."""
        if not self._step_locked:
            return
        self._step_locked = False
        self._lock.release()

    @contextlib.contextmanager
    def _observation_lock(self) -> Iterator[None]:
        """Hold the manager lock only for the duration of one observation.

        For the read-mostly hooks (has_pending_work, get_stats, take_events).
        They take no part in the lookup()->prepare_load() invariant that makes
        the step-wide hold necessary, and the engine calls them on every tick,
        including ticks that schedule nothing and run no model. Letting them
        extend a step's hold would leave almost no gap between one step and the
        next on an idle engine — exactly the state a producer waiting for a
        consumer to connect is in, and exactly when the control plane matters.
        """
        if self._control_thread_exc is not None:
            raise self._control_thread_exc
        if self._step_locked:
            # Already held for this step; the step's release point owns it.
            yield
            return
        with self._lock:
            yield

    @property
    def _transfer_jobs(self) -> dict[JobId, JobMetadata]:
        return self._jobs

    def _next_job_id(self) -> JobId:
        """Generate a unique job ID for async transfer tracking."""
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id

    def _register_job(self, transfer_job: TransferJob, tier_idx: int) -> None:
        job_metadata = JobMetadata(transfer_job, tier_idx)
        self._jobs[transfer_job.job_id] = job_metadata
        self._metrics.on_job_registered(job_metadata)

    def _pop_job(self, job_id: JobId) -> JobMetadata | None:
        return self._jobs.pop(job_id, None)

    def _maybe_process_finished_jobs(self):
        """Poll secondary tiers for completed jobs (at most once per step).

        Guarded by _processed_jobs_this_step: the first call in an engine step
        does the actual polling; subsequent calls are no-ops. The flag is reset
        in on_schedule_end() at the end of each step.

        A no-op once the control-plane thread is running: that thread owns tier
        polling, and repeating it here would put the very transport polls we
        moved off the scheduler thread straight back onto it.
        """
        if self._control_thread is not None:
            return
        if self._processed_jobs_this_step:
            return
        self._processed_jobs_this_step = True
        self._process_finished_jobs()

    def _complete_promotion(
        self, job_metadata: JobMetadata, completed_job: JobResult
    ) -> None:
        transfer_job = job_metadata.transfer_job
        successful_keys = completed_job.successful_keys
        failed_keys: Collection[OffloadKey]
        if completed_job.success:
            successful_keys = transfer_job.keys
            failed_keys = ()
        elif successful_keys:
            failed_keys_set = set(transfer_job.keys)
            assert failed_keys_set.issuperset(successful_keys), (
                f"Finished promotion job_id {completed_job.job_id} "
                "reported unknown successful keys"
            )
            failed_keys_set.difference_update(successful_keys)
            failed_keys = failed_keys_set
        else:
            successful_keys = ()
            failed_keys = transfer_job.keys

        if successful_keys:
            self.primary_tier.complete_write(
                successful_keys,
                transfer_job.req_context,
                True,
            )
        if failed_keys:
            self.primary_tier.complete_write(
                failed_keys,
                transfer_job.req_context,
                False,
            )

    def _process_finished_jobs(self):
        """Unconditionally poll all secondary tiers for completed jobs.

        This method:
        1. Calls get_finished_jobs() on each secondary tier
        2. For completed stores (primary→secondary): calls primary.complete_read()
           to decrement ref_cnt
        3. For completed loads (secondary→primary): calls primary.complete_write()
           to make chunks available
        """
        for i, tier in enumerate(self.secondary_tiers):
            for completed_job in tier.get_finished_jobs():
                job_id = completed_job.job_id
                job_metadata = self._pop_job(job_id)
                assert job_metadata is not None, (
                    f"Finished job_id {job_id} from tier #{i}"
                    f" ({tier.tier_type}) not in _jobs"
                )
                assert job_metadata.tier_idx == i, (
                    f"Finished job_id {job_id} reported by tier #{i}"
                    f" but belongs to tier #{job_metadata.tier_idx}"
                )
                transfer_job = job_metadata.transfer_job
                self._metrics.on_job_finished(job_metadata, completed_job)

                if transfer_job.is_promotion:
                    # secondary→primary transfer (promotion) completed.
                    # Make chunks available in primary tier.
                    self._complete_promotion(job_metadata, completed_job)
                else:
                    # primary→secondary transfer completed.
                    # Decrement ref_cnt on primary chunks.
                    self.primary_tier.complete_read(
                        transfer_job.keys, transfer_job.req_context
                    )

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        """Check whether a single chunk is offloaded and ready.

        See _lookup_unlocked for the algorithm and the return values.
        """
        self._enter_step()
        return self._lookup_unlocked(key, req_context)

    def _lookup_unlocked(
        self,
        key: OffloadKey,
        req_context: ReqContext,
        exclude_tier_idx: int | None = None,
    ) -> LookupResult:
        """Check whether a single chunk is offloaded and ready.

        The caller must hold the manager lock.

        Algorithm:
            1. Process any completed async jobs first.
            2. Query primary tier — short-circuit on hit or in-flight.
            3. On primary miss, query secondary tiers — stop on first
               hit and initiate promotion.

        Args:
            key: Chunk hash to look up.
            req_context: Per-request context.
            exclude_tier_idx: Skip this tier index during the lookup.

        Returns:
            HIT       — chunk is ready in the primary tier.
            HIT_PENDING — chunk found but not yet readable (write
                        in-flight on the primary tier).
            RETRY     — promotion started or a secondary tier is busy.
            MISS      — chunk not found in any tier, or primary is full
                        and cannot accept a promotion.

        """
        # Poll first so a promotion that finished since the last call is
        # already reflected as HIT (not stale HIT_PENDING/MISS) below, and
        # so chunks freed by cascade or promotion completions are evictable
        # in time for a promotion this lookup may initiate.
        self._maybe_process_finished_jobs()

        start_time = time.monotonic()
        primary_hit = self.primary_tier.lookup(key, req_context)
        lookup_duration = time.monotonic() - start_time
        self._metrics.on_lookup(
            req_context,
            key,
            self._metrics.primary_tier_label,
            primary_hit,
            lookup_duration,
        )
        if primary_hit is LookupResult.HIT:
            return LookupResult.HIT
        if primary_hit is LookupResult.HIT_PENDING:
            return LookupResult.HIT_PENDING

        any_retry = False
        for i, tier in enumerate(self.secondary_tiers):
            if i == exclude_tier_idx:
                continue
            if not req_context.load_tier_filter.allows(tier.medium, tier.locality):
                continue
            labelvalues = self._metrics.tier_label(i)
            start_time = time.monotonic()
            result = tier.lookup(key, req_context)
            lookup_duration = time.monotonic() - start_time
            if result is LookupResult.HIT:
                self._metrics.on_lookup(
                    req_context,
                    key,
                    labelvalues,
                    result,
                    lookup_duration,
                )
                promoted = self._initiate_promotion(i, key, req_context)
                return LookupResult.MISS if not promoted else LookupResult.HIT_PENDING
            if result is LookupResult.RETRY:
                any_retry = True
            self._metrics.on_lookup(
                req_context,
                key,
                labelvalues,
                result,
                lookup_duration,
            )

        if any_retry:
            return LookupResult.RETRY
        return LookupResult.MISS

    def _initiate_promotion(
        self,
        tier_idx: int,
        key: OffloadKey,
        req_context: ReqContext,
    ) -> bool:
        """Queue a chunk for promotion from a secondary tier to the primary tier.

        Allocates space in the primary tier immediately (sets ref_cnt=-1 so
        subsequent lookups within the same step see the slot as in-flight),
        then defers the actual submit_load() call to _flush_pending_promotions()
        so all chunks queued during one engine step are submitted as a single
        batched job.

        Args:
            tier_idx: The secondary tier index to promote from
            key: Chunk to promote
            req_context: Per-request context forwarded to primary.prepare_write().

        Returns:
            True if promotion was initiated, False if primary tier is full.

        """
        # Allocate space in primary tier for promoted chunk.
        # Must happen immediately so primary.lookup() returns None (in-flight)
        # for this key on any subsequent lookup() call within the same step,
        # preventing duplicate promotion attempts.
        primary_write_result = self.primary_tier.prepare_write([key], req_context)

        if primary_write_result is None:
            # Primary tier is full; caller should treat the chunk as unavailable
            # rather than retrying indefinitely.
            self._metrics.on_promotion_allocation_failure()
            return False

        store_spec = primary_write_result.store_spec
        assert isinstance(store_spec, CPULoadStoreSpec)
        # Defer submit_load to on_schedule_end(). Group by (tier, request) so
        # each request's chunks are submitted as one batched job per tier.
        tier_pending = self._pending_load_submissions.setdefault(tier_idx, {})
        ctx_id = req_context.req_id
        if ctx_id not in tier_pending:
            tier_pending[ctx_id] = PendingPromotion(
                keys=[], chunk_ids=[], req_context=req_context
            )
        entry = tier_pending[ctx_id]
        entry.keys.extend(primary_write_result.keys_to_store)
        entry.chunk_ids.extend(store_spec.chunk_ids)
        return True

    def _flush_pending_promotions(self) -> None:
        """Submit one batched submit_load() per (tier, request).

        Called from on_schedule_end() at the end of each scheduler step,
        flushing all promotion requests deferred during lookup().
        """
        if not self._pending_load_submissions:
            return

        for tier_idx, pending_by_ctx in self._pending_load_submissions.items():
            tier = self.secondary_tiers[tier_idx]
            for entry in pending_by_ctx.values():
                job_id = self._next_job_id()
                job_metadata = TransferJob(
                    job_id=job_id,
                    keys=entry.keys,
                    chunk_ids=np.array(entry.chunk_ids, dtype=np.int32),
                    is_promotion=True,
                    req_context=entry.req_context,
                )
                self._register_job(job_metadata, tier_idx)
                tier.submit_load(job_metadata)

        self._pending_load_submissions.clear()

    @override
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        """Prepare chunks to be loaded from primary tier to GPU.

        Callers only pass keys already confirmed HIT by lookup() earlier this
        step.

        This increments ref_cnt on the chunks in the primary tier, protecting
        them from eviction during the transfer.

        Args:
            keys: Chunks to prepare for loading.
            req_context: Per-request context.

        Returns:
            LoadStoreSpec for reading from primary tier.

        """
        self._enter_step()
        return self.primary_tier.prepare_load(keys, req_context)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """Mark chunks as recently used in all tiers.

        Args:
            keys: Chunks to mark as recently used.
            req_context: Per-request context.

        """
        self._enter_step()
        self.primary_tier.touch(keys, req_context)
        for tier in self.secondary_tiers:
            tier.touch(keys, req_context)

    @override
    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """Mark chunks as done loading from primary tier to GPU.

        This decrements ref_cnt on the chunks in the primary tier, allowing
        them to be evicted again.

        Args:
            keys: Chunks that finished loading.
            req_context: Per-request context.

        """
        self._enter_step()
        self.primary_tier.complete_load(keys, req_context)

    @override
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """Prepare chunks to be stored from GPU to primary tier.

        CRITICAL: This method calls _maybe_process_finished_jobs() FIRST to ensure
        that any completed async transfers have their ref_cnt decremented
        before the primary tier makes eviction decisions.

        For request-level tiers, chunks already present in the primary tier
        are immediately cascaded via submit_store().

        Args:
            keys: Chunks to prepare for storing.
            req_context: Per-request context.

        Returns:
            PrepareStoreOutput describing where to store chunks and what was
            evicted, or None if store cannot proceed.

        """
        # Step 1: Poll for completed async jobs FIRST
        # _process_finished_jobs() handles two kinds of completions here:
        #  - Cascade completions (store to a secondary tier, either a local
        #    cascade or a store job created for a remote requester via
        #    create_store_job()): decrements ref_cnt on the primary chunks
        #    that were read, making them evictable again once ref_cnt hits 0.
        #  - Promotion completions (secondary->primary loads): sets a
        #    not-yet-ready chunk's ref_cnt from -1 to 0 via complete_write(),
        #    making it evictable for the first time.
        # Both must be accounted for before the eviction decision below.
        self._enter_step()
        self._maybe_process_finished_jobs()

        # Step 2: Store to primary tier (new chunks only).
        # Cascading of these newly-stored chunks to ALL secondary tiers
        # happens later in complete_store(), after the GPU→Primary transfer
        # completes.
        primary_result = self.primary_tier.prepare_store(keys, req_context)

        if primary_result is None:
            return None

        if primary_result.keys_to_store:
            state = self._req_state[req_context.req_id]
            state.pending_primary_stores += 1

        # Step 3: For request-level tiers, cascade chunks already in primary
        request_level_tiers = self._req_state[req_context.req_id].request_level_tiers
        if request_level_tiers:
            keys_to_store_set = set(primary_result.keys_to_store)
            keys_already_in_primary = tuple(
                k for k in keys if k not in keys_to_store_set
            )
            if keys_already_in_primary:
                self._cascade_existing_chunks_to_request_level_tiers(
                    keys_already_in_primary, req_context, request_level_tiers
                )

        return primary_result

    def _cascade_existing_chunks_to_request_level_tiers(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext,
        request_level_tiers: set[int],
    ) -> None:
        """For tiers that requested request-level policy, submit_store() for
        chunks that are already present in the primary tier.

        A key whose primary write is still in flight (HIT_PENDING) cannot be
        dropped: prepare_store already excluded it as present, and the
        scheduler advances past its chunk, so no path offers it again. Park it
        instead. MISS keys are dropped, since nothing is there to read.

        The primary tier resolves every key it holds, so RETRY cannot reach
        here. Parking on it would have no guarantee of ever draining, which is
        what makes parking HIT_PENDING safe, so it is rejected rather than
        guessed at.
        """
        state = self._req_state[req_context.req_id]
        ready_keys = []
        for key in keys:
            result = self.primary_tier.lookup(key, req_context)
            if result is LookupResult.HIT:
                ready_keys.append(key)
            elif result is LookupResult.HIT_PENDING:
                state.pending_cascade_keys.append(key)
            else:
                assert result is LookupResult.MISS, (
                    f"primary tier returned {result} for a cascade key"
                )
        if not ready_keys:
            return

        for tier_idx in request_level_tiers:
            job_metadata = self._create_store_job_unlocked(
                ready_keys, req_context, tier_idx
            )
            tier = self.secondary_tiers[tier_idx]
            tier.submit_store(job_metadata)

    def _flush_pending_cascades(self) -> None:
        """Retry request-level cascades parked on an in-flight primary write.

        A parked key always resolves, to HIT or to MISS, so the set drains and
        a request cannot be held from finalization forever.
        """
        for req_id, state in list(self._req_state.items()):
            if not state.pending_cascade_keys:
                continue
            assert state.request_level_tiers
            keys, state.pending_cascade_keys = state.pending_cascade_keys, []
            self._cascade_existing_chunks_to_request_level_tiers(
                keys, state.req_context, state.request_level_tiers
            )
            self._maybe_finalize_request(req_id)

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        """Mark chunks as done storing from GPU to primary tier.

        This is where secondary tier cascading happens — after chunks are
        confirmed to be in the primary tier, they are cascaded to ALL
        secondary tiers.

        For each secondary tier:
        1. Call primary.prepare_read() to get LoadStoreSpec AND increment
           ref_cnt (protecting chunks during async transfer)
        2. Call tier.submit_store() to start async transfer: primary→secondary
        3. Track the job in _store_jobs dictionary

        Args:
            keys: Chunks that finished storing.
            success: Whether the GPU→primary transfer succeeded.
            req_context: Per-request context forwarded to primary.prepare_read().

        """
        self._enter_step()
        # Step 1: Complete store in primary tier (makes chunks loadable)
        self.primary_tier.complete_store(keys, req_context, success)

        if success:
            # Step 2: Cascade to ALL secondary tiers
            # For each secondary tier, call primary.prepare_read() to get the
            # LoadStoreSpec AND to increment ref_cnt (protecting chunks from
            # eviction during the async transfer). One prepare_read() call per
            # secondary tier.
            for tier_idx, tier in enumerate(self.secondary_tiers):
                job_metadata = self._create_store_job_unlocked(
                    keys, req_context, tier_idx
                )
                tier.submit_store(job_metadata)

        # Note: The async transfers are now in flight. Their completion is
        # tracked via get_finished_jobs() / _maybe_process_finished_jobs().
        req_id = req_context.req_id
        state = self._req_state[req_id]
        assert state.pending_primary_stores > 0
        state.pending_primary_stores -= 1
        self._maybe_finalize_request(req_id)

    def create_store_job(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        tier_idx: int = 0,
    ) -> TransferJob:
        """Pin chunks in the primary tier and create a tracked store job.

        See _create_store_job_unlocked.
        """
        self._enter_step()
        return self._create_store_job_unlocked(keys, req_context, tier_idx)

    def _create_store_job_unlocked(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        tier_idx: int = 0,
    ) -> TransferJob:
        """Pin chunks in the primary tier and create a tracked store job.

        Calls prepare_read() to increment ref_cnt (protecting chunks
        from eviction during the async transfer), allocates a job ID,
        and registers the job in _jobs. The caller must hold the manager lock.

        The caller is responsible for the actual data transfer and
        reporting completion via get_finished_jobs().
        """
        primary_chunks_spec = self.primary_tier.prepare_read(keys, req_context)
        assert isinstance(primary_chunks_spec, CPULoadStoreSpec)
        job_id = self._next_job_id()
        job_metadata = TransferJob(
            job_id=job_id,
            keys=keys,
            chunk_ids=primary_chunks_spec.chunk_ids,
            is_promotion=False,
            req_context=req_context,
        )
        self._register_job(job_metadata, tier_idx)
        return job_metadata

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """Query each secondary tier for its offload policy preference.

        See _on_new_request_unlocked.
        """
        self._enter_step()
        return self._on_new_request_unlocked(req_context)

    def _on_new_request_unlocked(
        self,
        req_context: ReqContext,
        exclude_tier_idx: int | None = None,
    ) -> RequestOffloadingContext:
        """Query each secondary tier for its offload policy preference.

        Returns REQUEST_LEVEL if ANY secondary tier wants request-level.
        Only stores REQUEST_LEVEL tier decisions for use in prepare_store.
        The caller must hold the manager lock.
        """
        state = RequestState(req_context=req_context)
        self._metrics.on_new_request(req_context)
        for tier_idx, tier in enumerate(self.secondary_tiers):
            if tier_idx == exclude_tier_idx:
                continue
            tier_ctx = tier.on_new_request(req_context)
            if tier_ctx.policy == OffloadPolicy.REQUEST_LEVEL:
                if state.request_level_tiers is None:
                    state.request_level_tiers = set()
                state.request_level_tiers.add(tier_idx)
        self._req_state[req_context.req_id] = state

        policy = (
            OffloadPolicy.REQUEST_LEVEL
            if state.request_level_tiers
            else OffloadPolicy.CHUNK_LEVEL
        )
        return RequestOffloadingContext(policy=policy)

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        self._enter_step()
        self._on_request_finished_unlocked(req_context)

    def _on_request_finished_unlocked(
        self,
        req_context: ReqContext,
        exclude_tier_idx: int | None = None,
    ) -> None:
        """Finalize a finished request. The caller must hold the manager lock."""
        self.primary_tier.on_request_finished(req_context)
        state = self._req_state[req_context.req_id]
        state.is_finished = True
        self._maybe_finalize_request(req_context.req_id, exclude_tier_idx)

    def _maybe_finalize_request(
        self,
        req_id: str,
        exclude_tier_idx: int | None = None,
    ) -> None:
        """Finalize secondary tiers once no more cascades can be submitted.

        Their finalization is delayed until pending GPU->primary stores
        finish, since those callbacks may still submit secondary stores.
        """
        state = self._req_state[req_id]
        if not state.is_finished:
            return
        if state.pending_primary_stores != 0:
            return
        if state.pending_cascade_keys:
            return

        for tier_idx, tier in enumerate(self.secondary_tiers):
            if tier_idx == exclude_tier_idx:
                continue
            tier.on_request_finished(state.req_context)
        self._metrics.on_request_finished(state.req_context)
        del self._req_state[req_id]

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        """End-of-schedule hook: process finished jobs, flush deferred
        promotions, and reset the per-step gate.

        Called once per scheduler step from
        OffloadingConnectorScheduler.build_connector_meta().

        Also where the step's hold on the manager lock is released, handing the
        control-plane thread the model-execution window.
        """
        self._enter_step()
        try:
            self._start_control_thread()

            if self._control_thread is None:
                # Catch-all poll: guarantees jobs are processed even on steps
                # where lookup()/prepare_store() were never called (e.g. no
                # requests scheduled but a tier still has_pending_work()).
                # Once the control thread runs, both of these are its job.
                self._maybe_process_finished_jobs()

                for tier in self.secondary_tiers:
                    tier.serve_external_requests(self._tier_parents[tier])

            # Reset the per-step gate AFTER serve_external_requests so that
            # lookup() calls within it skip redundant _process_finished_jobs().
            self._processed_jobs_this_step = False

            self._flush_pending_promotions()
            self._flush_pending_cascades()
            for tier in self.secondary_tiers:
                tier.on_schedule_end(context)

            for req_id in context.new_req_ids:
                state = self._req_state.get(req_id)
                if state is None:
                    continue
                self._metrics.on_request_allocated(state.req_context)
        finally:
            self._exit_step()

    # ------------------------------------------------------------------
    # Control plane
    # ------------------------------------------------------------------

    def _start_control_thread(self) -> None:
        """Start the control-plane thread, on the first scheduler step.

        Deferred out of __init__ so no sweep runs before the engine is
        stepping: accepting a peer builds a session, and a tier may need state
        that is only set up after this manager is constructed (the p2p tier
        resolves the block-hash seed, which init_none_hash() sets). It also
        keeps the thread out of the spec's partial-construction cleanup path.
        """
        if self._control_thread is not None or not self._control_thread_enabled:
            return
        self._control_thread = threading.Thread(
            target=self._control_plane_loop,
            name="vllm_tiering_control_plane",
            daemon=True,
        )
        self._control_thread.start()
        logger.info(
            "Tiering control-plane thread started for %d secondary tier(s), "
            "polling every %.3fs",
            len(self.secondary_tiers),
            self._control_poll_interval_s,
        )

    def _stop_control_thread(self) -> None:
        """Signal the control-plane thread and wait out its current sweep.

        Releases the step lock first: the thread may be blocked on it, and it
        only observes the stop event once it is no longer waiting. Because the
        loop re-checks that event after acquiring, a straggler outliving the
        join can no longer start a sweep against a torn-down tier.
        """
        thread = self._control_thread
        self._control_thread_stop.set()
        self._exit_step()
        if thread is None:
            return
        thread.join(timeout=_CONTROL_THREAD_JOIN_TIMEOUT_S)
        if thread.is_alive():
            logger.error(
                "Tiering control-plane thread did not stop within %.1fs; "
                "continuing with tier shutdown under the manager lock.",
                _CONTROL_THREAD_JOIN_TIMEOUT_S,
            )
        self._control_thread = None

    def _control_plane_loop(self) -> None:
        """Sweep the secondary tiers' control plane until shutdown.

        Runs between scheduler steps, holding the manager lock for each sweep.
        """
        stop = self._control_thread_stop
        starved_since: float | None = None
        while not stop.is_set():
            if not self._lock.acquire(timeout=_CONTROL_LOCK_TIMEOUT_S):
                now = time.monotonic()
                if starved_since is None:
                    starved_since = now
                elif now - starved_since >= _CONTROL_STARVATION_WARN_S:
                    logger.warning(
                        "Tiering control plane has not run for %.0fs: the "
                        "scheduler still holds the manager lock, so a step "
                        "never reached on_schedule_end.",
                        now - starved_since,
                    )
                    starved_since = now
                continue
            starved_since = None
            try:
                if stop.is_set():
                    # Shutdown began while we waited for the lock.
                    return
                self._sweep_control_plane()
            except BaseException as exc:
                # This work used to run on the scheduler thread, where a
                # failure took the engine down with a real traceback. Hand it
                # back there rather than leaving the control plane dead, which
                # from the outside is indistinguishable from a hang.
                logger.exception("Tiering control-plane sweep failed")
                self._control_thread_exc = exc
                return
            finally:
                self._lock.release()
            stop.wait(self._control_poll_interval_s)

    def _sweep_control_plane(self) -> None:
        """Poll tiers for finished jobs, then let them serve inbound peers.

        Same order as the on_schedule_end path this replaces: polling binds a
        fetch that just arrived and enqueues inbound lookups, so serving
        resolves them within the same sweep.
        """
        self._process_finished_jobs()
        for tier in self.secondary_tiers:
            tier.serve_external_requests(self._tier_parents[tier])

    @override
    def has_pending_work(self) -> bool:
        # In-flight primary<->secondary transfers (pending promotions are
        # translated to transfer jobs in on_schedule_end), plus any work the
        # secondary tiers themselves still have outstanding.
        with self._observation_lock():
            return (
                bool(self._jobs)
                or any(state.pending_cascade_keys for state in self._req_state.values())
                or any(tier.has_pending_work() for tier in self.secondary_tiers)
            )

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        """Collect events owned by the primary and secondary tiers.

        Materialized rather than generated: a generator body would not run
        until the caller iterates, so the step lock would be taken (and the
        tiers drained) at some later, unrelated point.

        Returns:
            New OffloadingEvents collected by each tier since the last call.

        """
        with self._observation_lock():
            events = list(self.primary_tier.take_events())
            for tier in self.secondary_tiers:
                events.extend(tier.take_events())
        return events

    @override
    def reset_cache(self) -> None:
        """Reset transfer bookkeeping and primary-tier cache.

        Called during sleep, weight update, or resume. Each secondary tier
        drains its in-flight transfers via drain_jobs() so no tier I/O is
        touching primary memory before the primary tier is reset. A stuck
        tier will block here visibly — preferable to silent corruption
        from reusing primary slots while a transfer is mid-copy.

        Secondary tiers are intentionally not reset: persistent stores
        (FS, network) keep their data across resets. Active request state is
        retained so those requests can continue after the reset; finished
        requests are finalized and removed.

        Runs outside a scheduler step, so it takes and drops the manager lock
        itself. The control-plane thread is locked out for the whole reset;
        drain_jobs() polls the tiers itself and does not need it.
        """
        self._enter_step()
        try:
            self._reset_cache_unlocked()
        finally:
            self._exit_step()

    def _reset_cache_unlocked(self) -> None:
        """Body of reset_cache(). The caller must hold the manager lock."""
        for tier in self.secondary_tiers:
            tier.drain_jobs()
        # All tier I/O has stopped; consume their completion notifications
        # so manager bookkeeping is consistent before the primary reset.
        self._process_finished_jobs()
        assert not self._jobs

        # Deferred promotion submissions reserve primary slots that the
        # reset below invalidates; their submit_load() has not yet been
        # called so no tier I/O is touching that memory.
        self._pending_load_submissions.clear()
        self._metrics.assert_idle()

        finished_req_ids = []
        for req_id, state in self._req_state.items():
            state.pending_primary_stores = 0
            state.pending_cascade_keys.clear()
            if not state.is_finished:
                continue
            for tier in self.secondary_tiers:
                tier.on_request_finished(state.req_context)
            self._metrics.on_request_finished(state.req_context)
            finished_req_ids.append(req_id)

        self.primary_tier.reset_cache()

        for req_id in finished_req_ids:
            del self._req_state[req_id]
        self._processed_jobs_this_step = False

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        with self._observation_lock():
            stats = self.primary_tier.get_stats()

            if stats is not None and stats.is_empty():
                stats = None

            metrics_stats = self._metrics.take_stats()
            if metrics_stats is not None:
                if stats is None:
                    stats = metrics_stats
                else:
                    stats.aggregate(metrics_stats)

            for tier in self.secondary_tiers:
                tier_stats = tier.get_stats()
                if tier_stats is None or tier_stats.is_empty():
                    continue
                if stats is None:
                    stats = tier_stats
                else:
                    stats.aggregate(tier_stats)

        return stats

    @override
    def shutdown(self) -> None:
        """Shut down secondary tiers before releasing primary resources.

        Every secondary tier is given a shutdown attempt. If any shutdown
        fails, preserve the primary mmap because a failed tier may still use it.

        The control-plane thread is stopped first, so nothing sweeps a tier
        while it is being torn down. A sweep that failed earlier is not
        re-raised here: it must not mask the teardown.
        """
        self._stop_control_thread()
        self._enter_step(raise_control_error=False)
        try:
            self._shutdown_unlocked()
        finally:
            self._exit_step()

    def _shutdown_unlocked(self) -> None:
        """Body of shutdown(). The caller must hold the manager lock."""
        shutdown_error: Exception | None = None
        for tier_idx, tier in enumerate(self.secondary_tiers):
            try:
                tier.shutdown()
            except Exception as exc:
                shutdown_error = exc
                logger.exception(
                    "Failed to shut down secondary tier #%d "
                    "(tier_type=%s, impl_class=%s)",
                    tier_idx,
                    tier.tier_type,
                    type(tier).__name__,
                )

        if shutdown_error is not None:
            raise shutdown_error

        self.primary_tier.shutdown()
