# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingManager: Multi-tier KV cache offloading orchestrator.

This manager coordinates between a CPU primary tier (with direct GPU access)
and zero or more secondary tiers (Storage, Network, etc.) to provide
hierarchical KV cache offloading.

Key Design Principles:
1. Pressure-driven placement — Device blocks are copied to CPU only when the
   scheduler's KV block pool is under pressure; CPU blocks are demoted only
   when the CPU tier approaches its configured watermark
2. Primary tier is the gateway — Secondary tiers cannot access GPU memory
   directly; all data flows through the CPU primary tier
3. Staged promotion — Blocks in secondary tiers must be promoted to the
   primary tier before GPU can access them
4. Transparent retry mechanism — Return None from lookup() to signal
   "data is being promoted, try later"
5. ref_cnt as eviction protection — primary.prepare_read() increments ref_cnt,
   protecting blocks from eviction until complete_read() is called
"""

import math
import time
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

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
    JobMetadata,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.lifecycle import (
    LifecycleConfig,
    SessionLifecycleManager,
    get_session_id,
)
from vllm.v1.kv_offload.tiering.residency import TieringMetrics

logger = init_logger(__name__)


@dataclass(slots=True)
class TieringPolicyConfig:
    """Pressure policy for device, CPU, and secondary-tier placement."""

    hbm_pressure_aware: bool = False
    hbm_high_watermark: float = 0.70
    hbm_low_watermark: float = 0.50
    secondary_pressure_aware: bool = False
    bypass_unknown_secondary_when_relaxed: bool = False
    min_session_requests_for_offload: int = 2
    min_reuse_probability: float = 0.5
    store_during_decode_only: bool = True
    max_device_store_blocks_per_step: int = 16
    max_device_store_blocks_per_request: int = 64
    max_device_store_blocks_per_pressure_episode: int = 128
    max_device_store_blocks_per_session_episode: int = 64
    max_inflight_device_store_jobs: int = 2
    reclaim_device_cache_after_store: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("tiering_hbm_high_watermark", self.hbm_high_watermark),
            ("tiering_hbm_low_watermark", self.hbm_low_watermark),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.hbm_low_watermark <= self.hbm_high_watermark <= 1:
            raise ValueError(
                "tiering_hbm_low_watermark and tiering_hbm_high_watermark "
                "must satisfy 0 < low <= high <= 1"
            )
        if self.min_session_requests_for_offload < 1:
            raise ValueError(
                "tiering_min_session_requests_for_offload must be at least 1"
            )
        if not math.isfinite(self.min_reuse_probability) or not (
            0 <= self.min_reuse_probability <= 1
        ):
            raise ValueError("tiering_min_reuse_probability must be between 0 and 1")
        for name, value in (
            (
                "tiering_max_device_store_blocks_per_step",
                self.max_device_store_blocks_per_step,
            ),
            (
                "tiering_max_device_store_blocks_per_request",
                self.max_device_store_blocks_per_request,
            ),
            (
                "tiering_max_device_store_blocks_per_pressure_episode",
                self.max_device_store_blocks_per_pressure_episode,
            ),
            (
                "tiering_max_device_store_blocks_per_session_episode",
                self.max_device_store_blocks_per_session_episode,
            ),
            (
                "tiering_max_inflight_device_store_jobs",
                self.max_inflight_device_store_jobs,
            ),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def reuse_aware(self) -> bool:
        return (
            self.min_session_requests_for_offload > 1 or self.min_reuse_probability > 0
        )

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any]) -> "TieringPolicyConfig":
        return cls(
            hbm_pressure_aware=bool(
                extra_config.get("tiering_hbm_pressure_aware", True)
            ),
            hbm_high_watermark=float(
                extra_config.get("tiering_hbm_high_watermark", 0.70)
            ),
            hbm_low_watermark=float(
                extra_config.get("tiering_hbm_low_watermark", 0.50)
            ),
            secondary_pressure_aware=bool(
                extra_config.get("tiering_secondary_pressure_aware", True)
            ),
            bypass_unknown_secondary_when_relaxed=bool(
                extra_config.get("tiering_bypass_unknown_secondary_when_relaxed", True)
            ),
            min_session_requests_for_offload=int(
                extra_config.get("tiering_min_session_requests_for_offload", 2)
            ),
            min_reuse_probability=float(
                extra_config.get("tiering_min_reuse_probability", 0.5)
            ),
            store_during_decode_only=bool(
                extra_config.get("tiering_store_during_decode_only", True)
            ),
            max_device_store_blocks_per_step=int(
                extra_config.get("tiering_max_device_store_blocks_per_step", 16)
            ),
            max_device_store_blocks_per_request=int(
                extra_config.get("tiering_max_device_store_blocks_per_request", 64)
            ),
            max_device_store_blocks_per_pressure_episode=int(
                extra_config.get(
                    "tiering_max_device_store_blocks_per_pressure_episode", 128
                )
            ),
            max_device_store_blocks_per_session_episode=int(
                extra_config.get(
                    "tiering_max_device_store_blocks_per_session_episode", 64
                )
            ),
            max_inflight_device_store_jobs=int(
                extra_config.get("tiering_max_inflight_device_store_jobs", 2)
            ),
            reclaim_device_cache_after_store=bool(
                extra_config.get("tiering_reclaim_device_cache_after_store", True)
            ),
        )


@dataclass
class PendingPromotion:
    """Accumulator for blocks awaiting submit_load() for one (tier, request)."""

    req_context: ReqContext
    keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class RequestState:
    req_context: ReqContext
    pending_primary_stores: int = 0
    is_finished: bool = False
    request_level_tiers: set[SecondaryTierManager] | None = None
    submitted_store_blocks: int = 0


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
        num_blocks: int,
        mmap_region: SharedOffloadRegion | None,
        block_size_bytes: int | None = None,
        cache_policy: str = "lru",
        enable_events: bool = False,
    ):
        super().__init__(
            num_blocks=num_blocks,
            cache_policy=cache_policy,  # type: ignore[arg-type]
            enable_events=enable_events,
        )
        self._mmap_region = mmap_region
        # read/write is for CPU<->secondary transfers,
        # load/store is for CPU<->GPU transfers.
        # These aliases avoid calling prepare_load inside a store path.
        self.prepare_read = self.prepare_load
        self.complete_read = self.complete_load
        self.prepare_write = self.prepare_store
        self.complete_write = self.complete_store

        self._kv_memoryview = (
            mmap_region.create_kv_memoryview() if mmap_region is not None else None
        )
        if block_size_bytes is None and self._kv_memoryview is not None:
            assert self._kv_memoryview.strides is not None
            block_size_bytes = self._kv_memoryview.strides[0]
        self._block_size_bytes = block_size_bytes

    def get_kv_memoryview(self) -> memoryview:
        """Return the memoryview over the primary tier's KV cache buffer.

        The view has shape (num_blocks, row_stride_bytes) and is backed by the
        SharedOffloadRegion mmap.  Secondary tiers address block *b* as
        ``view[b]``.
        """
        if self._kv_memoryview is None:
            raise RuntimeError(
                "Pinned CPU primary storage does not expose a shared memoryview"
            )
        return self._kv_memoryview

    @property
    def block_size_bytes(self) -> int:
        if self._block_size_bytes is None:
            raise RuntimeError("CPU primary block size is unavailable")
        return self._block_size_bytes

    @override
    def shutdown(self) -> None:
        super().shutdown()
        if self._kv_memoryview is not None:
            self._kv_memoryview.release()
            self._kv_memoryview = None
        if self._mmap_region is not None:
            self._mmap_region.cleanup()
            self._mmap_region = None


class TieringOffloadingManager(OffloadingManager):
    """
    Orchestrates multi-tier KV cache offloading.

    This manager coordinates between a CPU primary tier (with direct GPU access)
    and zero or more secondary tiers (Storage, Network, etc.) to provide
    hierarchical KV cache offloading.

    Key internal state:
      - Minimal state tracking; relies on secondary tiers to report completion
        via get_finished_jobs()
      - Secondary tiers return JobResult objects containing all necessary
        information
      - job_id_counter: monotonically increasing counter for job IDs
    """

    def __init__(
        self,
        primary_tier: CPUPrimaryTierOffloadingManager,
        secondary_tiers: list[SecondaryTierManager] | None = None,
        enable_events: bool = False,
        lifecycle_config: LifecycleConfig | None = None,
        policy_config: TieringPolicyConfig | None = None,
    ):
        """
        Initialize the TieringOffloadingManager.

        Args:
            primary_tier: The primary tier manager (CPU-based).
            secondary_tiers: List of secondary tier managers (e.g., Storage,
                            Network). Can be None or empty list.
            enable_events: Whether to track offloading events
        """
        self.primary_tier: CPUPrimaryTierOffloadingManager = primary_tier
        self.secondary_tiers = secondary_tiers or []
        self._tier_ids = {
            tier: f"{tier.tier_type}:{index}"
            for index, tier in enumerate(self.secondary_tiers)
        }

        self._job_id_counter: int = 0
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

        # Job tracking: maps job_id to metadata for all in-flight transfers.
        # JobMetadata.is_promotion distinguishes direction:
        #   True:  secondary → primary (promotion)
        #   False: primary → secondary (cascade)
        self._transfer_jobs: dict[JobId, JobMetadata] = {}

        # Pending promotion requests accumulated during lookup() calls; flushed
        # as one batched submit_load() per (tier, request) in on_schedule_end().
        # Outer key: tier. Inner key: req_context.req_id — the same ReqContext
        # object is reused for all block lookups of a given request per engine step.
        self._pending_load_submissions: dict[
            SecondaryTierManager, dict[str, PendingPromotion]
        ] = {}

        # Gate for once-per-step execution of _maybe_process_finished_jobs().
        # Reset at the end of each step in on_schedule_end().
        self._processed_jobs_this_step: bool = False

        # Per-request state for prepared GPU->primary stores and finalization.
        # Secondary tiers are finalized only after pending primary stores reach
        # complete_store(), since complete_store() can still submit cascades.
        self._req_state: dict[str, RequestState] = {}
        self._lifecycle = SessionLifecycleManager(lifecycle_config or LifecycleConfig())
        self._policy_config = policy_config or TieringPolicyConfig()
        self._device_usage = 0.0
        self._device_pressure_active = not self._policy_config.hbm_pressure_aware
        self._device_preempted = False
        self._cpu_pressure_requested = False
        self._step_submitted_store_blocks = 0
        self._episode_submitted_store_blocks = 0
        self._episode_session_store_blocks: dict[str, int] = {}
        self._metric_counters: dict[tuple[str, tuple[str, ...]], int | float] = {}
        self._metric_histograms: dict[tuple[str, tuple[str, ...]], list[float]] = {}

    @property
    def _track_residency(self) -> bool:
        return self._lifecycle.enabled

    @override
    def update_device_pressure(
        self,
        usage: float,
        *,
        preempted: bool = False,
    ) -> None:
        usage = min(1.0, max(0.0, usage))
        self._device_usage = usage
        self._device_preempted = preempted
        self._step_submitted_store_blocks = 0
        if not self._policy_config.hbm_pressure_aware:
            self._device_pressure_active = True
            return

        was_active = self._device_pressure_active
        if preempted or usage >= self._policy_config.hbm_high_watermark:
            self._device_pressure_active = True
        elif usage <= self._policy_config.hbm_low_watermark:
            self._device_pressure_active = False

        if was_active != self._device_pressure_active:
            logger.info(
                "KV tiering device pressure %s at %.1f%% usage",
                "activated" if self._device_pressure_active else "cleared",
                usage * 100,
            )
        if was_active and not self._device_pressure_active:
            self._episode_submitted_store_blocks = 0
            self._episode_session_store_blocks.clear()

    @override
    def should_store(self, req_context: ReqContext, num_blocks: int = 0) -> bool:
        params = req_context.kv_transfer_params or {}
        if params.get("force_kv_offload") is True:
            if num_blocks > 0:
                self._increase_counter(
                    TieringMetrics.STORE_DECISIONS,
                    num_blocks,
                    ("store", "request_override"),
                )
            return True

        if not self._device_pressure_active:
            if num_blocks > 0:
                self._increase_counter(
                    TieringMetrics.STORE_DECISIONS,
                    num_blocks,
                    ("skip", "hbm_relaxed"),
                )
            return False

        if num_blocks <= 0:
            return True

        if self._policy_config.secondary_pressure_aware:
            capacity = self.primary_tier.capacity_blocks
            resident = self.primary_tier.resident_blocks
            projected = resident + max(0, num_blocks)
            cpu_config = self._lifecycle.config
            if capacity > 0 and projected / capacity > cpu_config.cpu_high_watermark:
                self._cpu_pressure_requested = True
                if resident / capacity >= cpu_config.cpu_low_watermark:
                    self._increase_counter(
                        TieringMetrics.STORE_DECISIONS,
                        num_blocks,
                        ("skip", "cpu_pressure"),
                    )
                    return False

        self._increase_counter(
            TieringMetrics.STORE_DECISIONS,
            num_blocks,
            ("store", "hbm_pressure"),
        )
        return True

    @staticmethod
    def _remaining(limit: int, used: int) -> int | None:
        return None if limit == 0 else max(0, limit - used)

    @override
    def get_store_budget(
        self,
        req_context: ReqContext,
        requested_blocks: int,
        *,
        is_decode_phase: bool,
    ) -> int:
        if requested_blocks <= 0:
            return 0

        params = req_context.kv_transfer_params or {}
        force_store = params.get("force_kv_offload") is True
        if not force_store:
            if not self._device_pressure_active:
                return 0
            if (
                self._policy_config.store_during_decode_only
                and not is_decode_phase
                and not self._device_preempted
            ):
                self._increase_counter(
                    TieringMetrics.STORE_DECISIONS,
                    requested_blocks,
                    ("skip", "prefill_overlap"),
                )
                return 0

            request_count, reuse_probability = self._lifecycle.get_session_heat(
                req_context
            )
            if (
                request_count < self._policy_config.min_session_requests_for_offload
                or reuse_probability < self._policy_config.min_reuse_probability
            ):
                self._increase_counter(
                    TieringMetrics.STORE_DECISIONS,
                    requested_blocks,
                    ("skip", "cold_session"),
                )
                return 0

        state = self._req_state.get(req_context.req_id)
        if state is None:
            return 0

        inflight_limit = self._policy_config.max_inflight_device_store_jobs
        inflight_jobs = sum(
            request_state.pending_primary_stores
            for request_state in self._req_state.values()
        )
        if inflight_limit and inflight_jobs >= inflight_limit:
            self._increase_counter(
                TieringMetrics.STORE_DECISIONS,
                requested_blocks,
                ("skip", "inflight_budget"),
            )
            return 0

        limits = [requested_blocks]
        remaining_limits = [
            self._remaining(
                self._policy_config.max_device_store_blocks_per_step,
                self._step_submitted_store_blocks,
            ),
            self._remaining(
                self._policy_config.max_device_store_blocks_per_request,
                state.submitted_store_blocks,
            ),
        ]
        if self._policy_config.hbm_pressure_aware:
            remaining_limits.extend(
                (
                    self._remaining(
                        self._policy_config.max_device_store_blocks_per_pressure_episode,
                        self._episode_submitted_store_blocks,
                    ),
                    self._remaining(
                        self._policy_config.max_device_store_blocks_per_session_episode,
                        self._episode_session_store_blocks.get(
                            get_session_id(req_context), 0
                        ),
                    ),
                )
            )
        for remaining in remaining_limits:
            if remaining is not None:
                limits.append(remaining)

        budget = min(limits)
        if budget <= 0:
            self._increase_counter(
                TieringMetrics.STORE_DECISIONS,
                requested_blocks,
                ("skip", "migration_budget"),
            )
            return 0
        return budget

    @override
    def record_store_submission(
        self,
        req_context: ReqContext,
        submitted_blocks: int,
    ) -> None:
        if submitted_blocks <= 0:
            return
        state = self._req_state[req_context.req_id]
        state.submitted_store_blocks += submitted_blocks
        self._step_submitted_store_blocks += submitted_blocks
        self._episode_submitted_store_blocks += submitted_blocks
        session_id = get_session_id(req_context)
        self._episode_session_store_blocks[session_id] = (
            self._episode_session_store_blocks.get(session_id, 0) + submitted_blocks
        )
        self._increase_counter(
            TieringMetrics.STORE_DECISIONS,
            submitted_blocks,
            ("store", "budgeted"),
        )

    @override
    def should_reclaim_device_cache(self, req_context: ReqContext) -> bool:
        del req_context
        return (
            self._policy_config.reclaim_device_cache_after_store
            and self._device_pressure_active
        )

    def _tier_id(self, tier: SecondaryTierManager) -> str:
        return self._tier_ids[tier]

    def _increase_counter(
        self,
        name: str,
        value: int | float = 1,
        labels: tuple[str, ...] = (),
    ) -> None:
        key = (name, labels)
        self._metric_counters[key] = self._metric_counters.get(key, 0) + value

    def _observe_histogram(
        self,
        name: str,
        value: float,
        labels: tuple[str, ...] = (),
    ) -> None:
        self._metric_histograms.setdefault((name, labels), []).append(value)

    def _record_migration(
        self,
        keys: Collection[OffloadKey],
        source: str,
        target: str,
        latency: float | None = None,
    ) -> None:
        labels = (source, target)
        self._increase_counter(TieringMetrics.MIGRATION_BLOCKS, len(keys), labels)
        self._increase_counter(
            TieringMetrics.MIGRATION_BYTES,
            len(keys) * self.primary_tier.block_size_bytes,
            labels,
        )
        if latency is not None:
            self._observe_histogram(
                TieringMetrics.MIGRATION_LATENCY,
                latency,
                labels,
            )

    def _next_job_id(self) -> JobId:
        """Generate a unique job ID for async transfer tracking."""
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id

    def _maybe_process_finished_jobs(self):
        """
        Poll secondary tiers for completed jobs (at most once per step).

        Guarded by _processed_jobs_this_step: the first call in an engine step
        does the actual polling; subsequent calls are no-ops. The flag is reset
        in on_schedule_end() at the end of each step.
        """
        if self._processed_jobs_this_step:
            return
        self._processed_jobs_this_step = True
        self._process_finished_jobs()

    def _process_finished_jobs(self):
        """
        Unconditionally poll all secondary tiers for completed jobs.

        This method:
        1. Calls get_finished_jobs() on each secondary tier
        2. For completed stores (primary→secondary): calls primary.complete_read()
           to decrement ref_cnt
        3. For completed loads (secondary→primary): calls primary.complete_write()
           to make blocks available
        """
        for i, tier in enumerate(self.secondary_tiers):
            for completed_job in tier.get_finished_jobs():
                job_id = completed_job.job_id
                job_metadata = self._transfer_jobs.pop(job_id, None)
                assert job_metadata is not None, (
                    f"Finished job_id {job_id} from tier #{i}"
                    f" ({tier.tier_type}) not in _transfer_jobs"
                )

                if job_metadata.is_promotion:
                    # secondary→primary transfer (promotion) completed.
                    # Make blocks available in primary tier.
                    self.primary_tier.complete_write(
                        job_metadata.keys,
                        job_metadata.req_context,
                        completed_job.success,
                    )
                    if self._track_residency:
                        tier_id = self._tier_id(tier)
                        self._lifecycle.residency.finish_transfer(
                            job_metadata.keys, tier_id, "cpu"
                        )
                        self._lifecycle.residency.mark_cpu_resident(
                            job_metadata.keys, completed_job.success
                        )
                    if completed_job.success:
                        self._record_migration(
                            job_metadata.keys,
                            self._tier_id(tier),
                            "cpu",
                            time.monotonic() - job_metadata.started_at,
                        )
                else:
                    # primary→secondary transfer completed.
                    # Decrement ref_cnt on primary blocks.
                    self.primary_tier.complete_read(
                        job_metadata.keys, job_metadata.req_context
                    )
                    if self._track_residency:
                        tier_id = self._tier_id(tier)
                        self._lifecycle.residency.finish_transfer(
                            job_metadata.keys, "cpu", tier_id
                        )
                        if completed_job.success:
                            self._lifecycle.residency.mark_secondary_resident(
                                job_metadata.keys, tier_id
                            )
                    if completed_job.success:
                        self._record_migration(
                            job_metadata.keys,
                            "cpu",
                            self._tier_id(tier),
                            time.monotonic() - job_metadata.started_at,
                        )
                        if job_metadata.evict_primary_on_success:
                            self._reclaim_primary_keys(
                                job_metadata.keys,
                                reason=job_metadata.reclaim_reason
                                or "secondary_demotion",
                            )

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        """
        Check whether a single block is offloaded and ready.

        Algorithm:
            1. Process any completed async jobs first.
            2. Query primary tier — short-circuit on hit or in-flight.
            3. On primary miss, query secondary tiers — stop on first
               hit and initiate promotion.

        Args:
            key: Block hash to look up.
            req_context: Per-request context.

        Returns:
            HIT       — block is ready in the primary tier.
            HIT_PENDING — block found but not yet readable (write
                        in-flight on the primary tier).
            RETRY     — promotion started or a secondary tier is busy.
            MISS      — block not found in any tier, or primary is full
                        and cannot accept a promotion.
        """
        self._lifecycle.record_request_keys(req_context, (key,))
        self._maybe_process_finished_jobs()

        primary_hit = self.primary_tier.lookup(key, req_context)
        if primary_hit is LookupResult.HIT:
            self._lifecycle.record_reuse_hit(req_context)
            if self._track_residency:
                self._lifecycle.residency.mark_cpu_resident((key,))
            self._increase_counter(TieringMetrics.LOOKUPS, labels=("cpu", "hit"))
            return LookupResult.HIT
        if primary_hit is LookupResult.HIT_PENDING:
            self._increase_counter(
                TieringMetrics.LOOKUPS, labels=("cpu", "hit_pending")
            )
            return LookupResult.HIT_PENDING
        self._increase_counter(TieringMetrics.LOOKUPS, labels=("cpu", "miss"))

        if (
            self._policy_config.bypass_unknown_secondary_when_relaxed
            and not self._device_pressure_active
            and not self._lifecycle.residency.is_known(key)
        ):
            self._increase_counter(
                TieringMetrics.LOOKUPS,
                labels=("secondary", "pressure_bypass"),
            )
            return LookupResult.MISS

        any_retry = False
        for tier in self.secondary_tiers:
            result = tier.lookup(key, req_context)
            if result is LookupResult.HIT:
                self._lifecycle.record_reuse_hit(req_context)
                tier_id = self._tier_id(tier)
                if self._track_residency:
                    self._lifecycle.residency.mark_secondary_resident((key,), tier_id)
                self._increase_counter(TieringMetrics.LOOKUPS, labels=(tier_id, "hit"))
                if not self._initiate_promotion(tier, key, req_context):
                    self._increase_counter(
                        TieringMetrics.LOOKUPS,
                        labels=(tier_id, "promotion_rejected"),
                    )
                    return LookupResult.MISS
                return LookupResult.RETRY
            if result is LookupResult.RETRY:
                self._increase_counter(
                    TieringMetrics.LOOKUPS,
                    labels=(self._tier_id(tier), "retry"),
                )
                any_retry = True
            else:
                self._increase_counter(
                    TieringMetrics.LOOKUPS,
                    labels=(self._tier_id(tier), "miss"),
                )

        if any_retry:
            return LookupResult.RETRY
        return LookupResult.MISS

    def _initiate_promotion(
        self,
        tier: SecondaryTierManager,
        key: OffloadKey,
        req_context: ReqContext,
    ) -> bool:
        """
        Queue a block for promotion from a secondary tier to the primary tier.

        Allocates space in the primary tier immediately (sets ref_cnt=-1 so
        subsequent lookups within the same step see the slot as in-flight),
        then defers the actual submit_load() call to _flush_pending_promotions()
        so all blocks queued during one engine step are submitted as a single
        batched job.

        Args:
            tier: The secondary tier to promote from
            key: Block to promote
            req_context: Per-request context forwarded to primary.prepare_write().

        Returns:
            True if promotion was initiated, False if primary tier is full.
        """
        # Allocate space in primary tier for promoted block.
        # Must happen immediately so primary.lookup() returns None (in-flight)
        # for this key on any subsequent lookup() call within the same step,
        # preventing duplicate promotion attempts.
        primary_write_result = self.primary_tier.prepare_write([key], req_context)

        if primary_write_result is None:
            # Primary tier is full; caller should treat the block as unavailable
            # rather than retrying indefinitely.
            return False

        if self._track_residency:
            self._lifecycle.residency.mark_cpu_resident(
                primary_write_result.evicted_keys, False
            )
            self._lifecycle.residency.start_transfer(
                primary_write_result.keys_to_store,
                self._tier_id(tier),
                "cpu",
            )

        store_spec = primary_write_result.store_spec
        assert isinstance(store_spec, CPULoadStoreSpec)
        # Defer submit_load to on_schedule_end(). Group by (tier, request) so
        # each request's blocks are submitted as one batched job per tier.
        tier_pending = self._pending_load_submissions.setdefault(tier, {})
        ctx_id = req_context.req_id
        if ctx_id not in tier_pending:
            tier_pending[ctx_id] = PendingPromotion(
                keys=[], block_ids=[], req_context=req_context
            )
        entry = tier_pending[ctx_id]
        entry.keys.extend(primary_write_result.keys_to_store)
        entry.block_ids.extend(store_spec.block_ids)
        return True

    def _flush_pending_promotions(self) -> None:
        """Submit one batched submit_load() per (tier, request).

        Called from on_schedule_end() at the end of each scheduler step,
        flushing all promotion requests deferred during lookup().
        """
        if not self._pending_load_submissions:
            return

        for tier, pending_by_ctx in self._pending_load_submissions.items():
            for entry in pending_by_ctx.values():
                job_id = self._next_job_id()
                job_metadata = JobMetadata(
                    job_id=job_id,
                    keys=entry.keys,
                    block_ids=np.array(entry.block_ids, dtype=np.int64),
                    is_promotion=True,
                    req_context=entry.req_context,
                )
                self._transfer_jobs[job_id] = job_metadata
                tier.submit_load(job_metadata)

        self._pending_load_submissions.clear()

    @override
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        """
        Prepare blocks to be loaded from primary tier to GPU.

        CRITICAL: This method calls _maybe_process_finished_jobs() FIRST to ensure
        that any completed promotions have been finalized and blocks are ready.

        This increments ref_cnt on the blocks in the primary tier, protecting
        them from eviction during the transfer.

        Args:
            keys: Blocks to prepare for loading.
            req_context: Per-request context.

        Returns:
            LoadStoreSpec for reading from primary tier.
        """
        self._lifecycle.record_request_keys(req_context, keys)
        # Process completed promotions to ensure blocks are ready
        self._maybe_process_finished_jobs()

        load_spec = self.primary_tier.prepare_load(keys, req_context)
        if self._track_residency:
            self._lifecycle.residency.start_transfer(keys, "cpu", "device")
        return load_spec

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as recently used in all tiers.

        Args:
            keys: Blocks to mark as recently used.
            req_context: Per-request context.
        """
        self._lifecycle.record_request_keys(req_context, keys)
        self.primary_tier.touch(keys, req_context)
        for tier in self.secondary_tiers:
            tier.touch(keys, req_context)

    @override
    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as done loading from primary tier to GPU.

        This decrements ref_cnt on the blocks in the primary tier, allowing
        them to be evicted again.

        Args:
            keys: Blocks that finished loading.
            req_context: Per-request context.
        """
        self.primary_tier.complete_load(keys, req_context)
        if self._track_residency:
            self._lifecycle.residency.finish_transfer(keys, "cpu", "device")
        self._record_migration(keys, "cpu", "device")

    @override
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """
        Prepare blocks to be stored from GPU to primary tier.

        CRITICAL: This method calls _maybe_process_finished_jobs() FIRST to ensure
        that any completed async transfers have their ref_cnt decremented
        before the primary tier makes eviction decisions.

        For request-level tiers, blocks already present in the primary tier
        are immediately cascaded via submit_store().

        Args:
            keys: Blocks to prepare for storing.
            req_context: Per-request context.

        Returns:
            PrepareStoreOutput describing where to store blocks and what was
            evicted, or None if store cannot proceed.
        """
        self._lifecycle.record_request_keys(req_context, keys)
        # Step 1: Poll for completed async jobs FIRST
        # This decrements ref_cnt on primary blocks that have been
        # successfully transferred to secondary tiers.
        self._maybe_process_finished_jobs()

        # Step 2: Store to primary tier (new blocks only).
        # Cascading of these newly-stored blocks to ALL secondary tiers
        # happens later in complete_store(), after the GPU→Primary transfer
        # completes.
        primary_result = self.primary_tier.prepare_store(keys, req_context)

        if primary_result is None:
            return None

        if self._track_residency:
            self._lifecycle.residency.mark_cpu_resident(
                primary_result.evicted_keys, False
            )
            self._lifecycle.residency.start_transfer(
                primary_result.keys_to_store, "device", "cpu"
            )

        if primary_result.keys_to_store:
            state = self._req_state[req_context.req_id]
            state.pending_primary_stores += 1

        # Step 3: For request-level tiers, cascade blocks already in primary
        request_level_tiers = self._req_state[req_context.req_id].request_level_tiers
        if request_level_tiers:
            keys_to_store_set = set(primary_result.keys_to_store)
            keys_already_in_primary = tuple(
                k for k in keys if k not in keys_to_store_set
            )
            if keys_already_in_primary:
                self._cascade_existing_blocks_to_request_level_tiers(
                    keys_already_in_primary, req_context, request_level_tiers
                )

        return primary_result

    def _cascade_existing_blocks_to_request_level_tiers(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext,
        request_level_tiers: set[SecondaryTierManager],
    ) -> None:
        """
        For tiers that requested request-level policy, submit_store() for
        blocks that are already present in the primary tier.
        """
        # Filter out keys that are not ready in primary (e.g. in-flight)
        ready_keys = tuple(
            k
            for k in keys
            if self.primary_tier.lookup(k, req_context) is LookupResult.HIT
        )
        if not ready_keys:
            return

        self._submit_secondary_store(
            ready_keys,
            req_context,
            request_level_tiers,
        )

    def _submit_secondary_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        tiers: Iterable[SecondaryTierManager],
        *,
        evict_primary_on_success: bool = False,
        reclaim_reason: str | None = None,
    ) -> None:
        """Submit batched CPU-to-secondary stores for the selected tiers."""
        key_tuple = tuple(keys)
        if not key_tuple:
            return

        for tier in tiers:
            primary_blocks_spec = self.primary_tier.prepare_read(key_tuple, req_context)
            job_id = self._next_job_id()
            assert isinstance(primary_blocks_spec, CPULoadStoreSpec)
            job_metadata = JobMetadata(
                job_id=job_id,
                keys=key_tuple,
                block_ids=primary_blocks_spec.block_ids,
                is_promotion=False,
                req_context=req_context,
                evict_primary_on_success=evict_primary_on_success,
                reclaim_reason=reclaim_reason,
            )
            self._transfer_jobs[job_id] = job_metadata
            if self._track_residency:
                self._lifecycle.residency.start_transfer(
                    key_tuple, "cpu", self._tier_id(tier)
                )
            tier.submit_store(job_metadata)

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        """
        Mark blocks as done storing from GPU to primary tier.

        This is where secondary tier cascading happens — after blocks are
        confirmed to be in the primary tier, they are cascaded to ALL
        secondary tiers.

        For each secondary tier:
        1. Call primary.prepare_read() to get LoadStoreSpec AND increment
           ref_cnt (protecting blocks during async transfer)
        2. Call tier.submit_store() to start async transfer: primary→secondary
        3. Track the job in _store_jobs dictionary

        Args:
            keys: Blocks that finished storing.
            success: Whether the GPU→primary transfer succeeded.
            req_context: Per-request context forwarded to primary.prepare_read().
        """
        # Step 1: Complete store in primary tier (makes blocks loadable)
        self.primary_tier.complete_store(keys, req_context, success)
        if self._track_residency:
            self._lifecycle.residency.finish_transfer(keys, "device", "cpu")
            self._lifecycle.residency.mark_cpu_resident(keys, success)
        if success:
            self._record_migration(keys, "device", "cpu")

        if success:
            state = self._req_state[req_context.req_id]
            cascade_tiers: Iterable[SecondaryTierManager]
            if self._policy_config.secondary_pressure_aware:
                cascade_tiers = state.request_level_tiers or ()
            else:
                cascade_tiers = self.secondary_tiers
            self._submit_secondary_store(keys, req_context, cascade_tiers)

        # Note: The async transfers are now in flight. Their completion is
        # tracked via get_finished_jobs() / _maybe_process_finished_jobs().
        req_id = req_context.req_id
        state = self._req_state[req_id]
        assert state.pending_primary_stores > 0
        state.pending_primary_stores -= 1
        self._maybe_finalize_request(req_id)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """
        Query each secondary tier for its offload policy preference.

        Returns REQUEST_LEVEL if ANY secondary tier wants request-level.
        Only stores REQUEST_LEVEL tier decisions for use in prepare_store.
        """
        self._lifecycle.on_new_request(
            req_context,
            track_heat=self._policy_config.reuse_aware,
        )
        state = RequestState(req_context=req_context)
        for tier in self.secondary_tiers:
            tier_ctx = tier.on_new_request(req_context)
            if tier_ctx.policy == OffloadPolicy.REQUEST_LEVEL:
                if state.request_level_tiers is None:
                    state.request_level_tiers = set()
                state.request_level_tiers.add(tier)
        self._req_state[req_context.req_id] = state

        policy = (
            OffloadPolicy.REQUEST_LEVEL
            if state.request_level_tiers
            else OffloadPolicy.BLOCK_LEVEL
        )
        return RequestOffloadingContext(policy=policy)

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        self.primary_tier.on_request_finished(req_context)
        self._lifecycle.on_request_finished(
            req_context,
            track_heat=self._policy_config.reuse_aware,
        )
        state = self._req_state[req_context.req_id]
        state.is_finished = True
        self._maybe_finalize_request(req_context.req_id)

    def _maybe_finalize_request(self, req_id: str) -> None:
        """Finalize secondary tiers once no more store cascades can be submitted.

        Finalization means forwarding on_request_finished() to secondary tiers.
        It is delayed until pending GPU->primary stores finish, since their
        complete_store() callbacks may still submit primary->secondary stores.
        """
        state = self._req_state[req_id]
        if not state.is_finished:
            return
        if state.pending_primary_stores != 0:
            return

        for tier in self.secondary_tiers:
            tier.on_request_finished(state.req_context)
        del self._req_state[req_id]

    def _reclaim_primary_keys(
        self,
        keys: Collection[OffloadKey],
        *,
        reason: str,
    ) -> list[OffloadKey]:
        evicted = self.primary_tier.evict_keys(keys)
        if not evicted:
            return []
        if self._track_residency:
            self._lifecycle.residency.mark_cpu_resident(evicted, False)
        self._increase_counter(
            TieringMetrics.PRIMARY_RECLAIMED_BLOCKS,
            len(evicted),
            (reason,),
        )
        return evicted

    def _maintain_primary_residency(self) -> None:
        if not self._track_residency:
            return

        if self._policy_config.secondary_pressure_aware and self.secondary_tiers:
            self._maintain_pressure_driven_primary_residency()
            return

        config = self._lifecycle.config
        remaining = config.reclaim_batch_size
        if config.cpu_demote_after_sec > 0:
            idle_candidates = self._lifecycle.get_idle_cpu_candidates(
                limit=remaining,
                require_idle_age=True,
            )
            remaining -= len(
                self._reclaim_primary_keys(idle_candidates, reason="idle_ttl")
            )

        capacity = self.primary_tier.capacity_blocks
        resident = self.primary_tier.resident_blocks
        if (
            remaining <= 0
            or capacity <= 0
            or resident / capacity <= config.cpu_high_watermark
        ):
            return

        target_resident = int(capacity * config.cpu_low_watermark)
        num_to_reclaim = min(remaining, max(0, resident - target_resident))
        watermark_candidates = self._lifecycle.get_idle_cpu_candidates(
            limit=num_to_reclaim,
            require_idle_age=False,
        )
        self._reclaim_primary_keys(watermark_candidates, reason="watermark")

    def _maintain_pressure_driven_primary_residency(self) -> None:
        """Demote CPU blocks only when age or CPU watermarks require it."""
        config = self._lifecycle.config
        capacity = self.primary_tier.capacity_blocks
        resident = self.primary_tier.resident_blocks
        if capacity <= 0 or resident <= 0:
            self._cpu_pressure_requested = False
            return

        usage = resident / capacity
        watermark_pressure = (
            self._cpu_pressure_requested or usage > config.cpu_high_watermark
        )
        idle_before = None
        if not watermark_pressure:
            if config.cpu_demote_after_sec <= 0:
                return
            idle_before = time.monotonic() - config.cpu_demote_after_sec

        if watermark_pressure:
            target_resident = int(capacity * config.cpu_low_watermark)
            pending_reclaims = sum(
                len(metadata.keys)
                for metadata in self._transfer_jobs.values()
                if metadata.evict_primary_on_success
            )
            num_candidates = min(
                config.reclaim_batch_size,
                max(0, resident - target_resident - pending_reclaims),
            )
        else:
            num_candidates = config.reclaim_batch_size

        if num_candidates <= 0:
            self._cpu_pressure_requested = False
            return

        candidates = self._lifecycle.residency.get_cpu_pressure_candidates(
            limit=num_candidates,
            idle_before=idle_before,
            include_active=watermark_pressure,
        )
        if not candidates:
            return

        durable = [
            key
            for key in candidates
            if self._lifecycle.residency.has_secondary_copy(key)
        ]
        reclaimed = self._reclaim_primary_keys(
            durable,
            reason="watermark" if watermark_pressure else "idle_ttl",
        )
        remaining = num_candidates - len(reclaimed)
        if remaining <= 0:
            return

        durable_set = set(durable)
        to_demote = [key for key in candidates if key not in durable_set][:remaining]
        if not to_demote:
            return

        reason = "watermark" if watermark_pressure else "idle_ttl"
        demotion_context = ReqContext(req_id=f"tiering-{reason}-{self._job_id_counter}")
        self._submit_secondary_store(
            to_demote,
            demotion_context,
            (self.secondary_tiers[0],),
            evict_primary_on_success=True,
            reclaim_reason=reason,
        )

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        """End-of-schedule hook: process finished jobs, flush deferred
        promotions, and reset the per-step gate.

        Called once per scheduler step from
        OffloadingConnectorScheduler.build_connector_meta().
        """
        self._maybe_process_finished_jobs()
        self._processed_jobs_this_step = False
        self._flush_pending_promotions()
        for tier in self.secondary_tiers:
            tier.on_schedule_end(context)

        protected_keys = {
            key for metadata in self._transfer_jobs.values() for key in metadata.keys
        }
        protected_keys.update(
            key
            for pending_by_req in self._pending_load_submissions.values()
            for pending in pending_by_req.values()
            for key in pending.keys
        )
        if self._track_residency:
            protected_keys.update(self._lifecycle.residency.get_inflight_keys())
        protected_req_ids = {
            req_id
            for req_id, state in self._req_state.items()
            if state.pending_primary_stores > 0
        }
        expiration = self._lifecycle.expire_idle_sessions(
            protected_keys=protected_keys,
            protected_req_ids=protected_req_ids,
        )
        if expiration.expired_sessions:
            self._increase_counter(
                TieringMetrics.EXPIRED_SESSIONS,
                expiration.expired_sessions,
            )
            reclaimed = self._reclaim_primary_keys(
                expiration.unreferenced_keys,
                reason="expiration",
            )
            if reclaimed:
                self._increase_counter(
                    TieringMetrics.DELETED_BLOCKS,
                    len(reclaimed),
                    ("cpu",),
                )

            if self._lifecycle.config.delete_expired_secondary:
                for tier in self.secondary_tiers:
                    result = tier.delete(expiration.unreferenced_keys)
                    if self._track_residency and result.removed_keys:
                        self._lifecycle.residency.mark_secondary_resident(
                            result.removed_keys,
                            self._tier_id(tier),
                            False,
                        )
                    if result.deleted_count:
                        self._increase_counter(
                            TieringMetrics.DELETED_BLOCKS,
                            result.deleted_count,
                            (self._tier_id(tier),),
                        )

            logger.info(
                "Expired %d idle KV lifecycle session(s), %d block(s) "
                "became unreferenced",
                expiration.expired_sessions,
                len(expiration.unreferenced_keys),
            )

        self._maintain_primary_residency()
        if self._track_residency:
            pruned = self._lifecycle.residency.prune()
            if pruned:
                self._increase_counter(
                    TieringMetrics.PRUNED_TRACKING_ENTRIES,
                    pruned,
                )

    def record_request_keys(
        self, req_context: ReqContext, keys: Collection[OffloadKey]
    ) -> None:
        """Record retained keys for lifecycle observability and cleanup."""
        self._lifecycle.record_request_keys(req_context, keys)

    def get_lifecycle_snapshot(self) -> dict[str, int]:
        return self._lifecycle.snapshot()

    @override
    def has_pending_work(self) -> bool:
        # In-flight primary<->secondary transfers (pending promotions are
        # translated to transfer jobs in on_schedule_end), plus any work the
        # secondary tiers themselves still have outstanding.
        return (
            bool(self._transfer_jobs)
            or self._lifecycle.has_pending_expiration()
            or self._lifecycle.has_pending_cpu_demotion()
            or any(tier.has_pending_work() for tier in self.secondary_tiers)
        )

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        """Yield offloading events collected since the last call.

        Yields:
            New OffloadingEvents collected since the last call.
        """
        if self.events is not None:
            yield from self.events
            self.events.clear()

        yield from self.primary_tier.take_events()

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
        """
        for tier in self.secondary_tiers:
            tier.drain_jobs()
        # All tier I/O has stopped; consume their completion notifications
        # so manager bookkeeping is consistent before the primary reset.
        self._process_finished_jobs()

        # Deferred promotion submissions reserve primary slots that the
        # reset below invalidates; their submit_load() has not yet been
        # called so no tier I/O is touching that memory.
        if self._track_residency:
            for tier, pending_by_req in self._pending_load_submissions.items():
                for pending in pending_by_req.values():
                    self._lifecycle.residency.finish_transfer(
                        pending.keys,
                        self._tier_id(tier),
                        "cpu",
                    )
        self._pending_load_submissions.clear()

        finished_req_ids = []
        for req_id, state in self._req_state.items():
            state.pending_primary_stores = 0
            if not state.is_finished:
                continue
            for tier in self.secondary_tiers:
                tier.on_request_finished(state.req_context)
            finished_req_ids.append(req_id)

        self.primary_tier.reset_cache()
        if self._track_residency:
            self._lifecycle.residency.clear_cpu_residency()

        for req_id in finished_req_ids:
            del self._req_state[req_id]
        self._processed_jobs_this_step = False
        self._cpu_pressure_requested = False
        self._device_preempted = False
        self._step_submitted_store_blocks = 0
        self._episode_submitted_store_blocks = 0
        self._episode_session_store_blocks.clear()

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = self.primary_tier.get_stats()

        if stats is not None and stats.is_empty():
            stats = None

        for tier in self.secondary_tiers:
            tier_stats = tier.get_stats()
            if tier_stats is None or tier_stats.is_empty():
                continue
            if stats is None:
                stats = tier_stats
            else:
                stats.aggregate(tier_stats)

        if stats is None:
            stats = OffloadingConnectorStats()

        lifecycle_snapshot = self._lifecycle.snapshot()
        heat_snapshot = self._lifecycle.heat_snapshot()
        residency_snapshot = self._lifecycle.residency.snapshot()
        stats.set_gauge(
            TieringMetrics.SESSION_STATES,
            lifecycle_snapshot["active_sessions"],
            ("active",),
        )
        stats.set_gauge(
            TieringMetrics.SESSION_STATES,
            lifecycle_snapshot["idle_sessions"],
            ("idle_retained",),
        )
        stats.set_gauge(
            TieringMetrics.RESIDENT_BLOCKS,
            self.primary_tier.resident_blocks,
            ("cpu",),
        )
        secondary_counts = residency_snapshot["secondary_blocks"]
        assert isinstance(secondary_counts, dict)
        for tier in self.secondary_tiers:
            tier_id = self._tier_id(tier)
            stats.set_gauge(
                TieringMetrics.RESIDENT_BLOCKS,
                secondary_counts.get(tier_id, 0),
                (tier_id,),
            )
        tracked_blocks = residency_snapshot["tracked_blocks"]
        shared_blocks = residency_snapshot["shared_blocks"]
        active_blocks = residency_snapshot["active_blocks"]
        assert isinstance(tracked_blocks, int)
        assert isinstance(shared_blocks, int)
        assert isinstance(active_blocks, int)
        stats.set_gauge(TieringMetrics.TRACKED_BLOCKS, tracked_blocks)
        stats.set_gauge(TieringMetrics.SHARED_BLOCKS, shared_blocks)
        stats.set_gauge(TieringMetrics.ACTIVE_BLOCKS, active_blocks)
        stats.set_gauge(TieringMetrics.DEVICE_CACHE_USAGE, self._device_usage)
        stats.set_gauge(
            TieringMetrics.MIGRATION_BUDGET,
            self._step_submitted_store_blocks,
            ("step",),
        )
        stats.set_gauge(
            TieringMetrics.MIGRATION_BUDGET,
            self._episode_submitted_store_blocks,
            ("pressure_episode",),
        )
        stats.set_gauge(
            TieringMetrics.MIGRATION_BUDGET,
            sum(state.pending_primary_stores for state in self._req_state.values()),
            ("inflight_jobs",),
        )
        stats.set_gauge(
            TieringMetrics.REUSE_SIGNALS,
            heat_snapshot["reuse_requests"],
            ("reuse_requests",),
        )
        stats.set_gauge(
            TieringMetrics.REUSE_SIGNALS,
            heat_snapshot["reuse_hit_blocks"],
            ("external_hit_blocks",),
        )
        stats.set_gauge(
            TieringMetrics.PRESSURE_STATE,
            int(self._device_pressure_active),
            ("device",),
        )
        cpu_capacity = self.primary_tier.capacity_blocks
        cpu_usage = (
            self.primary_tier.resident_blocks / cpu_capacity
            if cpu_capacity > 0
            else 0.0
        )
        stats.set_gauge(
            TieringMetrics.PRESSURE_STATE,
            int(
                self._cpu_pressure_requested
                or cpu_usage > self._lifecycle.config.cpu_high_watermark
            ),
            ("cpu",),
        )

        for (name, labels), value in self._metric_counters.items():
            stats.increase_counter(name, value, labels)
        self._metric_counters.clear()
        for (name, labels), observations in self._metric_histograms.items():
            for observation in observations:
                stats.observe_histogram(name, observation, labels)
        self._metric_histograms.clear()

        return stats

    @override
    def shutdown(self) -> None:
        """Shutdown all tiers and release resources."""
        for tier in self.secondary_tiers:
            tier.shutdown()
        self.primary_tier.shutdown()
