# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    SecondaryTierManager,
    TieringOffloadingMetrics,
    TransferJob,
)

TierLabel = tuple[str]
PRIMARY_TIER_LABEL: TierLabel = ("0:primary",)

# Outcomes of a promoted chunk: how it left the primary tier's "promoted, not
# yet read" state. Each chunk a promotion lands records exactly one of them.
OUTCOME_LOADED = "loaded"  # a GPU load of it completed on every worker
OUTCOME_SERVED_PEER = "served_peer"  # a secondary tier sent it to a peer
OUTCOME_EVICTED_UNLOADED = "evicted_unloaded"  # evicted before any read
OUTCOME_RESET = "reset"  # dropped by reset_cache() before any read
# States of a promoted chunk still waiting for an outcome.
STATE_UNREAD = "unread"  # no read in flight; evictable
STATE_READING = "reading"  # a GPU load or a peer transfer is in flight


@dataclass(slots=True)
class _PromotedChunk:
    tier_idx: int
    landed_at: float
    # GPU loads and peer transfers of this chunk in flight. Each pins the
    # chunk in the primary tier, so it cannot be evicted while this is > 0.
    num_reads: int = 0


class _JobMetadataLike(Protocol):
    @property
    def transfer_job(self) -> TransferJob: ...

    @property
    def tier_idx(self) -> int: ...


@dataclass(slots=True)
class _RequestMetricsState:
    observed_lookups: dict[TierLabel, dict[OffloadKey, float | None]] | None = field(
        default_factory=dict
    )


@dataclass
class _TierState:
    active_promotion_count: int = 0
    active_cascade_count: int = 0
    primary_write_chunk_count: int = 0
    primary_read_chunk_count: int = 0
    deferred_release_chunk_count: int = 0
    promoted_unread_chunk_count: int = 0
    promoted_reading_chunk_count: int = 0


class TieringMetricsTracker:
    def __init__(
        self,
        tier_types: list[str],
        num_primary_chunks: int,
        primary_chunk_size: int,
    ) -> None:
        self._tier_types = tier_types
        self._num_primary_chunks = num_primary_chunks
        self._primary_chunk_size = primary_chunk_size
        self._request_states: dict[str, _RequestMetricsState] = {}
        self._tier_states = [_TierState() for _ in tier_types]
        self._stats = OffloadingConnectorStats()
        # Promoted chunks landed in the primary tier and waiting for an
        # outcome. Entries leave on read, eviction or reset, so this holds at
        # most one entry per primary-tier chunk.
        self._promoted: dict[OffloadKey, _PromotedChunk] = {}
        # Store jobs a secondary tier created to send primary chunks to a peer.
        self._peer_read_jobs: set[JobId] = set()
        # Chunks whose last promotion was evicted unread, oldest first.
        # Allocating a new promotion for one of them is a re-promotion.
        # Capped at the primary tier's capacity.
        self._unread_evictions: OrderedDict[OffloadKey, None] = OrderedDict()

    @functools.cache  # noqa: B019
    def tier_label(self, tier_idx: int) -> TierLabel:
        return (f"{tier_idx + 1}:{self._tier_types[tier_idx]}",)

    @property
    def primary_tier_label(self) -> TierLabel:
        return PRIMARY_TIER_LABEL

    def on_new_request(self, req_context: ReqContext) -> None:
        self._request_states[req_context.req_id] = _RequestMetricsState()

    def on_request_allocated(self, req_context: ReqContext) -> None:
        state = self._request_states.get(req_context.req_id)
        if state is None:
            return
        state.observed_lookups = None

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._request_states.pop(req_context.req_id, None)

    def on_lookup(
        self,
        req_context: ReqContext,
        key: OffloadKey,
        tier_label: TierLabel,
        result: LookupResult,
        lookup_duration: float,
    ) -> None:
        state = self._request_states.get(req_context.req_id)
        if state is None:
            state = _RequestMetricsState()
            self._request_states[req_context.req_id] = state

        if state.observed_lookups is not None and result in (
            LookupResult.HIT,
            LookupResult.MISS,
        ):
            observed = state.observed_lookups.setdefault(tier_label, {})
            start_time = observed.get(key)
            if key not in observed or start_time is not None:
                observed[key] = None
                self._observe_resolved_lookup(
                    tier_label,
                    result,
                    lookup_duration,
                    start_time,
                )
        elif result is LookupResult.RETRY and state.observed_lookups is not None:
            observed = state.observed_lookups.setdefault(tier_label, {})
            observed.setdefault(key, time.monotonic() - lookup_duration)

    def on_job_registered(self, job_metadata: _JobMetadataLike) -> None:
        transfer_job = job_metadata.transfer_job
        state = self._tier_states[job_metadata.tier_idx]
        chunk_count = len(transfer_job.chunk_ids)
        if transfer_job.is_promotion:
            state.active_promotion_count += 1
            state.primary_write_chunk_count += chunk_count
        else:
            state.active_cascade_count += 1
            state.primary_read_chunk_count += chunk_count

    def on_job_finished(
        self, job_metadata: _JobMetadataLike, result: JobResult
    ) -> None:
        if job_metadata.transfer_job.job_id in self._peer_read_jobs:
            self._peer_read_jobs.remove(job_metadata.transfer_job.job_id)
            self.on_promoted_reads_done(
                job_metadata.transfer_job.keys, result.success, by_gpu_load=False
            )
        self._observe_finished_job_stats(job_metadata, result)
        if job_metadata.transfer_job.is_promotion:
            self._observe_promotion_latency(job_metadata, result)
        self._decrement_tier_state(job_metadata)

    def on_release_deferred(self, tier_idx: int, chunk_count: int) -> None:
        self._tier_states[tier_idx].deferred_release_chunk_count += chunk_count

    def on_deferred_release_done(
        self, tier_idx: int, chunk_count: int, hold_duration: float
    ) -> None:
        state = self._tier_states[tier_idx]
        state.deferred_release_chunk_count -= chunk_count
        assert state.deferred_release_chunk_count >= 0
        self._stats.observe_histogram(
            TieringOffloadingMetrics.DEFERRED_RELEASE_HOLD,
            hold_duration,
            self.tier_label(tier_idx),
        )

    # ------------------------------------------------------------------
    # Promoted chunk outcomes. All hooks run under the tiering manager's lock.
    # ------------------------------------------------------------------

    def on_promotion_slots_allocated(
        self, tier_idx: int, keys: Iterable[OffloadKey]
    ) -> None:
        """Primary-tier slots were allocated to promote keys from `tier_idx`.

        Counts a re-promotion per key whose previous promotion was evicted
        unread. Whether this promotion then reads the tier or lands is not
        known yet.
        """
        if not self._unread_evictions:
            return
        num_repromoted = 0
        for key in keys:
            if key in self._unread_evictions:
                del self._unread_evictions[key]
                num_repromoted += 1
        if num_repromoted:
            self._stats.increase_counter(
                TieringOffloadingMetrics.CHUNK_REPROMOTIONS,
                num_repromoted,
                self.tier_label(tier_idx),
            )

    def on_promotion_landed(self, tier_idx: int, keys: Iterable[OffloadKey]) -> None:
        """Keys of a promotion from tier `tier_idx` became readable."""
        now = time.monotonic()
        state = self._tier_states[tier_idx]
        for key in keys:
            # A promotion starts only on a primary-tier miss, so a landing key
            # is not tracked. Keep the first entry if that ever breaks.
            if key in self._promoted:
                continue
            self._promoted[key] = _PromotedChunk(tier_idx, now)
            state.promoted_unread_chunk_count += 1

    def on_promoted_reads_started(self, keys: Iterable[OffloadKey]) -> None:
        """A GPU load or a peer transfer pinned keys for reading."""
        if not self._promoted:
            return
        for key in dict.fromkeys(keys):
            chunk = self._promoted.get(key)
            if chunk is None:
                continue
            if chunk.num_reads == 0:
                state = self._tier_states[chunk.tier_idx]
                state.promoted_unread_chunk_count -= 1
                state.promoted_reading_chunk_count += 1
            chunk.num_reads += 1

    def on_promoted_reads_done(
        self, keys: Iterable[OffloadKey], success: bool, by_gpu_load: bool
    ) -> None:
        """Reads started by on_promoted_reads_started() completed.

        The first successful read records the chunk's outcome; a failed read
        returns it to the unread state once no other read is in flight.
        """
        if not self._promoted:
            return
        now = time.monotonic()
        for key in dict.fromkeys(keys):
            chunk = self._promoted.get(key)
            if chunk is None or chunk.num_reads == 0:
                continue
            chunk.num_reads -= 1
            if success:
                del self._promoted[key]
                self._tier_states[chunk.tier_idx].promoted_reading_chunk_count -= 1
                self._record_outcome(
                    chunk,
                    OUTCOME_LOADED if by_gpu_load else OUTCOME_SERVED_PEER,
                    now,
                )
            elif chunk.num_reads == 0:
                state = self._tier_states[chunk.tier_idx]
                state.promoted_reading_chunk_count -= 1
                state.promoted_unread_chunk_count += 1

    def on_peer_read_job(self, job_id: JobId, keys: Iterable[OffloadKey]) -> None:
        """A secondary tier created store job `job_id` to serve a peer."""
        self._peer_read_jobs.add(job_id)
        self.on_promoted_reads_started(keys)

    def on_primary_evicted(self, keys: Iterable[OffloadKey]) -> None:
        if not self._promoted:
            return
        now = time.monotonic()
        for key in keys:
            chunk = self._promoted.pop(key, None)
            if chunk is None:
                continue
            state = self._tier_states[chunk.tier_idx]
            if chunk.num_reads:  # unreachable: reads pin the chunk
                state.promoted_reading_chunk_count -= 1
            else:
                state.promoted_unread_chunk_count -= 1
            self._record_outcome(chunk, OUTCOME_EVICTED_UNLOADED, now)
            self._unread_evictions[key] = None
            self._unread_evictions.move_to_end(key)
            if len(self._unread_evictions) > self._num_primary_chunks:
                self._unread_evictions.popitem(last=False)

    def on_primary_reset(self) -> None:
        """The primary tier dropped every chunk (reset_cache())."""
        now = time.monotonic()
        for chunk in self._promoted.values():
            self._record_outcome(chunk, OUTCOME_RESET, now)
        self._promoted.clear()
        self._peer_read_jobs.clear()
        self._unread_evictions.clear()
        for state in self._tier_states:
            state.promoted_unread_chunk_count = 0
            state.promoted_reading_chunk_count = 0

    def _record_outcome(self, chunk: _PromotedChunk, outcome: str, now: float) -> None:
        labelvalues = (*self.tier_label(chunk.tier_idx), outcome)
        self._stats.increase_counter(
            TieringOffloadingMetrics.PROMOTED_CHUNK_OUTCOMES, labelvalues=labelvalues
        )
        self._stats.observe_histogram(
            TieringOffloadingMetrics.PROMOTED_CHUNK_TIME_TO_OUTCOME,
            now - chunk.landed_at,
            labelvalues,
        )

    def on_promotion_allocation_failure(self) -> None:
        self._stats.increase_counter(
            TieringOffloadingMetrics.PROMOTION_ALLOCATION_FAILURES
        )

    def record_backpressure(
        self,
        tiers: list[SecondaryTierManager],
    ) -> None:
        for i, tier in enumerate(tiers):
            detector = tier.bp_detector
            if detector is None:
                continue
            label = self.tier_label(i)
            ema = detector.stats.get("store_latency_ema")
            if ema is not None:
                self._stats.set_gauge(
                    TieringOffloadingMetrics.BACKPRESSURE_STORE_LATENCY_EMA,
                    ema,
                    labelvalues=label,
                )
            stores_dropped, blocks_dropped = detector.policy.pop_stores_dropped()
            if stores_dropped > 0:
                self._stats.increase_counter(
                    TieringOffloadingMetrics.BACKPRESSURE_STORES_DROPPED,
                    stores_dropped,
                    labelvalues=label,
                )
            if blocks_dropped > 0:
                self._stats.increase_counter(
                    TieringOffloadingMetrics.BACKPRESSURE_BLOCKS_DROPPED,
                    blocks_dropped,
                    labelvalues=label,
                )

    def take_stats(self) -> OffloadingConnectorStats | None:
        active_transfer_stats = OffloadingConnectorStats()
        self._observe_active_transfer_stats(active_transfer_stats)

        stats = None
        if not active_transfer_stats.is_empty():
            stats = active_transfer_stats
        if not self._stats.is_empty():
            if stats is None:
                stats = self._stats
            else:
                stats.aggregate(self._stats)
            self._stats = OffloadingConnectorStats()
        return stats

    def assert_idle(self) -> None:
        assert all(
            state.active_promotion_count == 0
            and state.active_cascade_count == 0
            and state.primary_write_chunk_count == 0
            and state.primary_read_chunk_count == 0
            and state.deferred_release_chunk_count == 0
            for state in self._tier_states
        )

    def _decrement_tier_state(self, job_metadata: _JobMetadataLike) -> None:
        transfer_job = job_metadata.transfer_job
        state = self._tier_states[job_metadata.tier_idx]
        chunk_count = len(transfer_job.chunk_ids)
        if transfer_job.is_promotion:
            assert state.active_promotion_count > 0
            state.active_promotion_count -= 1
            state.primary_write_chunk_count -= chunk_count
            assert state.primary_write_chunk_count >= 0
        else:
            assert state.active_cascade_count > 0
            state.active_cascade_count -= 1
            state.primary_read_chunk_count -= chunk_count
            assert state.primary_read_chunk_count >= 0

    def _observe_finished_job_stats(
        self,
        job_metadata: _JobMetadataLike,
        completed_job: JobResult,
    ) -> None:
        transfer_job = job_metadata.transfer_job
        labelvalues = self.tier_label(job_metadata.tier_idx)
        completed_key_count = len(transfer_job.keys)
        if not completed_job.success:
            failure_metric = (
                TieringOffloadingMetrics.PROMOTION_JOB_FAILURES
                if transfer_job.is_promotion
                else TieringOffloadingMetrics.CASCADE_JOB_FAILURES
            )
            self._stats.increase_counter(failure_metric, labelvalues=labelvalues)
            if transfer_job.is_promotion and completed_job.successful_keys:
                completed_key_count = len(completed_job.successful_keys)
            else:
                return

        bytes_metric = (
            TieringOffloadingMetrics.READ_BYTES
            if transfer_job.is_promotion
            else TieringOffloadingMetrics.WRITE_BYTES
        )
        time_metric = (
            TieringOffloadingMetrics.READ_TIME
            if transfer_job.is_promotion
            else TieringOffloadingMetrics.WRITE_TIME
        )
        transfer_size = completed_key_count * self._primary_chunk_size
        self._stats.increase_counter(bytes_metric, transfer_size, labelvalues)
        if completed_job.transfer_time is not None:
            self._stats.increase_counter(
                time_metric, completed_job.transfer_time, labelvalues
            )

    def _observe_promotion_latency(
        self, job_metadata: _JobMetadataLike, result: JobResult
    ) -> None:
        """Record promotion time from job creation to reported completion.

        This is broader than the tier-reported `JobResult.transfer_time`, which
        covers only the active data movement, and so includes tier queueing.
        Failed promotions are observed too; PROMOTION_JOB_FAILURES counts them.
        """
        self._stats.observe_histogram(
            TieringOffloadingMetrics.PROMOTION_LATENCY,
            time.monotonic() - job_metadata.transfer_job.submit_time,
            (
                *self.tier_label(job_metadata.tier_idx),
                "success" if result.success else "failure",
            ),
        )

    def _observe_active_transfer_stats(self, stats: OffloadingConnectorStats) -> None:
        for tier_idx, state in enumerate(self._tier_states):
            labelvalues = self.tier_label(tier_idx)
            write_usage = (
                state.primary_write_chunk_count / self._num_primary_chunks
                if self._num_primary_chunks > 0
                else 0.0
            )
            read_usage = (
                state.primary_read_chunk_count / self._num_primary_chunks
                if self._num_primary_chunks > 0
                else 0.0
            )
            stats.set_gauge(
                TieringOffloadingMetrics.PRIMARY_WRITE_USAGE_PERC,
                write_usage,
                labelvalues,
            )
            stats.set_gauge(
                TieringOffloadingMetrics.PRIMARY_READ_USAGE_PERC,
                read_usage,
                labelvalues,
            )
            stats.set_gauge(
                TieringOffloadingMetrics.ACTIVE_PROMOTION_JOBS,
                state.active_promotion_count,
                labelvalues,
            )
            stats.set_gauge(
                TieringOffloadingMetrics.ACTIVE_CASCADE_JOBS,
                state.active_cascade_count,
                labelvalues,
            )
            stats.set_gauge(
                TieringOffloadingMetrics.DEFERRED_RELEASE_CHUNKS,
                state.deferred_release_chunk_count,
                labelvalues,
            )
            stats.set_gauge(
                TieringOffloadingMetrics.PROMOTED_CHUNKS_PENDING,
                state.promoted_unread_chunk_count,
                (*labelvalues, STATE_UNREAD),
            )
            stats.set_gauge(
                TieringOffloadingMetrics.PROMOTED_CHUNKS_PENDING,
                state.promoted_reading_chunk_count,
                (*labelvalues, STATE_READING),
            )

    def _observe_resolved_lookup(
        self,
        tier_label: TierLabel,
        result: LookupResult,
        lookup_duration: float,
        async_start_time: float | None,
    ) -> None:
        self._stats.increase_counter(
            TieringOffloadingMetrics.CHUNK_QUERIES,
            labelvalues=tier_label,
        )
        if result is LookupResult.HIT:
            self._stats.increase_counter(
                TieringOffloadingMetrics.CHUNK_HITS,
                labelvalues=tier_label,
            )
        self._stats.observe_histogram(
            TieringOffloadingMetrics.LOOKUP_SYNC_DELAY,
            lookup_duration,
            tier_label,
        )
        if async_start_time is not None:
            self._stats.observe_histogram(
                TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY,
                time.monotonic() - async_start_time,
                tier_label,
            )
