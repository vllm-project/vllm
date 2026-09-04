# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import time
from dataclasses import dataclass, field
from typing import Protocol

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    TieringOffloadingMetrics,
    TransferJob,
)

TierLabel = tuple[str]
PRIMARY_TIER_LABEL: TierLabel = ("0:primary",)


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
    primary_write_block_count: int = 0
    primary_read_block_count: int = 0


class TieringMetricsTracker:
    def __init__(
        self,
        tier_types: list[str],
        num_primary_blocks: int,
        primary_block_size: int,
    ) -> None:
        self._tier_types = tier_types
        self._num_primary_blocks = num_primary_blocks
        self._primary_block_size = primary_block_size
        self._request_states: dict[str, _RequestMetricsState] = {}
        self._tier_states = [_TierState() for _ in tier_types]
        self._stats = OffloadingConnectorStats()

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
        block_count = len(transfer_job.block_ids)
        if transfer_job.is_promotion:
            state.active_promotion_count += 1
            state.primary_write_block_count += block_count
        else:
            state.active_cascade_count += 1
            state.primary_read_block_count += block_count

    def on_job_finished(
        self, job_metadata: _JobMetadataLike, result: JobResult
    ) -> None:
        self._observe_finished_job_stats(job_metadata, result)
        self._decrement_tier_state(job_metadata)

    def on_promotion_allocation_failure(self) -> None:
        self._stats.increase_counter(
            TieringOffloadingMetrics.PROMOTION_ALLOCATION_FAILURES
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
            and state.primary_write_block_count == 0
            and state.primary_read_block_count == 0
            for state in self._tier_states
        )

    def _decrement_tier_state(self, job_metadata: _JobMetadataLike) -> None:
        transfer_job = job_metadata.transfer_job
        state = self._tier_states[job_metadata.tier_idx]
        block_count = len(transfer_job.block_ids)
        if transfer_job.is_promotion:
            assert state.active_promotion_count > 0
            state.active_promotion_count -= 1
            state.primary_write_block_count -= block_count
            assert state.primary_write_block_count >= 0
        else:
            assert state.active_cascade_count > 0
            state.active_cascade_count -= 1
            state.primary_read_block_count -= block_count
            assert state.primary_read_block_count >= 0

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
        transfer_size = completed_key_count * self._primary_block_size
        self._stats.increase_counter(bytes_metric, transfer_size, labelvalues)
        if completed_job.transfer_time is not None:
            self._stats.increase_counter(
                time_metric, completed_job.transfer_time, labelvalues
            )

    def _observe_active_transfer_stats(self, stats: OffloadingConnectorStats) -> None:
        for tier_idx, state in enumerate(self._tier_states):
            labelvalues = self.tier_label(tier_idx)
            write_usage = (
                state.primary_write_block_count / self._num_primary_blocks
                if self._num_primary_blocks > 0
                else 0.0
            )
            read_usage = (
                state.primary_read_block_count / self._num_primary_blocks
                if self._num_primary_blocks > 0
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

    def _observe_resolved_lookup(
        self,
        tier_label: TierLabel,
        result: LookupResult,
        lookup_duration: float,
        async_start_time: float | None,
    ) -> None:
        self._stats.increase_counter(
            TieringOffloadingMetrics.BLOCK_QUERIES,
            labelvalues=tier_label,
        )
        if result is LookupResult.HIT:
            self._stats.increase_counter(
                TieringOffloadingMetrics.BLOCK_HITS,
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
