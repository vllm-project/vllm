# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission policy adapting the per-tier backpressure detectors.

The existing ``BackpressureDetector`` on each secondary tier is reused
behind the ``TieringAdmissionPolicy`` interface:

- Cascades consult the detector policy; admission and rejection accounting
  happen in the corresponding lifecycle hooks.
- Promotions are always admitted, preserving existing behavior.
- Only successful cascade completions feed latency samples back into the
  detector (via ``detector.update``).
- The pressure activate/clear transition log, drop metrics, and detector
  reset behavior are unchanged; only the wiring moved.
"""

from collections.abc import Collection
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
)
from vllm.v1.kv_offload.tiering.admission.base import TieringAdmissionPolicy
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    SecondaryTierManager,
    TieringOffloadingMetrics,
)
from vllm.v1.kv_offload.tiering.metrics import make_tier_label

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.manager import JobMetadata

logger = init_logger(__name__)


class BackpressureAdmissionPolicy(TieringAdmissionPolicy):
    """Gate cascade stores on each tier's backpressure detector.

    Args:
        secondary_tiers: The manager's secondary tiers, in index order.
            The policy reads (but does not own) each tier's
            ``bp_detector``.

    """

    def __init__(self, secondary_tiers: list[SecondaryTierManager]):
        self._tiers = secondary_tiers
        self._admitted_jobs: set[int] = set()
        self._cascades_dropped = [0] * len(secondary_tiers)
        self._promotions_dropped = [0] * len(secondary_tiers)

    def should_admit(
        self,
        keys: Collection[OffloadKey],
        tier_idx: int,
        *,
        is_promotion: bool,
    ) -> bool:
        # Promotions remain admitted to preserve current tiering behavior.
        if is_promotion:
            return True
        detector = self._tiers[tier_idx].bp_detector
        if detector is None:
            return True
        return detector.should_store_without_side_effects()

    def on_rejected(
        self,
        keys: Collection[OffloadKey],
        tier_idx: int,
        *,
        is_promotion: bool,
    ) -> None:
        if is_promotion:
            self._promotions_dropped[tier_idx] += 1
            return
        self._cascades_dropped[tier_idx] += 1
        detector = self._tiers[tier_idx].bp_detector
        if detector is not None:
            detector.on_store_skipped(len(keys))

    def on_admitted(self, metadata: "JobMetadata") -> None:
        transfer_job = metadata.transfer_job
        self._admitted_jobs.add(transfer_job.job_id)
        if not transfer_job.is_promotion:
            detector = self._tiers[metadata.tier_idx].bp_detector
            if detector is not None:
                detector.on_store_admitted()

    def on_completed(self, metadata: "JobMetadata", result: JobResult) -> None:
        transfer_job = metadata.transfer_job
        if transfer_job.job_id not in self._admitted_jobs:
            return
        self._admitted_jobs.remove(transfer_job.job_id)
        if transfer_job.is_promotion or not result.success:
            return
        tier = self._tiers[metadata.tier_idx]
        detector = tier.bp_detector
        if detector is None:
            return
        was_under_pressure = detector.is_under_pressure()
        num_bytes = (
            result.transfer_bytes
            if result.transfer_bytes is not None
            else len(transfer_job.keys) * tier.block_size_bytes
        )
        detector.update(transfer_job.submit_time, num_bytes)
        if detector.is_under_pressure() != was_under_pressure:
            logger.info(
                "Tier #%d (%s) back-pressure %s (stats=%s)",
                metadata.tier_idx,
                tier.tier_type,
                "activated" if detector.is_under_pressure() else "cleared",
                detector.stats,
            )

    def reset(self) -> None:
        self._admitted_jobs.clear()
        self._cascades_dropped = [0] * len(self._tiers)
        self._promotions_dropped = [0] * len(self._tiers)
        for tier in self._tiers:
            if tier.bp_detector is not None:
                tier.bp_detector.reset()

    @classmethod
    def build_metric_definitions(cls) -> dict[str, OffloadingMetricMetadata]:
        return {
            TieringOffloadingMetrics.BACKPRESSURE_STORE_LATENCY_EMA: (
                OffloadingGaugeMetadata(
                    documentation=(
                        "Exponential moving average of store latency "
                        "for back-pressure detection, in s/MiB."
                    ),
                    labelnames=("tier",),
                )
            ),
            TieringOffloadingMetrics.BACKPRESSURE_STORES_DROPPED: (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of store operations dropped due to "
                        "back-pressure on a secondary tier."
                    ),
                    labelnames=("tier",),
                )
            ),
            TieringOffloadingMetrics.BACKPRESSURE_BLOCKS_DROPPED: (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of blocks dropped due to back-pressure "
                        "on a secondary tier."
                    ),
                    labelnames=("tier",),
                )
            ),
            TieringOffloadingMetrics.CASCADES_DROPPED: OffloadingCounterMetadata(
                documentation="Number of cascade stores rejected by admission.",
                labelnames=("tier",),
            ),
            TieringOffloadingMetrics.PROMOTIONS_DROPPED: OffloadingCounterMetadata(
                documentation="Number of promotions rejected by admission.",
                labelnames=("tier",),
            ),
        }

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()
        for i, tier in enumerate(self._tiers):
            detector = tier.bp_detector
            if detector is None:
                continue
            label = make_tier_label(i, tier.tier_type)
            ema = detector.stats.get("store_latency_ema")
            if ema is not None:
                stats.set_gauge(
                    TieringOffloadingMetrics.BACKPRESSURE_STORE_LATENCY_EMA,
                    ema,
                    labelvalues=label,
                )
            stores_dropped, blocks_dropped = detector.policy.pop_stores_dropped()
            if stores_dropped > 0:
                stats.increase_counter(
                    TieringOffloadingMetrics.BACKPRESSURE_STORES_DROPPED,
                    stores_dropped,
                    labelvalues=label,
                )
            if blocks_dropped > 0:
                stats.increase_counter(
                    TieringOffloadingMetrics.BACKPRESSURE_BLOCKS_DROPPED,
                    blocks_dropped,
                    labelvalues=label,
                )
            if self._cascades_dropped[i] > 0:
                stats.increase_counter(
                    TieringOffloadingMetrics.CASCADES_DROPPED,
                    self._cascades_dropped[i],
                    labelvalues=label,
                )
            if self._promotions_dropped[i] > 0:
                stats.increase_counter(
                    TieringOffloadingMetrics.PROMOTIONS_DROPPED,
                    self._promotions_dropped[i],
                    labelvalues=label,
                )
            self._cascades_dropped[i] = 0
            self._promotions_dropped[i] = 0
        return stats if not stats.is_empty() else None
