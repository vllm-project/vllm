# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission policy adapting the per-tier backpressure detectors (#50045).

This is the RFC #53485 deliverable: instead of a second detector framework,
the existing ``BackpressureDetector`` on each secondary tier is reused
behind the ``TieringAdmissionPolicy`` interface:

- Cascades consult ``detector.should_store(len(keys))``; rejections are
  recorded by the detector's drop policy exactly as before.
- Promotions are always admitted, preserving #50045 behavior.
- Only successful cascade completions feed latency samples back into the
  detector (via ``detector.update``).
- The pressure activate/clear transition log, drop metrics, and detector
  reset behavior are unchanged from #50045; only the wiring moved.
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

    def should_admit(
        self,
        keys: Collection[OffloadKey],
        tier_idx: int,
        is_promotion: bool,
    ) -> bool:
        # Promotions stay admitted under backpressure: dropping a promotion
        # only delays a hit the requester already paid a lookup for, while
        # the promotion itself relieves future load on the tier.
        if is_promotion:
            return True
        detector = self._tiers[tier_idx].bp_detector
        if detector is None:
            return True
        return detector.should_store(len(keys))

    def on_completed(self, metadata: "JobMetadata", result: JobResult) -> None:
        transfer_job = metadata.transfer_job
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
        }

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()
        for i, tier in enumerate(self._tiers):
            detector = tier.bp_detector
            if detector is None:
                continue
            label = (f"{i + 1}:{tier.tier_type}",)
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
        return stats if not stats.is_empty() else None
