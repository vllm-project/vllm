# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base interface for tiering admission policies."""

from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import TYPE_CHECKING

from vllm.v1.kv_offload.base import OffloadingMetricMetadata, OffloadKey
from vllm.v1.kv_offload.tiering.base import JobResult

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingConnectorStats,
    )
    from vllm.v1.kv_offload.tiering.manager import JobMetadata


class TieringAdmissionPolicy(ABC):
    """Decides whether a tiering transfer may be submitted.

    The manager consults the policy at these points:

    1. ``should_lookup`` — before querying a secondary tier. A ``False``
       return skips that tier for the current lookup.
    2. ``should_admit`` — before creating a transfer job. A ``False`` return
       means the transfer is skipped.
    3. ``on_admitted`` — after the real ``TransferJob`` is registered,
       giving the policy the job metadata (job ID, chunk IDs, tier index).
    4. ``on_completed`` — for every finished job, so the policy can feed
       completion signals (e.g. latency) back into its detectors.

    ``JobMetadata`` is referenced under ``TYPE_CHECKING`` only; it stays
    defined in ``manager.py``.
    """

    def should_lookup(
        self, key: OffloadKey, tier_idx: int, req_context: object
    ) -> bool:
        """Whether a secondary tier should be queried for this key."""
        return True

    @abstractmethod
    def should_admit(
        self,
        keys: Collection[OffloadKey],
        tier_idx: int,
        is_promotion: bool,
    ) -> bool:
        """Whether the transfer may be submitted.

        Args:
            keys: Chunk keys of the candidate transfer.
            tier_idx: Index of the secondary tier involved.
            is_promotion: True for secondary -> primary (promotion),
                False for primary -> secondary (cascade store).

        """
        ...

    def on_admitted(self, metadata: "JobMetadata") -> None:
        """Record a transfer that passed admission and was registered."""
        return

    def on_completed(self, metadata: "JobMetadata", result: JobResult) -> None:
        """Observe a finished transfer (success or failure)."""
        return

    def reset(self) -> None:
        """Reset policy state (e.g. detector EMAs)."""
        return

    @classmethod
    def build_metric_definitions(cls) -> dict[str, OffloadingMetricMetadata]:
        """Return Prometheus metric definitions emitted by this policy."""
        return {}

    def get_stats(self) -> "OffloadingConnectorStats | None":
        """Return and reset metric observations collected by this policy."""
        return None
