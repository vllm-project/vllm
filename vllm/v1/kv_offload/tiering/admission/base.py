# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from vllm.v1.kv_offload.base import OffloadingMetricMetadata
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingConnectorStats,
    )


class TieringAdmissionPolicy(ABC):
    """Gates job submission for cascades and promotions."""

    @abstractmethod
    def should_admit(self, job: JobMetadata) -> bool:
        """Pure predicate, no side effects: may `job` be submitted now?"""

    @abstractmethod
    def on_admitted(self, job: JobMetadata) -> None:
        """`job` was just committed to submission."""

    @abstractmethod
    def on_completed(self, job: JobMetadata, result: JobResult) -> None:
        """`job` finished with `result`."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all tracked state (used by reset_cache())."""

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        """Return Prometheus metric definitions emitted by this policy."""
        return {}

    def get_stats(self) -> "OffloadingConnectorStats | None":
        """Return and reset metric observations collected by this policy."""
        return None
