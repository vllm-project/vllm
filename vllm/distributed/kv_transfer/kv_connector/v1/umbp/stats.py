# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared UMBP connector transfer statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)


@dataclass
class UMBPStoreConnectorStats(KVConnectorStats):
    """Serializable interval counters shared by every UMBP runtime."""

    data: dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        operation: str,
        *,
        submitted: int = 0,
        completed: int = 0,
        failed: int = 0,
        num_bytes: int = 0,
    ) -> None:
        entry = self.data.setdefault(
            operation,
            {
                "submitted": 0,
                "completed": 0,
                "failed": 0,
                "num_bytes": 0,
            },
        )
        entry["submitted"] += submitted
        entry["completed"] += completed
        entry["failed"] += failed
        entry["num_bytes"] += num_bytes

    def reset(self) -> None:
        self.data.clear()

    def aggregate(self, other: KVConnectorStats) -> "UMBPStoreConnectorStats":
        if not isinstance(other, UMBPStoreConnectorStats):
            raise TypeError("cannot aggregate incompatible UMBP stats")
        result = UMBPStoreConnectorStats()
        for operation, values in [*self.data.items(), *other.data.items()]:
            entry = result.data.setdefault(
                operation,
                {
                    "submitted": 0,
                    "completed": 0,
                    "failed": 0,
                    "num_bytes": 0,
                },
            )
            for key in entry:
                entry[key] += values.get(key, 0)
        return result

    def reduce(self) -> dict[str, int | float]:
        result: dict[str, int | float] = {}
        for operation, values in self.data.items():
            for key, value in values.items():
                result[f"{operation}_{key}"] = value
        return result

    def is_empty(self) -> bool:
        return not self.data


class UMBPStorePromMetrics(KVConnectorPromMetrics):
    """Prometheus counters for shared UMBP transfer outcomes."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )
        labels = labelnames + ["operation"]
        self._submitted = self._counter_cls(
            name="vllm:umbp_transfer_submitted_total",
            documentation="Number of UMBP objects submitted.",
            labelnames=labels,
        )
        self._completed = self._counter_cls(
            name="vllm:umbp_transfer_completed_total",
            documentation="Number of UMBP objects completed.",
            labelnames=labels,
        )
        self._failed = self._counter_cls(
            name="vllm:umbp_transfer_failed_total",
            documentation="Number of UMBP objects failed.",
            labelnames=labels,
        )
        self._bytes = self._counter_cls(
            name="vllm:umbp_transfer_bytes_total",
            documentation="Number of bytes transferred by UMBP.",
            labelnames=labels,
        )

    def observe(
        self,
        transfer_stats_data: dict[str, Any] | None,
        engine_idx: int = 0,
    ) -> None:
        if not transfer_stats_data:
            return
        for operation, values in transfer_stats_data.items():
            labels = self.per_engine_labelvalues[engine_idx] + [operation]
            self._submitted.labels(*labels).inc(values.get("submitted", 0))
            self._completed.labels(*labels).inc(values.get("completed", 0))
            self._failed.labels(*labels).inc(values.get("failed", 0))
            self._bytes.labels(*labels).inc(values.get("num_bytes", 0))
