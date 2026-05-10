# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = max(
        0,
        min(
            len(sorted_values) - 1,
            int(percentile * len(sorted_values) - 1e-12),
        ),
    )
    return sorted_values[rank]


@dataclass
class MooncakeStoreConnectorStats(KVConnectorStats):
    """Serializable Mooncake store communication telemetry."""

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data: dict[str, list[dict[str, int | float | str]]] = {}

    def is_empty(self) -> bool:
        return not self.data

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if other.is_empty():
            return self
        for operation, records in other.data.items():
            self.data.setdefault(operation, []).extend(records)
        return self

    def reduce(self) -> dict[str, int | float]:
        reduced: dict[str, int | float] = {}
        for operation, records in sorted(self.data.items()):
            if not records:
                continue
            durations = [
                float(record["duration_seconds"]) for record in records
            ]
            reduced[f"{operation}_count"] = len(records)
            reduced[f"{operation}_avg_ms"] = round(fmean(durations) * 1e3, 3)
            reduced[f"{operation}_p90_ms"] = round(
                _nearest_rank_percentile(durations, 0.9) * 1e3, 3
            )
            reduced[f"{operation}_total_keys"] = sum(
                int(record["num_keys"]) for record in records
            )
            reduced[f"{operation}_total_bytes"] = sum(
                int(record["num_bytes"]) for record in records
            )
            reduced[f"{operation}_failed_keys"] = sum(
                int(record["num_failed_keys"]) for record in records
            )
            reduced[f"{operation}_error_count"] = sum(
                1 for record in records if record["status"] == "error"
            )
        return reduced

    def record_operation(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        self.data.setdefault(operation, []).append(
            {
                "duration_seconds": duration_seconds,
                "num_keys": num_keys,
                "num_bytes": num_bytes,
                "status": status,
                "num_failed_keys": num_failed_keys,
            }
        )


class MooncakeStorePromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for Mooncake store communication."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        metric_labelnames = labelnames + ["operation", "status"]
        self._metric_cache: dict[tuple[int, str, str], dict[str, PromMetricT]] = {}

        self._histogram_operation_time = self._histogram_cls(
            name="vllm:mooncake_store_operation_time_seconds",
            documentation="Histogram of Mooncake store communication time.",
            buckets=[
                1e-4,
                5e-4,
                1e-3,
                5e-3,
                1e-2,
                2.5e-2,
                5e-2,
                1e-1,
                2e-1,
                3e-1,
                4e-1,
                5e-1,
                7.5e-1,
                1.0,
                1.5,
                2.0,
            ],
            labelnames=metric_labelnames,
        )
        self._counter_operation_calls = self._counter_cls(
            name="vllm:mooncake_store_operation_total",
            documentation="Number of Mooncake store communication operations.",
            labelnames=metric_labelnames,
        )
        self._counter_operation_keys = self._counter_cls(
            name="vllm:mooncake_store_operation_keys_total",
            documentation="Number of Mooncake store keys touched by operations.",
            labelnames=metric_labelnames,
        )
        self._counter_operation_bytes = self._counter_cls(
            name="vllm:mooncake_store_operation_bytes_total",
            documentation="Number of bytes transferred by Mooncake store operations.",
            labelnames=metric_labelnames,
        )
        self._counter_failed_keys = self._counter_cls(
            name="vllm:mooncake_store_operation_failed_keys_total",
            documentation="Number of Mooncake store keys that failed in operations.",
            labelnames=metric_labelnames,
        )

    def _get_metrics(
        self,
        engine_idx: int,
        operation: str,
        status: str,
    ) -> dict[str, PromMetricT]:
        cache_key = (engine_idx, operation, status)
        if cache_key not in self._metric_cache:
            label_values = self.per_engine_labelvalues[engine_idx] + [operation, status]
            self._metric_cache[cache_key] = {
                "time": self._histogram_operation_time.labels(*label_values),
                "calls": self._counter_operation_calls.labels(*label_values),
                "keys": self._counter_operation_keys.labels(*label_values),
                "bytes": self._counter_operation_bytes.labels(*label_values),
                "failed_keys": self._counter_failed_keys.labels(*label_values),
            }
        return self._metric_cache[cache_key]

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for operation, records in transfer_stats_data.items():
            assert isinstance(records, list)
            for record in records:
                assert isinstance(record, dict)
                status = str(record["status"])
                metrics = self._get_metrics(engine_idx, operation, status)
                metrics["time"].observe(float(record["duration_seconds"]))
                metrics["calls"].inc()
                metrics["keys"].inc(int(record["num_keys"]))
                metrics["bytes"].inc(int(record["num_bytes"]))
                metrics["failed_keys"].inc(int(record["num_failed_keys"]))
