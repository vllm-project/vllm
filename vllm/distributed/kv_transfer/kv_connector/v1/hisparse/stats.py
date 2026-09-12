# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics for the HiSparse connector."""

from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

_HISPARSE_COUNTERS: tuple[tuple[str, str], ...] = (
    ("cache_hits", "Number of HiSparse device hot-buffer hits."),
    ("cache_misses", "Number of HiSparse device hot-buffer misses."),
    (
        "host_to_device_bytes",
        "Bytes transferred from host KV storage to HiSparse hot buffers.",
    ),
)


@dataclass
class HiSparseKVConnectorStats(KVConnectorStats):
    """Container for HiSparse hot-buffer residency metrics.

    Each list entry is the delta recorded over one device counter snapshot
    interval, so a list can hold multiple snapshots per logging interval.
    """

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        # Must be serializable
        self.data: dict[str, list[int]] = {
            "cache_hits": [],
            "cache_misses": [],
            "host_to_device_bytes": [],
        }

    def record_snapshot(self, hits: int, misses: int, host_to_device_bytes: int):
        self.data["cache_hits"].append(hits)
        self.data["cache_misses"].append(misses)
        self.data["host_to_device_bytes"].append(host_to_device_bytes)

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        # Compute compact representative stats suitable for CLI logging.
        return {
            "HiSparse hot-buffer hits": sum(self.data["cache_hits"]),
            "HiSparse hot-buffer misses": sum(self.data["cache_misses"]),
            "HiSparse host-to-device bytes": sum(self.data["host_to_device_bytes"]),
        }

    def is_empty(self) -> bool:
        return all(len(values) == 0 for values in self.data.values())


class HiSparsePromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        self.hisparse_counters: dict[str, Any] = {}
        for name, documentation in _HISPARSE_COUNTERS:
            counter = self._counter_cls(
                name=f"vllm:hisparse_{name}",
                documentation=documentation,
                labelnames=labelnames,
            )
            self.hisparse_counters[name] = create_metric_per_engine(
                counter, self.per_engine_labelvalues
            )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for name, counter in self.hisparse_counters.items():
            for value in transfer_stats_data.get(name, []):
                counter[engine_idx].inc(value)
