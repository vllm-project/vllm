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

_HISPARSE_LEVEL_KEYS = frozenset(
    {"host_blocks_used", "host_blocks_total", "host_blocks_usage", "pending_spills"}
)


@dataclass
class HiSparseKVConnectorStats(KVConnectorStats):
    """Container for HiSparse hot-buffer residency metrics.

    Each list entry is the delta recorded over one device counter snapshot
    interval, so a list can hold multiple snapshots per logging interval.
    The host-tier gauges are level values, not deltas: aggregation keeps
    only the most recent observation.
    """

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        # Must be serializable
        self.data: dict[str, Any] = {
            "cache_hits": [],
            "cache_misses": [],
            "host_to_device_bytes": [],
            "host_blocks_used": [],
            "host_blocks_total": [],
            "host_blocks_usage": [],
            "pending_spills": [],
        }

    def record_snapshot(self, hits: int, misses: int, host_to_device_bytes: int):
        self.data["cache_hits"].append(hits)
        self.data["cache_misses"].append(misses)
        self.data["host_to_device_bytes"].append(host_to_device_bytes)

    def record_host_usage(self, used: int, total: int, pending_spills: int):
        """Record scheduler-side host-tier level values."""
        self.data["host_blocks_used"].append(used)
        self.data["host_blocks_total"].append(total)
        self.data["host_blocks_usage"].append(used / total if total else 0.0)
        self.data["pending_spills"].append(pending_spills)

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                if k in _HISPARSE_LEVEL_KEYS:
                    # Level values: keep the most recent observation only.
                    accumulator.extend(v[-1:])
                else:
                    accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        # Compute compact representative stats suitable for CLI logging.
        # The host gauges are level values, so report the latest snapshot.
        reduced: dict[str, int | float] = {
            "HiSparse hot-buffer hits": sum(self.data["cache_hits"]),
            "HiSparse hot-buffer misses": sum(self.data["cache_misses"]),
            "HiSparse host-to-device bytes": sum(self.data["host_to_device_bytes"]),
        }
        if self.data["host_blocks_used"]:
            used = self.data["host_blocks_used"][-1]
            total = self.data["host_blocks_total"][-1]
            reduced["HiSparse host pool used blocks"] = used
            reduced["HiSparse host pool usage %"] = (
                round(100 * used / total, 1) if total else 0.0
            )
        return reduced

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

        counter_cache_hits = self._counter_cls(
            name="vllm:hisparse_cache_hits",
            documentation="Number of HiSparse device hot-buffer hits.",
            labelnames=labelnames,
        )
        counter_cache_misses = self._counter_cls(
            name="vllm:hisparse_cache_misses",
            documentation="Number of HiSparse device hot-buffer misses.",
            labelnames=labelnames,
        )
        counter_host_to_device_bytes = self._counter_cls(
            name="vllm:hisparse_host_to_device_bytes",
            documentation=(
                "Bytes transferred from host KV storage to HiSparse hot buffers."
            ),
            labelnames=labelnames,
        )
        self.hisparse_counters: dict[str, Any] = {
            "cache_hits": create_metric_per_engine(
                counter_cache_hits, self.per_engine_labelvalues
            ),
            "cache_misses": create_metric_per_engine(
                counter_cache_misses, self.per_engine_labelvalues
            ),
            "host_to_device_bytes": create_metric_per_engine(
                counter_host_to_device_bytes, self.per_engine_labelvalues
            ),
        }

        gauge_host_blocks_used = self._gauge_cls(
            name="vllm:hisparse_host_blocks_used",
            documentation="HiSparse host KV blocks backing live or cached prefixes.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        gauge_host_blocks_total = self._gauge_cls(
            name="vllm:hisparse_host_blocks_total",
            documentation="Total HiSparse host KV blocks (fixed at startup).",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        gauge_host_blocks_usage = self._gauge_cls(
            name="vllm:hisparse_host_blocks_usage",
            documentation="HiSparse host KV pool usage. 1 means 100 percent usage.",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        gauge_pending_spills = self._gauge_cls(
            name="vllm:hisparse_pending_spills",
            documentation=(
                "HiSparse page transfers enqueued but not yet completed on host."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.hisparse_gauges: dict[str, Any] = {
            "host_blocks_used": create_metric_per_engine(
                gauge_host_blocks_used, self.per_engine_labelvalues
            ),
            "host_blocks_total": create_metric_per_engine(
                gauge_host_blocks_total, self.per_engine_labelvalues
            ),
            "host_blocks_usage": create_metric_per_engine(
                gauge_host_blocks_usage, self.per_engine_labelvalues
            ),
            "pending_spills": create_metric_per_engine(
                gauge_pending_spills, self.per_engine_labelvalues
            ),
        }

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for name, counter in self.hisparse_counters.items():
            for value in transfer_stats_data.get(name, []):
                counter[engine_idx].inc(value)
        for name, gauge in self.hisparse_gauges.items():
            values = transfer_stats_data.get(name, [])
            if values:
                # Level values: each observation replaces the previous one.
                gauge[engine_idx].set(values[-1])
