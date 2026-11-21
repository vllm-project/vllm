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
from vllm.v1.metrics.buckets import histogram_buckets
from vllm.v1.metrics.stats import KVCacheEvictionEvent
from vllm.v1.metrics.utils import create_metric_per_engine

_HISPARSE_LEVEL_KEYS = frozenset({"host_cache_usage_perc", "pending_page_transfers"})


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
        self.data: dict[str, list[int | float]] = {
            "cache_hits": [],
            "cache_misses": [],
            "host_to_device_bytes": [],
            "host_cache_usage_perc": [],
            "pending_page_transfers": [],
            "host_block_lifetime_seconds": [],
            "host_block_idle_before_evict_seconds": [],
            "host_block_reuse_gap_seconds": [],
        }

    def record_snapshot(self, hits: int, misses: int, host_to_device_bytes: int):
        self.data["cache_hits"].append(hits)
        self.data["cache_misses"].append(misses)
        self.data["host_to_device_bytes"].append(host_to_device_bytes)

    def record_host_usage(self, usage: float, pending_page_transfers: int) -> None:
        self.data["host_cache_usage_perc"].append(usage)
        self.data["pending_page_transfers"].append(pending_page_transfers)

    def record_host_evictions(self, events: list[KVCacheEvictionEvent]) -> None:
        for event in events:
            self.data["host_block_lifetime_seconds"].append(event.lifetime_seconds)
            self.data["host_block_idle_before_evict_seconds"].append(event.idle_seconds)
            self.data["host_block_reuse_gap_seconds"].extend(event.reuse_gaps_seconds)

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                if k in _HISPARSE_LEVEL_KEYS:
                    accumulator.extend(v[-1:])
                else:
                    accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        # Compute compact representative stats suitable for CLI logging.
        reduced: dict[str, int | float] = {
            "HiSparse hot-buffer hits": sum(self.data["cache_hits"]),
            "HiSparse hot-buffer misses": sum(self.data["cache_misses"]),
            "HiSparse host-to-device bytes": sum(self.data["host_to_device_bytes"]),
        }
        if self.data["host_cache_usage_perc"]:
            usage = self.data["host_cache_usage_perc"][-1]
            reduced["HiSparse host KV cache usage %"] = round(100 * usage, 1)
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

        gauge_host_cache_usage = self._gauge_cls(
            name="vllm:hisparse_host_cache_usage_perc",
            documentation=(
                "HiSparse host KV-cache usage. 1 means 100 percent usage. "
                "Evictable cached blocks count as free."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        gauge_pending_page_transfers = self._gauge_cls(
            name="vllm:hisparse_pending_page_transfers",
            documentation=(
                "HiSparse page transfers (write-backs and restores) not yet completed."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.hisparse_gauges: dict[str, Any] = {
            "host_cache_usage_perc": create_metric_per_engine(
                gauge_host_cache_usage, self.per_engine_labelvalues
            ),
            "pending_page_transfers": create_metric_per_engine(
                gauge_pending_page_transfers, self.per_engine_labelvalues
            ),
        }

        self.hisparse_histograms: dict[str, Any] = {}
        if vllm_config.observability_config.kv_cache_metrics:
            residency_buckets = histogram_buckets(
                "kv_cache_residency",
                overrides=vllm_config.observability_config.custom_histogram_buckets,
            )
            histogram_host_block_lifetime = self._histogram_cls(
                name="vllm:hisparse_host_block_lifetime_seconds",
                documentation=(
                    "Histogram of HiSparse host KV block lifetime from allocation "
                    "to eviction. Sampled metrics (controlled by "
                    "--kv-cache-metrics-sample)."
                ),
                buckets=residency_buckets,
                labelnames=labelnames,
            )
            histogram_host_block_idle_before_evict = self._histogram_cls(
                name="vllm:hisparse_host_block_idle_before_evict_seconds",
                documentation=(
                    "Histogram of HiSparse host KV block idle time before eviction. "
                    "Sampled metrics (controlled by --kv-cache-metrics-sample)."
                ),
                buckets=residency_buckets,
                labelnames=labelnames,
            )
            histogram_host_block_reuse_gap = self._histogram_cls(
                name="vllm:hisparse_host_block_reuse_gap_seconds",
                documentation=(
                    "Histogram of time gaps between consecutive HiSparse host KV "
                    "block accesses. Sampled metrics (controlled by "
                    "--kv-cache-metrics-sample)."
                ),
                buckets=residency_buckets,
                labelnames=labelnames,
            )
            self.hisparse_histograms = {
                "host_block_lifetime_seconds": create_metric_per_engine(
                    histogram_host_block_lifetime, self.per_engine_labelvalues
                ),
                "host_block_idle_before_evict_seconds": create_metric_per_engine(
                    histogram_host_block_idle_before_evict, self.per_engine_labelvalues
                ),
                "host_block_reuse_gap_seconds": create_metric_per_engine(
                    histogram_host_block_reuse_gap, self.per_engine_labelvalues
                ),
            }

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for name, counter in self.hisparse_counters.items():
            for value in transfer_stats_data.get(name, []):
                counter[engine_idx].inc(value)
        for name, gauge in self.hisparse_gauges.items():
            values = transfer_stats_data.get(name, [])
            if values:
                gauge[engine_idx].set(values[-1])
        for name, histogram in self.hisparse_histograms.items():
            for value in transfer_stats_data.get(name, []):
                histogram[engine_idx].observe(value)
