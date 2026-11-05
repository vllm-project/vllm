# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Metrics for encoder-cache CPU and filesystem tiering."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.metrics import (
    ECConnectorPromMetrics,
    ECConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

COUNTER_KEYS = (
    "cpu_hits",
    "recomputes",
    "fs_reads",
    "fs_prefetch_reads",
    "fs_cpu_alloc_failures",
    "fs_read_completions",
    "fs_read_failures",
    "fs_read_bytes",
    "fs_writes",
    "fs_write_completions",
    "fs_write_failures",
    "fs_write_bytes",
    "fs_evictions",
    "fs_recoveries",
)
HISTOGRAM_KEYS = (
    "fs_read_queue_seconds",
    "fs_read_transfer_seconds",
    "fs_read_publish_seconds",
    "fs_read_completion_delay_seconds",
    "fs_request_wait_seconds",
    "fs_prefetch_submit_seconds",
    "fs_admission_check_seconds",
    "fs_ready_to_load_seconds",
    "h2d_host_submit_seconds",
    "h2d_device_seconds",
    "d2h_host_submit_seconds",
    "d2h_device_seconds",
    "fs_write_queue_seconds",
    "fs_write_transfer_seconds",
)
GAUGE_KEYS = (
    "fs_used_bytes",
    "fs_capacity_bytes",
    "pending_read_jobs",
    "pending_read_bytes",
    "pending_write_jobs",
    "pending_write_bytes",
    "oldest_read_queue_seconds",
    "oldest_write_queue_seconds",
    "fs_recovery_seconds",
    "fs_recovered_entries",
    "fs_recovered_bytes",
)


@dataclass
class ECTieringStats(ECConnectorStats):
    """Serializable interval counters, timings, and pressure snapshots."""

    def __post_init__(self) -> None:
        if not self.data:
            self.reset()

    def reset(self) -> None:
        self.data = {
            "counters": {key: 0 for key in COUNTER_KEYS},
            "histograms": {key: [] for key in HISTOGRAM_KEYS},
            "gauges": {key: 0 for key in GAUGE_KEYS},
            "gauges_updated": False,
        }

    def increment(self, key: str, value: int = 1) -> None:
        self.data["counters"][key] += value

    def observe(self, key: str, value: float) -> None:
        self.data["histograms"][key].append(value)

    def set_gauge(self, key: str, value: int | float) -> None:
        self.data["gauges"][key] = value
        self.data["gauges_updated"] = True

    def clone_and_reset(self) -> ECTieringStats:
        old = copy.deepcopy(self)
        self.reset()
        return old

    def aggregate(self, other: ECConnectorStats) -> ECConnectorStats:
        assert isinstance(other, ECTieringStats)
        for key, value in other.data["counters"].items():
            self.data["counters"][key] += value
        for key, values in other.data["histograms"].items():
            self.data["histograms"][key].extend(values)
        if other.data.get("gauges_updated", False):
            self.data["gauges"].update(other.data["gauges"])
        self.data["gauges_updated"] |= other.data.get("gauges_updated", False)
        return self

    def reduce(self) -> dict[str, int | float]:
        reduced = dict(self.data["counters"])
        reduced.update(self.data["gauges"])
        for key, values in self.data["histograms"].items():
            reduced[f"{key}_count"] = len(values)
            reduced[f"{key}_sum"] = sum(values)
        return reduced

    def is_empty(self) -> bool:
        return (
            not self.data.get("gauges_updated", False)
            and not any(self.data["counters"].values())
            and not any(self.data["histograms"].values())
        )


class ECTieringPromMetrics(ECConnectorPromMetrics):
    """Prometheus projection of :class:`ECTieringStats`."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._counters = {
            key: create_metric_per_engine(
                self._counter_cls(
                    name=f"vllm:ec_tiering_{key}",
                    documentation=f"Encoder-cache tiering {key.replace('_', ' ')}.",
                    labelnames=labelnames,
                ),
                per_engine_labelvalues,
            )
            for key in COUNTER_KEYS
        }
        latency_buckets = (
            0.00001,
            0.00005,
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1,
            5,
        )
        self._histograms = {
            key: create_metric_per_engine(
                self._histogram_cls(
                    name=f"vllm:ec_tiering_{key}",
                    documentation=f"Encoder-cache tiering {key.replace('_', ' ')}.",
                    buckets=latency_buckets,
                    labelnames=labelnames,
                ),
                per_engine_labelvalues,
            )
            for key in HISTOGRAM_KEYS
        }
        self._gauges = {
            key: create_metric_per_engine(
                self._gauge_cls(
                    name=f"vllm:ec_tiering_{key}",
                    documentation=f"Encoder-cache tiering {key.replace('_', ' ')}.",
                    multiprocess_mode="mostrecent",
                    labelnames=labelnames,
                ),
                per_engine_labelvalues,
            )
            for key in GAUGE_KEYS
        }

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for key, value in transfer_stats_data["counters"].items():
            self._counters[key][engine_idx].inc(value)
        for key, values in transfer_stats_data["histograms"].items():
            histogram = self._histograms[key][engine_idx]
            for value in values:
                histogram.observe(value)
        if transfer_stats_data.get("gauges_updated", False):
            for key, value in transfer_stats_data["gauges"].items():
                self._gauges[key][engine_idx].set(value)
