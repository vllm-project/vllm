# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for SimpleCPUOffloadConnector.

The stats payload uses the same self-describing wire format as the
offloading connector, so it survives IPC serialization without the
metric classes on the receiving side.
"""

from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)


class MetricName:
    """Metric names for SimpleCPUOffloadConnector."""

    SAVE_OUTCOMES = "vllm:simple_kv_offload_save_outcomes_total"
    LOAD_BLOCKS = "vllm:simple_kv_offload_load_blocks_total"
    USED_BLOCKS = "vllm:simple_kv_offload_used_blocks"
    PENDING_STORE_BLOCKS = "vllm:simple_kv_offload_pending_store_blocks"
    INFO = "vllm:simple_kv_offload_info"


# outcome label value -> BoundaryStoreStats field name.
# ``published`` is the sum of all outcomes and is not labeled.
OUTCOME_TO_FIELD = {
    "stored": "stored",
    "dropped_cpu_full": "dropped_cpu_full",
    "dropped_request_gone": "dropped_request_gone",
    "dropped_null_block": "dropped_null_block",
    "dropped_not_hashed": "dropped_not_hashed",
    "skipped_already_cached": "skipped_already_cached",
    "skipped_in_flight": "skipped_in_flight",
}

LOAD_PHASE_ISSUED = "issued"
LOAD_PHASE_COMPLETED = "completed"

INFO_LABELS = ["backend", "page_cache", "lazy_offload", "capacity_blocks"]


class _MetricType:
    """Type tags embedded in the serialized stats payload."""

    COUNTER = "counter"
    GAUGE = "gauge"


class _StatsKey:
    """Top-level keys in the serialized stats dict."""

    # Maps metric name -> _MetricType value
    TYPES = "types"
    # Maps metric name -> {label values tuple -> observed value}
    DATA = "data"


@dataclass
class SimpleCPUOffloadStats(KVConnectorStats):
    """Stats payload for SimpleCPUOffloadConnector.

    The ``data`` dict is structured as::

        {
            _StatsKey.TYPES: {name: _MetricType.*, ...},
            _StatsKey.DATA:  {name: {labelvalues: value, ...}, ...},
        }

    Counter values are summed when aggregating, gauge values keep the
    latest snapshot, and unlabeled metrics use ``()`` as their
    labelvalues tuple.
    """

    def __post_init__(self):
        if _StatsKey.DATA not in self.data:
            self.reset()

    def reset(self):
        self.data: dict[str, Any] = {
            _StatsKey.TYPES: {},
            _StatsKey.DATA: {},
        }

    @property
    def _types(self) -> dict[str, str]:
        return self.data[_StatsKey.TYPES]

    @property
    def _values(self) -> dict[str, Any]:
        return self.data[_StatsKey.DATA]

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if other.is_empty():
            return self
        assert isinstance(other, SimpleCPUOffloadStats)
        for key, other_label_values in other._values.items():
            type_str = other._types.get(key)
            if type_str is None:
                raise AssertionError(f"Unknown simple KV offload stats key: {key}")
            self._types.setdefault(key, type_str)
            current_label_values = self._values.setdefault(key, {})
            for labelvalues, value in other_label_values.items():
                if type_str == _MetricType.COUNTER:
                    assert isinstance(value, int | float)
                    current_label_values[labelvalues] = (
                        current_label_values.get(labelvalues, 0) + value
                    )
                elif type_str == _MetricType.GAUGE:
                    assert isinstance(value, int | float)
                    current_label_values[labelvalues] = value
                else:
                    raise AssertionError(
                        f"Unknown metric type '{type_str}' for key: {key}"
                    )
        return self

    def reduce(self) -> dict[str, int | float]:
        """Reduce the interval observations to a flat summary for logging."""
        return_dict: dict[str, int | float] = {}
        for key, label_value_map in self._values.items():
            if key == MetricName.INFO:
                # Constant config fingerprint; exclude from the log summary.
                continue
            type_str = self._types.get(key)
            if type_str is None:
                raise AssertionError(f"Unknown simple KV offload stats key: {key}")
            for labelvalues, value in label_value_map.items():
                key_with_labels = f"{key}:{labelvalues}" if labelvalues else key
                if type_str in (_MetricType.COUNTER, _MetricType.GAUGE):
                    assert isinstance(value, int | float)
                    return_dict[key_with_labels] = value
                else:
                    raise AssertionError(
                        f"Unknown metric type '{type_str}' for key: {key}"
                    )
        return return_dict

    def is_empty(self) -> bool:
        return not self.data.get(_StatsKey.DATA)

    def increase_counter(
        self,
        counter_name: str,
        counter_increase_value: int | float = 1,
        labelvalues: tuple[str, ...] = (),
    ) -> None:
        """Increase a counter on the stats payload."""
        self._types.setdefault(counter_name, _MetricType.COUNTER)
        counter_values = self._values.setdefault(counter_name, {})
        counter_values[labelvalues] = (
            counter_values.get(labelvalues, 0) + counter_increase_value
        )

    def set_gauge(
        self,
        gauge_name: str,
        gauge_value: int | float,
        labelvalues: tuple[str, ...] = (),
    ) -> None:
        """Set a gauge snapshot on the stats payload."""
        self._types.setdefault(gauge_name, _MetricType.GAUGE)
        gauge_values = self._values.setdefault(gauge_name, {})
        gauge_values[labelvalues] = gauge_value


class SimpleCPUOffloadPromMetrics(KVConnectorPromMetrics):
    """Prometheus registration and recording for SimpleCPUOffloadConnector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        # (engine_idx, metric_name, labelvalues) -> bound child metric
        self._metrics: dict[tuple[int, str, tuple[str, ...]], PromMetricT] = {}

        self._counter_save_outcomes = self._counter_cls(
            name="vllm:simple_kv_offload_save_outcomes_total",
            documentation=(
                "Store admission decisions in SimpleCPUOffloadConnector, by "
                "outcome. Outcomes classify eager-mode boundary hand-off "
                "stores; lazy-mode stores are not classified."
            ),
            labelnames=self._labelnames + ["outcome"],
        )
        self._counter_load_blocks = self._counter_cls(
            name="vllm:simple_kv_offload_load_blocks_total",
            documentation=(
                "KV blocks involved in offload loads by phase: issued when "
                "dispatched to workers, completed when the load finished "
                "successfully."
            ),
            labelnames=self._labelnames + ["phase"],
        )
        self._gauge_used_blocks = self._gauge_cls(
            name="vllm:simple_kv_offload_used_blocks",
            documentation=(
                "Offload-pool blocks currently pinned by in-flight "
                "transfers or cache hits (capacity minus free). Evictable "
                "cached blocks are not counted; capacity_blocks is on "
                "vllm:simple_kv_offload_info."
            ),
            labelnames=self._labelnames,
            multiprocess_mode="mostrecent",
        )
        self._gauge_pending_store_blocks = self._gauge_cls(
            name="vllm:simple_kv_offload_pending_store_blocks",
            documentation=(
                "Offload blocks in pending stores: queued for dispatch, "
                "issued to workers, or awaiting release after a cache "
                "reset. A persistently growing value indicates a stuck "
                "transfer."
            ),
            labelnames=self._labelnames,
            multiprocess_mode="mostrecent",
        )
        self._gauge_info = self._gauge_cls(
            name="vllm:simple_kv_offload_info",
            documentation=(
                "SimpleCPUOffloadConnector deployment facts. Value is "
                "always 1. backend is cpu or disk; page_cache and "
                "lazy_offload are true/false; capacity_blocks is the "
                "offload pool size."
            ),
            labelnames=self._labelnames + INFO_LABELS,
            multiprocess_mode="mostrecent",
        )

        self._defs: dict[str, PromMetricT] = {
            MetricName.SAVE_OUTCOMES: self._counter_save_outcomes,
            MetricName.LOAD_BLOCKS: self._counter_load_blocks,
            MetricName.USED_BLOCKS: self._gauge_used_blocks,
            MetricName.PENDING_STORE_BLOCKS: self._gauge_pending_store_blocks,
            MetricName.INFO: self._gauge_info,
        }
        self._num_label_values: dict[str, int] = {
            MetricName.SAVE_OUTCOMES: 1,
            MetricName.LOAD_BLOCKS: 1,
            MetricName.USED_BLOCKS: 0,
            MetricName.PENDING_STORE_BLOCKS: 0,
            MetricName.INFO: len(INFO_LABELS),
        }

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Observe transfer statistics."""
        metric_types = transfer_stats_data.get(_StatsKey.TYPES, {})
        metric_data = transfer_stats_data.get(_StatsKey.DATA, {})
        for key, label_value_map in metric_data.items():
            prom_metric = self._defs.get(key)
            if prom_metric is None:
                raise AssertionError(f"Unknown simple KV offload stats key: {key}")
            type_str = metric_types.get(key)
            for labelvalues, value in label_value_map.items():
                if len(labelvalues) != self._num_label_values[key]:
                    raise AssertionError(
                        f"Metric {key} expects "
                        f"{self._num_label_values[key]} labels, "
                        f"got {len(labelvalues)}"
                    )
                child = self._metrics.get((engine_idx, key, labelvalues))
                if child is None:
                    engine_labelvalues = self.per_engine_labelvalues[engine_idx]
                    child = prom_metric.labels(
                        *(engine_labelvalues + list(labelvalues))
                    )
                    self._metrics[(engine_idx, key, labelvalues)] = child
                if type_str == _MetricType.COUNTER:
                    assert isinstance(value, int | float)
                    child.inc(value)
                elif type_str == _MetricType.GAUGE:
                    assert isinstance(value, int | float)
                    child.set(value)
                else:
                    raise AssertionError(
                        f"Unknown metric type '{type_str}' for key: {key}"
                    )
