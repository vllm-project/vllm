# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats container for the CPU EC Connector.

Mirrors OffloadingConnectorStats
(vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py), but
rooted in ECConnectorStats. Flat metric names are keys; a metric-type
registry is needed (rather than inferring the type from the value's shape)
because a gauge's value is a plain number, same shape as a counter's, but
aggregates differently (latest-wins vs. sum).
"""

from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.metrics import (
    ECConnectorPromMetrics,
    ECConnectorStats,
    PromMetric,
    PromMetricT,
)


class _MetricType:
    """Type tags embedded in the serialized stats payload."""

    COUNTER = "counter"
    HISTOGRAM = "histogram"


class _StatsKey:
    """Top-level keys in the serialized stats dict."""

    # Maps metric name -> _MetricType value
    TYPES = "types"
    # Maps metric name -> observed value (number for a counter, list for a
    # histogram)
    DATA = "data"


@dataclass
class ECCPUConnectorStats(ECConnectorStats):
    """Counter values are aggregated by summing; histogram values are lists of
    observed samples.
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

    def is_empty(self) -> bool:
        return not self._values

    def increase_counter(
        self, counter_name: str, increase_value: int | float = 1
    ) -> None:
        self._types.setdefault(counter_name, _MetricType.COUNTER)
        self._values[counter_name] = self._values.get(counter_name, 0) + increase_value

    def observe_histogram(self, histogram_name: str, value: int | float) -> None:
        self._types.setdefault(histogram_name, _MetricType.HISTOGRAM)
        self._values.setdefault(histogram_name, []).append(value)

    def aggregate(self, other: "ECConnectorStats") -> "ECConnectorStats":
        if other.is_empty():
            return self
        assert isinstance(other, ECCPUConnectorStats)
        for key, other_value in other._values.items():
            type_str = other._types[key]
            self._types.setdefault(key, type_str)
            if type_str == _MetricType.HISTOGRAM:
                assert isinstance(other_value, list)
                self._values.setdefault(key, []).extend(other_value)
            elif type_str == _MetricType.COUNTER:
                assert isinstance(other_value, int | float)
                self._values[key] = self._values.get(key, 0) + other_value
            else:
                raise AssertionError(f"Unknown metric type '{type_str}' for key: {key}")
        return self

    def reduce(self) -> dict[str, int | float]:
        return_dict: dict[str, int | float] = {}
        for key, value in self._values.items():
            type_str = self._types[key]
            if type_str == _MetricType.HISTOGRAM:
                assert isinstance(value, list)
                return_dict[f"{key}_count"] = len(value)
                return_dict[f"{key}_sum"] = sum(value)
            elif type_str == _MetricType.COUNTER:
                assert isinstance(value, int | float)
                return_dict[key] = value
            else:
                raise AssertionError(f"Unknown metric type '{type_str}' for key: {key}")
        return return_dict


class ECCPUMetricName:
    """Flat metric names for the GPU<->CPU mmap save/load path (Tier 1)."""

    SAVE_BYTES = "vllm:ec_cpu_save_bytes"
    SAVE_TIME = "vllm:ec_cpu_save_time_seconds"
    SAVE_SIZE = "vllm:ec_cpu_save_size_bytes"
    LOAD_BYTES = "vllm:ec_cpu_load_bytes"
    LOAD_TIME = "vllm:ec_cpu_load_time_seconds"
    LOAD_SIZE = "vllm:ec_cpu_load_size_bytes"


_COUNTER_METRICS: dict[str, str] = {
    ECCPUMetricName.SAVE_BYTES: (
        "Total bytes copied from GPU to the CPU mmap region (encoder cache save)."
    ),
    ECCPUMetricName.SAVE_TIME: (
        "Total time spent copying GPU to CPU for encoder cache saves, in seconds."
    ),
    ECCPUMetricName.LOAD_BYTES: (
        "Total bytes copied from the CPU mmap region to GPU (encoder cache load)."
    ),
    ECCPUMetricName.LOAD_TIME: (
        "Total time spent copying CPU to GPU for encoder cache loads, in seconds."
    ),
}
_HISTOGRAM_METRICS: dict[str, str] = {
    ECCPUMetricName.SAVE_SIZE: (
        "Histogram of per-batch GPU->CPU encoder cache save size, in bytes."
    ),
    ECCPUMetricName.LOAD_SIZE: (
        "Histogram of per-batch CPU->GPU encoder cache load size, in bytes."
    ),
}
# Sized for encoder-cache save/load batches, not KV-cache block transfers:
# a single mainstream-VLM embedding is ~1-20MB (e.g. LLaVA-1.5 ~4.7MB,
# Qwen2-VL ~8.8MB, Pixtral ~42MB), and a step's flush can batch several
# concurrent requests' embeddings into one transfer.
_TRANSFER_SIZE_BUCKETS = (
    0.5e6,
    1e6,
    2e6,
    5e6,
    10e6,
    20e6,
    50e6,
    100e6,
    250e6,
)


class ECCPUConnectorProm(ECConnectorPromMetrics):
    """Registers the CPU EC connector's Prometheus metrics and replays a
    serialized ECCPUConnectorStats payload into them."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        metric_defs: dict[str, PromMetricT] = {
            name: self._counter_cls(name=name, documentation=doc, labelnames=labelnames)
            for name, doc in _COUNTER_METRICS.items()
        }
        metric_defs.update(
            {
                name: self._histogram_cls(
                    name=name,
                    documentation=doc,
                    buckets=_TRANSFER_SIZE_BUCKETS,
                    labelnames=labelnames,
                )
                for name, doc in _HISTOGRAM_METRICS.items()
            }
        )

        # (engine_idx, metric_name) -> metric bound to that engine's labels.
        self._bound: dict[tuple[int, str], PromMetricT] = {
            (engine_idx, name): metric.labels(*labelvalues)
            for engine_idx, labelvalues in per_engine_labelvalues.items()
            for name, metric in metric_defs.items()
        }

    def observe(self, transfer_stats_data: dict, engine_idx: int = 0) -> None:
        metric_types = transfer_stats_data.get(_StatsKey.TYPES, {})
        metric_data = transfer_stats_data.get(_StatsKey.DATA, {})
        for key, value in metric_data.items():
            metric = self._bound[(engine_idx, key)]
            if metric_types[key] == _MetricType.HISTOGRAM:
                assert isinstance(value, list)
                for sample in value:
                    metric.observe(sample)
            else:
                assert isinstance(value, int | float)
                metric.inc(value)
