# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
)
from vllm.v1.kv_offload.factory import OffloadingSpecFactory


class _TransferMetricName:
    """Flat metric names for GPU↔offload-medium transfer operations."""

    LOAD_BYTES = "vllm:kv_offload_load_bytes"
    LOAD_TIME = "vllm:kv_offload_load_time"
    LOAD_SIZE = "vllm:kv_offload_load_size"
    STORE_BYTES = "vllm:kv_offload_store_bytes"
    STORE_TIME = "vllm:kv_offload_store_time"
    STORE_SIZE = "vllm:kv_offload_store_size"


class _TransferType:
    """Transfer direction labels for deprecated CPU offload metrics."""

    LOAD = "CPU_to_GPU"
    STORE = "GPU_to_CPU"
    ALL = (LOAD, STORE)


TRANSFER_SIZE_BUCKETS = (
    1e6,
    5e6,
    10e6,
    20e6,
    40e6,
    60e6,
    80e6,
    100e6,
    150e6,
    200e6,
)


def get_connector_metric_definitions() -> dict[str, OffloadingMetricMetadata]:
    return {
        _TransferMetricName.LOAD_BYTES: OffloadingCounterMetadata(
            documentation="Total bytes loaded from offload storage to GPU.",
        ),
        _TransferMetricName.LOAD_TIME: OffloadingCounterMetadata(
            documentation="Total load time from offload storage to GPU, in seconds.",
        ),
        _TransferMetricName.LOAD_SIZE: OffloadingHistogramMetadata(
            documentation="Histogram of KV offload load operation size, in bytes.",
            buckets=TRANSFER_SIZE_BUCKETS,
        ),
        _TransferMetricName.STORE_BYTES: OffloadingCounterMetadata(
            documentation="Total bytes stored from GPU to offload storage.",
        ),
        _TransferMetricName.STORE_TIME: OffloadingCounterMetadata(
            documentation="Total store time from GPU to offload storage, in seconds.",
        ),
        _TransferMetricName.STORE_SIZE: OffloadingHistogramMetadata(
            documentation="Histogram of KV offload store operation size, in bytes.",
            buckets=TRANSFER_SIZE_BUCKETS,
        ),
    }


_DEPRECATED_TOTAL_BYTES = "vllm:kv_offload_total_bytes"
_DEPRECATED_TOTAL_TIME = "vllm:kv_offload_total_time"
_DEPRECATED_SIZE = "vllm:kv_offload_size"

# Deprecated legacy transfer metrics, kept during the migration to the flat
# metric names above. These stay in a separate definition block because they
# use a transfer_type label, but are emitted from the same flat stats payload
# for compatibility.
_DEPRECATED_CONNECTOR_METRIC_DEFINITIONS: dict[str, OffloadingMetricMetadata] = {
    _DEPRECATED_TOTAL_BYTES: OffloadingCounterMetadata(
        documentation="Number of bytes offloaded by KV connector",
    ),
    _DEPRECATED_TOTAL_TIME: OffloadingCounterMetadata(
        documentation="Total time measured by all KV offloading operations",
    ),
    _DEPRECATED_SIZE: OffloadingHistogramMetadata(
        documentation="Histogram of KV offload transfer size, in bytes.",
        buckets=TRANSFER_SIZE_BUCKETS,
    ),
}


class _MetricType:
    """Type tags embedded in the serialized stats payload."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class _StatsKey:
    """Top-level keys in the serialized stats dict."""

    # Maps metric name -> _MetricType value
    TYPES = "types"
    # Maps metric name -> observed value (number or list)
    DATA = "data"


@dataclass
class OffloadingConnectorStats(KVConnectorStats):
    """
    Offloading connector stats use flat metric names as keys.

    The ``data`` dict is structured using ``_StatsKey`` / ``_MetricType``::

        {
            _StatsKey.TYPES: {name: _MetricType.*, ...},
            _StatsKey.DATA:  {name: value, ...},
        }

    This structure is self-describing: it survives IPC serialization
    without needing the full ``OffloadingMetricMetadata`` objects on the
    receiving side.

    Counter values are aggregated by summing, gauge values use the latest
    snapshot, and histogram values are lists of observed samples.
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
        assert isinstance(other, OffloadingConnectorStats)
        other_types = other._types
        other_values = other._values
        for key, value in other_values.items():
            type_str = other_types.get(key)
            if type_str is None:
                raise AssertionError(f"Unknown offloading stats key: {key}")
            self._types.setdefault(key, type_str)
            if type_str == _MetricType.HISTOGRAM:
                assert isinstance(value, list)
                if key not in self._values:
                    self._values[key] = value
                else:
                    assert isinstance(self._values[key], list)
                    self._values[key].extend(value)
            elif type_str == _MetricType.COUNTER:
                assert isinstance(value, int | float)
                self._values[key] = self._values.get(key, 0) + value
            elif type_str == _MetricType.GAUGE:
                assert isinstance(value, int | float)
                self._values[key] = value
            else:
                raise AssertionError(f"Unknown metric type '{type_str}' for key: {key}")
        return self

    def reduce(self) -> dict[str, int | float]:
        """
        Reduce the observations collected during a time interval to one or
        more representative values (eg avg/median/sum of the series).
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        return_dict: dict[str, int | float] = {}
        for key, value in self._values.items():
            type_str = self._types.get(key)
            if type_str is None:
                raise AssertionError(f"Unknown offloading stats key: {key}")
            if type_str == _MetricType.HISTOGRAM:
                assert isinstance(value, list)
                return_dict[f"{key}_count"] = len(value)
                return_dict[f"{key}_sum"] = sum(value)
            elif type_str in (_MetricType.COUNTER, _MetricType.GAUGE):
                assert isinstance(value, int | float)
                return_dict[key] = value
            else:
                raise AssertionError(f"Unknown metric type '{type_str}' for key: {key}")
        return return_dict

    def is_empty(self) -> bool:
        return not self.data.get(_StatsKey.DATA)

    def increase_counter(
        self, counter_name: str, counter_increase_value: int | float
    ) -> None:
        """Increase a counter on the stats payload."""
        self._types.setdefault(counter_name, _MetricType.COUNTER)
        self._values[counter_name] = (
            self._values.get(counter_name, 0) + counter_increase_value
        )

    def set_gauge(self, gauge_name: str, gauge_value: int | float) -> None:
        """Set a gauge snapshot on the stats payload."""
        self._types.setdefault(gauge_name, _MetricType.GAUGE)
        self._values[gauge_name] = gauge_value

    def observe_histogram(
        self, histogram_name: str, histogram_value: int | float
    ) -> None:
        """Record a histogram observation on the stats payload."""
        self._types.setdefault(histogram_name, _MetricType.HISTOGRAM)
        self._values.setdefault(histogram_name, []).append(histogram_value)


class OffloadPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        # (engine_idx, transfer_type) -> (metric with bounded labels)
        self.histogram_transfer_size: dict[tuple[int, str], PromMetricT] = {}
        self.counter_kv_bytes: dict[tuple[int, str], PromMetricT] = {}
        self.counter_kv_transfer_time: dict[tuple[int, str], PromMetricT] = {}
        spec_cls = OffloadingSpecFactory.get_spec_cls(vllm_config)
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        extra_config = kv_transfer_config.kv_connector_extra_config
        self._offloading_metric_metadata: dict[str, OffloadingMetricMetadata] = {
            **spec_cls.build_metric_definitions(extra_config),
            **get_connector_metric_definitions(),
        }
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

        self._observe_deprecated_metrics = issubclass(spec_cls, CPUOffloadingSpec)
        self._offloading_metric_defs: dict[str, PromMetricT] = {}
        self.offloading_metrics: dict[tuple[int, str], PromMetricT] = {}

        self._counter_kv_bytes = self._counter_cls(
            name=_DEPRECATED_TOTAL_BYTES,
            documentation=_DEPRECATED_CONNECTOR_METRIC_DEFINITIONS[
                _DEPRECATED_TOTAL_BYTES
            ].documentation,
            labelnames=labelnames + ["transfer_type"],
        )

        self._counter_kv_transfer_time = self._counter_cls(
            name=_DEPRECATED_TOTAL_TIME,
            documentation=_DEPRECATED_CONNECTOR_METRIC_DEFINITIONS[
                _DEPRECATED_TOTAL_TIME
            ].documentation,
            labelnames=labelnames + ["transfer_type"],
        )

        deprecated_size_metadata = _DEPRECATED_CONNECTOR_METRIC_DEFINITIONS[
            _DEPRECATED_SIZE
        ]
        assert isinstance(deprecated_size_metadata, OffloadingHistogramMetadata)
        self._histogram_transfer_size = self._histogram_cls(
            name=_DEPRECATED_SIZE,
            documentation=deprecated_size_metadata.documentation,
            buckets=deprecated_size_metadata.buckets,
            labelnames=labelnames + ["transfer_type"],
        )

        for engine_idx, labelvalues in per_engine_labelvalues.items():
            for transfer_type in _TransferType.ALL:
                bounded_labelvalues = labelvalues + [transfer_type]
                self.histogram_transfer_size[(engine_idx, transfer_type)] = (
                    self._histogram_transfer_size.labels(*bounded_labelvalues)
                )
                self.counter_kv_bytes[(engine_idx, transfer_type)] = (
                    self._counter_kv_bytes.labels(*bounded_labelvalues)
                )
                self.counter_kv_transfer_time[(engine_idx, transfer_type)] = (
                    self._counter_kv_transfer_time.labels(*bounded_labelvalues)
                )

        for metric_name, metadata in self._offloading_metric_metadata.items():
            self._offloading_metric_defs[metric_name] = self._create_metric(
                metric_name, metadata
            )
            for engine_idx, labelvalues in per_engine_labelvalues.items():
                self.offloading_metrics[(engine_idx, metric_name)] = (
                    self._offloading_metric_defs[metric_name].labels(*labelvalues)
                )

    def _create_metric(
        self, metric_name: str, metadata: OffloadingMetricMetadata
    ) -> Any:
        kwargs: dict[str, Any] = {
            "name": metric_name,
            "documentation": metadata.documentation,
            "labelnames": self._labelnames,
        }
        if isinstance(metadata, OffloadingCounterMetadata):
            metric_cls = self._counter_cls
        elif isinstance(metadata, OffloadingGaugeMetadata):
            metric_cls = self._gauge_cls
        elif isinstance(metadata, OffloadingHistogramMetadata):
            metric_cls = self._histogram_cls
            if metadata.buckets is not None:
                kwargs["buckets"] = metadata.buckets
        else:
            raise AssertionError(f"Unknown offloading metric metadata: {metadata}")
        return metric_cls(**kwargs)

    def _increase_counter(
        self, metric_name: str, value: int | float, engine_idx: int
    ) -> None:
        self.offloading_metrics[(engine_idx, metric_name)].inc(value)
        if not self._observe_deprecated_metrics:
            return
        # Keep deprecated CPU offload transfer metrics updated during the
        # transition to flat metric names.
        if metric_name == _TransferMetricName.LOAD_BYTES:
            self.counter_kv_bytes[(engine_idx, _TransferType.LOAD)].inc(value)
        elif metric_name == _TransferMetricName.LOAD_TIME:
            self.counter_kv_transfer_time[(engine_idx, _TransferType.LOAD)].inc(value)
        elif metric_name == _TransferMetricName.STORE_BYTES:
            self.counter_kv_bytes[(engine_idx, _TransferType.STORE)].inc(value)
        elif metric_name == _TransferMetricName.STORE_TIME:
            self.counter_kv_transfer_time[(engine_idx, _TransferType.STORE)].inc(value)

    def _set_gauge(self, metric_name: str, value: int | float, engine_idx: int) -> None:
        self.offloading_metrics[(engine_idx, metric_name)].set(value)

    def _observe_histogram(
        self, metric_name: str, value: list[int | float], engine_idx: int
    ) -> None:
        for observation in value:
            self.offloading_metrics[(engine_idx, metric_name)].observe(observation)
            if not self._observe_deprecated_metrics:
                continue
            # Keep deprecated CPU offload transfer metrics updated during the
            # transition to flat metric names.
            if metric_name == _TransferMetricName.LOAD_SIZE:
                self.histogram_transfer_size[(engine_idx, _TransferType.LOAD)].observe(
                    observation
                )
            elif metric_name == _TransferMetricName.STORE_SIZE:
                self.histogram_transfer_size[(engine_idx, _TransferType.STORE)].observe(
                    observation
                )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Observe transfer statistics."""
        metric_types = transfer_stats_data.get(_StatsKey.TYPES, {})
        metric_data = transfer_stats_data.get(_StatsKey.DATA, {})
        for key, value in metric_data.items():
            type_str = metric_types.get(key)
            if type_str is None:
                raise AssertionError(f"Unknown offloading stats key: {key}")
            assert key in self._offloading_metric_defs
            if type_str == _MetricType.COUNTER:
                assert isinstance(value, int | float)
                self._increase_counter(key, value, engine_idx)
            elif type_str == _MetricType.GAUGE:
                assert isinstance(value, int | float)
                self._set_gauge(key, value, engine_idx)
            elif type_str == _MetricType.HISTOGRAM:
                assert isinstance(value, list)
                assert all(isinstance(v, int | float) for v in value)
                self._observe_histogram(key, value, engine_idx)
            else:
                raise AssertionError(f"Unknown metric type '{type_str}' for key: {key}")
