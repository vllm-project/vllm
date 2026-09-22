# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    OffloadingSpec,
)
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

logger = init_logger(__name__)

KV_OFFLOAD_CONFIG_INFO = "vllm:kv_offload_config_info"

_INFO_METRIC_HELP = (
    "Static configuration of the KV offload managers of this engine instance. "
    "The configured offloading spec declares the label names, and each manager "
    "fills the values, so a series appears from the first scheduler step of its "
    "engine. Each engine reports its own configuration, not the instance total."
)


class _TransferMetricName:
    """Flat metric names for GPU↔offload-medium transfer operations."""

    LOAD_BYTES = "vllm:kv_offload_load_bytes"
    LOAD_TIME = "vllm:kv_offload_load_time"
    LOAD_SIZE = "vllm:kv_offload_load_size"
    STORE_BYTES = "vllm:kv_offload_store_bytes"
    STORE_TIME = "vllm:kv_offload_store_time"
    STORE_SIZE = "vllm:kv_offload_store_size"


class _ConnectorMetricName:
    """Connector-side metrics emitted by scheduler-side offloading code."""

    LOOKUP_SYNC_DELAY = "vllm:kv_offload_lookup_sync_delay_seconds"
    LOOKUP_ASYNC_DELAY = "vllm:kv_offload_lookup_async_delay_seconds"
    ALLOCATION_FAILURE = "vllm:kv_offload_allocation_failure"


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
        _ConnectorMetricName.LOOKUP_SYNC_DELAY: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of the time spent in a single offload lookup call, "
                "in seconds."
            ),
            buckets=(
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
            ),
        ),
        _ConnectorMetricName.LOOKUP_ASYNC_DELAY: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of time between a request's offload lookup first "
                "deferring and the following lookup resolving, or request "
                "finish, in seconds."
            ),
            buckets=(
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
                10,
            ),
        ),
        _ConnectorMetricName.ALLOCATION_FAILURE: OffloadingCounterMetadata(
            documentation=(
                "Number of KV offload store allocation attempts that failed."
            ),
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
    # Maps metric name -> {label values tuple -> observed value (number or list)}
    DATA = "data"
    # Maps info metric label name -> label value. Absent until a manager
    # reports it, and an empty dict once a manager reports no facts.
    INFO = "info"


@dataclass
class OffloadingConnectorStats(KVConnectorStats):
    """Offloading connector stats use flat metric names as keys.

    The ``data`` dict is structured using ``_StatsKey`` / ``_MetricType``::

        {
            _StatsKey.TYPES: {name: _MetricType.*, ...},
            _StatsKey.DATA:  {name: {labelvalues: value, ...}, ...},
            _StatsKey.INFO:  {labelname: labelvalue, ...} | None,
        }

    This structure is self-describing: it survives IPC serialization
    without needing the full ``OffloadingMetricMetadata`` objects on the
    receiving side.

    Counter values are aggregated by summing per-label-tuple, gauge values
    use the latest snapshot per-label-tuple, and histogram values are lists of
    observed samples per-label-tuple. Unlabeled metrics use ``()`` as their
    labelvalues tuple.

    ``INFO`` holds static config facts, which the scheduler sends once per
    process. It stays out of ``DATA``, because its label names are known only
    when the payload arrives.
    """

    def __post_init__(self):
        if _StatsKey.DATA not in self.data:
            self.reset()

    def reset(self):
        self.data: dict[str, Any] = {
            _StatsKey.TYPES: {},
            _StatsKey.DATA: {},
            _StatsKey.INFO: None,
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
        # The scheduler payload carries the info and merges into the worker
        # payload (v1/core/sched/scheduler.py), so the info must survive the
        # merge in this direction. The scheduler sends it once, so a later
        # payload with no info must not clear it.
        other_info = other.data.get(_StatsKey.INFO)
        if other_info is not None:
            self.data[_StatsKey.INFO] = other_info
        other_types = other._types
        other_values = other._values
        for key, other_label_values in other_values.items():
            type_str = other_types.get(key)
            if type_str is None:
                raise AssertionError(f"Unknown offloading stats key: {key}")
            self._types.setdefault(key, type_str)
            current_label_values = self._values.setdefault(key, {})
            for labelvalues, value in other_label_values.items():
                if type_str == _MetricType.HISTOGRAM:
                    assert isinstance(value, list)
                    if labelvalues not in current_label_values:
                        current_label_values[labelvalues] = list(value)
                    else:
                        assert isinstance(current_label_values[labelvalues], list)
                        current_label_values[labelvalues].extend(value)
                elif type_str == _MetricType.COUNTER:
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
        """Reduce the observations collected during a time interval to one or
        more representative values (eg avg/median/sum of the series).
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        return_dict: dict[str, int | float] = {}
        for key, label_value_map in self._values.items():
            type_str = self._types.get(key)
            if type_str is None:
                raise AssertionError(f"Unknown offloading stats key: {key}")
            for labelvalues, value in label_value_map.items():
                key_with_labels = f"{key}:{labelvalues}" if labelvalues else key
                if type_str == _MetricType.HISTOGRAM:
                    assert isinstance(value, list)
                    return_dict[f"{key_with_labels}_count"] = len(value)
                    return_dict[f"{key_with_labels}_sum"] = sum(value)
                elif type_str in (_MetricType.COUNTER, _MetricType.GAUGE):
                    assert isinstance(value, int | float)
                    return_dict[key_with_labels] = value
                else:
                    raise AssertionError(
                        f"Unknown metric type '{type_str}' for key: {key}"
                    )
        return return_dict

    def is_empty(self) -> bool:
        # An info-only payload is not empty. It carries no observation, but the
        # caller drops an empty payload, and the info must reach the frontend.
        return (
            not self.data.get(_StatsKey.DATA) and self.data.get(_StatsKey.INFO) is None
        )

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

    def set_info(self, info: Mapping[str, str | int | float | bool]) -> None:
        """Put the static config facts of this engine on the stats payload.

        Args:
            info: Mapping of info metric label name to label value, as
                OffloadingManager.config_info() returns it. An empty mapping
                still yields the metric, with an empty value on every label
                the spec declared.

        """
        self.data[_StatsKey.INFO] = dict(info)

    def observe_histogram(
        self,
        histogram_name: str,
        histogram_value: int | float,
        labelvalues: tuple[str, ...] = (),
    ) -> None:
        """Record a histogram observation on the stats payload."""
        self._types.setdefault(histogram_name, _MetricType.HISTOGRAM)
        histogram_values = self._values.setdefault(histogram_name, {})
        histogram_values.setdefault(labelvalues, []).append(histogram_value)


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
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        extra_config = kv_transfer_config.kv_connector_extra_config
        self._spec_cls: type[OffloadingSpec] = OffloadingSpecFactory.get_spec_cls(
            extra_config
        )
        # The spec declares the info label names here; a manager fills the
        # values later, and _observe_info() aligns the two.
        self._info_keys: tuple[str, ...] = self._spec_cls.config_info_keys(extra_config)
        self._offloading_metric_metadata: dict[str, OffloadingMetricMetadata] = {
            **self._spec_cls.build_metric_definitions(extra_config),
            **get_connector_metric_definitions(),
            KV_OFFLOAD_CONFIG_INFO: OffloadingGaugeMetadata(
                documentation=_INFO_METRIC_HELP, labelnames=self._info_keys
            ),
        }
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

        self._observe_deprecated_metrics = issubclass(self._spec_cls, CPUOffloadingSpec)
        self._offloading_metric_defs: dict[str, PromMetricT] = {}
        # (engine_idx, metric_name, labelvalues) -> metric with bound labels
        self.offloading_metrics: dict[
            tuple[int, str, tuple[str, ...]], PromMetricT
        ] = {}

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

    def _create_metric(
        self, metric_name: str, metadata: OffloadingMetricMetadata
    ) -> Any:
        kwargs: dict[str, Any] = {
            "name": metric_name,
            "documentation": metadata.documentation,
            "labelnames": self._labelnames + list(metadata.labelnames),
        }
        if isinstance(metadata, OffloadingCounterMetadata):
            metric_cls = self._counter_cls
        elif isinstance(metadata, OffloadingGaugeMetadata):
            metric_cls = self._gauge_cls
            kwargs["multiprocess_mode"] = metadata.multiprocess_mode
        elif isinstance(metadata, OffloadingHistogramMetadata):
            metric_cls = self._histogram_cls
            if metadata.buckets is not None:
                kwargs["buckets"] = metadata.buckets
        else:
            raise AssertionError(f"Unknown offloading metric metadata: {metadata}")
        return metric_cls(**kwargs)

    def _get_prometheus_metric(
        self,
        metric_name: str,
        labelvalues: tuple[str, ...],
        engine_idx: int,
    ) -> PromMetric:
        metadata = self._offloading_metric_metadata[metric_name]
        if len(labelvalues) != len(metadata.labelnames):
            raise AssertionError(
                f"Metric {metric_name} expects {len(metadata.labelnames)} labels, "
                f"got {len(labelvalues)}"
            )
        key = (engine_idx, metric_name, labelvalues)
        prom_metric = self.offloading_metrics.get(key)
        if prom_metric is None:
            engine_labelvalues = self.per_engine_labelvalues[engine_idx]
            prom_metric = self._offloading_metric_defs[metric_name].labels(
                *(engine_labelvalues + list(labelvalues))
            )
            self.offloading_metrics[key] = prom_metric
        return prom_metric

    def _increase_counter(
        self,
        metric_name: str,
        value: int | float,
        labelvalues: tuple[str, ...],
        engine_idx: int,
    ) -> None:
        self._get_prometheus_metric(metric_name, labelvalues, engine_idx).inc(value)
        if labelvalues or not self._observe_deprecated_metrics:
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

    def _set_gauge(
        self,
        metric_name: str,
        value: int | float,
        labelvalues: tuple[str, ...],
        engine_idx: int,
    ) -> None:
        self._get_prometheus_metric(metric_name, labelvalues, engine_idx).set(value)

    def _observe_histogram(
        self,
        metric_name: str,
        value: list[int | float],
        labelvalues: tuple[str, ...],
        engine_idx: int,
    ) -> None:
        prom_metric = self._get_prometheus_metric(metric_name, labelvalues, engine_idx)
        for observation in value:
            prom_metric.observe(observation)
            if labelvalues or not self._observe_deprecated_metrics:
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

    def _observe_info(self, info: dict[str, Any], engine_idx: int) -> None:
        """Publish the static config facts of one engine.

        The spec declared the label names in __init__, so this only aligns one
        payload with them: it reads every declared name out of the payload,
        which makes the payload order irrelevant. A declared name the payload
        does not hold gets an empty value, and a payload name the spec did not
        declare is dropped, because the label names of a live Prometheus
        metric cannot change. Both gaps mean the spec and the manager disagree,
        so each one is logged.

        Args:
            info: Mapping of label name to label value, as
                OffloadingManager.config_info() returns it.
            engine_idx: Index of the reporting engine.

        """
        empty = tuple(key for key in self._info_keys if key not in info)
        dropped = tuple(key for key in info if key not in self._info_keys)
        if empty or dropped:
            logger.warning_once(
                "%s: spec %s and the manager of engine %d disagree on the KV "
                "offload config labels. Declared but not filled, so empty: %s. "
                "Filled but not declared, so dropped: %s.",
                KV_OFFLOAD_CONFIG_INFO,
                self._spec_cls.__name__,
                engine_idx,
                empty,
                dropped,
            )

        labelvalues = tuple(str(info.get(key, "")) for key in self._info_keys)
        self._set_gauge(KV_OFFLOAD_CONFIG_INFO, 1, labelvalues, engine_idx)

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Observe transfer statistics."""
        info = transfer_stats_data.get(_StatsKey.INFO)
        if info is not None:
            self._observe_info(info, engine_idx)
        metric_types = transfer_stats_data.get(_StatsKey.TYPES, {})
        metric_data = transfer_stats_data.get(_StatsKey.DATA, {})
        for key, label_value_map in metric_data.items():
            type_str = metric_types.get(key)
            if type_str is None:
                raise AssertionError(f"Unknown offloading stats key: {key}")
            assert key in self._offloading_metric_defs
            for labelvalues, value in label_value_map.items():
                if type_str == _MetricType.COUNTER:
                    assert isinstance(value, int | float)
                    self._increase_counter(key, value, labelvalues, engine_idx)
                elif type_str == _MetricType.GAUGE:
                    assert isinstance(value, int | float)
                    self._set_gauge(key, value, labelvalues, engine_idx)
                elif type_str == _MetricType.HISTOGRAM:
                    assert isinstance(value, list)
                    assert all(isinstance(v, int | float) for v in value)
                    self._observe_histogram(key, value, labelvalues, engine_idx)
                else:
                    raise AssertionError(
                        f"Unknown metric type '{type_str}' for key: {key}"
                    )
