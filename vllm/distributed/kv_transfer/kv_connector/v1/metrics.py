# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar

from prometheus_client import Counter, Gauge, Histogram

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.logger import init_logger

PromMetric: TypeAlias = Gauge | Counter | Histogram
PromMetricT = TypeVar("PromMetricT", bound=PromMetric)

logger = init_logger(__name__)


@dataclass
class KVConnectorStats:
    """
    Base class for KV Connector Stats, a container for transfer performance
    metrics or otherwise important telemetry from the connector.
    All sub-classes need to be serializable as stats are sent from worker to
    logger process.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable connector stats payload."""
        return self.data

    def reset(self):
        """Reset the stats, clear the state."""
        raise NotImplementedError

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        """
        Aggregate stats with another `KVConnectorStats` object.
        """
        raise NotImplementedError

    def reduce(self) -> dict[str, int | float]:
        """
        Reduce the observations collected during a time interval to one or
        more representative values (eg avg/median/sum of the series).
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Return True if the stats are empty."""
        raise NotImplementedError


@dataclass(frozen=True)
class MetricMetadata:
    """Declaration of a connector metric; the subclass selects its type."""

    documentation: str
    labelnames: tuple[str, ...] = ()


@dataclass(frozen=True)
class CounterMetadata(MetricMetadata):
    pass


@dataclass(frozen=True)
class GaugeMetadata(MetricMetadata):
    pass


@dataclass(frozen=True)
class HistogramMetadata(MetricMetadata):
    buckets: tuple[float, ...] | None = None


MetricDefinitions: TypeAlias = dict[str, MetricMetadata]


class MetricType:
    """Type tags embedded in the serialized stats payload."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class StatsKey:
    """Top-level keys in the serialized stats dict."""

    # Maps metric name -> MetricType value
    TYPES = "types"
    # Maps metric name -> {label values tuple -> observed value (number or list)}
    DATA = "data"


@dataclass
class TypedKVConnectorStats(KVConnectorStats):
    """
    Connector stats keyed by flat metric name, with each metric's type carried
    in the payload so that aggregation and Prometheus recording work without a
    per-connector subclass. Connectors record into one instance with
    ``increase_counter``, ``set_gauge`` and ``observe_histogram``; stats from
    different metric families compose by sharing the instance or aggregating.

    The ``data`` dict is structured using ``StatsKey`` / ``MetricType``::

        {
            StatsKey.TYPES: {name: MetricType.*, ...},
            StatsKey.DATA:  {name: {labelvalues: value, ...}, ...},
        }

    This structure is self-describing: it survives IPC serialization
    without needing the ``MetricMetadata`` objects on the receiving side.

    Counter values are aggregated by summing per-label-tuple, gauge values
    use the latest snapshot per-label-tuple, and histogram values are lists of
    observed samples per-label-tuple. Unlabeled metrics use ``()`` as their
    labelvalues tuple.
    """

    def __post_init__(self):
        if StatsKey.DATA not in self.data:
            self.reset()

    def reset(self):
        self.data: dict[str, Any] = {
            StatsKey.TYPES: {},
            StatsKey.DATA: {},
        }

    @property
    def _types(self) -> dict[str, str]:
        return self.data[StatsKey.TYPES]

    @property
    def _values(self) -> dict[str, Any]:
        return self.data[StatsKey.DATA]

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if other.is_empty():
            return self
        assert isinstance(other, TypedKVConnectorStats)
        other_types = other._types
        other_values = other._values
        for key, other_label_values in other_values.items():
            type_str = other_types.get(key)
            if type_str is None:
                raise AssertionError(f"Unknown connector stats key: {key}")
            self._types.setdefault(key, type_str)
            current_label_values = self._values.setdefault(key, {})
            for labelvalues, value in other_label_values.items():
                if type_str == MetricType.HISTOGRAM:
                    assert isinstance(value, list)
                    if labelvalues not in current_label_values:
                        current_label_values[labelvalues] = list(value)
                    else:
                        assert isinstance(current_label_values[labelvalues], list)
                        current_label_values[labelvalues].extend(value)
                elif type_str == MetricType.COUNTER:
                    assert isinstance(value, int | float)
                    current_label_values[labelvalues] = (
                        current_label_values.get(labelvalues, 0) + value
                    )
                elif type_str == MetricType.GAUGE:
                    assert isinstance(value, int | float)
                    current_label_values[labelvalues] = value
                else:
                    raise AssertionError(
                        f"Unknown metric type '{type_str}' for key: {key}"
                    )
        return self

    def reduce(self) -> dict[str, int | float]:
        return_dict: dict[str, int | float] = {}
        for key, label_value_map in self._values.items():
            type_str = self._types.get(key)
            if type_str is None:
                raise AssertionError(f"Unknown connector stats key: {key}")
            for labelvalues, value in label_value_map.items():
                key_with_labels = f"{key}:{labelvalues}" if labelvalues else key
                if type_str == MetricType.HISTOGRAM:
                    assert isinstance(value, list)
                    return_dict[f"{key_with_labels}_count"] = len(value)
                    return_dict[f"{key_with_labels}_sum"] = sum(value)
                elif type_str in (MetricType.COUNTER, MetricType.GAUGE):
                    assert isinstance(value, int | float)
                    return_dict[key_with_labels] = value
                else:
                    raise AssertionError(
                        f"Unknown metric type '{type_str}' for key: {key}"
                    )
        return return_dict

    def is_empty(self) -> bool:
        return not self.data.get(StatsKey.DATA)

    def increase_counter(
        self,
        counter_name: str,
        counter_increase_value: int | float = 1,
        labelvalues: tuple[str, ...] = (),
    ) -> None:
        """Increase a counter on the stats payload."""
        self._types.setdefault(counter_name, MetricType.COUNTER)
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
        self._types.setdefault(gauge_name, MetricType.GAUGE)
        gauge_values = self._values.setdefault(gauge_name, {})
        gauge_values[labelvalues] = gauge_value

    def observe_histogram(
        self,
        histogram_name: str,
        histogram_value: int | float,
        labelvalues: tuple[str, ...] = (),
    ) -> None:
        """Record a histogram observation on the stats payload."""
        self._types.setdefault(histogram_name, MetricType.HISTOGRAM)
        histogram_values = self._values.setdefault(histogram_name, {})
        histogram_values.setdefault(labelvalues, []).append(histogram_value)


class KVConnectorLogging:
    def __init__(self, kv_transfer_config: KVTransferConfig | None):
        # Instantiate the connector's stats class.
        if kv_transfer_config and kv_transfer_config.kv_connector:
            self.connector_cls = KVConnectorFactory.get_connector_class(
                kv_transfer_config
            )
        self.reset()

    def reset(self):
        self.transfer_stats_accumulator: KVConnectorStats | None = None

    def observe(self, transfer_stats_data: dict[str, Any]):
        # Should not be called when a KVConnector is not configured.
        assert self.connector_cls is not None
        # Called periodically when connector syncs with the scheduler.
        # Note that this is not the same as the logging interval.
        # We expect transfer_stats_data to be aggregated across all workers and
        # consist of observations from a single connector or a MultiConnector.
        transfer_stats = self.connector_cls.build_kv_connector_stats(
            transfer_stats_data
        )
        if transfer_stats is None:
            logger.warning_once(
                "The connector %s is collecting stats but "
                "does not implement the "
                "`build_kv_connector_stats` method. "
                "Stats will not be logged.",
                self.connector_cls,
            )
            return

        if self.transfer_stats_accumulator is None:
            self.transfer_stats_accumulator = transfer_stats
        else:
            # Accumulate last interval stats.
            self.transfer_stats_accumulator = self.transfer_stats_accumulator.aggregate(
                transfer_stats
            )

    def log(self, log_fn=logger.info):
        """Log transfer metrics periodically, similar to throughput logging"""
        if (
            self.transfer_stats_accumulator
            and not self.transfer_stats_accumulator.is_empty()
        ):
            # Produce a single cumulative stats object for the last time
            # interval from the recorded observations.
            xfer_metrics = self.transfer_stats_accumulator.reduce()
            xfer_metrics_str = ", ".join(f"{k}={v}" for k, v in xfer_metrics.items())
            log_fn("KV Transfer metrics: %s", xfer_metrics_str)

            # Reset metrics for next interval
            self.reset()


class KVConnectorPromMetrics:
    """
    A base class for per-connector Prometheus metric registration
    and recording.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._gauge_cls = metric_types[Gauge]
        self._counter_cls = metric_types[Counter]
        self._histogram_cls = metric_types[Histogram]
        self._labelnames = labelnames
        self.per_engine_labelvalues = per_engine_labelvalues

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """
        Record the supplied transfer statistics to Prometheus metrics. These
        statistics are engine-specific, and should be recorded to a metric
        with the appropriate 'engine' label. These metric instances can be
        created using the create_metric_per_engine() helper method.
        """
        raise NotImplementedError


class TypedKVConnectorPromMetrics(KVConnectorPromMetrics):
    """
    Registers one Prometheus metric per entry of ``metric_definitions`` and
    records ``TypedKVConnectorStats`` payloads against them. Connectors that
    declare their metrics can return this class from ``build_prom_metrics``
    directly; subclass only to layer extra behavior on top of a recording.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
        metric_definitions: MetricDefinitions,
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self.metric_definitions = metric_definitions
        self._metric_defs: dict[str, PromMetricT] = {
            metric_name: self._create_metric(metric_name, metadata)
            for metric_name, metadata in metric_definitions.items()
        }
        # (engine_idx, metric_name, labelvalues) -> metric with bound labels
        self.metrics: dict[tuple[int, str, tuple[str, ...]], PromMetricT] = {}

    def _create_metric(self, metric_name: str, metadata: MetricMetadata) -> Any:
        kwargs: dict[str, Any] = {
            "name": metric_name,
            "documentation": metadata.documentation,
            "labelnames": self._labelnames + list(metadata.labelnames),
        }
        if isinstance(metadata, CounterMetadata):
            metric_cls = self._counter_cls
        elif isinstance(metadata, GaugeMetadata):
            metric_cls = self._gauge_cls
        elif isinstance(metadata, HistogramMetadata):
            metric_cls = self._histogram_cls
            if metadata.buckets is not None:
                kwargs["buckets"] = metadata.buckets
        else:
            raise AssertionError(f"Unknown metric metadata: {metadata}")
        return metric_cls(**kwargs)

    def _get_prometheus_metric(
        self,
        metric_name: str,
        labelvalues: tuple[str, ...],
        engine_idx: int,
    ) -> PromMetric:
        metadata = self.metric_definitions[metric_name]
        if len(labelvalues) != len(metadata.labelnames):
            raise AssertionError(
                f"Metric {metric_name} expects {len(metadata.labelnames)} labels, "
                f"got {len(labelvalues)}"
            )
        key = (engine_idx, metric_name, labelvalues)
        prom_metric = self.metrics.get(key)
        if prom_metric is None:
            engine_labelvalues = self.per_engine_labelvalues[engine_idx]
            prom_metric = self._metric_defs[metric_name].labels(
                *(engine_labelvalues + list(labelvalues))
            )
            self.metrics[key] = prom_metric
        return prom_metric

    def _increase_counter(
        self,
        metric_name: str,
        value: int | float,
        labelvalues: tuple[str, ...],
        engine_idx: int,
    ) -> None:
        self._get_prometheus_metric(metric_name, labelvalues, engine_idx).inc(value)

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

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        metric_types = transfer_stats_data.get(StatsKey.TYPES, {})
        metric_data = transfer_stats_data.get(StatsKey.DATA, {})
        for key, label_value_map in metric_data.items():
            type_str = metric_types.get(key)
            if type_str is None:
                raise AssertionError(f"Unknown connector stats key: {key}")
            assert key in self._metric_defs, f"Undeclared connector metric: {key}"
            for labelvalues, value in label_value_map.items():
                if type_str == MetricType.COUNTER:
                    assert isinstance(value, int | float)
                    self._increase_counter(key, value, labelvalues, engine_idx)
                elif type_str == MetricType.GAUGE:
                    assert isinstance(value, int | float)
                    self._set_gauge(key, value, labelvalues, engine_idx)
                elif type_str == MetricType.HISTOGRAM:
                    assert isinstance(value, list)
                    assert all(isinstance(v, int | float) for v in value)
                    self._observe_histogram(key, value, labelvalues, engine_idx)
                else:
                    raise AssertionError(
                        f"Unknown metric type '{type_str}' for key: {key}"
                    )


class KVConnectorProm:
    """
    Support for registering per-connector Prometheus metrics, and
    recording transfer statistics to those metrics. Uses
    KVConnectorBase.build_prom_metrics().
    """

    _gauge_cls = Gauge
    _counter_cls = Counter
    _histogram_cls = Histogram

    def __init__(
        self,
        vllm_config: VllmConfig,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.prom_metrics: KVConnectorPromMetrics | None = None
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config and kv_transfer_config.kv_connector:
            connector_cls = KVConnectorFactory.get_connector_class(kv_transfer_config)
            metric_types = {
                Gauge: self._gauge_cls,
                Counter: self._counter_cls,
                Histogram: self._histogram_cls,
            }
            self.prom_metrics = connector_cls.build_prom_metrics(
                vllm_config,
                metric_types,
                labelnames,
                per_engine_labelvalues,
            )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        if self.prom_metrics is None:
            return
        self.prom_metrics.observe(transfer_stats_data, engine_idx)
