# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar

from prometheus_client import Counter, Gauge, Histogram

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.logger import init_logger
from vllm.v1.metrics.utils import create_metric_per_engine

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


# Bucket edges shared by the connectors that export per-transfer telemetry,
# named after the series each one feeds. Timing runs 1ms to 5s; post time gets
# one extra sub-millisecond edge below the copy's range. Payload sizes run 2KiB
# to 8GiB, doubling every other power of two, and descriptor counts run 10 to
# 50k.
KV_TRANSFER_XFER_TIME_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.2,
    0.3,
    0.5,
    0.75,
    1.0,
    5.0,
)
KV_TRANSFER_POST_TIME_BUCKETS = (0.001, *KV_TRANSFER_XFER_TIME_BUCKETS)
KV_TRANSFER_BYTES_BUCKETS = tuple(2 ** (10 + i) for i in range(1, 25, 2))
KV_TRANSFER_DESCRIPTOR_BUCKETS = (
    10,
    20,
    30,
    50,
    75,
    100,
    200,
    400,
    1000,
    2000,
    4000,
    10000,
    20000,
    50000,
)


class KVTransferPromMetrics(KVConnectorPromMetrics):
    """
    A base class for connectors that record one row per KV transfer.

    Subclasses declare their series with `declare_histogram()` and
    `declare_counter()`, naming the key of the stats snapshot that feeds each
    one. `observe()` then fans a snapshot out to every declared series, so a
    connector describes what it exports rather than how it is recorded.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._histograms: list[tuple[dict[int, PromMetric], str]] = []
        self._counters: list[tuple[dict[int, PromMetric], str]] = []

    def declare_histogram(
        self,
        *,
        name: str,
        documentation: str,
        stats_key: str,
        buckets: Sequence[float],
    ) -> dict[int, PromMetric]:
        """Declare a histogram observing every value stored under `stats_key`."""
        return self._declare(
            self._histogram_cls(
                name=name,
                documentation=documentation,
                buckets=buckets,
                labelnames=self._labelnames,
            ),
            stats_key,
            declared=self._histograms,
        )

    def declare_counter(
        self,
        *,
        name: str,
        documentation: str,
        stats_key: str,
    ) -> dict[int, PromMetric]:
        """Declare a counter of the events recorded under `stats_key`."""
        return self._declare(
            self._counter_cls(
                name=name,
                documentation=documentation,
                labelnames=self._labelnames,
            ),
            stats_key,
            declared=self._counters,
        )

    def _declare(
        self,
        metric: PromMetric,
        stats_key: str,
        declared: list[tuple[dict[int, PromMetric], str]],
    ) -> dict[int, PromMetric]:
        per_engine_metrics = create_metric_per_engine(
            metric, self.per_engine_labelvalues
        )
        declared.append((per_engine_metrics, stats_key))
        return per_engine_metrics

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        # A snapshot always carries every declared key, so a miss here is a
        # connector bug and should surface rather than silently drop samples.
        for per_engine_metrics, stats_key in self._histograms:
            for value in transfer_stats_data[stats_key]:
                per_engine_metrics[engine_idx].observe(value)

        # Each failure appends a single unit, so incrementing by the sum costs
        # one Prometheus client call per series instead of one per event.
        for per_engine_metrics, stats_key in self._counters:
            num_events = sum(transfer_stats_data[stats_key])
            if num_events:
                per_engine_metrics[engine_idx].inc(num_events)


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
