# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeAlias, TypeVar

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
    """Base class for KV Connector Stats, a container for transfer performance
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
        """Aggregate stats with another `KVConnectorStats` object."""
        raise NotImplementedError

    def reduce(self) -> dict[str, int | float]:
        """Reduce the observations collected during a time interval to one or
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
        """Log transfer metrics periodically, similar to throughput logging."""
        if (
            self.transfer_stats_accumulator
            and not self.transfer_stats_accumulator.is_empty()
        ):
            # Produce a single cumulative stats object for the last time
            # interval from the recorded observations.
            xfer_metrics = self.transfer_stats_accumulator.reduce()
            # A payload can hold static config facts alone (info metric), which reduce()
            # drops, because they belong to the Prometheus metrics only.
            if xfer_metrics:
                xfer_metrics_str = ", ".join(
                    f"{k}={v}" for k, v in xfer_metrics.items()
                )
                log_fn("KV Transfer metrics: %s", xfer_metrics_str)

            # Reset metrics for next interval
            self.reset()


# Default bucket edges for the per-transfer series. A connector that needs
# different edges passes its own tuple to the spec.
#
# Timing covers 5ms to 5s. Payload sizes run 2KiB to 8GiB, doubling every other
# power of two, and descriptor counts run 10 to 50k.
XFER_TIME_BUCKETS: tuple[float, ...] = (
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
BYTES_BUCKETS: tuple[float, ...] = tuple(2 ** (10 + i) for i in range(1, 25, 2))
DESCRIPTOR_BUCKETS: tuple[float, ...] = (
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


@dataclass(frozen=True)
class HistogramSpec:
    """A per-engine histogram series and the stats key that feeds it.

    Attributes:
        stats_key: Key in the connector stats payload holding the observations.
        name: Prometheus series name.
        documentation: Prometheus help text.
        buckets: Upper bounds of the histogram buckets.

    """

    stats_key: str
    name: str
    documentation: str
    buckets: tuple[float, ...]


@dataclass(frozen=True)
class CounterSpec:
    """A per-engine counter series and the stats keys that feed it.

    Every value under the listed keys counts events, so recording a payload adds
    their sum to the counter. Several keys share one series when different code
    paths report the same kind of event.

    Attributes:
        stats_keys: Keys in the connector stats payload holding event counts.
        name: Prometheus series name.
        documentation: Prometheus help text.

    """

    stats_keys: tuple[str, ...]
    name: str
    documentation: str

    def __post_init__(self) -> None:
        if not self.stats_keys:
            raise ValueError(f"counter {self.name} declares no stats keys")


class KVConnectorPromMetrics:
    """A base class for per-connector Prometheus metric registration
    and recording.

    Subclasses declare the series they publish in `HISTOGRAM_SPECS` and
    `COUNTER_SPECS`; registration and recording are shared. A connector that
    needs a different metric shape overrides `observe` and keeps its own metric
    attributes.
    """

    HISTOGRAM_SPECS: ClassVar[tuple[HistogramSpec, ...]] = ()
    COUNTER_SPECS: ClassVar[tuple[CounterSpec, ...]] = ()

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
        self._histogram_metrics: list[dict[int, PromMetric]] = []
        self._counter_metrics: list[dict[int, PromMetric]] = []
        for histogram_spec in self.HISTOGRAM_SPECS:
            histogram = self._histogram_cls(
                name=histogram_spec.name,
                documentation=histogram_spec.documentation,
                buckets=histogram_spec.buckets,
                labelnames=self._labelnames,
            )
            self._histogram_metrics.append(
                create_metric_per_engine(histogram, per_engine_labelvalues)
            )
        for counter_spec in self.COUNTER_SPECS:
            counter = self._counter_cls(
                name=counter_spec.name,
                documentation=counter_spec.documentation,
                labelnames=self._labelnames,
            )
            self._counter_metrics.append(
                create_metric_per_engine(counter, per_engine_labelvalues)
            )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Record the supplied transfer statistics to the declared metrics.

        These statistics are engine-specific, and are recorded to the series
        carrying the matching 'engine' label.
        """
        if not (self.HISTOGRAM_SPECS or self.COUNTER_SPECS):
            # Nothing to record into: the connector forgot to declare its series
            # or to override this method.
            raise NotImplementedError(
                f"{type(self).__name__} declares no metrics and does not"
                " override observe()"
            )
        # A stats payload always carries every declared key, so a miss is a
        # connector bug and should surface rather than drop samples silently.
        for histogram_spec, per_engine in zip(
            self.HISTOGRAM_SPECS, self._histogram_metrics
        ):
            histogram = per_engine[engine_idx]
            for value in transfer_stats_data[histogram_spec.stats_key]:
                histogram.observe(value)
        for counter_spec, per_engine in zip(self.COUNTER_SPECS, self._counter_metrics):
            # One increment by the sum costs a single client call per series
            # instead of one per event.
            num_events = sum(
                sum(transfer_stats_data[stats_key])
                for stats_key in counter_spec.stats_keys
            )
            if num_events:
                per_engine[engine_idx].inc(num_events)


class KVConnectorProm:
    """Support for registering per-connector Prometheus metrics, and
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
