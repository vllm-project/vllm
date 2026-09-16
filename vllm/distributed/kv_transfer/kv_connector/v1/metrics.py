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

    Pipeline:
    1. Worker calls `record_transfer()` / `record_failed_*()` on its local
       stats object (per-engine).
    2. Stats from all workers are aggregated across workers (for NIXL, all
       TP ranks) into a single cumulative `KVConnectorStats` object.
    3. Logger calls `aggregate()` to accumulate the current interval's stats
       into the logging accumulator (list-wise concatenation), not to combine
       workers.
    4. Logger calls `reduce()` to produce a summary dict for CLI logging.
    5. Logger calls `log()` to print the summary.
    6. Logger calls `reset()` to clear for the next interval.

    For NIXL connector, aggregation uses `list.extend()` so the combined
    pool contains observations from ALL TP ranks. Reduction then computes
    statistics (mean, P90, throughput) over this combined pool.
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

        The default expectation is list-wise concatenation (for NIXL this
        combines observations from all TP ranks into a single list per metric).
        """
        raise NotImplementedError

    def reduce(self) -> dict[str, int | float]:
        """
        Reduce the observations collected during a time interval to one or
        more representative values (e.g., avg, P90, sum of the series).
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.

        For NIXL, the input is already the **combined pool of all TP ranks'
        observations** (see `aggregate()`). The reduction computes stats over
        this combined pool, not per-rank or per-engine.
        """
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Return True if the stats are empty."""
        raise NotImplementedError


class KVConnectorLogging:
    """
    Manages the observe -> aggregate -> reduce -> log pipeline for KV
    connector transfer metrics.

    Pipeline per logging interval:
    1. `observe()` - Called periodically when connector syncs with scheduler.
       Receives `transfer_stats_data` already aggregated across ALL workers
       (for MultiConnector, all TP ranks are already combined). Builds a
       stats object via connector's `build_kv_connector_stats()`.
    2. `aggregate()` - Accumulates current interval stats into the logging
       accumulator (list-wise concatenation), NOT cross-worker aggregation.
    3. `log()` - Calls `reduce()` on accumulator to get summary dict,
       formats and logs it, then calls `reset()`.

    Cross-worker aggregation happens before `observe()`: the
    `transfer_stats_data` arrives already combined across all workers.
    """

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
