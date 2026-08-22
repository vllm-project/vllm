# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Base classes for KV Connector metrics logging and Prometheus recording.

The full data flow for NIXL transfer metrics is:

1. Each TP rank records per-transfer telemetry via
   :meth:`NixlKVConnectorStats.record_transfer` during worker execution.
2. The connector periodically snapshots and resets the per-rank collectors
   (:meth:`NixlKVConnectorStats.clone_and_reset`), shipping the snapshots
   to the scheduler.
3. Snapshots from all ranks are concatenated upstream, so the dict passed
   to :meth:`KVConnectorLogging.observe` already contains observations from
   every TP rank.
4. ``observe`` rebuilds the stats object and merges it into the running
   accumulator via :meth:`KVConnectorStats.aggregate`.
5. On the logging interval, :meth:`KVConnectorLogging.log` calls
   :meth:`KVConnectorStats.reduce` to produce the summary line and resets
   the accumulator.
6. Separately, :meth:`KVConnectorPromMetrics.observe` records every
   observation into Prometheus histograms/counters.

Because stats arrive pre-aggregated across workers, the summaries produced
by ``reduce`` reflect the combined pool of observations from all TP ranks
rather than any single rank. See
:class:`NixlKVConnectorStats` for the exact semantics of each metric.
"""

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
        """Accumulate connector stats received from the scheduler.

        Called whenever the connector syncs with the scheduler, which is
        not necessarily the logging interval. ``transfer_stats_data`` is
        expected to be aggregated across all workers (each TP rank's
        observations are concatenated upstream) and consist of observations
        from a single connector or a MultiConnector. The data is rebuilt
        into a stats object and merged into the running accumulator via
        :meth:`KVConnectorStats.aggregate`, so the accumulator spans all
        observations since the last :meth:`log` call.
        """
        # Should not be called when a KVConnector is not configured.
        assert self.connector_cls is not None
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
        """Log the accumulated transfer metrics for the current interval.

        Reduces the accumulator (which may hold observations aggregated across
        all TP ranks) to a summary line via :meth:`KVConnectorStats.reduce`,
        logs it, and resets the accumulator for the next interval. Intervals
        with only failures still produce a log line with zero values so they
        are not silently dropped — failure counts are surfaced through
        Prometheus instead.
        """
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
