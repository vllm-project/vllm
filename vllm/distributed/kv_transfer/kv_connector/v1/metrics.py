# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV Connector metrics: stats containers, logging pipeline, and Prometheus
metric registration.

Pipeline overview (``observe → aggregate → reduce → log``)
==========================================================

1. **observe()** — Each TP rank's worker periodically pushes its transfer
   telemetry (as a serializable ``dict``) to the logger process via
   ``KVConnectorLogging.observe()``. The data is already aggregated across
   all workers of that rank by the connector's ``build_kv_connector_stats``.

2. **aggregate()** — ``KVConnectorStats.aggregate()`` merges observations
   from successive syncs into a single accumulator by concatenating raw
   observation lists. After several intervals the accumulator holds a flat
   pool of per-transfer observations from **all TP ranks**.

3. **reduce()** — ``KVConnectorStats.reduce()`` computes summary statistics
   (mean, P90, throughput, counts) over the combined observation pool.
   Because the pool spans all ranks, the results describe the *pooled*
   distribution across ranks rather than per-rank or system-aggregate values.

4. **log()** — The reduced metrics are formatted into a single log line and
   the accumulator is reset for the next interval.

Key caveat for multi-rank (TP > 1) deployments
-----------------------------------------------
All metrics are **aggregated across all TP ranks** before summary stats are
computed. Users should not interpret them as per-rank or per-engine values.
See ``NixlKVConnectorStats`` for the exact semantics of each metric.
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
    """Base class for KV Connector Stats, a container for transfer performance
    metrics or otherwise important telemetry from the connector.

    All sub-classes need to be serializable as stats are sent from worker to
    logger process.

    Lifecycle (called by :class:`KVConnectorLogging`):

    1. ``build_kv_connector_stats()`` deserializes raw worker data into a
       stats object.
    2. ``aggregate()`` merges successive stats objects by concatenating
       observation lists (not by merging summaries).
    3. ``reduce()`` computes summary values over the combined observation
       pool for the current logging interval.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable connector stats payload."""
        return self.data

    def reset(self):
        """Reset the stats, clear the state."""
        raise NotImplementedError

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        """Merge another ``KVConnectorStats`` into this one.

        Implementations should extend (concatenate) their raw observation
        lists with those from *other*, so that subsequent ``reduce()`` calls
        see the full pool of observations across all ranks and syncs.
        """
        raise NotImplementedError

    def reduce(self) -> dict[str, int | float]:
        """Reduce the observations collected during a time interval to one or
        more representative values (eg avg/median/sum of the series).

        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval. The input pool contains
        observations from **all TP ranks** (see ``aggregate``), so the
        returned values describe the pooled distribution across ranks — not
        per-rank or system-aggregate figures.
        """
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Return True if the stats are empty."""
        raise NotImplementedError


class KVConnectorLogging:
    """Periodic transfer-metrics logging for KV connectors.

    Drives the four-stage pipeline:

    ``observe() → aggregate() → reduce() → log()``

    - ``observe()`` is called when the connector syncs with the scheduler.
      The incoming ``transfer_stats_data`` is already aggregated across all
      workers (one connector or a MultiConnector).
    - ``aggregate()`` accumulates observations across successive syncs.
    - ``log()`` calls ``reduce()`` on the accumulator, emits a single log
      line, then resets for the next interval.

    Note that the connector sync interval and the logging interval are
    independent: ``observe()`` may be called multiple times between
    ``log()`` calls.
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
        """Ingest a batch of transfer telemetry from the connector.

        Called periodically when the connector syncs with the scheduler.
        The ``transfer_stats_data`` is expected to be **already aggregated
        across all workers** (one connector or a MultiConnector) and
        deserialized into a ``KVConnectorStats`` instance.

        Observations are accumulated across successive calls until ``log()``
        consumes and resets them.
        """
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
        """Emit a single ``KV Transfer metrics`` log line for the last interval.

        Calls ``reduce()`` on the accumulator to compute summary statistics,
        formats them as a key=value string, and resets the accumulator. The
        reduced values are pooled across all TP ranks — see
        ``KVConnectorStats.reduce()`` for per-metric semantics.
        """
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


class KVConnectorPromMetrics:
    """A base class for per-connector Prometheus metric registration
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
        """Record the supplied transfer statistics to Prometheus metrics. These
        statistics are engine-specific, and should be recorded to a metric
        with the appropriate 'engine' label. These metric instances can be
        created using the create_metric_per_engine() helper method.
        """
        raise NotImplementedError


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
