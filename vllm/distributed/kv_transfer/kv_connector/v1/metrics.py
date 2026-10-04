# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base classes and logging pipeline for KV connector transfer metrics.

This module defines the metrics infrastructure shared by all KV connectors
(not just NIXL). It provides two output paths for transfer telemetry:

1. **CLI logging** via ``KVConnectorLogging``:
   ``observe() → aggregate() → reduce() → log()``

2. **Prometheus metrics** via ``KVConnectorProm`` / ``KVConnectorPromMetrics``:
   ``observe()`` feeds individual observations into Prometheus histograms
   and counters for external monitoring and alerting.

**Multi-rank aggregation semantics:**

In a multi-rank deployment (TP > 1), each rank independently records
per-transfer telemetry. Before the metrics reach this module, the
worker process has already merged all local ranks' stats into a single
``KVConnectorStats`` object via the connector's ``aggregate()`` method.
This means:

- The ``KVConnectorLogging.observe()`` method receives **pre-aggregated**
  data — observations from all TP ranks on this worker are already
  concatenated into flat lists.
- If there are multiple workers (e.g., with pipeline parallelism or
  multiple engine instances), each worker's aggregated stats are
  accumulated independently by ``KVConnectorLogging`` via successive
  ``observe()`` calls, then combined in ``reduce()``.

The ``KVConnectorStats`` subclass (e.g., ``NixlKVConnectorStats``) is
responsible for defining the exact semantics of aggregation. See the
subclass docstrings for details on how to interpret individual metrics.
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
    """Base class for KV connector transfer performance metrics.

    This is a serializable container for per-transfer telemetry. Subclasses
    (e.g., ``NixlKVConnectorStats``) define the specific metrics collected
    and the semantics of aggregation and reduction.

    **Serialization requirement:** Stats objects are sent from worker
    processes to the logger process via inter-process communication.
    All subclass data must be serializable (e.g., lists of primitives,
    not numpy arrays or GPU tensors).

    **Lifecycle in a multi-rank deployment:**

    1. Each TP rank records observations into its own ``KVConnectorStats``
       instance (e.g., via ``record_transfer()``).
    2. Ranks' stats are merged via ``aggregate()`` — typically by
       concatenating observation lists — producing a single stats
       object per worker.
    3. The merged stats are serialized and sent to the logger process.
    4. The logger accumulates stats from multiple workers via
       ``KVConnectorLogging.observe()``.
    5. At the end of a reporting interval, ``reduce()`` computes
       summary statistics over the fully aggregated data.
    6. ``log()`` prints the summary; Prometheus ``observe()`` feeds
       the data to monitoring systems.

    Subclasses must implement: ``reset()``, ``aggregate()``,
    ``reduce()``, ``is_empty()``.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable representation of this stats object.

        Used when sending stats from a worker process to the logger
        process. The returned dict should contain only JSON-serializable
        or otherwise IPC-safe types (lists, dicts, numbers, strings).

        Returns:
            The ``data`` dict containing all raw observations.
        """
        return self.data

    def reset(self):
        """Reset the stats container to an empty state.

        Called after ``clone_and_reset()`` snapshots the current data,
        or at initialization. Subclasses must clear all accumulated
        observations so that the next reporting interval starts fresh.
        """
        raise NotImplementedError

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        """Merge another stats object's observations into this one.

        In multi-rank (TP > 1) deployments, each rank produces its own
        ``KVConnectorStats``. This method combines them into a single
        object — typically by concatenating observation lists — so that
        ``reduce()`` can compute summary statistics over the full pool.

        After aggregation, rank-level identity is lost. All observations
        are treated as a single combined dataset for statistical
        purposes (averages, percentiles, throughput).

        Args:
            other: Another rank's (or worker's) stats to merge in.

        Returns:
            ``self``, now containing the merged data.
        """
        raise NotImplementedError

    def reduce(self) -> dict[str, int | float]:
        """Compute summary statistics over the aggregated observation pool.

        Called by the logger at the end of a reporting interval, after
        all workers' stats have been accumulated. Subclasses implement
        this to produce human-readable summaries (averages, percentiles,
        throughput, counts) from the raw observation lists.

        The returned dict is used for CLI logging, formatted as::

            KV Transfer metrics: key1=value1, key2=value2, ...

        **Important:** The data passed to ``reduce()`` has already been
        aggregated across all TP ranks (and potentially across multiple
        workers). See the implementing subclass docstring for exact
        semantics of each output metric.

        Returns:
            A dict mapping human-readable metric names (e.g.,
            ``"Avg xfer time (ms)"``) to their computed values.
        """
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Check whether this stats object contains any reportable data.

        Returns True if there are no observations of any kind (no
        successful transfers, no failures, no expired requests).
        Used by ``KVConnectorLogging`` to skip logging when there is
        nothing to report for the current interval.

        Subclasses should ensure that failure-only data is NOT
        considered empty — failure counts are valuable even when no
        transfers succeeded.
        """
        raise NotImplementedError


class KVConnectorLogging:
    """Manages the CLI logging pipeline for KV connector transfer metrics.

    This class orchestrates the ``observe() → aggregate() → reduce() → log()``
    pipeline for periodic CLI metric reporting. It lives in the **logger
    process** (not the worker processes) and receives stats that have
    already been aggregated across all TP ranks within each worker.

    **Pipeline overview:**

    1. ``observe(data)``: Called periodically (not on a fixed logging
       interval) when a connector syncs with the scheduler. Each call
       provides stats from one worker, pre-aggregated across its TP
       ranks. Multiple calls within a single logging interval are
       accumulated via ``aggregate()``.

    2. ``log()``: Called on the logging interval. Reduces the accumulated
       observations to summary stats and prints them. Then resets.

    **Multi-worker accumulation:**

    If there are multiple workers (e.g., multiple engine instances),
    each worker's aggregated stats arrive as separate ``observe()``
    calls. These are accumulated into ``transfer_stats_accumulator``
    via successive ``aggregate()`` calls, so that ``reduce()`` at log
    time sees the combined data from all workers for the interval.

    Example log output::

        KV Transfer metrics: Num successful transfers=4, Avg xfer time (ms)=1.381, ...
    """

    def __init__(self, kv_transfer_config: KVTransferConfig | None):
        """Initialize the logging pipeline.

        Looks up the connector class from the KV transfer config so that
        it can later instantiate the correct ``KVConnectorStats`` subclass
        via ``build_kv_connector_stats()``.

        Args:
            kv_transfer_config: The KV transfer configuration specifying
                which connector to use. If None or has no connector
                configured, logging is effectively disabled.
        """
        if kv_transfer_config and kv_transfer_config.kv_connector:
            self.connector_cls = KVConnectorFactory.get_connector_class(
                kv_transfer_config
            )
        self.reset()

    def reset(self):
        """Clear the accumulator for the next logging interval.

        Sets ``transfer_stats_accumulator`` to None so that the next
        ``observe()`` call starts a fresh accumulation cycle. Called
        at initialization and at the end of each ``log()`` invocation.
        """
        self.transfer_stats_accumulator: KVConnectorStats | None = None

    def observe(self, transfer_stats_data: dict[str, Any]):
        """Receive and accumulate pre-aggregated stats from a worker.

        This is the entry point for the CLI logging pipeline. It is
        called periodically when a connector syncs with the scheduler
        — **not** on the logging interval itself. Multiple calls may
        occur between log invocations.

        Each call provides stats from a single worker that have already
        been **aggregated across all TP ranks** within that worker
        (via the connector's ``aggregate()`` method). The raw data is
        a dict of lists, where each list contains per-transfer
        observations from all ranks on that worker.

        If multiple workers exist (e.g., multiple engine instances),
        successive ``observe()`` calls accumulate their data via
        ``KVConnectorStats.aggregate()``, so that ``log()`` sees the
        combined observations from all workers.

        Args:
            transfer_stats_data: The ``data`` dict from a worker's
                aggregated ``KVConnectorStats`` object. Contains lists
                of per-transfer observations (durations, byte counts,
                etc.) already merged across all TP ranks.

        Example flow with 2 workers and 30s log interval::

            t=0s:   worker 1 sends stats  → accumulated
            t=5s:   worker 2 sends stats  → accumulated (merged)
            t=10s:  worker 1 sends stats  → accumulated (merged)
            t=30s:  log() called → reduce() over all accumulated data → reset
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
        """Reduce accumulated observations and log a summary line.

        Called once per logging interval (e.g., every 30 seconds). If
        there are accumulated observations, computes summary statistics
        via ``reduce()``, formats them into a human-readable log line,
        prints it, and resets the accumulator for the next interval.

        The log line has the format::

            KV Transfer metrics: Key1=value1, Key2=value2, ...

        If no data was accumulated (no transfers or failures in this
        interval), this is a no-op — no log line is emitted.

        **Interpreting the log line in multi-rank deployments:**

        The values in the log line are computed over the combined pool
        of observations from all TP ranks and all workers. See the
        connector's ``KVConnectorStats`` subclass docstring (e.g.,
        ``NixlKVConnectorStats``) for the exact semantics of each
        metric in this context.

        Args:
            log_fn: The logging function to use. Defaults to
                ``logger.info``. Can be overridden for testing.
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
    """Base class for per-connector Prometheus metric registration and recording.

    Subclasses (e.g., ``NixlPromMetrics``) define the specific Prometheus
    histograms, counters, and gauges for their connector and implement
    ``observe()`` to feed data into those metrics.

    Unlike the CLI logging path (which ``reduce()``s observations into
    summary stats), the Prometheus path records **each individual
    observation** into a histogram or increments a counter. This means
    Prometheus retains the full distribution of values, enabling
    external tools (Grafana, etc.) to compute arbitrary percentiles,
    rates, and aggregations.

    **Multi-rank semantics for Prometheus:**

    The ``observe()`` method receives pre-aggregated data (all TP
    ranks' observations concatenated). Each observation is recorded
    individually into Prometheus histograms, so the histogram sees
    the same combined distribution as the CLI log line. This is
    deliberate — it keeps the two output paths consistent.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        """Initialize Prometheus metric base classes and label configuration.

        Stores the metric class references (``Gauge``, ``Counter``,
        ``Histogram``) and label configuration so that subclasses can
        register their specific metrics in their own ``__init__``.

        Args:
            vllm_config: The full vLLM configuration, used to access
                ``kv_transfer_config`` and other engine settings.
            metric_types: Mapping from abstract metric types (``Gauge``,
                ``Counter``, ``Histogram``) to the concrete Prometheus
                client classes to use. This indirection supports custom
                metric implementations (e.g., for testing or alternative
                backends).
            labelnames: List of Prometheus label names that all metrics
                in this connector must include (e.g., ``["model_name",
                "engine"]``).
            per_engine_labelvalues: Mapping from engine index to the
                label values for that engine. Used by
                ``create_metric_per_engine()`` to create per-engine
                metric instances with the correct label values.
        """

        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._gauge_cls = metric_types[Gauge]
        self._counter_cls = metric_types[Counter]
        self._histogram_cls = metric_types[Histogram]
        self._labelnames = labelnames
        self.per_engine_labelvalues = per_engine_labelvalues

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Record transfer statistics to Prometheus metrics.

        Subclasses must implement this to feed each individual observation
        into the appropriate Prometheus histogram or counter. The data
        arrives **pre-aggregated** across all TP ranks — observations
        from all ranks are already concatenated into the lists in
        ``transfer_stats_data``.

        For **histogram** metrics (transfer time, bytes, etc.), each
        list item is recorded as a separate histogram sample via
        ``histogram.observe(value)``. This means Prometheus sees the
        full combined distribution from all ranks.

        For **counter** metrics (failure counts), each list item
        increments the counter via ``counter.inc(value)``.

        Args:
            transfer_stats_data: The ``data`` dict from an aggregated
                ``KVConnectorStats`` object. Contains lists of
                per-transfer observations from all ranks.
            engine_idx: Index of the engine instance for multi-engine
                setups. Used to select the correct per-engine Prometheus
                metric instance. Defaults to 0.
        """
        raise NotImplementedError


class KVConnectorProm:
    """Support for registering and recording per-connector Prometheus metrics.

    This is the Prometheus counterpart to ``KVConnectorLogging``. While
    ``KVConnectorLogging`` handles CLI log output (``reduce()`` to
    summary stats), this class handles Prometheus output (record each
    observation individually into histograms/counters).

    Lifecycle:

    1. ``__init__()``: Looks up the connector class and calls
       ``build_prom_metrics()`` to register all Prometheus metrics
       (histograms, counters, gauges) for this connector.
    2. ``observe(data, engine_idx)``: Called by the metrics system to
       feed pre-aggregated transfer data into Prometheus metrics.
       Delegates to the connector's ``KVConnectorPromMetrics.observe()``.

    **Relationship to CLI logging:**

    Both paths receive the same pre-aggregated data, but they process
    it differently:

    - CLI: ``reduce()`` computes summary stats (avg, P90, throughput)
      from the pooled observations.
    - Prometheus: Each observation is recorded individually, preserving
      the full distribution for external analysis.

    This means Prometheus metrics and CLI log lines reflect the same
    underlying data with the same aggregation semantics.
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
        """Register Prometheus metrics for the configured KV connector.

        Looks up the connector class from the KV transfer config and
        calls ``build_prom_metrics()`` to instantiate all Prometheus
        metric objects (histograms, counters, gauges) for this connector.
        If no connector is configured, ``self.prom_metrics`` remains
        None and all subsequent ``observe()`` calls are no-ops.

        Args:
            vllm_config: The full vLLM configuration.
            labelnames: Prometheus label names for all registered metrics.
            per_engine_labelvalues: Mapping from engine index to label
                values, used to create per-engine metric instances.
        """

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
