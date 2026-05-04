# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics for the NIXL KV cache connector.

This module implements the **observe → aggregate → reduce → log** pipeline
for NIXL disaggregated-prefill KV cache transfers:

1. **Observe** – ``NixlConnectorWorker`` calls ``record_transfer()``
   (or ``record_failed_*``/``record_kv_expired_req()``) after each NIXL
   operation completes, appending raw telemetry to per-metric lists.

2. **Aggregate** – The ``MultiprocExecutor`` gathers per-worker stats
   via ``get_kv_connector_stats()`` (which calls ``clone_and_reset()``)
   and merges them into a single ``NixlKVConnectorStats`` with
   ``aggregate()``.  The aggregated ``data`` dict is then serialized and
   forwarded to the engine/logger process.

3. **Reduce** – ``KVConnectorLogging.log()`` calls ``reduce()`` to
   collapse the accumulated lists into human-readable scalars (averages,
   P90 latencies, throughput) suitable for periodic CLI log lines.

4. **Log / Export** – Two parallel sinks consume the raw ``data`` dict:
   - ``KVConnectorLogging`` logs the reduced summary via the Python
     logger at each logging interval.
   - ``NixlPromMetrics.observe()`` pushes every individual observation
     into Prometheus histograms/counters for scraping.

Metrics tracked
~~~~~~~~~~~~~~~

+---------------------------+------+-----------------------------------------+
| Key                       | Unit | Description                             |
+===========================+======+=========================================+
| transfer_duration         | s    | RDMA transfer wall-clock time           |
+---------------------------+------+-----------------------------------------+
| post_duration             | s    | Post-transfer bookkeeping time          |
+---------------------------+------+-----------------------------------------+
| bytes_transferred         | B    | Payload size of the transfer            |
+---------------------------+------+-----------------------------------------+
| num_descriptors           | –    | NIXL descriptors used per transfer      |
+---------------------------+------+-----------------------------------------+
| num_failed_transfers      | –    | Counter of failed transfer operations   |
+---------------------------+------+-----------------------------------------+
| num_failed_notifications  | –    | Counter of failed send_notif calls      |
+---------------------------+------+-----------------------------------------+
| num_kv_expired_reqs       | –    | Counter of requests whose KV expired    |
|                           |      | before being consumed (P-instance only) |
+---------------------------+------+-----------------------------------------+
"""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.distributed.nixl_utils import nixlXferTelemetry
from vllm.v1.metrics.utils import create_metric_per_engine


@dataclass
class NixlKVConnectorStats(KVConnectorStats):
    """Accumulates per-transfer NIXL telemetry for a single logging interval.

    Each worker owns one instance.  Individual RDMA completions are
    recorded via ``record_transfer()``; failure events via the
    ``record_failed_*`` helpers.  At the end of each scheduler step the
    worker snapshots its stats with ``clone_and_reset()`` and hands them
    to the executor, which ``aggregate()``s across TP ranks before the
    data reaches ``reduce()`` (for CLI logging) or
    ``NixlPromMetrics.observe()`` (for Prometheus export).

    All fields in ``data`` are plain lists so the object is
    pickle-serializable across process boundaries.

    See Also:
        :class:`KVConnectorStats` – abstract base class defining the
        ``reset``/``aggregate``/``reduce``/``is_empty`` contract.
    """

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        """Clear all accumulated observations for the next interval.

        Each key maps to a list that ``record_*`` methods append to.
        Durations are stored in **seconds**; byte counts in **bytes**.
        """
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],
            "post_duration": [],
            "bytes_transferred": [],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_notifications": [],
            "num_kv_expired_reqs": [],
        }

    def record_transfer(self, res: nixlXferTelemetry):
        """Append telemetry from one successful NIXL transfer.

        Args:
            res: Telemetry struct returned by
                ``nixl_wrapper.get_xfer_telemetry()``.  Durations in the
                struct are in **microseconds**; they are converted to
                seconds here so all time values share a common unit.
        """
        self.data["transfer_duration"].append(res.xferDuration / 1e6)
        self.data["post_duration"].append(res.postDuration / 1e6)
        self.data["bytes_transferred"].append(res.totalBytes)
        self.data["num_descriptors"].append(res.descCount)

    def record_failed_transfer(self):
        """Record a failed NIXL transfer operation."""
        self.data["num_failed_transfers"].append(1)

    def record_failed_notification(self):
        """Record a failed NIXL notification (send_notif)."""
        self.data["num_failed_notifications"].append(1)

    def record_kv_expired_req(self):
        """Record a request whose KV blocks expired before consumption."""
        self.data["num_kv_expired_reqs"].append(1)

    def clone_and_reset(self) -> "NixlKVConnectorStats":
        """Snapshot the current stats and reset for the next interval.

        Returns a shallow copy of this object (with the accumulated
        lists) while resetting ``self`` to empty lists.  Called by the
        worker's ``get_kv_connector_stats()`` so the executor can read
        the snapshot without racing against new ``record_*`` calls.
        """
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        """True when no observations (successes *or* failures) were recorded.

        Failure-only intervals are **not** considered empty so that
        Prometheus counters still get incremented even when every
        transfer in the interval failed.
        """
        return (
            self.num_successful_transfers == 0
            and len(self.data["num_failed_transfers"]) == 0
            and len(self.data["num_failed_notifications"]) == 0
            and len(self.data["num_kv_expired_reqs"]) == 0
        )

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        """Merge another stats snapshot (typically from a different TP rank).

        Extends each list in ``self.data`` with the corresponding list
        from *other*, combining observations from multiple workers into
        a single container before ``reduce()`` is called.
        """
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        """Collapse accumulated observations into CLI-friendly scalars.

        Produces a dict whose keys are human-readable metric labels and
        whose values are rounded numbers suitable for a single log line.
        This is the final step of the pipeline before
        ``KVConnectorLogging.log()`` emits the summary.

        Returns:
            A dict with keys like ``"Avg xfer time (ms)"``,
            ``"Throughput (MB/s)"``, etc.  When no successful transfers
            were recorded the dict is all-zeros—failure-only intervals
            are reported exclusively through Prometheus counters.
        """
        if self.num_successful_transfers == 0:
            # Nothing to summarize for the CLI.  Failures are still
            # exported by NixlPromMetrics.observe() through the
            # counter_nixl_num_failed_* counters.
            return {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg post time (ms)": 0,
                "P90 post time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
            }

        # Convert the per-transfer lists to numpy arrays for vectorized
        # statistics.  Durations are in seconds (converted at record time).
        xfer_time = np.asarray(self.data["transfer_duration"])
        post_time = np.asarray(self.data["post_duration"])
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20  # B → MiB
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = mb.sum()
        avg_mb = total_mb / n

        # Throughput = total payload / total wall-clock transfer time,
        # giving an aggregate MB/s across all transfers in the interval.
        total_time_seconds = xfer_time.sum()
        throughput_mb_s = total_mb / total_time_seconds

        return {
            "Num successful transfers": n,
            "Avg xfer time (ms)": round(xfer_time.mean() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90).item() * 1e3, 3),
            "Avg post time (ms)": round(post_time.mean() * 1e3, 3),
            "P90 post time (ms)": round(np.percentile(post_time, 90).item() * 1e3, 3),
            "Avg MB per transfer": round(avg_mb, 3),
            "Throughput (MB/s)": round(throughput_mb_s, 3),
            "Avg number of descriptors": round(descs.mean(), 1),
        }

    @property
    def num_successful_transfers(self) -> int:
        """Count of successful transfers (those with recorded telemetry)."""
        return len(self.data["transfer_duration"])


class NixlPromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for NIXL KV cache transfers.

    Registers per-engine histograms for transfer latency, post-processing
    time, bytes transferred, and descriptor counts, plus counters for
    failure events.  ``observe()`` is called by ``KVConnectorProm`` with
    the raw ``data`` dict from ``NixlKVConnectorStats`` so every
    individual observation is recorded (as opposed to the reduced
    averages emitted by CLI logging).

    Metric names all share the ``vllm:nixl_`` prefix.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        buckets = [
            0.001,
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
        ]
        nixl_histogram_xfer_time = self._histogram_cls(
            name="vllm:nixl_xfer_time_seconds",
            documentation="Histogram of transfer duration for NIXL KV Cache transfers.",
            buckets=buckets[1:],
            labelnames=labelnames,
        )
        self.nixl_histogram_xfer_time = create_metric_per_engine(
            nixl_histogram_xfer_time, self.per_engine_labelvalues
        )
        nixl_histogram_post_time = self._histogram_cls(
            name="vllm:nixl_post_time_seconds",
            documentation="Histogram of transfer post time for NIXL KV"
            " Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_post_time = create_metric_per_engine(
            nixl_histogram_post_time, self.per_engine_labelvalues
        )
        buckets = [2 ** (10 + i) for i in range(1, 25, 2)]  # 2 KiB → 16 GiB
        nixl_histogram_bytes_transferred = self._histogram_cls(
            name="vllm:nixl_bytes_transferred",
            documentation="Histogram of bytes transferred per NIXL KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_bytes_transferred = create_metric_per_engine(
            nixl_histogram_bytes_transferred, self.per_engine_labelvalues
        )
        buckets = [
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
        ]
        nixl_histogram_num_descriptors = self._histogram_cls(
            name="vllm:nixl_num_descriptors",
            documentation="Histogram of number of descriptors per NIXL"
            "  KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_num_descriptors = create_metric_per_engine(
            nixl_histogram_num_descriptors, self.per_engine_labelvalues
        )
        counter_nixl_num_failed_transfers = self._counter_cls(
            name="vllm:nixl_num_failed_transfers",
            documentation="Number of failed NIXL KV Cache transfers.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_failed_transfers = create_metric_per_engine(
            counter_nixl_num_failed_transfers, self.per_engine_labelvalues
        )
        counter_nixl_num_failed_notifications = self._counter_cls(
            name="vllm:nixl_num_failed_notifications",
            documentation="Number of failed NIXL KV Cache notifications.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_failed_notifications = create_metric_per_engine(
            counter_nixl_num_failed_notifications, self.per_engine_labelvalues
        )

        counter_nixl_num_kv_expired_reqs = self._counter_cls(
            name="vllm:nixl_num_kv_expired_reqs",
            documentation="Number of requests that had their KV expire. "
            "NOTE: This metric is tracked on the P instance.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_kv_expired_reqs = create_metric_per_engine(
            counter_nixl_num_kv_expired_reqs, self.per_engine_labelvalues
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Push raw per-transfer observations to Prometheus metrics.

        Unlike ``reduce()`` which computes summary statistics, this
        method feeds each individual value into the corresponding
        histogram or counter so Prometheus can compute its own
        aggregations over arbitrary time windows.

        Args:
            transfer_stats_data: The ``data`` dict from a
                ``NixlKVConnectorStats`` instance.
            engine_idx: Index identifying which engine's label set to
                record against (for multi-engine deployments).
        """
        for prom_obj, list_item_key in zip(
            [
                self.nixl_histogram_xfer_time,
                self.nixl_histogram_post_time,
                self.nixl_histogram_bytes_transferred,
                self.nixl_histogram_num_descriptors,
            ],
            [
                "transfer_duration",
                "post_duration",
                "bytes_transferred",
                "num_descriptors",
            ],
        ):
            for list_item in transfer_stats_data[list_item_key]:
                prom_obj[engine_idx].observe(list_item)
        for counter_obj, counter_item_key in zip(
            [
                self.counter_nixl_num_failed_transfers,
                self.counter_nixl_num_failed_notifications,
                self.counter_nixl_num_kv_expired_reqs,
            ],
            ["num_failed_transfers", "num_failed_notifications", "num_kv_expired_reqs"],
        ):
            for list_item in transfer_stats_data[counter_item_key]:
                counter_obj[engine_idx].inc(list_item)
