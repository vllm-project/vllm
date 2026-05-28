# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics for the NIXL KV connector.

Metrics Pipeline
================

This module implements the observe → aggregate → reduce → log pipeline for
NIXL KV cache transfer telemetry. The pipeline operates across tensor-parallel
(TP) ranks and produces both periodic CLI log summaries and Prometheus metrics.

Data flow::

    ┌─────────────────────────────────────────────────────────────────┐
    │  Worker Process (one per TP rank)                                │
    │                                                                  │
    │  NixlConnectorWorker._pop_done_transfers()                       │
    │    └─> nixl_wrapper.get_xfer_telemetry(handle)                   │
    │        └─> xfer_stats.record_transfer(telemetry)  ← per-rank    │
    │                                                                  │
    │  NixlConnectorWorker.get_kv_connector_stats()                    │
    │    └─> xfer_stats.clone_and_reset()  ← snapshot & reset         │
    └─────────────────────────────────────────────────────────────────┘
                              │ (serialized dict via IPC)
                              ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Scheduler / Engine Core                                         │
    │                                                                  │
    │  For each TP rank's stats:                                       │
    │    accumulator.aggregate(rank_stats)  ← list.extend()            │
    │                                                                  │
    │  All ranks' observations are pooled into one flat list.          │
    └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  KVConnectorLogging.log()                                        │
    │    └─> accumulator.reduce()  ← summary stats over pooled data   │
    │        └─> logger.info("KV Transfer metrics: ...")               │
    │                                                                  │
    │  KVConnectorProm.observe()                                       │
    │    └─> NixlPromMetrics.observe()  ← per-observation histograms  │
    └─────────────────────────────────────────────────────────────────┘

Multi-Rank Aggregation Semantics
================================

All metrics are **aggregated across all TP ranks** before summary statistics
are computed. Each TP rank independently records per-transfer telemetry, and
the scheduler concatenates observations from all ranks via ``aggregate()``
(which calls ``list.extend()``). The ``reduce()`` method then computes
averages, percentiles, and throughput over the **combined** pool.

Implications for users reading the CLI log line:

- **"Num successful transfers"**: Total count across ALL TP ranks, not
  per-rank. With TP=8 and 10 requests, you may see up to 80 transfers
  (each rank transfers its shard independently).

- **"Avg xfer time (ms)"**: Mean transfer duration across all individual
  rank-level transfer operations.

- **"P90 xfer time (ms)"**: 90th percentile computed over the combined
  distribution of all ranks' transfer times.

- **"Avg MB per transfer"**: Average bytes per individual rank-level
  transfer, NOT the total bytes moved for a single logical KV transfer.
  With TP=8, multiply by 8 to estimate the total data moved per request.

- **"Throughput (MB/s)"**: Computed as ``total_MB_all_ranks /
  total_time_all_ranks``. This represents the average per-rank throughput,
  NOT the aggregate system throughput. For aggregate throughput with TP=N
  ranks transferring concurrently, multiply by N.

- **"Avg number of descriptors"**: Mean descriptor count per rank-level
  transfer. Each descriptor maps to one NIXL memory region (typically one
  KV cache block).

Metric Definitions
==================

+---------------------------+-------+------------------------------------------+
| Raw field                 | Unit  | Description                              |
+===========================+=======+==========================================+
| transfer_duration         | s     | Wall-clock time from initiating the RDMA |
|                           |       | transfer to completion notification.     |
|                           |       | Includes network RTT + data movement.    |
+---------------------------+-------+------------------------------------------+
| post_duration             | s     | Time to post (submit) the transfer       |
|                           |       | request to the NIXL/transport layer.     |
|                           |       | High values indicate contention on the   |
|                           |       | transport submission path.               |
+---------------------------+-------+------------------------------------------+
| bytes_transferred         | bytes | Total bytes in one rank-level transfer   |
|                           |       | operation (sum of all descriptors).      |
+---------------------------+-------+------------------------------------------+
| num_descriptors           | count | Number of NIXL descriptors (memory       |
|                           |       | regions / KV blocks) in one transfer.    |
+---------------------------+-------+------------------------------------------+
| num_failed_transfers      | count | Transfers that reached a terminal error  |
|                           |       | state (not DONE, not PROC).              |
+---------------------------+-------+------------------------------------------+
| num_failed_notifications  | count | Failed send_notif() calls (inability to  |
|                           |       | notify remote that a transfer completed  |
|                           |       | or a full-prefix-cache-hit occurred).    |
+---------------------------+-------+------------------------------------------+
| num_kv_expired_reqs       | count | Requests whose KV blocks expired before  |
|                           |       | being consumed (tracked on P instance).  |
+---------------------------+-------+------------------------------------------+

Unit Conversions
================

- NIXL telemetry reports durations in **microseconds (µs)**.
  ``record_transfer()`` converts to **seconds** (÷ 1e6) for internal storage.
- ``reduce()`` converts seconds to **milliseconds** (× 1e3) for CLI display.
- ``reduce()`` converts bytes to **MiB** (÷ 2^20) for CLI display.
- Prometheus histograms store durations in **seconds** and bytes as raw
  **bytes** (no conversion), per OpenMetrics conventions.
"""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

if TYPE_CHECKING:
    from vllm.distributed.nixl_utils import nixlXferTelemetry


@dataclass
class NixlKVConnectorStats(KVConnectorStats):
    """Per-interval container for NIXL transfer telemetry.

    Lifecycle:
        1. Each NixlConnectorWorker (one per TP rank) owns an instance.
        2. As transfers complete, ``record_transfer()`` appends observations.
        3. Periodically, the engine calls ``clone_and_reset()`` to snapshot
           the accumulated data and clear the container for the next interval.
        4. The scheduler receives serialized snapshots from all TP workers and
           merges them via ``aggregate()`` (flat concatenation).
        5. ``reduce()`` is called once on the merged data to produce the CLI
           log line summary.

    Serialization:
        The ``data`` dict contains only lists of floats/ints, making it
        safe for IPC between worker and scheduler processes. No NIXL handles
        or CUDA tensors are stored.
    """

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        """Clear all accumulated observations for the next collection interval.

        Called after ``clone_and_reset()`` snapshots the current state, and
        during initial construction.
        """
        # Must be serializable (sent via IPC from worker to scheduler).
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],
            "post_duration": [],
            "bytes_transferred": [],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_notifications": [],
            "num_kv_expired_reqs": [],
        }

    def record_transfer(self, res: "nixlXferTelemetry"):
        """Record telemetry for one successful rank-level transfer.

        Called by ``NixlConnectorWorker._pop_done_transfers()`` when a NIXL
        transfer handle reaches the DONE state. Each TP rank records its own
        transfers independently.

        Args:
            res: NIXL telemetry struct with durations in microseconds (µs)
                 and byte counts. Durations are converted to seconds here.
        """
        # Convert µs → s for internal storage.
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
        """Record a request that had its KV blocks expire."""
        self.data["num_kv_expired_reqs"].append(1)

    def clone_and_reset(self) -> "NixlKVConnectorStats":
        """Atomically snapshot current observations and reset for next interval.

        Returns a shallow copy of this object containing the accumulated data.
        The original instance is reset to empty. Called by the engine to
        collect stats without missing observations that arrive concurrently.
        """
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        # Do not discard metrics update that are entirely failures related.
        return (
            self.num_successful_transfers == 0
            and len(self.data["num_failed_transfers"]) == 0
            and len(self.data["num_failed_notifications"]) == 0
            and len(self.data["num_kv_expired_reqs"]) == 0
        )

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        """Merge another stats snapshot into this accumulator.

        Uses flat concatenation (list.extend) to pool observations from
        multiple TP ranks or multiple collection intervals. After aggregation,
        the combined lists contain interleaved observations from all ranks
        with no per-rank attribution.

        This is the step where per-rank isolation is lost: once aggregated,
        individual rank contributions cannot be separated.
        """
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        """Compute summary statistics from the pooled observations.

        Produces a dict suitable for the CLI log line::

            KV Transfer metrics: Num successful transfers=80,
            Avg xfer time (ms)=1.381, P90 xfer time (ms)=2.601, ...

        All statistics are computed over the **combined** observations from
        all TP ranks (see module docstring for multi-rank semantics).

        Returns:
            Dict of metric_name → value. Times in ms, sizes in MiB.
        """
        if self.num_successful_transfers == 0:
            # CLI logging only reports successful transfers stats. If all requests in
            # the interval were unsuccessful, Prom will report failures stats instead.
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

        # All arrays contain interleaved observations from all TP ranks.
        xfer_time = np.asarray(self.data["transfer_duration"])  # seconds
        post_time = np.asarray(self.data["post_duration"])  # seconds
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20  # bytes → MiB
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = mb.sum()  # Sum of MiB across ALL ranks' transfers.
        avg_mb = total_mb / n  # Average MiB per rank-level transfer.

        # Throughput = total_MiB_all_ranks / total_time_all_ranks.
        # This is the average per-rank throughput, not aggregate system
        # throughput. For TP=N with concurrent transfers, multiply by N.
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
        return len(self.data["transfer_duration"])


class NixlPromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for NIXL KV cache transfers.

    Registers histograms and counters under the ``vllm:nixl_*`` namespace.
    Unlike CLI logging (which reports aggregated summaries), Prometheus
    receives every individual observation, enabling per-percentile queries
    and Grafana dashboards.

    Metric naming follows OpenMetrics conventions:
        - ``vllm:nixl_xfer_time_seconds`` — transfer duration histogram (s)
        - ``vllm:nixl_post_time_seconds`` — post/submit duration histogram (s)
        - ``vllm:nixl_bytes_transferred`` — bytes per transfer histogram
        - ``vllm:nixl_num_descriptors`` — descriptors per transfer histogram
        - ``vllm:nixl_num_failed_transfers`` — failure counter
        - ``vllm:nixl_num_failed_notifications`` — notification failure counter
        - ``vllm:nixl_num_kv_expired_reqs`` — KV expiration counter (P-side)

    All metrics are labeled per-engine, allowing multi-engine deployments to
    be monitored independently.
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
        # uniform 2kb to 16gb range
        buckets = [2 ** (10 + i) for i in range(1, 25, 2)]
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
        """Record all observations from the stats data dict to Prometheus.

        Unlike ``reduce()`` which produces summaries, this method feeds every
        individual observation into histograms, preserving the full
        distribution for percentile queries via PromQL.

        Args:
            transfer_stats_data: The raw ``NixlKVConnectorStats.data`` dict
                (already aggregated across TP ranks).
            engine_idx: Engine index for multi-engine label selection.
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
