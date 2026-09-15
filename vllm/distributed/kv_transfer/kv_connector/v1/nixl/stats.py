# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics for the NIXL connector."""

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
    """
    Container for NIXL KV cache transfer performance metrics.

    This class collects per-transfer telemetry from each TP rank and aggregates
    them for periodic logging and Prometheus metric emission.

    IMPORTANT: Metrics Aggregation Semantics (Multi-Rank / TP > 1)
    ---------------------------------------------------------------
    In tensor-parallel deployments (TP > 1), each TP rank independently records
    its own transfer telemetry via `record_transfer()`. The stats from all ranks
    are then concatenated (via `aggregate()` using `list.extend()`) into a single
    observation pool. The `reduce()` method computes summary statistics (averages,
    percentiles, throughput) over this **combined pool of observations from all
    ranks**.

    This means the logged metrics represent **per-rank averages over the combined
    observation pool**, NOT per-engine totals or aggregate system throughput.
    Specifically:
    - "Num successful transfers" = total count across all TP ranks
    - "Avg MB per transfer" = average over all individual rank-level transfers
    - "Throughput (MB/s)" = total_MB_all_ranks / total_time_all_ranks
      (effectively an average per-rank throughput)
    - Percentiles (P90) = computed over the combined distribution of all ranks

    This design uses fire-and-forget reporting from workers; the logger receives
    pre-aggregated stats and computes final summaries.
    """

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        # Must be serializable for IPC transmission from worker to logger.
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],      # seconds; async data movement time (xferDuration)
            "post_duration": [],          # seconds; RDMA post/submit time (postDuration)
            "bytes_transferred": [],      # bytes; total bytes moved per transfer
            "num_descriptors": [],        # count; RDMA work request descriptors used
            "num_failed_transfers": [],   # count; failed transfer operations
            "num_failed_notifications": [],  # count; failed send_notif calls
            "num_kv_expired_reqs": [],    # count; requests with expired KV blocks (tracked on P)
        }

    def record_transfer(self, res: "nixlXferTelemetry"):
        """Record a successful NIXL transfer's telemetry.

        Args:
            res: NIXL transfer telemetry containing duration, bytes, and descriptors.
                 Time units are converted from microseconds to seconds for consistency.
        """
        # Keep metrics units consistent with rest of the code: time us->s
        self.data["transfer_duration"].append(res.xferDuration / 1e6)
        self.data["post_duration"].append(res.postDuration / 1e6)
        self.data["bytes_transferred"].append(res.totalBytes)
        self.data["num_descriptors"].append(res.descCount)

    def record_failed_transfer(self):
        """Record a failed NIXL transfer operation (data movement failure)."""
        self.data["num_failed_transfers"].append(1)

    def record_failed_notification(self):
        """Record a failed NIXL notification (send_notif failure, pre-transfer)."""
        self.data["num_failed_notifications"].append(1)

    def record_kv_expired_req(self):
        """Record a request that had its KV blocks expire (lease timeout).

        Tracked on the prefiller (P) instance when KV blocks are evicted before
        the decoder can consume them.
        """
        self.data["num_kv_expired_reqs"].append(1)

    def clone_and_reset(self) -> "NixlKVConnectorStats":
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
        """
        Aggregate stats from another instance (typically from a different TP rank).

        Uses `list.extend()` to concatenate observations, building a combined pool
        across all ranks. Called by the logger when collecting stats from workers.
        """
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        """
        Reduce collected observations to summary statistics for CLI logging.

        Computes averages, percentiles, and throughput over the **combined pool
        of observations from all TP ranks**. The returned dict represents
        per-rank averages over the aggregated observation pool, NOT per-engine
        totals or aggregate system throughput.

        Returns:
            Dict with keys:
            - "Num successful transfers": total count across all ranks
            - "Avg xfer time (ms)": mean transfer duration in milliseconds
            - "P90 xfer time (ms)": 90th percentile transfer duration
            - "Avg post time (ms)": mean RDMA post/submit time
            - "P90 post time (ms)": 90th percentile post time
            - "Avg MB per transfer": mean bytes per transfer (converted to MB)
            - "Throughput (MB/s)": total_MB / total_time_seconds (per-rank average)
            - "Avg number of descriptors": mean RDMA descriptors per transfer
        """
        # Compute compact representative stats suitable for CLI logging
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

        xfer_time = np.asarray(self.data["transfer_duration"])
        post_time = np.asarray(self.data["post_duration"])
        # Convert to MB for CLI logging.
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = mb.sum()
        avg_mb = total_mb / n

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
    """
    Prometheus metrics implementation for the NIXL KV connector.

    Registers the following metrics (all per-engine via 'engine' label):
    - vllm:nixl_xfer_time_seconds (Histogram): Transfer duration (async data movement)
    - vllm:nixl_post_time_seconds (Histogram): RDMA post/submit time
    - vllm:nixl_bytes_transferred (Histogram): Bytes moved per transfer
    - vllm:nixl_num_descriptors (Histogram): RDMA work request descriptors per transfer
    - vllm:nixl_num_failed_transfers (Counter): Failed transfer operations
    - vllm:nixl_num_failed_notifications (Counter): Failed send_notif calls
    - vllm:nixl_num_kv_expired_reqs (Counter): Requests with expired KV blocks (P instance)

    Metrics are recorded from pre-aggregated stats data (combined across all TP ranks).
    The observe() method processes each observation in the aggregated lists.
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
        """
        Record NIXL transfer statistics to Prometheus metrics.

        Processes pre-aggregated stats data (combined across all TP ranks) and
        updates the corresponding histograms and counters for the given engine.

        Args:
            transfer_stats_data: Dictionary containing lists of observations from
                all TP ranks (keys: transfer_duration, post_duration,
                bytes_transferred, num_descriptors, num_failed_transfers,
                num_failed_notifications, num_kv_expired_reqs).
            engine_idx: Engine index for multi-engine deployments (default 0).
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
