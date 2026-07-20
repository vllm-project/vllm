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
    Collects and aggregates NIXL KV transfer telemetry across all TP ranks.

    Aggregation workflow:
    1. Each TP rank independently records individual transfer observations via record_transfer()
    2. aggregate() concatenates all observation lists from every TP rank using list.extend()
    3. reduce() computes averages, P90 percentiles and throughput over the merged cross-rank dataset

    Metric semantics after aggregation across all ranks:
    - Num successful transfers: Total completed transfers summed across all TP ranks, not per-rank count
    - Avg / P90 xfer time (ms): Mean and P90 transfer latency across all individual
      rank-level transfers.
    - Avg / P90 post time (ms): Time spent submitting the transfer request to the RDMA backend before
      asynchronous data transfer begins.
    - Avg MB per transfer: Average data size of every single rank-level transfer, not total bytes of one global KV cache shift
    - Throughput (MB/s): Total MB across all ranks divided by the total transfer
      time across all ranks. This represents average per-rank throughput, not
      aggregate system throughput.
    - Avg number of descriptors: Average descriptor count across all individual
      rank transfer events

    These semantics are particularly important when interpreting metrics from
    TP > 1 deployments, where a single logical KV transfer may generate
    multiple per-rank transfer observations.
    """

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        # Must be serializable
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
        # Keep metrics units consistent with rest of the code: time us->s
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
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        # All stats are calculated on observations merged from every TP rank.
        # Derived values (averages, P90, throughput) are computed over the
        # combined cross-rank data pool.
        # Throughput metric is per-rank throughput, not total system aggregate.
        # Raw data is stored in seconds and bytes; reduce() converts to
        # milliseconds and MB for display.
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
