# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics for the MoRI IO connector."""

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
from vllm.v1.metrics.utils import create_metric_per_engine


@dataclass
class MoRIIOKVConnectorStats(KVConnectorStats):
    """Container for MoRI IO transfer performance metrics.

    Field schema and reduce() output mirror NixlKVConnectorStats so dashboards
    and log parsers see identical shapes across connectors.
    """

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        # Must be serializable: worker -> scheduler over IPC.
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],
            "post_duration": [],
            "bytes_transferred": [],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_notifications": [],
            "num_kv_expired_reqs": [],
        }

    def record_transfer(
        self,
        transfer_duration_s: float,
        post_duration_s: float,
        bytes_transferred: int,
        num_descriptors: int,
    ) -> None:
        self.data["transfer_duration"].append(transfer_duration_s)
        self.data["post_duration"].append(post_duration_s)
        self.data["bytes_transferred"].append(bytes_transferred)
        self.data["num_descriptors"].append(num_descriptors)

    def record_failed_transfer(self) -> None:
        self.data["num_failed_transfers"].append(1)

    def record_failed_notification(self) -> None:
        self.data["num_failed_notifications"].append(1)

    def record_kv_expired_req(self) -> None:
        self.data["num_kv_expired_reqs"].append(1)

    def clone_and_reset(self) -> "MoRIIOKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
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
        if self.num_successful_transfers == 0:
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
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = mb.sum()
        avg_mb = total_mb / n

        total_time_seconds = xfer_time.sum()
        throughput_mb_s = total_mb / total_time_seconds if total_time_seconds > 0 else 0

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


class MoRIIOPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        time_buckets = [
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
        moriio_histogram_xfer_time = self._histogram_cls(
            name="vllm:moriio_xfer_time_seconds",
            documentation="Histogram of transfer duration for MoRI IO"
            " KV Cache transfers.",
            buckets=time_buckets[1:],
            labelnames=labelnames,
        )
        self.moriio_histogram_xfer_time = create_metric_per_engine(
            moriio_histogram_xfer_time, self.per_engine_labelvalues
        )
        moriio_histogram_post_time = self._histogram_cls(
            name="vllm:moriio_post_time_seconds",
            documentation="Histogram of transfer post time for MoRI IO"
            " KV Cache transfers.",
            buckets=time_buckets,
            labelnames=labelnames,
        )
        self.moriio_histogram_post_time = create_metric_per_engine(
            moriio_histogram_post_time, self.per_engine_labelvalues
        )
        # 2kb to 16gb range, same as NIXL.
        byte_buckets = [2 ** (10 + i) for i in range(1, 25, 2)]
        moriio_histogram_bytes_transferred = self._histogram_cls(
            name="vllm:moriio_bytes_transferred",
            documentation="Histogram of bytes transferred per MoRI IO"
            " KV Cache transfers.",
            buckets=byte_buckets,
            labelnames=labelnames,
        )
        self.moriio_histogram_bytes_transferred = create_metric_per_engine(
            moriio_histogram_bytes_transferred, self.per_engine_labelvalues
        )
        desc_buckets = [
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
        moriio_histogram_num_descriptors = self._histogram_cls(
            name="vllm:moriio_num_descriptors",
            documentation="Histogram of number of descriptors per MoRI IO"
            " KV Cache transfers.",
            buckets=desc_buckets,
            labelnames=labelnames,
        )
        self.moriio_histogram_num_descriptors = create_metric_per_engine(
            moriio_histogram_num_descriptors, self.per_engine_labelvalues
        )
        counter_moriio_num_failed_transfers = self._counter_cls(
            name="vllm:moriio_num_failed_transfers",
            documentation="Number of failed MoRI IO KV Cache transfers.",
            labelnames=labelnames,
        )
        self.counter_moriio_num_failed_transfers = create_metric_per_engine(
            counter_moriio_num_failed_transfers, self.per_engine_labelvalues
        )
        counter_moriio_num_failed_notifications = self._counter_cls(
            name="vllm:moriio_num_failed_notifications",
            documentation="Number of failed MoRI IO KV Cache notifications.",
            labelnames=labelnames,
        )
        self.counter_moriio_num_failed_notifications = create_metric_per_engine(
            counter_moriio_num_failed_notifications, self.per_engine_labelvalues
        )
        counter_moriio_num_kv_expired_reqs = self._counter_cls(
            name="vllm:moriio_num_kv_expired_reqs",
            documentation="Number of requests that had their KV expire. "
            "NOTE: This metric is tracked on the P instance.",
            labelnames=labelnames,
        )
        self.counter_moriio_num_kv_expired_reqs = create_metric_per_engine(
            counter_moriio_num_kv_expired_reqs, self.per_engine_labelvalues
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for prom_obj, list_item_key in zip(
            [
                self.moriio_histogram_xfer_time,
                self.moriio_histogram_post_time,
                self.moriio_histogram_bytes_transferred,
                self.moriio_histogram_num_descriptors,
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
                self.counter_moriio_num_failed_transfers,
                self.counter_moriio_num_failed_notifications,
                self.counter_moriio_num_kv_expired_reqs,
            ],
            [
                "num_failed_transfers",
                "num_failed_notifications",
                "num_kv_expired_reqs",
            ],
        ):
            for list_item in transfer_stats_data[counter_item_key]:
                counter_obj[engine_idx].inc(list_item)
