# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats container for the Mooncake connector."""

import threading
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
class MooncakeKVConnectorStats(KVConnectorStats):
    """Container for Mooncake KV transfer performance metrics.

    `_lock` serializes record_* against clone_and_reset so each row's
    appends are atomic and column lengths stay aligned. Writers run on
    the sender pool / receiver loop / sender loop; reader runs on the
    main worker thread.
    """

    def __post_init__(self):
        self._lock = threading.Lock()
        if not self.data:
            self.reset()

    # threading.Lock is not picklable; strip it from the wire form and
    # rebuild a fresh per-process lock on the receiver side.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def reset(self):
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],
            "bytes_transferred": [],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_recvs": [],
            "num_kv_expired_reqs": [],
        }

    def record_transfer(self, duration_s: float, total_bytes: int, num_descs: int):
        with self._lock:
            self.data["transfer_duration"].append(duration_s)
            self.data["bytes_transferred"].append(total_bytes)
            self.data["num_descriptors"].append(num_descs)

    # Failure counters store a list of 1s so a future Prom counter can iterate
    # with .inc(list_item), mirroring NIXL's NixlPromMetrics.observe.
    def record_failed_transfer(self):
        with self._lock:
            self.data["num_failed_transfers"].append(1)

    def record_failed_recv(self):
        with self._lock:
            self.data["num_failed_recvs"].append(1)

    def record_kv_expired_req(self):
        with self._lock:
            self.data["num_kv_expired_reqs"].append(1)

    def clone_and_reset(self) -> "MooncakeKVConnectorStats":
        # Copy lists under the lock for length alignment; return a fresh
        # instance so the snapshot has its own _lock.
        with self._lock:
            snapshot_data: dict[str, list[float | int]] = {
                k: list(v) for k, v in self.data.items()
            }
            self.reset()
        return MooncakeKVConnectorStats(data=snapshot_data)

    def is_empty(self) -> bool:
        return (
            self.num_successful_transfers == 0
            and len(self.data["num_failed_transfers"]) == 0
            and len(self.data["num_failed_recvs"]) == 0
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
        num_failed_transfers = len(self.data["num_failed_transfers"])
        num_failed_recvs = len(self.data["num_failed_recvs"])
        num_kv_expired_reqs = len(self.data["num_kv_expired_reqs"])

        if self.num_successful_transfers == 0:
            return {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
                "Num failed transfers": num_failed_transfers,
                "Num failed recvs": num_failed_recvs,
                "Num KV expired reqs": num_kv_expired_reqs,
            }

        xfer_time = np.asarray(self.data["transfer_duration"])
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = mb.sum()
        avg_mb = total_mb / n
        total_time_seconds = xfer_time.sum()
        throughput_mb_s = (
            total_mb / total_time_seconds if total_time_seconds > 0 else 0.0
        )

        return {
            "Num successful transfers": n,
            "Avg xfer time (ms)": round(xfer_time.mean() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90).item() * 1e3, 3),
            "Avg MB per transfer": round(avg_mb, 3),
            "Throughput (MB/s)": round(throughput_mb_s, 3),
            "Avg number of descriptors": round(descs.mean(), 1),
            "Num failed transfers": num_failed_transfers,
            "Num failed recvs": num_failed_recvs,
            "Num KV expired reqs": num_kv_expired_reqs,
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


class MooncakePromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for Mooncake KV transfer."""

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
        mooncake_histogram_xfer_time = self._histogram_cls(
            name="vllm:mooncake_xfer_time_seconds",
            documentation=(
                "Histogram of transfer duration for Mooncake"
                " KV Cache transfers."
            ),
            buckets=buckets[1:],
            labelnames=labelnames,
        )
        self.mooncake_histogram_xfer_time = create_metric_per_engine(
            mooncake_histogram_xfer_time, self.per_engine_labelvalues
        )
        buckets = [2 ** (10 + i) for i in range(1, 25, 2)]
        mooncake_histogram_bytes_transferred = self._histogram_cls(
            name="vllm:mooncake_bytes_transferred",
            documentation=(
                "Histogram of bytes transferred per Mooncake"
                " KV Cache transfer."
            ),
            buckets=buckets,
            labelnames=labelnames,
        )
        self.mooncake_histogram_bytes_transferred = create_metric_per_engine(
            mooncake_histogram_bytes_transferred, self.per_engine_labelvalues
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
        mooncake_histogram_num_descriptors = self._histogram_cls(
            name="vllm:mooncake_num_descriptors",
            documentation=(
                "Histogram of number of descriptors per Mooncake"
                " KV Cache transfer."
            ),
            buckets=buckets,
            labelnames=labelnames,
        )
        self.mooncake_histogram_num_descriptors = create_metric_per_engine(
            mooncake_histogram_num_descriptors, self.per_engine_labelvalues
        )
        counter_mooncake_num_failed_transfers = self._counter_cls(
            name="vllm:mooncake_num_failed_transfers",
            documentation="Number of failed Mooncake KV Cache transfers.",
            labelnames=labelnames,
        )
        self.counter_mooncake_num_failed_transfers = create_metric_per_engine(
            counter_mooncake_num_failed_transfers, self.per_engine_labelvalues
        )
        counter_mooncake_num_failed_recvs = self._counter_cls(
            name="vllm:mooncake_num_failed_recvs",
            documentation="Number of failed Mooncake KV Cache receives.",
            labelnames=labelnames,
        )
        self.counter_mooncake_num_failed_recvs = create_metric_per_engine(
            counter_mooncake_num_failed_recvs, self.per_engine_labelvalues
        )
        counter_mooncake_num_kv_expired_reqs = self._counter_cls(
            name="vllm:mooncake_num_kv_expired_reqs",
            documentation="Number of requests that had their KV expire. "
            "NOTE: This metric is tracked on the P instance.",
            labelnames=labelnames,
        )
        self.counter_mooncake_num_kv_expired_reqs = create_metric_per_engine(
            counter_mooncake_num_kv_expired_reqs, self.per_engine_labelvalues
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for prom_obj, list_item_key in zip(
            [
                self.mooncake_histogram_xfer_time,
                self.mooncake_histogram_bytes_transferred,
                self.mooncake_histogram_num_descriptors,
            ],
            [
                "transfer_duration",
                "bytes_transferred",
                "num_descriptors",
            ],
        ):
            for list_item in transfer_stats_data[list_item_key]:
                prom_obj[engine_idx].observe(list_item)
        for counter_obj, counter_item_key in zip(
            [
                self.counter_mooncake_num_failed_transfers,
                self.counter_mooncake_num_failed_recvs,
                self.counter_mooncake_num_kv_expired_reqs,
            ],
            [
                "num_failed_transfers",
                "num_failed_recvs",
                "num_kv_expired_reqs",
            ],
        ):
            for list_item in transfer_stats_data[counter_item_key]:
                counter_obj[engine_idx].inc(list_item)
