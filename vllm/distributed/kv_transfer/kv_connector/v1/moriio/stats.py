# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics for the MoRIIO connector."""

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

_XFER_TIME_BUCKETS = (
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
)
_BYTES_BUCKETS = tuple(2 ** (10 + i) for i in range(1, 25, 2))
_DESCRIPTOR_BUCKETS = (
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
)


@dataclass
class MoRIIOKVConnectorStats(KVConnectorStats):
    """Container for MoRIIO KV transfer metrics.

    MoRI-IO reports no per-transfer telemetry, so a transfer's duration runs
    from posting its first RDMA op to the connector observing completion. On
    the READ path that is rounded up to the step in which it is polled; on the
    layerwise WRITE path it includes the forward of the later layers.
    Successful transfers are recorded where the data moves: on D in READ mode
    and on P in WRITE mode.

    `_lock` keeps rows aligned between the writer thread recording WRITEs and
    the main thread taking snapshots.
    """

    def __post_init__(self):
        self._lock = threading.Lock()
        if not self.data:
            self.reset()

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
            "num_failed_notifications": [],
            "num_kv_expired_reqs": [],
        }

    def record_transfer(self, duration_s: float, total_bytes: int, num_descs: int):
        with self._lock:
            self.data["transfer_duration"].append(duration_s)
            self.data["bytes_transferred"].append(total_bytes)
            self.data["num_descriptors"].append(num_descs)

    def record_failed_transfer(self, num_transfers: int = 1):
        with self._lock:
            self.data["num_failed_transfers"].extend([1] * num_transfers)

    def record_failed_notification(self):
        with self._lock:
            self.data["num_failed_notifications"].append(1)

    def record_kv_expired_req(self, num_reqs: int = 1):
        with self._lock:
            self.data["num_kv_expired_reqs"].extend([1] * num_reqs)

    def clone_and_reset(self) -> "MoRIIOKVConnectorStats":
        with self._lock:
            snapshot_data: dict[str, list[float | int]] = {
                k: list(v) for k, v in self.data.items()
            }
            self.reset()
        return MoRIIOKVConnectorStats(data=snapshot_data)

    def is_empty(self) -> bool:
        return self.num_successful_transfers == 0 and not any(
            self.data[k]
            for k in (
                "num_failed_transfers",
                "num_failed_notifications",
                "num_kv_expired_reqs",
            )
        )

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        failure_counts = {
            "Num failed transfers": len(self.data["num_failed_transfers"]),
            "Num failed notifications": len(self.data["num_failed_notifications"]),
            "Num KV expired reqs": len(self.data["num_kv_expired_reqs"]),
        }
        if self.num_successful_transfers == 0:
            return {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
                **failure_counts,
            }

        xfer_time = np.asarray(self.data["transfer_duration"])
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = mb.sum()
        total_time_seconds = xfer_time.sum()
        throughput_mb_s = (
            (total_mb / total_time_seconds).item() if total_time_seconds > 0 else 0.0
        )

        return {
            "Num successful transfers": n,
            "Avg xfer time (ms)": round(xfer_time.mean().item() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90).item() * 1e3, 3),
            "Avg MB per transfer": round((total_mb / n).item(), 3),
            "Throughput (MB/s)": round(throughput_mb_s, 3),
            "Avg number of descriptors": round(descs.mean().item(), 1),
            **failure_counts,
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


class MoRIIOPromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for MoRIIO KV Cache transfers.

    The histograms are populated on D in READ mode and on P in WRITE mode;
    the counters are bumped wherever the failure was detected.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        self.histogram_xfer_time = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:moriio_xfer_time_seconds",
                documentation="Histogram of MoRIIO KV Cache transfer duration, "
                "from posting the transfer to observing its completion.",
                buckets=_XFER_TIME_BUCKETS,
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        self.histogram_bytes_transferred = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:moriio_bytes_transferred",
                documentation="Histogram of bytes per MoRIIO KV Cache transfer.",
                buckets=_BYTES_BUCKETS,
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        self.histogram_num_descriptors = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:moriio_num_descriptors",
                documentation="Histogram of RDMA descriptors per MoRIIO KV Cache "
                "transfer.",
                buckets=_DESCRIPTOR_BUCKETS,
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        self.counter_num_failed_transfers = create_metric_per_engine(
            self._counter_cls(
                name="vllm:moriio_num_failed_transfers",
                documentation="Number of failed or timed-out MoRIIO KV Cache "
                "transfers.",
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        self.counter_num_failed_notifications = create_metric_per_engine(
            self._counter_cls(
                name="vllm:moriio_num_failed_notifications",
                documentation="Number of MoRIIO completion, release or failure "
                "notifications that could not be sent to the peer.",
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        self.counter_num_kv_expired_reqs = create_metric_per_engine(
            self._counter_cls(
                name="vllm:moriio_num_kv_expired_reqs",
                documentation="Number of requests whose KV blocks were freed "
                "after defer_timeout without a completed transfer. NOTE: This "
                "metric is tracked on the P instance.",
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        for per_engine_metrics, stats_key in (
            (self.histogram_xfer_time, "transfer_duration"),
            (self.histogram_bytes_transferred, "bytes_transferred"),
            (self.histogram_num_descriptors, "num_descriptors"),
        ):
            for value in transfer_stats_data[stats_key]:
                per_engine_metrics[engine_idx].observe(value)

        for per_engine_metrics, stats_key in (
            (self.counter_num_failed_transfers, "num_failed_transfers"),
            (self.counter_num_failed_notifications, "num_failed_notifications"),
            (self.counter_num_kv_expired_reqs, "num_kv_expired_reqs"),
        ):
            num_events = sum(transfer_stats_data[stats_key])
            if num_events:
                per_engine_metrics[engine_idx].inc(num_events)
