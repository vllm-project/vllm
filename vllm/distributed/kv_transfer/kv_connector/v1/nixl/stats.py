# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics for the NIXL connector."""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    BYTES_BUCKETS,
    DESCRIPTOR_BUCKETS,
    XFER_TIME_BUCKETS,
    CounterSpec,
    HistogramSpec,
    KVConnectorPromMetrics,
    KVConnectorStats,
)

if TYPE_CHECKING:
    from vllm.distributed.nixl_utils import nixlXferTelemetry


@dataclass
class NixlKVConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics."""

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
            "num_failed_handshakes": [],
            "num_kv_expired_reqs": [],
            "num_notifications_after_expiry": [],
        }

    def record_transfer(self, res: "nixlXferTelemetry | None"):
        # Keep metrics units consistent with rest of the code: time us->s
        if res is None:
            return
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

    def record_failed_handshake(self):
        """Record a failed NIXL handshake."""
        self.data["num_failed_handshakes"].append(1)

    def record_kv_expired_req(self):
        """Record a request that had its KV blocks expire."""
        self.data["num_kv_expired_reqs"].append(1)

    def record_notification_after_expiry(self):
        """Record a completion notification that arrived after its request expired."""
        self.data["num_notifications_after_expiry"].append(1)

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
            and len(self.data["num_failed_handshakes"]) == 0
            and len(self.data["num_kv_expired_reqs"]) == 0
            and len(self.data["num_notifications_after_expiry"]) == 0
        )

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        # Compute compact representative stats suitable for CLI logging.
        # Failure counts are reported on every interval: transfer, handshake
        # and notification failures are grouped as sporadic
        # lower-transport-layer events, while KV expiry is reported separately
        # as it is an actionable autoscaler signal rather than a transport
        # issue. Notifications arriving after expiry are surfaced too, as the
        # reader may have consumed KV blocks that were already reused.
        failure_counts = {
            "Num failed transfers": len(self.data["num_failed_transfers"])
            + len(self.data["num_failed_handshakes"])
            + len(self.data["num_failed_notifications"]),
            "Num KV expired reqs": len(self.data["num_kv_expired_reqs"]),
            "Num notifs after expiry": len(self.data["num_notifications_after_expiry"]),
        }
        if self.num_successful_transfers == 0:
            # Timing / throughput stats only cover successful transfers. If
            # all requests in the interval were unsuccessful, the failure
            # counts above still surface what happened.
            return {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg post time (ms)": 0,
                "P90 post time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
                **failure_counts,
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
            "Avg xfer time (ms)": round(xfer_time.mean().item() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90).item() * 1e3, 3),
            "Avg post time (ms)": round(post_time.mean().item() * 1e3, 3),
            "P90 post time (ms)": round(np.percentile(post_time, 90).item() * 1e3, 3),
            "Avg MB per transfer": round(avg_mb.item(), 3),
            "Throughput (MB/s)": round(throughput_mb_s.item(), 3),
            "Avg number of descriptors": round(descs.mean().item(), 1),
            **failure_counts,
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


class NixlPromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for NIXL KV Cache transfers."""

    HISTOGRAM_SPECS = (
        HistogramSpec(
            stats_key="transfer_duration",
            name="vllm:nixl_xfer_time_seconds",
            documentation="Histogram of transfer duration for NIXL KV Cache transfers.",
            buckets=XFER_TIME_BUCKETS,
        ),
        HistogramSpec(
            stats_key="post_duration",
            name="vllm:nixl_post_time_seconds",
            documentation="Histogram of transfer post time for NIXL KV Cache"
            " transfers.",
            buckets=(0.001, *XFER_TIME_BUCKETS),
        ),
        HistogramSpec(
            stats_key="bytes_transferred",
            name="vllm:nixl_bytes_transferred",
            documentation="Histogram of bytes transferred per NIXL KV Cache transfer.",
            buckets=BYTES_BUCKETS,
        ),
        HistogramSpec(
            stats_key="num_descriptors",
            name="vllm:nixl_num_descriptors",
            documentation="Histogram of number of descriptors per NIXL KV Cache"
            " transfer.",
            buckets=DESCRIPTOR_BUCKETS,
        ),
    )
    COUNTER_SPECS = (
        # Transfer, handshake and notification failures are grouped: all are
        # sporadic lower-transport-layer events. KV expiry is reported separately
        # as it signals autoscaler behavior, not transport health.
        CounterSpec(
            stats_keys=(
                "num_failed_transfers",
                "num_failed_handshakes",
                "num_failed_notifications",
            ),
            name="vllm:nixl_num_failed_transfers",
            documentation="Number of failed NIXL KV Cache transfers, including"
            " handshake and notification failures. NOTE: KV expiry is tracked"
            " separately in vllm:nixl_num_kv_expired_reqs.",
        ),
        CounterSpec(
            stats_keys=("num_failed_notifications",),
            name="vllm:nixl_num_failed_notifications",
            documentation="Number of failed NIXL KV Cache notifications. "
            "Retained for compatibility; these failures are also included in "
            "vllm:nixl_num_failed_transfers.",
        ),
        CounterSpec(
            stats_keys=("num_kv_expired_reqs",),
            name="vllm:nixl_num_kv_expired_reqs",
            documentation="Number of requests that had their KV expire. "
            "NOTE: This metric is tracked on the P instance.",
        ),
        CounterSpec(
            stats_keys=("num_notifications_after_expiry",),
            name="vllm:nixl_num_notifications_after_expiry",
            documentation="Number of completion notifications for requests that "
            "were no longer tracked, usually because their KV lease expired. The "
            "KV blocks may have been reused before the transfer finished. Counted "
            "per notification (one per remote rank), not per request.",
        ),
    )
