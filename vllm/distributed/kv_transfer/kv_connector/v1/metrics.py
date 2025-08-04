# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Union

import msgspec

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorType)


# Sent to scheduler process to aggregate stats.
class KVTransferStats(msgspec.Struct,
                      array_like=True,
                      omit_defaults=True,
                      tag_field="type"):

    def reset(self):
        raise NotImplementedError

    def aggregate(self, other: "KVTransferStats") -> "KVTransferStats":
        raise NotImplementedError

    def reduce(self) -> dict[str, Union[int, float]]:
        raise NotImplementedError

    def is_empty(self) -> bool:
        raise NotImplementedError


class NixlKVTransferStats(KVTransferStats, tag="NIXL"):
    """Container for transfer performance metrics"""
    # Setup buffers
    # TODO we could use specialized data structures to avoid copying the data
    # or even just maintaining order when merging. Let's keep it simple for now.
    transfer_durations: list[float] = msgspec.field(
        default_factory=list)  # Transfer durations in seconds
    bytes_transferred: list[int] = msgspec.field(
        default_factory=list)  # Bytes transferred per transfer
    num_blocks_transferred: list[int] = msgspec.field(
        default_factory=list)  # Number of blocks per transfer
    num_successful_transfers: int = 0

    def reset(self):
        self.transfer_durations = []
        self.bytes_transferred = []
        self.num_blocks_transferred = []
        self.num_successful_transfers = 0

    def record_transfer(self, duration: float, bytes_count: int,
                        num_blocks: int):
        self.transfer_durations.append(duration)
        self.bytes_transferred.append(bytes_count)
        self.num_blocks_transferred.append(num_blocks)
        self.num_successful_transfers += 1

    def get_throughput_stats(self, now: float) -> tuple[float, float, float]:
        """Get transfer throughput statistics"""
        pass

    def get_latency_stats(self) -> tuple[float, float, float]:
        """Get transfer latency statistics"""
        # TODO possible use
        import numpy as np
        durations = np.array(self.transfer_durations)
        avg_latency = float(np.mean(durations))
        p50_latency = float(np.percentile(durations, 50))
        p95_latency = float(np.percentile(durations, 95))

        return avg_latency, p50_latency, p95_latency

    def clone_and_reset(self) -> "NixlKVTransferStats":
        if self.is_empty():
            return EMPTY_KV_TRANSFER_STATS
        old = copy.deepcopy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        return self.num_successful_transfers == 0

    def aggregate(self, other: "NixlKVTransferStats") -> "NixlKVTransferStats":
        if not other.is_empty():
            self.transfer_durations.extend(other.transfer_durations)
            self.bytes_transferred.extend(other.bytes_transferred)
            self.num_blocks_transferred.extend(other.num_blocks_transferred)
            self.num_successful_transfers += other.num_successful_transfers
        return self

    def reduce(self) -> dict[str, Union[int, float]]:
        # TODO should you return a pair with a string for the unit?
        # # Format throughput for readability
        # if bytes_per_sec >= 1024**3:  # GB/s
        #     bytes_throughput_str = f"{bytes_per_sec / (1024**3):.2f} GB/s"
        # elif bytes_per_sec >= 1024**2:  # MB/s
        #     bytes_throughput_str = f"{bytes_per_sec / (1024**2):.1f} MB/s"
        # elif bytes_per_sec >= 1024:  # KB/s
        #     bytes_throughput_str = f"{bytes_per_sec / 1024:.1f} KB/s"
        # else:  # B/s
        #     bytes_throughput_str = f"{bytes_per_sec:.1f} B/s"
        return {
            "avg_transfer_durations": 11,
            "avg_bytes_transferred": 110,
            "num_blocks_transferred": 77,
            "num_successful_transfers": 8
        }


# Union type for serialization/deserialization
KVTransferStatsType = Union[NixlKVTransferStats]


class KVTransferLogging:

    def __init__(self):
        self.reset()

    def reset(self):
        self.transfer_stats = defaultdict[KVConnectorType,
                                          list[KVTransferStats]](list)

    def observe(self, transfer_stats: dict[KVConnectorType,
                                           KVTransferStatsType]):
        # Called periodically when connector syncs with the scheduler.
        # Note that this is not the same as the logging interval.
        # We expect transfer_stats to be aggregated across all workers.
        if len(transfer_stats):
            for connector_type, stats in transfer_stats.items():
                if not stats.is_empty():
                    self.transfer_stats[connector_type].append(stats)

    def log(self, log_fn=logger.info):
        """Log transfer metrics periodically, similar to throughput logging"""
        if self.transfer_stats:
            # Produce a single cumulative stats object for the last time
            # interval from the recorded observations. This allows different
            # connectors to log their own set of stats.
            for connector_type, stats_list in self.transfer_stats.items():
                if len(stats_list) > 0:
                    cumulative_stats = stats_list[0]
                    for stats in stats_list[1:]:
                        cumulative_stats.aggregate(stats)
                    cumulative_stats = cumulative_stats.reduce()
                    fields = ", ".join(f"{k}={v}"
                                       for k, v in cumulative_stats.items())
                    log_fn("KVConnectorType: %s, KV Transfer metrics: %s",
                           connector_type, fields)

            # example
            # logger.info(
            #     "Engine %s: KV Transfer metrics: "
            #     "Avg transfer throughput: %s, "
            #     "Blocks/s: %.1f, Transfers/s: %.1f, "
            #     "Avg latency: %.3fs, P50: %.3fs, P95: %.3fs, "
            #     "Total transfers: %d",
            #     self.engine_id,
            #     bytes_throughput_str,
            #     blocks_per_sec,
            #     transfers_per_sec,
            #     avg_latency,
            #     p50_latency,
            #     p95_latency,
            #     len(self.transfer_metrics.transfer_durations)
            # )

        # Reset metrics for next interval
        self.reset()


EMPTY_KV_TRANSFER_STATS = NixlKVTransferStats()
