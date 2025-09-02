# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from typing import Union

import msgspec

from vllm.logger import init_logger

logger = init_logger(__name__)


# Sent to scheduler process to aggregate stats.
class KVTransferStats(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        tag_field="type"):  # type: ignore[call-arg]

    def reset(self):
        raise NotImplementedError

    def aggregate(self, other: "KVTransferStats") -> "KVTransferStats":
        raise NotImplementedError

    def reduce(self) -> dict[str, Union[int, float]]:
        # TODO docs
        raise NotImplementedError

    def is_empty(self) -> bool:
        raise NotImplementedError


class NixlKVTransferStats(KVTransferStats,
                          tag="NIXL"):  # type: ignore[call-arg]
    """Container for transfer performance metrics"""
    # Setup buffers
    # We could use specialized data structures to avoid copying the data
    # or even just maintaining order when merging. Let's keep it simple for now
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

    def record_transfer(self):
        # TODO: record actual transfer stats when available
        self.num_successful_transfers += 1

    def clone_and_reset(self) -> "NixlKVTransferStats":
        # if self.is_empty():
        # return EMPTY_KV_TRANSFER_STATS
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
        # TODO: reduce stats to a single value, calculate latency/throughput
        return {"num_successful_transfers": self.num_successful_transfers}


# Union type for serialization/deserialization
KVTransferStatsType = Union[NixlKVTransferStats]


class KVTransferLogging:

    def __init__(self):
        self.reset()

    def reset(self):
        self.transfer_stats_accumulator: KVTransferStats = None

    def observe(self, transfer_stats: KVTransferStats):
        # Called periodically when connector syncs with the scheduler.
        # Note that this is not the same as the logging interval.
        # We expect transfer_stats to be aggregated across all workers and
        # consist of observations from a single connector or a MultiConnector.
        if self.transfer_stats_accumulator is None:
            self.transfer_stats_accumulator = transfer_stats
        elif not transfer_stats.is_empty():
            print("OBSERVING TRANSFER STATS", transfer_stats, "\n\n")
            self.transfer_stats_accumulator.aggregate(transfer_stats)
            print("OBSERVING TRANSFER STATS", self.transfer_stats_accumulator,
                  "\n\n")

    def log(self, log_fn=logger.info):
        """Log transfer metrics periodically, similar to throughput logging"""
        if (self.transfer_stats_accumulator
                and not self.transfer_stats_accumulator.is_empty()):
            # Produce a single cumulative stats object for the last time
            # interval from the recorded observations.
            print("LOGGINGTRANSFER STATS", self.transfer_stats_accumulator,
                  "\n\n")
            xfer_metrics = self.transfer_stats_accumulator.reduce()
            xfer_metrics_str = ", ".join(f"{k}={v}"
                                         for k, v in xfer_metrics.items())
            log_fn("KV Transfer metrics: %s", xfer_metrics_str)

            # Reset metrics for next interval
            self.reset()


EMPTY_KV_TRANSFER_STATS = NixlKVTransferStats()
