# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    has_kv_transfer_group)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class KVTransferStats:
    """
    Base class for KV Transfer Stats, a container for transfer performance 
    metrics. All sub-classes need to be serializable as stats are sent
    from worker to logger process.
    """
    data: dict[str, Any] = field(default_factory=dict)

    def reset(self):
        """Reset the stats, clear the state."""
        raise NotImplementedError

    def aggregate(self, other: "KVTransferStats") -> "KVTransferStats":
        """
        Aggregate stats with another `KVTransferStats` object.
        """
        raise NotImplementedError

    def reduce(self) -> dict[str, Union[int, float]]:
        """
        Reduce the observations collected during a time interval to one or 
        more representative values (eg avg/median/sum of the series). 
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Return True if the stats are empty."""
        raise NotImplementedError


class KVTransferLogging:

    def __init__(self, connector_cls: type[KVConnectorBase_V1]):
        # This should be called on frontend process.
        assert not has_kv_transfer_group()
        self.connector_cls = connector_cls
        self.reset()

    def reset(self):
        self.transfer_stats_accumulator: Optional[KVTransferStats] = None

    def observe(self, transfer_stats_data: dict[str, Any]):
        # Called periodically when connector syncs with the scheduler.
        # Note that this is not the same as the logging interval.
        # We expect transfer_stats_data to be aggregated across all workers and
        # consist of observations from a single connector or a MultiConnector.
        transfer_stats = self.connector_cls.build_kv_transfer_stats(
            transfer_stats_data)
        if self.transfer_stats_accumulator is None:
            self.transfer_stats_accumulator = transfer_stats
        else:
            # Accumulate last interval stats.
            self.transfer_stats_accumulator = \
                self.transfer_stats_accumulator.aggregate(transfer_stats)

    def log(self, log_fn=logger.info):
        """Log transfer metrics periodically, similar to throughput logging"""
        if (self.transfer_stats_accumulator
                and not self.transfer_stats_accumulator.is_empty()):
            # Produce a single cumulative stats object for the last time
            # interval from the recorded observations.
            xfer_metrics = self.transfer_stats_accumulator.reduce()
            xfer_metrics_str = ", ".join(f"{k}={v}"
                                         for k, v in xfer_metrics.items())
            log_fn("KV Transfer metrics: %s", xfer_metrics_str)

            # Reset metrics for next interval
            self.reset()


EMPTY_KV_TRANSFER_STATS = KVTransferStats()
