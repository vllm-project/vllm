from dataclasses import dataclass, field
import abc
from vllm.logger import init_logger
from typing import Union

logger = init_logger(__name__)

# TODO should be subclassed by connectors to add specific stats
class KVTransferStats(abc.ABC):

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def aggregate(self, other: "KVTransferStats")->"KVTransferStats":
        raise NotImplementedError
    @abc.abstractmethod
    def reduce(self)->dict[str, Union[int, float]]:
        raise NotImplementedError

@dataclass 
class KVTransferAggregatedStats:
    """Container for aggregating performance metrics across engines"""
    avg_transfer_durations: float = 0.0
    avg_bytes_transferred: float = 0.0
    num_blocks_transferred: int = 0
    num_successful_transfers: int = 0

    def aggregate(self, other: "KVTransferAggregatedStats"):
        if other.is_empty():
            return
        
        # Reduce stats
        self.num_successful_transfers += other.num_successful_transfers
    

    
@dataclass 
class NixlKVTransferStats(KVTransferStats):
    """Container for transfer performance metrics"""
    # TODO we could use specialized data structures to avoid copying the data 
    # or even just maintaining order when merging. Let's keep it simple for now.
    transfer_durations: list[float]=field(default_factory=list)  # Transfer durations in seconds
    bytes_transferred: list[int]=field(default_factory=list)     # Bytes transferred per transfer
    num_blocks_transferred: list[int]=field(default_factory=list) # Number of blocks per transfer
    num_successful_transfers: int = 0
    
    def reset(self):
        self.transfer_durations = []
        self.bytes_transferred = []
        self.num_blocks_transferred = []
        self.num_successful_transfers = 0

    def record_transfer(self, duration: float, bytes_count: int, num_blocks: int):
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
        pass

    def is_empty(self) -> bool:
        return self.num_successful_transfers == 0

    def aggregate(self, other: "NixlKVTransferStats")->"NixlKVTransferStats":
        self.transfer_durations.extend(other.transfer_durations)
        self.bytes_transferred.extend(other.bytes_transferred)
        self.num_blocks_transferred.extend(other.num_blocks_transferred)
        self.num_successful_transfers += other.num_successful_transfers
        return self

    def reduce(self)->dict[str, Union[int, float]]:
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
            "avg_transfer_durations": self.avg_transfer_durations,
            "avg_bytes_transferred": self.avg_bytes_transferred,
            "num_blocks_transferred": self.num_blocks_transferred,
            "num_successful_transfers": self.num_successful_transfers
            }

class KVTransferLogging:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.transfer_stats: list[KVTransferStats] = []

    def observe(self, transfer_stats: KVTransferStats):
        # Called periodically when connector syncs with the scheduler.
        # Note that this is not the same as the logging interval.
        self.transfer_stats.append(transfer_stats)
    
    def log(self, log_fn=logger.info):
        """Log transfer metrics periodically, similar to throughput logging"""
        if self.transfer_stats:
            # Produce a single cumulative stats object for the last time 
            # interval. Resulting key/value mapping is logged. This allows
            # different connectors to log their own stats.
            cumulative_stats = self.transfer_stats[0]
            for stats in self.transfer_stats[1:]:
                cumulative_stats.aggregate(stats)
            log_fn("KV Transfer metrics: %s", cumulative_stats.reduce())
            
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