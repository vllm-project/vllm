from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)

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
    
    def is_empty(self) -> bool:
        return self.num_successful_transfers == 0
    
@dataclass 
class KVTransferStats:
    """Container for transfer performance metrics"""
    transfer_durations: list[float]=field(default_factory=list)  # Transfer durations in seconds
    bytes_transferred: list[int]=field(default_factory=list)     # Bytes transferred per transfer
    num_blocks_transferred: list[int]=field(default_factory=list) # Number of blocks per transfer
    num_transfers: int = 0
    
    def reset(self):
        self.transfer_durations = []
        self.bytes_transferred = []
        self.num_blocks_transferred = []

    def observe(self):
        # TODO finish this
        self.num_transfers += 1
    
    def reduce_and_reset(self) -> KVTransferAggregatedStats:
        # NOTE (NickLucche): to have statistical significance, we assume the 
        # size of the measurements groups to be the same. This allows to bound
        # the size of the messages.
        # TODO finish this
        stats = KVTransferAggregatedStats(
            avg_transfer_durations=11.0,
            avg_bytes_transferred=0.0,
            num_blocks_transferred=0,
            num_successful_transfers=self.num_transfers)
        self.num_transfers = 0
        return stats
    
    def record_transfer(self, duration: float, bytes_count: int, num_blocks: int):
        self.transfer_durations.append(duration)
        self.bytes_transferred.append(bytes_count)
        self.num_blocks_transferred.append(num_blocks)
    
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

class KVTransferLogging:
    def __init__(self):
        self.reset()
        self.transfer_stats = None
    
    def reset(self):
        self.transfer_durations = []
        self.bytes_transferred = []
        self.num_blocks_transferred: int = 0

    def observe(self, transfer_stats: KVTransferAggregatedStats):
        self.transfer_stats = transfer_stats
        # self.transfer_durations.append(transfer_stats.transfer_durations)
        # self.bytes_transferred.append(transfer_stats.bytes_transferred)
        # self.num_blocks_transferred += transfer_stats.num_blocks_transferred
    
    def log(self, log_fn=logger.info):
        """Log transfer metrics periodically, similar to throughput logging"""
        # Only log if we have transfer data
        if self.transfer_stats is not None:
            log_fn("KV Transfer metrics: %s", self.transfer_stats)
            # bytes_per_sec, blocks_per_sec, transfers_per_sec = \
            #     self.transfer_metrics.get_throughput_stats(now)
                
            # # Get latency stats
            # avg_latency, p50_latency, p95_latency = \
            #     self.transfer_metrics.get_latency_stats()
            
            # # Format throughput for readability
            # if bytes_per_sec >= 1024**3:  # GB/s
            #     bytes_throughput_str = f"{bytes_per_sec / (1024**3):.2f} GB/s"
            # elif bytes_per_sec >= 1024**2:  # MB/s
            #     bytes_throughput_str = f"{bytes_per_sec / (1024**2):.1f} MB/s"
            # elif bytes_per_sec >= 1024:  # KB/s
            #     bytes_throughput_str = f"{bytes_per_sec / 1024:.1f} KB/s"
            # else:  # B/s
            #     bytes_throughput_str = f"{bytes_per_sec:.1f} B/s"
            
            # # Log the metrics in a format similar to the existing throughput logs
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
        
        # # Reset metrics for next interval
        # self.transfer_metrics.reset(now)