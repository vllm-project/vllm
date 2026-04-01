# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Instrumentation for KV cache offloading: access tracing and metrics.

Provides AccessTracer for recording block-level access patterns and
OffloadingMetrics for aggregating performance statistics. Trace data
can be exported for offline workload characterization and analysis.
"""
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class BlockAccessRecord:
    """A single block access event."""
    timestamp: float
    block_hash: int  # Stored as int for serialization
    event_type: str  # "lookup", "load", "store", "evict", "touch", "prefetch"
    hit: bool = True
    attention_score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class OffloadingMetrics:
    """Aggregated metrics for a time window."""
    window_start: float = 0.0
    window_end: float = 0.0
    total_lookups: int = 0
    total_hits: int = 0
    total_misses: int = 0
    total_stores: int = 0
    total_loads: int = 0
    total_evictions: int = 0
    total_prefetches: int = 0
    prefetch_hits: int = 0
    bytes_transferred_gpu_to_cpu: int = 0
    bytes_transferred_cpu_to_gpu: int = 0
    avg_transfer_time_gpu_to_cpu: float = 0.0
    avg_transfer_time_cpu_to_gpu: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    @property
    def prefetch_accuracy(self) -> float:
        return (
            self.prefetch_hits / self.total_prefetches
            if self.total_prefetches > 0
            else 0.0
        )


class AccessTracer:
    """
    Records block-level access traces for workload characterization.

    Tracks lookups, loads, stores, evictions, and prefetches with
    timestamps. Computes reuse distance, access frequency distributions,
    and hot/cold block classification.

    Usage:
        tracer = AccessTracer(max_records=100000)
        tracer.record_lookup(block_hash, hit=True)
        tracer.record_store(block_hash)
        ...
        tracer.export_traces("/path/to/traces.jsonl")
        analysis = tracer.compute_workload_stats()
    """

    def __init__(
        self,
        max_records: int = 100_000,
        metrics_window_seconds: float = 10.0,
    ):
        self._records: list[BlockAccessRecord] = []
        self._max_records: int = max_records
        self._metrics_window: float = metrics_window_seconds
        self._current_metrics: OffloadingMetrics = OffloadingMetrics(
            window_start=time.monotonic()
        )

        # Per-block tracking for workload analysis
        self._block_access_counts: defaultdict[int, int] = defaultdict(int)
        self._block_last_access: dict[int, float] = {}
        # Reuse distance tracking: maps block_hash -> access index
        self._access_index: int = 0
        self._last_access_index: dict[int, int] = {}
        self._reuse_distances: list[int] = []

        # Transfer timing accumulators
        self._gpu_to_cpu_times: list[float] = []
        self._cpu_to_gpu_times: list[float] = []

    def _hash_to_int(self, block_hash: BlockHash) -> int:
        """Convert BlockHash to int for serialization."""
        if isinstance(block_hash, int):
            return block_hash
        return hash(block_hash)

    def _add_record(self, record: BlockAccessRecord) -> None:
        if len(self._records) < self._max_records:
            self._records.append(record)

    def _track_reuse(self, block_hash_int: int) -> None:
        """Track reuse distance for the given block."""
        prev_index = self._last_access_index.get(block_hash_int)
        if prev_index is not None:
            distance = self._access_index - prev_index
            self._reuse_distances.append(distance)
        self._last_access_index[block_hash_int] = self._access_index
        self._access_index += 1
        self._block_access_counts[block_hash_int] += 1
        self._block_last_access[block_hash_int] = time.monotonic()

    def record_lookup(
        self,
        block_hash: BlockHash,
        hit: bool,
        attention_score: float = 0.0,
    ) -> None:
        now = time.monotonic()
        bh_int = self._hash_to_int(block_hash)
        self._add_record(
            BlockAccessRecord(
                timestamp=now,
                block_hash=bh_int,
                event_type="lookup",
                hit=hit,
                attention_score=attention_score,
            )
        )
        self._current_metrics.total_lookups += 1
        if hit:
            self._current_metrics.total_hits += 1
            self._track_reuse(bh_int)
        else:
            self._current_metrics.total_misses += 1

    def record_store(self, block_hash: BlockHash) -> None:
        now = time.monotonic()
        bh_int = self._hash_to_int(block_hash)
        self._add_record(
            BlockAccessRecord(
                timestamp=now,
                block_hash=bh_int,
                event_type="store",
            )
        )
        self._current_metrics.total_stores += 1

    def record_load(self, block_hash: BlockHash) -> None:
        now = time.monotonic()
        bh_int = self._hash_to_int(block_hash)
        self._add_record(
            BlockAccessRecord(
                timestamp=now,
                block_hash=bh_int,
                event_type="load",
            )
        )
        self._current_metrics.total_loads += 1
        self._track_reuse(bh_int)

    def record_eviction(self, block_hash: BlockHash) -> None:
        now = time.monotonic()
        bh_int = self._hash_to_int(block_hash)
        self._add_record(
            BlockAccessRecord(
                timestamp=now,
                block_hash=bh_int,
                event_type="evict",
            )
        )
        self._current_metrics.total_evictions += 1

    def record_touch(self, block_hash: BlockHash) -> None:
        now = time.monotonic()
        bh_int = self._hash_to_int(block_hash)
        self._add_record(
            BlockAccessRecord(
                timestamp=now,
                block_hash=bh_int,
                event_type="touch",
            )
        )
        self._track_reuse(bh_int)

    def record_prefetch(
        self, block_hash: BlockHash, hit: bool = False
    ) -> None:
        now = time.monotonic()
        bh_int = self._hash_to_int(block_hash)
        self._add_record(
            BlockAccessRecord(
                timestamp=now,
                block_hash=bh_int,
                event_type="prefetch",
                hit=hit,
            )
        )
        self._current_metrics.total_prefetches += 1
        if hit:
            self._current_metrics.prefetch_hits += 1

    def record_transfer(
        self,
        direction: str,
        size_bytes: int,
        transfer_time: float,
    ) -> None:
        """Record a completed transfer for bandwidth tracking."""
        if direction == "gpu_to_cpu":
            self._current_metrics.bytes_transferred_gpu_to_cpu += size_bytes
            self._gpu_to_cpu_times.append(transfer_time)
        elif direction == "cpu_to_gpu":
            self._current_metrics.bytes_transferred_cpu_to_gpu += size_bytes
            self._cpu_to_gpu_times.append(transfer_time)

    def get_metrics(self) -> OffloadingMetrics:
        """Get current metrics window and start a new one."""
        now = time.monotonic()
        metrics = self._current_metrics
        metrics.window_end = now

        if self._gpu_to_cpu_times:
            metrics.avg_transfer_time_gpu_to_cpu = (
                sum(self._gpu_to_cpu_times) / len(self._gpu_to_cpu_times)
            )
        if self._cpu_to_gpu_times:
            metrics.avg_transfer_time_cpu_to_gpu = (
                sum(self._cpu_to_gpu_times) / len(self._cpu_to_gpu_times)
            )

        # Reset for next window
        self._current_metrics = OffloadingMetrics(window_start=now)
        self._gpu_to_cpu_times.clear()
        self._cpu_to_gpu_times.clear()

        return metrics

    def compute_workload_stats(self) -> dict:
        """
        Compute workload characterization statistics from traces.

        Returns:
            Dictionary with:
            - reuse_distance: {mean, median, p95, p99, max}
            - access_frequency: {mean, max, num_unique_blocks}
            - hot_blocks: list of (block_hash, count) for top-10 blocks
            - temporal_locality: fraction of accesses within reuse dist 100
        """
        stats: dict = {}

        # Reuse distance analysis
        if self._reuse_distances:
            sorted_rd = sorted(self._reuse_distances)
            n = len(sorted_rd)
            stats["reuse_distance"] = {
                "mean": sum(sorted_rd) / n,
                "median": sorted_rd[n // 2],
                "p95": sorted_rd[int(n * 0.95)],
                "p99": sorted_rd[int(n * 0.99)],
                "max": sorted_rd[-1],
                "count": n,
            }
            # Temporal locality: fraction with reuse dist <= 100
            close_reuses = sum(1 for d in sorted_rd if d <= 100)
            stats["temporal_locality"] = close_reuses / n
        else:
            stats["reuse_distance"] = {}
            stats["temporal_locality"] = 0.0

        # Access frequency distribution
        if self._block_access_counts:
            counts = list(self._block_access_counts.values())
            stats["access_frequency"] = {
                "mean": sum(counts) / len(counts),
                "max": max(counts),
                "num_unique_blocks": len(counts),
            }
            # Hot blocks (top 10 by access count)
            top_blocks = sorted(
                self._block_access_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            stats["hot_blocks"] = [
                {"block_hash": bh, "access_count": cnt}
                for bh, cnt in top_blocks
            ]
        else:
            stats["access_frequency"] = {}
            stats["hot_blocks"] = []

        return stats

    def export_traces(self, path: str) -> int:
        """
        Export access traces to a JSONL file.

        Args:
            path: output file path (will be created/overwritten).

        Returns:
            Number of records written.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(output_path, "w") as f:
            for record in self._records:
                f.write(json.dumps(asdict(record)) + "\n")
                count += 1
        return count

    def clear(self) -> None:
        """Reset all trace data."""
        self._records.clear()
        self._block_access_counts.clear()
        self._block_last_access.clear()
        self._last_access_index.clear()
        self._reuse_distances.clear()
        self._access_index = 0
        self._current_metrics = OffloadingMetrics(
            window_start=time.monotonic()
        )
        self._gpu_to_cpu_times.clear()
        self._cpu_to_gpu_times.clear()
