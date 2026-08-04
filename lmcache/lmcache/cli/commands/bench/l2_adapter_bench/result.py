# SPDX-License-Identifier: Apache-2.0
"""Aggregated benchmark statistics for L2 adapter operations."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import statistics

_KB = 1024
_MB = 1024 * 1024


def _percentile(values: list[float], pct: float) -> float:
    """Return the percentile *pct* (0..100) using nearest-rank.

    Returns 0.0 for an empty list.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    # Nearest-rank method: rank = ceil(pct/100 * N)
    rank = max(1, int((pct / 100.0) * len(sorted_vals) + 0.999999))
    rank = min(rank, len(sorted_vals))
    return sorted_vals[rank - 1]


@dataclass
class BenchResult:
    """Aggregated benchmark statistics for one operation type.

    Each measured *round* issues ``in_flight`` submits, where each submit
    carries ``num_keys`` keys. ``round_durations[r]`` is the wall-clock
    elapsed for the whole round (from issuing the first submit to all
    submits of that round completing).
    """

    operation: str
    in_flight: int
    num_keys: int
    data_size_bytes: int
    round_durations: list[float] = field(default_factory=list)
    success_counts: list[int] = field(default_factory=list)
    # Lookup-specific metadata (left as defaults for store/load).
    expected_max_hit_rate: float = 0.0
    expected_hit_count: int = 0

    # ------------------------------------------------------------------
    # Derived counts
    # ------------------------------------------------------------------

    @property
    def keys_per_round(self) -> int:
        return self.in_flight * self.num_keys

    @property
    def total_keys(self) -> int:
        return self.keys_per_round * len(self.round_durations)

    @property
    def total_data_bytes_per_round(self) -> int:
        return self.keys_per_round * self.data_size_bytes

    @property
    def total_data_bytes(self) -> int:
        return self.total_keys * self.data_size_bytes

    @property
    def total_success(self) -> int:
        return sum(self.success_counts)

    # ------------------------------------------------------------------
    # Duration stats (seconds)
    # ------------------------------------------------------------------

    @property
    def avg_duration(self) -> float:
        return statistics.mean(self.round_durations) if self.round_durations else 0.0

    @property
    def min_duration(self) -> float:
        return min(self.round_durations) if self.round_durations else 0.0

    @property
    def max_duration(self) -> float:
        return max(self.round_durations) if self.round_durations else 0.0

    @property
    def std_duration(self) -> float:
        if len(self.round_durations) > 1:
            return statistics.stdev(self.round_durations)
        return 0.0

    @property
    def p50_duration(self) -> float:
        return _percentile(self.round_durations, 50.0)

    @property
    def p99_duration(self) -> float:
        return _percentile(self.round_durations, 99.0)

    # ------------------------------------------------------------------
    # Throughput stats (per-round, MB/s)
    # ------------------------------------------------------------------

    @property
    def per_round_throughput_mbps(self) -> list[float]:
        if self.data_size_bytes <= 0:
            return []
        bytes_per_round = self.total_data_bytes_per_round
        out: list[float] = []
        for d in self.round_durations:
            if d <= 0:
                out.append(float("inf"))
            else:
                out.append((bytes_per_round / _MB) / d)
        return out

    @property
    def avg_throughput_mbps(self) -> float:
        vals = self.per_round_throughput_mbps
        return statistics.mean(vals) if vals else 0.0

    @property
    def min_throughput_mbps(self) -> float:
        vals = self.per_round_throughput_mbps
        return min(vals) if vals else 0.0

    @property
    def max_throughput_mbps(self) -> float:
        vals = self.per_round_throughput_mbps
        return max(vals) if vals else 0.0

    # ------------------------------------------------------------------
    # Ops/sec (key-rate) stats — useful for lookup which has no payload
    # ------------------------------------------------------------------

    @property
    def per_round_ops_per_sec(self) -> list[float]:
        out: list[float] = []
        for d in self.round_durations:
            if d <= 0:
                out.append(float("inf"))
            else:
                out.append(self.keys_per_round / d)
        return out

    @property
    def avg_ops_per_sec(self) -> float:
        vals = self.per_round_ops_per_sec
        return statistics.mean(vals) if vals else 0.0

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def avg_success_per_round(self) -> float:
        return statistics.mean(self.success_counts) if self.success_counts else 0.0

    @property
    def avg_latency_per_key_ms(self) -> float:
        if self.keys_per_round <= 0:
            return 0.0
        return (self.avg_duration / self.keys_per_round) * 1000

    @property
    def actual_hit_rate(self) -> float:
        if self.total_keys <= 0:
            return 0.0
        return self.total_success / self.total_keys
