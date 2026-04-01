#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Workload characterization and trace analysis for KV cache access patterns.

Reads access traces exported by AccessTracer and computes:
- Reuse distance distributions
- Working set size over time
- Hot/cold block classification
- Temporal and spatial locality metrics
- Access frequency distributions (Zipf analysis)

Usage:
    python -m kv_cache_tiering.analysis.analyze_traces \
        --traces traces.jsonl \
        --output analysis_results.json
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TraceAnalysis:
    """Complete workload characterization results."""
    # Basic stats
    total_events: int = 0
    unique_blocks: int = 0
    event_type_counts: dict = field(default_factory=dict)

    # Reuse distance
    reuse_distance_mean: float = 0.0
    reuse_distance_median: float = 0.0
    reuse_distance_p95: float = 0.0
    reuse_distance_p99: float = 0.0
    reuse_distance_histogram: dict = field(default_factory=dict)

    # Temporal locality
    temporal_locality_10: float = 0.0  # fraction with reuse dist <= 10
    temporal_locality_100: float = 0.0  # fraction with reuse dist <= 100
    temporal_locality_1000: float = 0.0

    # Access frequency
    frequency_mean: float = 0.0
    frequency_max: int = 0
    frequency_gini: float = 0.0  # Gini coefficient (inequality)
    zipf_exponent: float = 0.0  # estimated Zipf exponent

    # Working set
    working_set_sizes: list = field(default_factory=list)
    avg_working_set_size: float = 0.0

    # Hot/cold classification
    hot_block_count: int = 0  # blocks accessed > 2x average
    cold_block_count: int = 0  # blocks accessed <= 1x
    hot_block_fraction: float = 0.0

    # Hit rate analysis per policy
    simulated_hit_rates: dict = field(default_factory=dict)


def load_traces(path: str) -> list[dict]:
    """Load JSONL traces from file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_reuse_distances(records: list[dict]) -> list[int]:
    """Compute stack reuse distances from access records."""
    last_access: dict[int, int] = {}
    distances: list[int] = []
    access_idx = 0

    for record in records:
        if record["event_type"] in ("lookup", "load", "touch"):
            bh = record["block_hash"]
            if bh in last_access:
                dist = access_idx - last_access[bh]
                distances.append(dist)
            last_access[bh] = access_idx
            access_idx += 1

    return distances


def compute_working_set(
    records: list[dict], window_size: float = 1.0
) -> list[int]:
    """
    Compute working set size over time windows.

    Args:
        records: access trace records with timestamps.
        window_size: time window in seconds.

    Returns:
        List of unique block counts per window.
    """
    if not records:
        return []

    start_time = records[0]["timestamp"]
    current_window_blocks: set[int] = set()
    window_sizes: list[int] = []
    current_window_end = start_time + window_size

    for record in records:
        if record["event_type"] not in ("lookup", "load", "touch"):
            continue

        while record["timestamp"] >= current_window_end:
            window_sizes.append(len(current_window_blocks))
            current_window_blocks = set()
            current_window_end += window_size

        current_window_blocks.add(record["block_hash"])

    if current_window_blocks:
        window_sizes.append(len(current_window_blocks))

    return window_sizes


def compute_gini(values: list[int]) -> float:
    """Compute Gini coefficient for access frequency distribution."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumulative = sum(
        (2 * i - n + 1) * val for i, val in enumerate(sorted_vals)
    )
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    return cumulative / (n * total)


def estimate_zipf_exponent(frequencies: list[int]) -> float:
    """
    Estimate Zipf exponent via log-log linear regression.

    For a Zipf distribution: freq(rank) ~ rank^(-alpha)
    Taking logs: log(freq) = -alpha * log(rank) + const
    """
    if len(frequencies) < 2:
        return 0.0

    sorted_freq = sorted(frequencies, reverse=True)
    # Filter out zeros
    valid = [(i + 1, f) for i, f in enumerate(sorted_freq) if f > 0]
    if len(valid) < 2:
        return 0.0

    # Log-log linear regression
    n = len(valid)
    sum_log_rank = sum(math.log(r) for r, _ in valid)
    sum_log_freq = sum(math.log(f) for _, f in valid)
    sum_log_rank_sq = sum(math.log(r) ** 2 for r, _ in valid)
    sum_log_rank_freq = sum(
        math.log(r) * math.log(f) for r, f in valid
    )

    denom = n * sum_log_rank_sq - sum_log_rank ** 2
    if abs(denom) < 1e-12:
        return 0.0

    slope = (n * sum_log_rank_freq - sum_log_rank * sum_log_freq) / denom
    return -slope  # Zipf exponent is negative of slope


def simulate_lru_hit_rate(
    records: list[dict], cache_size: int
) -> float:
    """Simulate LRU cache hit rate for a given cache size."""
    from collections import OrderedDict
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    total = 0

    for record in records:
        if record["event_type"] not in ("lookup", "load"):
            continue
        bh = record["block_hash"]
        total += 1
        if bh in cache:
            hits += 1
            cache.move_to_end(bh)
        else:
            if len(cache) >= cache_size:
                cache.popitem(last=False)
            cache[bh] = None

    return hits / total if total > 0 else 0.0


def analyze_traces(
    records: list[dict],
    cache_sizes: list[int] | None = None,
) -> TraceAnalysis:
    """
    Perform complete workload characterization.

    Args:
        records: list of trace records.
        cache_sizes: optional list of cache sizes for hit rate simulation.

    Returns:
        TraceAnalysis with all computed metrics.
    """
    analysis = TraceAnalysis()
    analysis.total_events = len(records)

    if not records:
        return analysis

    # Event type counts
    type_counts: Counter[str] = Counter()
    block_hashes: set[int] = set()
    freq_counter: Counter[int] = Counter()

    for record in records:
        type_counts[record["event_type"]] += 1
        bh = record["block_hash"]
        block_hashes.add(bh)
        if record["event_type"] in ("lookup", "load", "touch"):
            freq_counter[bh] += 1

    analysis.event_type_counts = dict(type_counts)
    analysis.unique_blocks = len(block_hashes)

    # Reuse distance analysis
    distances = compute_reuse_distances(records)
    if distances:
        sorted_d = sorted(distances)
        n = len(sorted_d)
        analysis.reuse_distance_mean = sum(sorted_d) / n
        analysis.reuse_distance_median = sorted_d[n // 2]
        analysis.reuse_distance_p95 = sorted_d[int(n * 0.95)]
        analysis.reuse_distance_p99 = sorted_d[int(n * 0.99)]

        # Histogram with log-scale buckets
        buckets = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
        hist: defaultdict[str, int] = defaultdict(int)
        for d in sorted_d:
            for b in buckets:
                if d <= b:
                    hist[f"<={b}"] += 1
                    break
            else:
                hist[f">{buckets[-1]}"] += 1
        analysis.reuse_distance_histogram = dict(hist)

        # Temporal locality
        analysis.temporal_locality_10 = (
            sum(1 for d in sorted_d if d <= 10) / n
        )
        analysis.temporal_locality_100 = (
            sum(1 for d in sorted_d if d <= 100) / n
        )
        analysis.temporal_locality_1000 = (
            sum(1 for d in sorted_d if d <= 1000) / n
        )

    # Access frequency analysis
    if freq_counter:
        freqs = list(freq_counter.values())
        analysis.frequency_mean = sum(freqs) / len(freqs)
        analysis.frequency_max = max(freqs)
        analysis.frequency_gini = compute_gini(freqs)
        analysis.zipf_exponent = estimate_zipf_exponent(freqs)

        # Hot/cold classification
        avg = analysis.frequency_mean
        analysis.hot_block_count = sum(1 for f in freqs if f > 2 * avg)
        analysis.cold_block_count = sum(1 for f in freqs if f <= 1)
        analysis.hot_block_fraction = (
            analysis.hot_block_count / len(freqs) if freqs else 0.0
        )

    # Working set analysis
    analysis.working_set_sizes = compute_working_set(records)
    if analysis.working_set_sizes:
        analysis.avg_working_set_size = (
            sum(analysis.working_set_sizes) / len(analysis.working_set_sizes)
        )

    # Simulated hit rates
    if cache_sizes:
        for size in cache_sizes:
            rate = simulate_lru_hit_rate(records, size)
            analysis.simulated_hit_rates[str(size)] = rate

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze KV cache access traces"
    )
    parser.add_argument("--traces", required=True, help="JSONL trace file")
    parser.add_argument("--output", default="analysis_results.json")
    parser.add_argument(
        "--cache-sizes", nargs="*", type=int,
        default=[64, 128, 256, 512, 1024],
        help="Cache sizes for hit rate simulation",
    )

    args = parser.parse_args()

    records = load_traces(args.traces)
    print(f"Loaded {len(records)} trace records")

    analysis = analyze_traces(records, cache_sizes=args.cache_sizes)

    output_path = Path(args.output)
    from dataclasses import asdict
    with open(output_path, "w") as f:
        json.dump(asdict(analysis), f, indent=2)
    print(f"Analysis saved to {output_path}")

    # Print summary
    print(f"\nWorkload Summary:")
    print(f"  Total events: {analysis.total_events}")
    print(f"  Unique blocks: {analysis.unique_blocks}")
    print(f"  Event types: {analysis.event_type_counts}")
    if analysis.reuse_distance_mean > 0:
        print(f"  Reuse distance (mean): {analysis.reuse_distance_mean:.1f}")
        print(f"  Reuse distance (p95): {analysis.reuse_distance_p95:.1f}")
        print(f"  Temporal locality (<100): "
              f"{analysis.temporal_locality_100:.1%}")
    print(f"  Frequency Gini: {analysis.frequency_gini:.3f}")
    print(f"  Zipf exponent: {analysis.zipf_exponent:.3f}")
    print(f"  Hot blocks: {analysis.hot_block_fraction:.1%}")
    if analysis.simulated_hit_rates:
        print(f"  LRU hit rates by cache size:")
        for size, rate in analysis.simulated_hit_rates.items():
            print(f"    {size}: {rate:.1%}")


if __name__ == "__main__":
    main()
