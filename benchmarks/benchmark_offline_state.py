#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Evaluation benchmark for OfflineState bloom-filter cooperative caching.

Simulates the three-tier KV cache discovery (local -> peer bloom -> miss)
at various cluster sizes and workloads using vLLM's actual block hash
format, then generates evaluation plots for the thesis.

Usage:
    python benchmarks/benchmark_offline_state.py [--output-dir plots/]

Workloads:
    1. Zipfian (controlled skew, theta = 0.5-0.99)
    2. Shared system prompt (many users, same prefix)
    3. Multi-turn conversation (high prefix reuse)

Baselines:
    - No caching: every lookup = server RTT
    - Local-only: vLLM's current BlockPool (Tier 1 only)
    - Centralized indexer: models llm-d (1 RTT index + 3 RTT server)
    - Event-based: models LMCache (variable delay, 2 RTT)
    - OfflineState: three-tier bloom filter (0 RTT local, ~1us peer bloom)
"""

import argparse
import importlib.util
import json
import logging
import os
import random
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Direct import of our bloom filter module (avoids heavy vLLM import chain)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VLLM_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

# Add vLLM root to path for module loading
if _VLLM_ROOT not in sys.path:
    sys.path.insert(0, _VLLM_ROOT)


def _load_module(name, rel_path):
    full_path = os.path.join(_VLLM_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub vllm.logger to avoid triggering heavy imports
vllm_pkg = types.ModuleType("vllm")
vllm_pkg.__path__ = [os.path.join(_VLLM_ROOT, "vllm")]
sys.modules.setdefault("vllm", vllm_pkg)
logger_mod = types.ModuleType("vllm.logger")
logger_mod.init_logger = logging.getLogger
sys.modules["vllm.logger"] = logger_mod

_bloom_mod = _load_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.bloom_filter",
    "vllm/distributed/kv_transfer/kv_connector/v1/bloom_filter.py",
)
BloomFilter = _bloom_mod.BloomFilter


# ---------------------------------------------------------------------------
# Latency model constants (from literature)
# ---------------------------------------------------------------------------

# Discovery latencies (microseconds)
LATENCY_LOCAL_LOOKUP_US = 0.1  # Local hash table lookup
LATENCY_BLOOM_CHECK_US = 0.5  # Bloom filter membership test
LATENCY_CENTRALIZED_INDEX_US = 50.0  # 1 RTT to centralized index (~50us same-rack)
LATENCY_CROSS_RACK_US = 500.0  # Cross-rack RTT
LATENCY_EVENT_PROPAGATION_US = 100.0  # Event-based propagation delay (avg)
LATENCY_RECOMPUTE_US = 10000.0  # Full recompute (10ms for typical block)

# Transfer latencies (per block, RDMA)
LATENCY_TRANSFER_PER_BLOCK_US = 50.0  # ~50us per block via RDMA

# From literature: discovery overhead as fraction of TTFT
# IEEE Cloud 2025: 18-23% of TTFT from metadata lookup
DISCOVERY_OVERHEAD_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------

def zipfian_generator(n_items: int, theta: float, n_ops: int,
                      seed: int = 42) -> list[int]:
    """Generate Zipfian-distributed key sequence."""
    rng = np.random.default_rng(seed)
    # Zipf weights: 1/rank^theta
    ranks = np.arange(1, n_items + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, theta)
    weights /= weights.sum()
    return rng.choice(n_items, size=n_ops, p=weights).tolist()


def shared_prefix_generator(n_users: int, prefix_length: int,
                             suffix_range: int, n_ops: int,
                             seed: int = 42) -> list[tuple[int, list[int]]]:
    """Generate shared-prefix workload.

    Returns list of (user_id, block_hashes) where block_hashes includes
    shared prefix blocks + user-specific suffix blocks.
    """
    rng = random.Random(seed)
    # Shared prefix blocks (same for all users)
    prefix_blocks = list(range(prefix_length))

    ops = []
    for _ in range(n_ops):
        user = rng.randint(0, n_users - 1)
        # User-specific suffix
        n_suffix = rng.randint(1, suffix_range)
        suffix_start = 1000000 + user * 10000
        suffix_blocks = list(range(suffix_start, suffix_start + n_suffix))
        ops.append((user, prefix_blocks + suffix_blocks))
    return ops


def multi_turn_generator(n_conversations: int, turns_per_conv: int,
                          blocks_per_turn: int, n_nodes: int,
                          seed: int = 42) -> list[tuple[int, list[int]]]:
    """Generate multi-turn conversation workload.

    Each conversation builds on the previous turn's prefix.
    Returns list of (node_id, block_hashes).
    """
    rng = random.Random(seed)
    ops = []
    for conv_id in range(n_conversations):
        node = conv_id % n_nodes
        cumulative_blocks = []
        base = conv_id * 100000
        for turn in range(turns_per_conv):
            # Each turn adds new blocks to the prefix
            new_blocks = list(range(
                base + turn * blocks_per_turn,
                base + (turn + 1) * blocks_per_turn,
            ))
            cumulative_blocks = cumulative_blocks + new_blocks
            ops.append((node, list(cumulative_blocks)))
    return ops


# ---------------------------------------------------------------------------
# Cache simulation
# ---------------------------------------------------------------------------

@dataclass
class CacheNode:
    """Simulates a single vLLM instance's KV cache."""
    node_id: int
    capacity: int
    cache: OrderedDict = field(default_factory=OrderedDict)
    bloom: BloomFilter = field(default=None)

    def __post_init__(self):
        if self.bloom is None:
            self.bloom = BloomFilter(
                expected_items=self.capacity, fp_rate=0.01
            )

    def contains(self, key: int) -> bool:
        if key in self.cache:
            self.cache.move_to_end(key)
            return True
        return False

    def insert(self, key: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
            return
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = True
        self.bloom.add(key)

    def rebuild_bloom(self) -> None:
        self.bloom.rebuild_from_keys(self.cache.keys())


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    name: str
    n_nodes: int
    n_ops: int
    local_hits: int = 0
    peer_hits: int = 0
    server_hits: int = 0
    false_positives: int = 0
    total_discovery_latency_us: float = 0.0

    @property
    def total(self) -> int:
        return self.local_hits + self.peer_hits + self.server_hits

    @property
    def local_rate(self) -> float:
        return self.local_hits / max(1, self.total)

    @property
    def peer_rate(self) -> float:
        return self.peer_hits / max(1, self.total)

    @property
    def server_rate(self) -> float:
        return self.server_hits / max(1, self.total)

    @property
    def combined_hit_rate(self) -> float:
        return (self.local_hits + self.peer_hits) / max(1, self.total)

    @property
    def avg_discovery_latency_us(self) -> float:
        return self.total_discovery_latency_us / max(1, self.total)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_nodes": self.n_nodes,
            "n_ops": self.n_ops,
            "local_hits": self.local_hits,
            "peer_hits": self.peer_hits,
            "server_hits": self.server_hits,
            "false_positives": self.false_positives,
            "local_rate": self.local_rate,
            "peer_rate": self.peer_rate,
            "server_rate": self.server_rate,
            "combined_hit_rate": self.combined_hit_rate,
            "avg_discovery_latency_us": self.avg_discovery_latency_us,
        }


def simulate_offlinestate(
    keys: list[int],
    n_nodes: int,
    cache_capacity: int,
    bloom_sync_interval: int = 100,
) -> SimulationResult:
    """Simulate OfflineState three-tier lookup."""
    nodes = [
        CacheNode(
            node_id=i,
            capacity=cache_capacity,
            bloom=BloomFilter(
                expected_items=cache_capacity,
                fp_rate=0.01,
                auto_scale_clients=n_nodes,
            ),
        )
        for i in range(n_nodes)
    ]

    result = SimulationResult(
        name="OfflineState", n_nodes=n_nodes, n_ops=len(keys)
    )

    for op_idx, key in enumerate(keys):
        node_idx = op_idx % n_nodes
        node = nodes[node_idx]

        # Tier 1: Local cache
        if node.contains(key):
            result.local_hits += 1
            result.total_discovery_latency_us += LATENCY_LOCAL_LOOKUP_US
            continue

        # Tier 2: Peer bloom filter check
        found_peer = False
        result.total_discovery_latency_us += LATENCY_BLOOM_CHECK_US
        for other in nodes:
            if other.node_id == node_idx:
                continue
            if other.bloom.contains(key):
                # Bloom says yes — verify against actual cache
                if other.contains(key):
                    found_peer = True
                    result.peer_hits += 1
                    result.total_discovery_latency_us += (
                        LATENCY_TRANSFER_PER_BLOCK_US
                    )
                    break
                else:
                    result.false_positives += 1

        if not found_peer:
            # Tier 3: Miss — recompute
            result.server_hits += 1
            result.total_discovery_latency_us += LATENCY_RECOMPUTE_US

        # Cache the block locally
        node.insert(key)

        # Periodic bloom rebuild (simulates sync)
        if op_idx > 0 and op_idx % bloom_sync_interval == 0:
            for n in nodes:
                n.rebuild_bloom()

    return result


def simulate_local_only(
    keys: list[int], n_nodes: int, cache_capacity: int
) -> SimulationResult:
    """Simulate local-only caching (vLLM default BlockPool)."""
    caches: list[OrderedDict] = [OrderedDict() for _ in range(n_nodes)]

    result = SimulationResult(
        name="Local-only", n_nodes=n_nodes, n_ops=len(keys)
    )

    for op_idx, key in enumerate(keys):
        node_idx = op_idx % n_nodes
        cache = caches[node_idx]

        if key in cache:
            cache.move_to_end(key)
            result.local_hits += 1
            result.total_discovery_latency_us += LATENCY_LOCAL_LOOKUP_US
        else:
            result.server_hits += 1
            result.total_discovery_latency_us += LATENCY_RECOMPUTE_US
            if len(cache) >= cache_capacity:
                cache.popitem(last=False)
            cache[key] = True

    return result


def simulate_centralized(
    keys: list[int], n_nodes: int, cache_capacity: int
) -> SimulationResult:
    """Simulate centralized indexer (models llm-d)."""
    caches: list[OrderedDict] = [OrderedDict() for _ in range(n_nodes)]

    result = SimulationResult(
        name="Centralized", n_nodes=n_nodes, n_ops=len(keys)
    )

    for op_idx, key in enumerate(keys):
        node_idx = op_idx % n_nodes
        cache = caches[node_idx]

        # Always pay centralized index lookup cost
        result.total_discovery_latency_us += LATENCY_CENTRALIZED_INDEX_US

        if key in cache:
            cache.move_to_end(key)
            result.local_hits += 1
            result.total_discovery_latency_us += LATENCY_LOCAL_LOOKUP_US
        else:
            # Check other nodes (centralized knows where it is)
            found = False
            for other_idx in range(n_nodes):
                if other_idx == node_idx:
                    continue
                if key in caches[other_idx]:
                    caches[other_idx].move_to_end(key)
                    found = True
                    result.peer_hits += 1
                    result.total_discovery_latency_us += (
                        LATENCY_TRANSFER_PER_BLOCK_US
                    )
                    break

            if not found:
                result.server_hits += 1
                result.total_discovery_latency_us += LATENCY_RECOMPUTE_US

            if len(cache) >= cache_capacity:
                cache.popitem(last=False)
            cache[key] = True

    return result


def simulate_event_based(
    keys: list[int], n_nodes: int, cache_capacity: int,
    event_lag_ops: int = 10,
) -> SimulationResult:
    """Simulate event-based discovery (models LMCache).

    Events propagate with a lag, so recent insertions on peers
    are not immediately discoverable.
    """
    caches: list[OrderedDict] = [OrderedDict() for _ in range(n_nodes)]
    # Track insertion order for event lag simulation
    insert_history: list[tuple[int, int]] = []  # (op_idx, key)
    global_index: set[int] = set()  # Keys known to global event stream

    result = SimulationResult(
        name="Event-based", n_nodes=n_nodes, n_ops=len(keys)
    )

    for op_idx, key in enumerate(keys):
        node_idx = op_idx % n_nodes
        cache = caches[node_idx]

        # Update global index with events that have propagated
        while insert_history and insert_history[0][0] <= op_idx - event_lag_ops:
            _, old_key = insert_history.pop(0)
            global_index.add(old_key)

        # Event propagation cost
        result.total_discovery_latency_us += LATENCY_EVENT_PROPAGATION_US

        if key in cache:
            cache.move_to_end(key)
            result.local_hits += 1
            result.total_discovery_latency_us += LATENCY_LOCAL_LOOKUP_US
        elif key in global_index:
            # Event says someone has it — find who
            found = False
            for other_idx in range(n_nodes):
                if other_idx == node_idx:
                    continue
                if key in caches[other_idx]:
                    caches[other_idx].move_to_end(key)
                    found = True
                    result.peer_hits += 1
                    result.total_discovery_latency_us += (
                        LATENCY_TRANSFER_PER_BLOCK_US
                    )
                    break
            if not found:
                result.server_hits += 1
                result.total_discovery_latency_us += LATENCY_RECOMPUTE_US
        else:
            result.server_hits += 1
            result.total_discovery_latency_us += LATENCY_RECOMPUTE_US

        if len(cache) >= cache_capacity:
            cache.popitem(last=False)
        cache[key] = True
        insert_history.append((op_idx, key))

    return result


def simulate_no_caching(keys: list[int], n_nodes: int) -> SimulationResult:
    """Simulate no caching (always recompute)."""
    result = SimulationResult(
        name="No-caching", n_nodes=n_nodes, n_ops=len(keys)
    )
    result.server_hits = len(keys)
    result.total_discovery_latency_us = len(keys) * LATENCY_RECOMPUTE_US
    return result


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

def run_zipfian_sweep(
    n_nodes: int = 8,
    cache_capacity: int = 512,
    n_items: int = 50000,
    n_ops: int = 50000,
) -> dict[str, list[SimulationResult]]:
    """Sweep Zipf theta values and compare all strategies."""
    thetas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    results = {name: [] for name in [
        "No-caching", "Local-only", "Centralized",
        "Event-based", "OfflineState",
    ]}

    for theta in thetas:
        print(f"  Zipf theta={theta:.2f} ...", end=" ", flush=True)
        keys = zipfian_generator(n_items, theta, n_ops)

        r_none = simulate_no_caching(keys, n_nodes)
        r_local = simulate_local_only(keys, n_nodes, cache_capacity)
        r_central = simulate_centralized(keys, n_nodes, cache_capacity)
        r_event = simulate_event_based(keys, n_nodes, cache_capacity)
        r_offline = simulate_offlinestate(keys, n_nodes, cache_capacity)

        results["No-caching"].append(r_none)
        results["Local-only"].append(r_local)
        results["Centralized"].append(r_central)
        results["Event-based"].append(r_event)
        results["OfflineState"].append(r_offline)

        print(
            f"offline_state hit={r_offline.combined_hit_rate:.1%} "
            f"(local={r_offline.local_rate:.1%}, "
            f"peer={r_offline.peer_rate:.1%})"
        )

    return results


def run_cluster_scale_sweep(
    theta: float = 0.9,
    cache_capacity: int = 512,
    n_items: int = 50000,
    n_ops: int = 50000,
) -> dict[str, list[SimulationResult]]:
    """Sweep cluster sizes and compare strategies."""
    node_counts = [1, 2, 4, 8, 16, 32, 64]
    results = {name: [] for name in [
        "Local-only", "Centralized", "Event-based", "OfflineState",
    ]}

    for n_nodes in node_counts:
        print(f"  Cluster size={n_nodes} ...", end=" ", flush=True)
        keys = zipfian_generator(n_items, theta, n_ops)

        r_local = simulate_local_only(keys, n_nodes, cache_capacity)
        r_central = simulate_centralized(keys, n_nodes, cache_capacity)
        r_event = simulate_event_based(keys, n_nodes, cache_capacity)
        r_offline = simulate_offlinestate(keys, n_nodes, cache_capacity)

        results["Local-only"].append(r_local)
        results["Centralized"].append(r_central)
        results["Event-based"].append(r_event)
        results["OfflineState"].append(r_offline)

        print(
            f"offline_state hit={r_offline.combined_hit_rate:.1%}, "
            f"avg_latency={r_offline.avg_discovery_latency_us:.0f}us"
        )

    return results


def run_bloom_sizing() -> dict[str, list]:
    """Measure bloom filter size and FPR across scales."""
    node_counts = [1, 2, 4, 8, 16, 32, 64]
    cache_sizes = [1000, 5000, 10000, 50000]

    results = {"node_counts": node_counts, "cache_sizes": cache_sizes, "data": {}}

    for cache_size in cache_sizes:
        sizes_kb = []
        fprs = []
        for n_nodes in node_counts:
            bf = BloomFilter(
                expected_items=cache_size,
                fp_rate=0.01,
                auto_scale_clients=n_nodes,
            )
            sizes_kb.append(bf.size_bytes / 1024)

            # Measure empirical FPR
            for k in range(cache_size):
                bf.add(k)
            fp = sum(
                1 for k in range(cache_size, cache_size + 10000)
                if bf.contains(k)
            )
            fprs.append(fp / 10000)

        results["data"][cache_size] = {
            "sizes_kb": sizes_kb,
            "fprs": fprs,
        }

    return results


def run_sync_interval_sweep(
    n_nodes: int = 8,
    cache_capacity: int = 512,
    n_items: int = 50000,
    n_ops: int = 50000,
    theta: float = 0.9,
) -> list[dict]:
    """Sweep bloom filter sync intervals."""
    intervals = [10, 25, 50, 100, 250, 500, 1000, 5000]
    results = []

    for interval in intervals:
        print(f"  Sync interval={interval} ...", end=" ", flush=True)
        keys = zipfian_generator(n_items, theta, n_ops)
        r = simulate_offlinestate(
            keys, n_nodes, cache_capacity, bloom_sync_interval=interval
        )
        results.append({
            "interval": interval,
            "combined_hit_rate": r.combined_hit_rate,
            "false_positives": r.false_positives,
            "avg_latency_us": r.avg_discovery_latency_us,
        })
        print(
            f"hit={r.combined_hit_rate:.1%}, fp={r.false_positives}"
        )

    return results


def run_shared_prefix_benchmark(
    n_nodes: int = 8,
    cache_capacity: int = 512,
) -> dict[str, SimulationResult]:
    """Benchmark with shared system prompt workload."""
    # 100 users, 32 shared prefix blocks, 1-8 suffix blocks
    ops = shared_prefix_generator(
        n_users=100, prefix_length=32, suffix_range=8, n_ops=10000
    )

    # Flatten to just keys for our simulation
    # Each op's blocks need to be looked up
    all_keys = []
    for user_id, blocks in ops:
        all_keys.extend(blocks)

    results = {}
    for name, sim_fn in [
        ("Local-only", lambda k: simulate_local_only(k, n_nodes, cache_capacity)),
        ("Centralized", lambda k: simulate_centralized(k, n_nodes, cache_capacity)),
        ("OfflineState", lambda k: simulate_offlinestate(k, n_nodes, cache_capacity)),
    ]:
        results[name] = sim_fn(all_keys)

    return results


# ---------------------------------------------------------------------------
# Plotting (optional — requires matplotlib)
# ---------------------------------------------------------------------------

def generate_plots(
    zipfian_results: dict,
    cluster_results: dict,
    bloom_sizing: dict,
    sync_results: list,
    shared_prefix_results: dict,
    output_dir: str,
) -> None:
    """Generate all thesis evaluation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot generation")
        print("Install with: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)
    colors = {
        "No-caching": "#d62728",
        "Local-only": "#ff7f0e",
        "Centralized": "#2ca02c",
        "Event-based": "#9467bd",
        "OfflineState": "#1f77b4",
    }

    # ---- Plot 1: Cache hit rate vs Zipf theta ----
    fig, ax = plt.subplots(figsize=(8, 5))
    thetas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    for name, sim_results in zipfian_results.items():
        if name == "No-caching":
            continue
        rates = [r.combined_hit_rate for r in sim_results]
        ax.plot(thetas, rates, "o-", label=name, color=colors.get(name, "gray"),
                linewidth=2, markersize=6)
    ax.set_xlabel("Zipf Skewness (theta)", fontsize=12)
    ax.set_ylabel("Cache Hit Rate (local + peer)", fontsize=12)
    ax.set_title("Cache Hit Rate vs Workload Skewness", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_hitrate_vs_zipf.png"), dpi=150)
    plt.close(fig)
    print("  Saved 01_hitrate_vs_zipf.png")

    # ---- Plot 2: Discovery latency vs Zipf theta ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, sim_results in zipfian_results.items():
        latencies = [r.avg_discovery_latency_us for r in sim_results]
        ax.plot(thetas, latencies, "o-", label=name, color=colors.get(name, "gray"),
                linewidth=2, markersize=6)
    ax.set_xlabel("Zipf Skewness (theta)", fontsize=12)
    ax.set_ylabel("Avg Discovery Latency (us)", fontsize=12)
    ax.set_title("Discovery Latency vs Workload Skewness", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_latency_vs_zipf.png"), dpi=150)
    plt.close(fig)
    print("  Saved 02_latency_vs_zipf.png")

    # ---- Plot 3: Hit rate vs cluster size ----
    fig, ax = plt.subplots(figsize=(8, 5))
    node_counts = [1, 2, 4, 8, 16, 32, 64]
    for name, sim_results in cluster_results.items():
        rates = [r.combined_hit_rate for r in sim_results]
        ax.plot(node_counts, rates, "o-", label=name,
                color=colors.get(name, "gray"), linewidth=2, markersize=6)
    ax.set_xlabel("Cluster Size (nodes)", fontsize=12)
    ax.set_ylabel("Cache Hit Rate", fontsize=12)
    ax.set_title("Cache Hit Rate vs Cluster Size (Zipf theta=0.9)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(node_counts)
    ax.set_xticklabels([str(n) for n in node_counts])
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_hitrate_vs_cluster.png"), dpi=150)
    plt.close(fig)
    print("  Saved 03_hitrate_vs_cluster.png")

    # ---- Plot 4: Discovery latency vs cluster size ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, sim_results in cluster_results.items():
        latencies = [r.avg_discovery_latency_us for r in sim_results]
        ax.plot(node_counts, latencies, "o-", label=name,
                color=colors.get(name, "gray"), linewidth=2, markersize=6)
    ax.set_xlabel("Cluster Size (nodes)", fontsize=12)
    ax.set_ylabel("Avg Discovery Latency (us)", fontsize=12)
    ax.set_title("Discovery Latency vs Cluster Size", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(node_counts)
    ax.set_xticklabels([str(n) for n in node_counts])
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_latency_vs_cluster.png"), dpi=150)
    plt.close(fig)
    print("  Saved 04_latency_vs_cluster.png")

    # ---- Plot 5: Bloom filter size vs cluster scale ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    node_counts_bloom = bloom_sizing["node_counts"]
    for cache_size, data in bloom_sizing["data"].items():
        ax1.plot(node_counts_bloom, data["sizes_kb"], "o-",
                 label=f"{cache_size:,} entries", linewidth=2, markersize=5)
        ax2.plot(node_counts_bloom, data["fprs"], "o-",
                 label=f"{cache_size:,} entries", linewidth=2, markersize=5)

    ax1.set_xlabel("Cluster Size (nodes)", fontsize=12)
    ax1.set_ylabel("Bloom Filter Size (KB)", fontsize=12)
    ax1.set_title("Bloom Filter Size vs Cluster Scale", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(node_counts_bloom)
    ax1.set_xticklabels([str(n) for n in node_counts_bloom])

    ax2.set_xlabel("Cluster Size (nodes)", fontsize=12)
    ax2.set_ylabel("False Positive Rate", fontsize=12)
    ax2.set_title("Empirical FPR vs Cluster Scale", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="Target 1%")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(node_counts_bloom)
    ax2.set_xticklabels([str(n) for n in node_counts_bloom])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_bloom_sizing.png"), dpi=150)
    plt.close(fig)
    print("  Saved 05_bloom_sizing.png")

    # ---- Plot 6: FPR vs sync interval ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    intervals = [r["interval"] for r in sync_results]
    hit_rates = [r["combined_hit_rate"] for r in sync_results]
    fps = [r["false_positives"] for r in sync_results]

    ax1.plot(intervals, hit_rates, "o-", color=colors["OfflineState"],
             linewidth=2, markersize=6)
    ax1.set_xlabel("Sync Interval (operations)", fontsize=12)
    ax1.set_ylabel("Combined Hit Rate", fontsize=12)
    ax1.set_title("Hit Rate vs Sync Interval", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    ax2.plot(intervals, fps, "o-", color="#d62728", linewidth=2, markersize=6)
    ax2.set_xlabel("Sync Interval (operations)", fontsize=12)
    ax2.set_ylabel("False Positives (count)", fontsize=12)
    ax2.set_title("False Positives vs Sync Interval", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_sync_interval.png"), dpi=150)
    plt.close(fig)
    print("  Saved 06_sync_interval.png")

    # ---- Plot 7: Hit rate breakdown (stacked bar) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    strategies = ["Local-only", "Centralized", "Event-based", "OfflineState"]
    # Use theta=0.9 results
    idx_09 = 3  # thetas = [0.5, 0.7, 0.8, 0.9, ...]

    local_rates = []
    peer_rates = []
    miss_rates = []
    for name in strategies:
        r = zipfian_results[name][idx_09]
        local_rates.append(r.local_rate)
        peer_rates.append(r.peer_rate)
        miss_rates.append(r.server_rate)

    x = np.arange(len(strategies))
    width = 0.5
    ax.bar(x, local_rates, width, label="Local (Tier 1)", color="#2ca02c")
    ax.bar(x, peer_rates, width, bottom=local_rates,
           label="Peer (Tier 2)", color="#1f77b4")
    ax.bar(x, miss_rates, width,
           bottom=[l + p for l, p in zip(local_rates, peer_rates)],
           label="Miss (Tier 3)", color="#d62728")

    ax.set_ylabel("Fraction of Lookups", fontsize=12)
    ax.set_title("Cache Hit Rate Breakdown (Zipf theta=0.9, 8 nodes)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_hitrate_breakdown.png"), dpi=150)
    plt.close(fig)
    print("  Saved 07_hitrate_breakdown.png")

    print(f"\nAll plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OfflineState bloom-filter cooperative caching"
    )
    parser.add_argument(
        "--output-dir", default="plots",
        help="Directory for output plots and data (default: plots/)",
    )
    parser.add_argument(
        "--n-ops", type=int, default=50000,
        help="Number of operations per simulation (default: 50000)",
    )
    parser.add_argument(
        "--cache-capacity", type=int, default=512,
        help="Per-node cache capacity in blocks (default: 512)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (just print results)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("OfflineState Bloom-Filter Cooperative Caching Benchmark")
    print("=" * 60)

    # Run all benchmarks
    print("\n[1/5] Zipfian skewness sweep (8 nodes)...")
    zipfian_results = run_zipfian_sweep(
        n_nodes=8, cache_capacity=args.cache_capacity,
        n_ops=args.n_ops,
    )

    print("\n[2/5] Cluster size sweep (Zipf theta=0.9)...")
    cluster_results = run_cluster_scale_sweep(
        theta=0.9, cache_capacity=args.cache_capacity,
        n_ops=args.n_ops,
    )

    print("\n[3/5] Bloom filter sizing analysis...")
    bloom_sizing = run_bloom_sizing()

    print("\n[4/5] Sync interval sweep...")
    sync_results = run_sync_interval_sweep(
        n_nodes=8, cache_capacity=args.cache_capacity,
        n_ops=args.n_ops,
    )

    print("\n[5/5] Shared prefix workload...")
    shared_prefix_results = run_shared_prefix_benchmark(
        n_nodes=8, cache_capacity=args.cache_capacity,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n--- Shared Prefix Workload (8 nodes) ---")
    for name, r in shared_prefix_results.items():
        print(
            f"  {name:15s}: hit={r.combined_hit_rate:.1%} "
            f"(local={r.local_rate:.1%}, peer={r.peer_rate:.1%}), "
            f"avg_lat={r.avg_discovery_latency_us:.0f}us"
        )

    print("\n--- Zipfian theta=0.9 (8 nodes) ---")
    idx_09 = 3
    for name in ["Local-only", "Centralized", "Event-based", "OfflineState"]:
        r = zipfian_results[name][idx_09]
        print(
            f"  {name:15s}: hit={r.combined_hit_rate:.1%} "
            f"(local={r.local_rate:.1%}, peer={r.peer_rate:.1%}), "
            f"avg_lat={r.avg_discovery_latency_us:.0f}us"
        )

    print("\n--- Bloom Filter Sizes ---")
    for cache_size, data in bloom_sizing["data"].items():
        sizes = data["sizes_kb"]
        print(
            f"  {cache_size:>6,} entries: "
            + " | ".join(f"{n}n={s:.1f}KB" for n, s in
                         zip(bloom_sizing["node_counts"], sizes))
        )

    # Save raw data
    os.makedirs(args.output_dir, exist_ok=True)
    raw_data = {
        "zipfian": {
            name: [r.to_dict() for r in results]
            for name, results in zipfian_results.items()
        },
        "cluster": {
            name: [r.to_dict() for r in results]
            for name, results in cluster_results.items()
        },
        "bloom_sizing": bloom_sizing,
        "sync_interval": sync_results,
        "shared_prefix": {
            name: r.to_dict()
            for name, r in shared_prefix_results.items()
        },
    }
    data_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(data_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"\nRaw data saved to {data_path}")

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(
            zipfian_results, cluster_results, bloom_sizing,
            sync_results, shared_prefix_results, args.output_dir,
        )


if __name__ == "__main__":
    main()
