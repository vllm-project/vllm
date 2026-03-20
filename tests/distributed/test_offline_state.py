# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the OfflineState bloom-filter cooperative caching system.

Tests cover:
1. BloomFilter: correctness, FPR, auto-scaling, serialization
2. BloomFilterPeerDiscovery: three-tier lookup, sync protocol
3. OfflineStateConnector: integration with vLLM connector interface

Uses direct module loading to avoid importing the full vLLM dependency chain.
"""

import importlib.util
import os
import random
import sys

import pytest

# ---------------------------------------------------------------------------
# Direct module loading helpers — avoids pulling in vLLM's heavy import chain
# (torch distributed, psutil, etc.) that the unit-under-test doesn't need.
# ---------------------------------------------------------------------------

_VLLM_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def _load_module(name: str, rel_path: str):
    """Load a Python module directly from file path."""
    full_path = os.path.join(_VLLM_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _setup_stubs():
    """Register stub modules so peer_discovery.py can import from
    vllm.logger and bloom_filter without triggering the heavy vLLM
    import chain."""
    import logging
    import types

    # Stub vllm.logger.init_logger -> returns standard logging.getLogger
    vllm_pkg = types.ModuleType("vllm")
    vllm_pkg.__path__ = [os.path.join(_VLLM_ROOT, "vllm")]
    sys.modules.setdefault("vllm", vllm_pkg)

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = logging.getLogger  # type: ignore[attr-defined]
    sys.modules["vllm.logger"] = logger_mod


_setup_stubs()

# Pre-load the two modules under test
_bloom_mod = _load_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.bloom_filter",
    "vllm/distributed/kv_transfer/kv_connector/v1/bloom_filter.py",
)
BloomFilter = _bloom_mod.BloomFilter

_peer_mod = _load_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.peer_discovery",
    "vllm/distributed/kv_transfer/kv_connector/v1/peer_discovery.py",
)
BloomFilterPeerDiscovery = _peer_mod.BloomFilterPeerDiscovery
PeerCacheEntry = _peer_mod.PeerCacheEntry


# ============================================================
# BloomFilter Tests
# ============================================================

class TestBloomFilter:
    """Tests for the BloomFilter data structure."""

    def _make_filter(self, **kwargs):
        return BloomFilter(**kwargs)

    def test_no_false_negatives(self):
        """Items added to the filter must always be found."""
        bf = self._make_filter(expected_items=1000, fp_rate=0.01)
        keys = list(range(1000))
        for k in keys:
            bf.add(k)
        for k in keys:
            assert bf.contains(k), f"False negative for key {k}"

    def test_empty_filter_returns_false(self):
        """Empty filter should return False for any key."""
        bf = self._make_filter(expected_items=100)
        for k in range(100):
            assert not bf.contains(k)

    def test_false_positive_rate_bounded(self):
        """Empirical FPR should be close to target."""
        target_fpr = 0.01
        n_items = 10000
        bf = self._make_filter(expected_items=n_items, fp_rate=target_fpr)

        # Add items
        for k in range(n_items):
            bf.add(k)

        # Test with keys that were NOT added
        n_test = 100000
        false_positives = 0
        for k in range(n_items, n_items + n_test):
            if bf.contains(k):
                false_positives += 1

        empirical_fpr = false_positives / n_test
        # Allow 3x tolerance (theoretical is approximate)
        assert empirical_fpr < target_fpr * 3, (
            f"Empirical FPR {empirical_fpr:.4f} exceeds "
            f"3x target {target_fpr:.4f}"
        )

    def test_theoretical_fpr_matches(self):
        """Theoretical FPR calculation should match formula."""
        bf = self._make_filter(expected_items=1000, fp_rate=0.01)
        for k in range(1000):
            bf.add(k)

        theoretical = bf.theoretical_fpr()
        # Should be close to 0.01
        assert 0.001 < theoretical < 0.05, (
            f"Theoretical FPR {theoretical} out of expected range"
        )

    def test_auto_scaling_reduces_fpr(self):
        """Auto-scaling with more clients should keep FPR bounded."""
        n_items = 1000

        # Without scaling (1 client)
        bf1 = self._make_filter(
            expected_items=n_items, fp_rate=0.01, auto_scale_clients=1
        )
        for k in range(n_items):
            bf1.add(k)

        # With scaling (100 clients)
        bf100 = self._make_filter(
            expected_items=n_items, fp_rate=0.01, auto_scale_clients=100
        )
        for k in range(n_items):
            bf100.add(k)

        # Scaled filter should be larger
        assert bf100.size_bits > bf1.size_bits

        # Scaled filter should have lower FPR for same number of items
        assert bf100.theoretical_fpr() <= bf1.theoretical_fpr()

    def test_clear(self):
        """Clear should reset the filter."""
        bf = self._make_filter(expected_items=100)
        for k in range(100):
            bf.add(k)
        assert bf.fill_rate > 0

        bf.clear()
        assert bf.fill_rate == 0.0
        assert bf.item_count == 0
        for k in range(100):
            assert not bf.contains(k)

    def test_rebuild_from_keys(self):
        """Rebuild should reflect only current keys."""
        bf = self._make_filter(expected_items=100)

        # Add keys 0-99
        for k in range(100):
            bf.add(k)

        # Rebuild with only keys 50-99
        bf.rebuild_from_keys(range(50, 100))

        # Keys 50-99 should still be found
        for k in range(50, 100):
            assert bf.contains(k)

        # Keys 0-49 should mostly not be found (some FP expected)
        not_found = sum(1 for k in range(50) if not bf.contains(k))
        assert not_found > 30, "Too many false positives after rebuild"

    def test_bitwise_or(self):
        """Merged filter should contain items from all filters."""
        bf1 = self._make_filter(expected_items=100)
        bf2 = self._make_filter(expected_items=100)

        for k in range(50):
            bf1.add(k)
        for k in range(50, 100):
            bf2.add(k)

        merged = BloomFilter.bitwise_or([bf1, bf2])

        # All items from both filters should be in merged
        for k in range(100):
            assert merged.contains(k), f"Merged filter missing key {k}"

    def test_serialization_roundtrip(self):
        """to_bytes/from_bytes should preserve filter state."""
        bf = self._make_filter(expected_items=1000, fp_rate=0.01)
        keys = list(range(500))
        for k in keys:
            bf.add(k)

        # Serialize and deserialize
        data = bf.to_bytes()
        bf2 = BloomFilter.from_bytes(data)

        # Check all keys still found
        for k in keys:
            assert bf2.contains(k), f"Key {k} lost in serialization"

        # Check properties preserved
        assert bf2.size_bits == bf.size_bits
        assert bf2.num_hashes == bf.num_hashes
        assert bf2.item_count == bf.item_count

    def test_copy(self):
        """Copy should be independent."""
        bf = self._make_filter(expected_items=100)
        bf.add(42)

        bf2 = bf.copy()
        bf2.add(99)

        assert bf.contains(42)
        assert bf2.contains(42)
        assert bf2.contains(99)
        # Original should not have 99 (unless false positive)
        # Can't assert this deterministically due to FP

    def test_fill_rate(self):
        """Fill rate should increase as items are added."""
        bf = self._make_filter(expected_items=10000)
        assert bf.fill_rate == 0.0

        for k in range(1000):
            bf.add(k)
        rate_1k = bf.fill_rate
        assert rate_1k > 0

        for k in range(1000, 5000):
            bf.add(k)
        rate_5k = bf.fill_rate
        assert rate_5k > rate_1k

    def test_size_bytes(self):
        """Bloom filter size should be reasonable."""
        # 10K items at 1% FPR should be ~12 KB
        bf = self._make_filter(expected_items=10000, fp_rate=0.01)
        size_kb = bf.size_bytes / 1024
        assert 5 < size_kb < 200, f"Unexpected size: {size_kb:.1f} KB"


# ============================================================
# BloomFilterPeerDiscovery Tests
# ============================================================

class TestBloomFilterPeerDiscovery:
    """Tests for the peer discovery service."""

    def _make_discovery(self, **kwargs):
        defaults = {
            "node_id": 0,
            "num_nodes": 1,
            "sync_interval_ms": 1000,
            "bloom_fp_rate": 0.01,
            "max_cache_entries": 10000,
        }
        defaults.update(kwargs)
        return BloomFilterPeerDiscovery(**defaults)

    def test_register_and_has_local(self):
        """Registered blocks should be found locally."""
        disc = self._make_discovery()
        disc.register_block(42)
        assert disc.has_local_block(42)
        assert not disc.has_local_block(99)

    def test_register_blocks_batch(self):
        """Batch registration should work."""
        disc = self._make_discovery()
        disc.register_blocks([1, 2, 3, 4, 5])
        for h in [1, 2, 3, 4, 5]:
            assert disc.has_local_block(h)
        assert disc.get_local_cache_size() == 5

    def test_unregister_block(self):
        """Unregistered blocks should not be found."""
        disc = self._make_discovery()
        disc.register_block(42)
        assert disc.has_local_block(42)

        disc.unregister_block(42)
        assert not disc.has_local_block(42)

    def test_lru_eviction(self):
        """Cache should evict LRU entries when at capacity."""
        disc = self._make_discovery(max_cache_entries=5)
        for i in range(10):
            disc.register_block(i)

        # Only last 5 should be present
        assert disc.get_local_cache_size() == 5
        for i in range(5, 10):
            assert disc.has_local_block(i)
        for i in range(5):
            assert not disc.has_local_block(i)

    def test_single_node_peer_lookup_returns_none(self):
        """In single-node mode, peer lookup should always return None."""
        disc = self._make_discovery(num_nodes=1)
        disc.register_block(42)
        # Peer lookup (not local) should return None since no peers
        result = disc.find_peer_with_block(99)
        assert result is None

    def test_force_sync_updates_bloom(self):
        """Force sync should rebuild bloom filter from cache."""
        disc = self._make_discovery()
        disc.register_block(42)
        disc.register_block(100)

        # Force sync rebuilds bloom
        disc.force_sync()

        stats = disc.stats
        assert stats["syncs_completed"] == 1

    def test_stats_tracking(self):
        """Statistics should be tracked correctly."""
        disc = self._make_discovery()
        disc.find_peer_with_block(42)  # miss

        stats = disc.stats
        assert stats["misses"] >= 1


# ============================================================
# Multi-Node Discovery Integration Tests
# ============================================================

class TestMultiNodeDiscovery:
    """Tests for multi-node bloom filter exchange."""

    def test_two_node_discovery(self):
        """Two nodes should discover each other's blocks via bloom sync."""
        # Simulate two nodes without ZMQ (manual bloom exchange)
        node0 = BloomFilterPeerDiscovery(
            node_id=0, num_nodes=2, sync_interval_ms=10000,
            max_cache_entries=1000,
        )
        node1 = BloomFilterPeerDiscovery(
            node_id=1, num_nodes=2, sync_interval_ms=10000,
            max_cache_entries=1000,
        )

        # Node 0 has blocks 0-49, Node 1 has blocks 50-99
        for i in range(50):
            node0.register_block(i)
        for i in range(50, 100):
            node1.register_block(i)

        # Manually exchange bloom filters (simulating ZMQ sync)
        node0._local_bloom.rebuild_from_keys(node0._local_cache.keys())
        node1._local_bloom.rebuild_from_keys(node1._local_cache.keys())

        # Set peer blooms manually
        node0._peer_blooms[1] = PeerCacheEntry(1, node1._local_bloom.copy())
        node1._peer_blooms[0] = PeerCacheEntry(0, node0._local_bloom.copy())

        # Merge
        node0._merged_bloom = BloomFilter.bitwise_or(
            [node0._local_bloom, node1._local_bloom]
        )
        node1._merged_bloom = BloomFilter.bitwise_or(
            [node0._local_bloom, node1._local_bloom]
        )

        # Node 0 should find blocks from Node 1 via bloom filter
        peer = node0.find_peer_with_block(75)
        assert peer == 1, f"Expected peer 1, got {peer}"

        # Node 1 should find blocks from Node 0 via bloom filter
        peer = node1.find_peer_with_block(25)
        assert peer == 0, f"Expected peer 0, got {peer}"

    def test_bloom_merge_preserves_all_entries(self):
        """Merged bloom should contain entries from all nodes."""
        n_nodes = 4
        n_items_per_node = 100
        blooms = []

        for node in range(n_nodes):
            bf = BloomFilter(expected_items=n_items_per_node, fp_rate=0.01)
            for i in range(n_items_per_node):
                bf.add(node * 1000 + i)
            blooms.append(bf)

        merged = BloomFilter.bitwise_or(blooms)

        # All items from all nodes should be found
        for node in range(n_nodes):
            for i in range(n_items_per_node):
                key = node * 1000 + i
                assert merged.contains(key), (
                    f"Merged bloom missing key {key} from node {node}"
                )


# ============================================================
# Evaluation / Simulation Tests
# ============================================================

class TestOfflineStateSimulation:
    """Tests that validate the core claim: bloom filter cooperative
    caching reduces discovery overhead."""

    def test_three_tier_hit_rates(self):
        """Simulate three-tier lookup with Zipfian workload and verify
        that local + peer hits bypass server significantly."""
        # Setup: 4 nodes, each with LRU cache of 256 entries
        n_nodes = 4
        cache_size = 256
        n_items = 10000
        n_ops = 10000

        # Create per-node LRU caches and bloom filters
        from collections import OrderedDict
        caches: list[OrderedDict] = [OrderedDict() for _ in range(n_nodes)]
        blooms = [
            BloomFilter(expected_items=cache_size, fp_rate=0.01,
                        auto_scale_clients=n_nodes)
            for _ in range(n_nodes)
        ]

        # Generate Zipfian-like workload (power law)
        random.seed(42)
        # Approximate Zipfian: popular items accessed much more often
        weights = [1.0 / (i + 1) ** 0.99 for i in range(n_items)]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        local_hits = 0
        peer_hits = 0
        server_hits = 0

        for op in range(n_ops):
            # Pick a random node and key
            node = op % n_nodes
            key = random.choices(range(n_items), weights=weights, k=1)[0]

            # Tier 1: Local cache
            if key in caches[node]:
                caches[node].move_to_end(key)
                local_hits += 1
                continue

            # Tier 2: Check merged bloom filter
            found_peer = False
            for other in range(n_nodes):
                if other == node:
                    continue
                if blooms[other].contains(key) and key in caches[other]:
                    found_peer = True
                    break

            if found_peer:
                peer_hits += 1
            else:
                server_hits += 1

            # Cache the item locally
            if len(caches[node]) >= cache_size:
                caches[node].popitem(last=False)
            caches[node][key] = True
            blooms[node].add(key)

        total = local_hits + peer_hits + server_hits
        local_rate = local_hits / total
        peer_rate = peer_hits / total
        server_rate = server_hits / total

        # With Zipfian workload, we expect significant local + peer hits
        # This validates the core OfflineState claim
        assert local_rate > 0.15, (
            f"Local hit rate {local_rate:.2%} too low for Zipfian workload"
        )
        assert local_rate + peer_rate > 0.30, (
            f"Combined local+peer rate {local_rate + peer_rate:.2%} "
            "doesn't show significant server bypass"
        )

    def test_bloom_filter_scales_with_nodes(self):
        """Bloom filter overhead should scale linearly with node count."""
        results = {}
        for n_nodes in [4, 8, 16, 32, 64]:
            bf = BloomFilter(
                expected_items=10000,
                fp_rate=0.01,
                auto_scale_clients=n_nodes,
            )
            results[n_nodes] = bf.size_bytes

        # Size should grow sub-linearly (sqrt scaling)
        # 64 nodes should be < 10x the size of 4 nodes
        ratio = results[64] / results[4]
        assert ratio < 10, (
            f"Bloom filter scaling ratio {ratio:.1f}x for 64/4 nodes "
            "is too high (expected < 10x due to sqrt scaling)"
        )
