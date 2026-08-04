# SPDX-License-Identifier: Apache-2.0
"""Tests for the LMCache cache simulator tool."""

# Standard
from pathlib import Path
import json
import tempfile

# Third Party
import pytest

# First Party
from lmcache.tools.cache_simulator.lru_cache import LRUCache, LRUCacheFast
from lmcache.tools.cache_simulator.simulator import (
    compute_kv_bytes_per_chunk,
    load_lookup_events,
    simulate,
)

# ---------------------------------------------------------------------------
# LRUCacheFast
# ---------------------------------------------------------------------------


class TestLRUCacheFast:
    def test_basic_insert_and_contains(self):
        cache = LRUCacheFast(3)
        cache.insert("a")
        assert cache.contains("a")
        assert not cache.contains("b")

    def test_evicts_lru_when_full(self):
        cache = LRUCacheFast(2)
        cache.insert("a")
        cache.insert("b")
        cache.insert("c")  # should evict "a"
        assert not cache.contains("a")
        assert cache.contains("b")
        assert cache.contains("c")
        assert cache.eviction_count == 1

    def test_access_refreshes_lru_order(self):
        cache = LRUCacheFast(2)
        cache.insert("a")
        cache.insert("b")
        cache.access("a")  # "a" is now MRU; "b" is LRU
        cache.insert("c")  # should evict "b"
        assert cache.contains("a")
        assert not cache.contains("b")
        assert cache.contains("c")

    def test_insert_existing_key_does_not_grow(self):
        cache = LRUCacheFast(2)
        cache.insert("a")
        cache.insert("a")
        assert len(cache) == 1
        assert cache.eviction_count == 0

    def test_capacity_one(self):
        cache = LRUCacheFast(1)
        cache.insert("a")
        cache.insert("b")
        assert not cache.contains("a")
        assert cache.contains("b")
        assert cache.eviction_count == 1

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            LRUCacheFast(0)


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------


class TestLRUCache:
    def test_basic_insert_and_contains(self):
        cache = LRUCache(3)
        cache.insert("a")
        assert cache.contains("a")
        assert not cache.contains("z")

    def test_evicts_lru_when_full(self):
        cache = LRUCache(2)
        cache.insert("a")
        cache.insert("b")
        cache.insert("c")  # evicts "a"
        assert not cache.contains("a")
        assert cache.eviction_count == 1

    def test_position_mru_is_zero(self):
        cache = LRUCache(3)
        cache.insert("a")
        cache.insert("b")
        cache.insert("c")
        # "c" was inserted last → MRU → position 0
        assert cache.position("c") == 0
        # "a" was inserted first → LRU → position 2
        assert cache.position("a") == 2

    def test_access_moves_to_mru(self):
        cache = LRUCache(3)
        cache.insert("a")
        cache.insert("b")
        cache.insert("c")
        cache.access("a")  # "a" becomes MRU
        assert cache.position("a") == 0
        assert cache.position("c") == 1
        assert cache.position("b") == 2

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            LRUCache(0)


# ---------------------------------------------------------------------------
# compute_kv_bytes_per_chunk
# ---------------------------------------------------------------------------


class TestComputeKvBytesPerChunk:
    def test_float16(self):
        event = {"shapes": [[32, 256, 128]], "dtypes": ["float16"]}
        assert compute_kv_bytes_per_chunk(event) == 32 * 256 * 128 * 2

    def test_multiple_tensors(self):
        # Two tensors: one float16, one float8_e4m3fn
        event = {
            "shapes": [[4, 256, 64], [4, 256, 64]],
            "dtypes": ["float16", "float8_e4m3fn"],
        }
        expected = 4 * 256 * 64 * 2 + 4 * 256 * 64 * 1
        assert compute_kv_bytes_per_chunk(event) == expected

    def test_empty_shapes_returns_zero(self):
        assert compute_kv_bytes_per_chunk({"shapes": [], "dtypes": []}) == 0

    def test_missing_fields_returns_zero(self):
        assert compute_kv_bytes_per_chunk({}) == 0


# ---------------------------------------------------------------------------
# load_lookup_events
# ---------------------------------------------------------------------------


class TestLoadLookupEvents:
    def _write_jsonl(self, lines):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, dir="/tmp")
        for line in lines:
            f.write(json.dumps(line) + "\n")
        f.close()
        return Path(f.name)

    def test_loads_and_sorts_by_timestamp(self):
        path = self._write_jsonl(
            [
                {
                    "timestamp": 2.0,
                    "chunk_hashes": ["0xbb"],
                    "seq_len": 256,
                    "chunk_size": 256,
                },
                {
                    "timestamp": 1.0,
                    "chunk_hashes": ["0xaa"],
                    "seq_len": 256,
                    "chunk_size": 256,
                },
            ]
        )
        events = load_lookup_events([path])
        assert events[0]["chunk_hashes"] == ["0xaa"]
        assert events[1]["chunk_hashes"] == ["0xbb"]
        path.unlink()

    def test_model_filter(self):
        path = self._write_jsonl(
            [
                {
                    "timestamp": 1.0,
                    "model_name": "A",
                    "chunk_hashes": [],
                    "seq_len": 0,
                    "chunk_size": 256,
                },
                {
                    "timestamp": 2.0,
                    "model_name": "B",
                    "chunk_hashes": [],
                    "seq_len": 0,
                    "chunk_size": 256,
                },
            ]
        )
        events = load_lookup_events([path], model="A")
        assert len(events) == 1
        assert events[0]["model_name"] == "A"
        path.unlink()

    def test_max_samples(self):
        path = self._write_jsonl(
            [
                {
                    "timestamp": float(i),
                    "chunk_hashes": [],
                    "seq_len": 0,
                    "chunk_size": 256,
                }
                for i in range(10)
            ]
        )
        events = load_lookup_events([path], max_samples=3)
        assert len(events) == 3
        path.unlink()

    def test_skips_malformed_lines(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, dir="/tmp")
        f.write(
            '{"timestamp": 1.0, "chunk_hashes": [], "seq_len": 0, "chunk_size": 256}\n'
        )
        f.write("not json\n")
        f.write(
            '{"timestamp": 2.0, "chunk_hashes": [], "seq_len": 0, "chunk_size": 256}\n'
        )
        f.close()
        # Standard
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            events = load_lookup_events([Path(f.name)])
        assert len(events) == 2
        Path(f.name).unlink()


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


class TestSimulate:
    """Core simulation correctness tests."""

    def _make_events(self, specs):
        """Build minimal event dicts from (chunk_hashes, seq_len) pairs."""
        return [
            {"chunk_hashes": hashes, "seq_len": seq_len, "chunk_size": 256}
            for hashes, seq_len in specs
        ]

    def test_cold_cache_zero_hits(self):
        events = self._make_events(
            [
                (["0xaa", "0xbb"], 600),
            ]
        )
        res = simulate(events, cache_capacity_bytes=100, kv_bytes_per_chunk=1)
        assert res["token_hit_rate"] == 0.0
        assert res["total_hit_tokens"] == 0
        assert res["total_tokens"] == 600

    def test_second_identical_request_hits(self):
        # r1: cold miss; r2: full prefix hit (2 chunks × 256 = 512 tokens)
        events = self._make_events(
            [
                (["0xaa", "0xbb"], 600),
                (["0xaa", "0xbb"], 600),
            ]
        )
        res = simulate(events, cache_capacity_bytes=100, kv_bytes_per_chunk=1)
        assert res["total_hit_tokens"] == 512
        assert res["total_tokens"] == 1200
        assert abs(res["token_hit_rate"] - 512 / 1200) < 1e-9

    def test_prefix_semantics_breaks_on_first_miss(self):
        # r1 populates 0xaa, 0xbb, 0xcc
        # r2: 0xaa hits, 0xdd misses → prefix=1; 0xbb is in cache but not counted
        events = self._make_events(
            [
                (["0xaa", "0xbb", "0xcc"], 768),
                (["0xaa", "0xdd", "0xbb"], 768),
            ]
        )
        res = simulate(events, cache_capacity_bytes=100, kv_bytes_per_chunk=1)
        # Only 0xaa is a prefix hit in r2 → 1 × 256 = 256 hit tokens
        assert res["total_hit_tokens"] == 256

    def test_tail_tokens_always_miss(self):
        # seq_len=300, chunk_size=256 → 1 full chunk + 44 tail tokens
        # On second request the full chunk hits but tail is always a miss
        events = self._make_events(
            [
                (["0xaa"], 300),
                (["0xaa"], 300),
            ]
        )
        res = simulate(events, cache_capacity_bytes=100, kv_bytes_per_chunk=1)
        # hit tokens = 256, total = 600 → tail (44 tokens per request) is miss
        assert res["total_hit_tokens"] == 256
        assert res["total_tokens"] == 600

    def test_eviction_reduces_hits(self):
        # capacity = 1 chunk; insert 0xaa then 0xbb evicts 0xaa
        events = self._make_events(
            [
                (["0xaa"], 256),  # populates 0xaa
                (["0xbb"], 256),  # evicts 0xaa, populates 0xbb
                (["0xaa"], 256),  # 0xaa was evicted → miss; evicts 0xbb
            ]
        )
        res = simulate(events, cache_capacity_bytes=1, kv_bytes_per_chunk=1)
        assert res["total_hit_tokens"] == 0
        assert res["eviction_count"] == 2

    def test_fast_mode_matches_normal_token_hit_rate(self):
        events = self._make_events(
            [
                (["0xaa", "0xbb"], 600),
                (["0xaa", "0xbb"], 600),
                (["0xcc"], 300),
            ]
        )
        cap, bpc = 100, 1
        res_normal = simulate(events, cap, bpc, fast=False)
        res_fast = simulate(events, cap, bpc, fast=True)
        assert abs(res_normal["token_hit_rate"] - res_fast["token_hit_rate"]) < 1e-9

    def test_invalid_kv_bytes_raises(self):
        with pytest.raises(ValueError):
            simulate([], cache_capacity_bytes=100, kv_bytes_per_chunk=0)
