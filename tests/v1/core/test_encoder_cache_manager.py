# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderCacheStats,
)

pytestmark = pytest.mark.cpu_test


# ------------------ Mock Classes ------------------ #
class MockRequest:
    def __init__(self, request_id, mm_hashes, token_counts):
        self.request_id = request_id
        self._token_counts = token_counts
        self.mm_features = []
        for i, mm_hash in enumerate(mm_hashes):
            feature = MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=mm_hash,
                mm_position=PlaceholderRange(offset=0, length=self._token_counts[i]),
            )
            self.mm_features.append(feature)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        return self._token_counts[input_id]


# ------------------ Unit Tests ------------------ #
def test_basic_allocate_and_reuse():
    cache = EncoderCacheManager(cache_size=10)
    req = MockRequest("r1", ["imgA"], [4])

    assert not cache.check_and_update_cache(req, 0)
    assert cache.can_allocate(req, 0, int(1e9), 0)

    cache.allocate(req, 0)

    assert cache.check_and_update_cache(req, 0)
    assert "r1" in cache.cached["imgA"]
    assert cache.num_free_slots == 6

    # Free twice to bring refcount to 0.
    cache.free_encoder_input(req, 0)
    cache.free_encoder_input(req, 0)

    assert not cache.cached["imgA"]
    assert "imgA" in cache.freeable
    assert cache.num_freeable_slots == 10
    assert cache.num_free_slots == 6


def test_freeing_decreases_refcount_and_moves_to_freeable():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("req2", ["img3"], [5])

    assert manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)

    assert len(manager.cached["img3"]) == 1

    manager.free_encoder_input(req, 0)

    assert not manager.cached["img3"]
    assert "img3" in manager.freeable
    assert manager.num_freeable_slots == 10


def test_free_request_frees_all_inputs():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("req3", ["a", "b"], [2, 3])

    assert manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)

    assert manager.can_allocate(req, 1, int(1e9), 0)
    manager.allocate(req, 1)

    assert len(manager.cached["a"]) == 1
    assert len(manager.cached["b"]) == 1

    manager.free(req)

    assert not manager.cached["a"]
    assert not manager.cached["b"]
    assert "a" in manager.freeable
    assert "b" in manager.freeable
    assert manager.num_freeable_slots == 10


def test_eviction_when_cache_is_full():
    manager = EncoderCacheManager(cache_size=10)

    req1 = MockRequest("req1", ["x"], [6])
    req2 = MockRequest("req2", ["y"], [5])

    assert manager.can_allocate(req1, 0, int(1e9), 0)
    manager.allocate(req1, 0)
    manager.free_encoder_input(req1, 0)

    assert manager.can_allocate(req2, 0, int(1e9), 0)
    manager.allocate(req2, 0)

    # 'x' should have been evicted.
    assert "x" not in manager.cached
    assert "x" in manager.get_freed_mm_hashes()


def test_get_cached_input_ids():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("reqX", ["m", "n", "o"], [2, 4, 3])

    assert manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)

    assert manager.can_allocate(req, 2, int(1e9), 0)
    manager.allocate(req, 2)

    cached_ids = manager.get_cached_input_ids(req)
    assert cached_ids == {0, 2}


def test_has_cache_restores_from_freeable():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("reqY", ["imgZ"], [4])

    assert manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)

    manager.free_encoder_input(req, 0)

    # Should restore from freeable.
    assert manager.check_and_update_cache(req, 0)
    assert len(manager.cached["imgZ"]) == 1
    assert "imgZ" not in manager.freeable
    assert manager.num_freeable_slots == 6


def test_get_freed_mm_hashes_clears_freed_list():
    manager = EncoderCacheManager(cache_size=10)
    req1 = MockRequest("reqA", ["a"], [5])
    req2 = MockRequest("reqB", ["b"], [6])

    assert manager.can_allocate(req1, 0, int(1e9), 0)
    manager.allocate(req1, 0)
    manager.free_encoder_input(req1, 0)

    # Should trigger eviction of 'a'.
    assert manager.can_allocate(req2, 0, int(1e9), 0)
    manager.allocate(req2, 0)

    freed = manager.get_freed_mm_hashes()
    assert "a" in freed
    assert manager.get_freed_mm_hashes() == []


def test_schedule_request_multi_images_respect_space_limit():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("reqA", ["a", "b"], [5, 6])
    compute_budget = 100

    num_tokens_to_schedule = 0
    assert manager.can_allocate(req, 0, compute_budget, num_tokens_to_schedule)
    num_tokens_to_schedule += req.get_num_encoder_tokens(0)
    compute_budget -= req.get_num_encoder_tokens(0)

    assert not manager.can_allocate(req, 1, compute_budget, num_tokens_to_schedule)


def test_schedule_request_multi_images_respect_compute_limit():
    manager = EncoderCacheManager(cache_size=100)
    req = MockRequest("reqA", ["a", "b"], [5, 6])
    compute_budget = 10
    num_tokens_to_schedule = 0
    assert manager.can_allocate(req, 0, compute_budget, num_tokens_to_schedule)
    num_tokens_to_schedule += req.get_num_encoder_tokens(0)
    compute_budget -= req.get_num_encoder_tokens(0)

    assert not manager.can_allocate(req, 1, compute_budget, num_tokens_to_schedule)


# ---------- Computation time tracking tests ----------


def test_cache_hit_miss_stats():
    """Cache hits and misses are counted correctly."""
    manager = EncoderCacheManager(cache_size=20)
    req = MockRequest("r1", ["imgA"], [4])

    # Miss
    assert not manager.check_and_update_cache(req, 0)
    assert manager.stats.num_cache_misses == 1
    assert manager.stats.num_cache_hits == 0

    manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)

    # Hit
    req2 = MockRequest("r2", ["imgA"], [4])
    assert manager.check_and_update_cache(req2, 0)
    assert manager.stats.num_cache_hits == 1
    assert manager.stats.num_cache_misses == 1
    assert manager.stats.hit_rate == pytest.approx(0.5)


def test_record_compute_time():
    """record_compute_time stores time in entry_meta and aggregates."""
    manager = EncoderCacheManager(cache_size=20)
    req = MockRequest("r1", ["imgA", "imgB"], [4, 6])

    manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)
    manager.record_compute_time("imgA", 0.05)

    manager.can_allocate(req, 1, int(1e9), 0)
    manager.allocate(req, 1)
    manager.record_compute_time("imgB", 0.10)

    assert manager.stats.total_compute_time == pytest.approx(0.15)

    meta_a = manager.get_entry_meta("imgA")
    assert meta_a is not None
    assert meta_a.compute_time == pytest.approx(0.05)
    assert meta_a.num_tokens == 4

    meta_b = manager.get_entry_meta("imgB")
    assert meta_b is not None
    assert meta_b.compute_time == pytest.approx(0.10)
    assert meta_b.num_tokens == 6


def test_time_saved_on_cache_hit():
    """Cache hits accumulate total_time_saved from recorded compute_time."""
    manager = EncoderCacheManager(cache_size=20)
    req = MockRequest("r1", ["imgA"], [4])

    manager.check_and_update_cache(req, 0)  # miss
    manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)
    manager.record_compute_time("imgA", 0.05)

    # Hit from a different request
    req2 = MockRequest("r2", ["imgA"], [4])
    assert manager.check_and_update_cache(req2, 0)
    assert manager.stats.total_time_saved == pytest.approx(0.05)

    # Another hit
    req3 = MockRequest("r3", ["imgA"], [4])
    assert manager.check_and_update_cache(req3, 0)
    assert manager.stats.total_time_saved == pytest.approx(0.10)
    assert manager.stats.num_cache_hits == 2


def test_eviction_preserves_meta_for_recompute_tracking():
    """Evicted entries' metadata is preserved so re-computation cost is
    tracked."""
    manager = EncoderCacheManager(cache_size=10)

    req1 = MockRequest("r1", ["x"], [6])
    req2 = MockRequest("r2", ["y"], [5])

    manager.can_allocate(req1, 0, int(1e9), 0)
    manager.allocate(req1, 0)
    manager.record_compute_time("x", 0.08)
    manager.free_encoder_input(req1, 0)

    # Eviction of 'x' happens here
    manager.can_allocate(req2, 0, int(1e9), 0)
    manager.allocate(req2, 0)

    assert "x" not in manager.entry_meta
    assert "x" in manager.evicted_meta
    assert manager.evicted_meta["x"].compute_time == pytest.approx(0.08)


def test_recompute_after_eviction_stats():
    """Re-computing a previously evicted entry updates eviction cost stats."""
    manager = EncoderCacheManager(cache_size=10)

    req1 = MockRequest("r1", ["x"], [6])
    req2 = MockRequest("r2", ["y"], [5])
    req3 = MockRequest("r3", ["x"], [6])

    # Allocate and record time for 'x'
    manager.check_and_update_cache(req1, 0)  # miss
    manager.can_allocate(req1, 0, int(1e9), 0)
    manager.allocate(req1, 0)
    manager.record_compute_time("x", 0.08)
    manager.free_encoder_input(req1, 0)

    # Evict 'x' by allocating 'y'
    manager.check_and_update_cache(req2, 0)  # miss
    manager.can_allocate(req2, 0, int(1e9), 0)
    manager.allocate(req2, 0)
    manager.free_encoder_input(req2, 0)

    # Now re-allocate 'x' (miss after eviction)
    manager.check_and_update_cache(req3, 0)  # miss — 'x' was evicted
    # Evict 'y' to make room for 'x' again
    manager.can_allocate(req3, 0, int(1e9), 0)
    manager.allocate(req3, 0)
    manager.record_compute_time("x", 0.09)

    assert manager.stats.num_recomputed_after_eviction == 1
    assert manager.stats.total_time_lost_to_eviction == pytest.approx(0.09)
    assert "x" not in manager.evicted_meta  # cleaned up after re-compute


def test_cost_density():
    """cost_density = compute_time / num_tokens."""
    manager = EncoderCacheManager(cache_size=20)
    req = MockRequest("r1", ["imgA"], [4])

    manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)
    manager.record_compute_time("imgA", 0.08)

    meta = manager.get_entry_meta("imgA")
    assert meta is not None
    assert meta.cost_density == pytest.approx(0.02)


def test_get_all_entry_meta():
    """get_all_entry_meta returns all tracked entries."""
    manager = EncoderCacheManager(cache_size=20)
    req = MockRequest("r1", ["a", "b"], [3, 5])

    manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)
    manager.can_allocate(req, 1, int(1e9), 0)
    manager.allocate(req, 1)

    all_meta = manager.get_all_entry_meta()
    assert set(all_meta.keys()) == {"a", "b"}
    assert all_meta["a"].num_tokens == 3
    assert all_meta["b"].num_tokens == 5


def test_hit_increments_num_hits_in_entry_meta():
    """Each cache hit increments the entry's num_hits counter."""
    manager = EncoderCacheManager(cache_size=20)
    req = MockRequest("r1", ["imgA"], [4])

    manager.check_and_update_cache(req, 0)  # miss
    manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)
    manager.record_compute_time("imgA", 0.05)

    # Two hits from different requests
    for i in range(2):
        r = MockRequest(f"r{i+2}", ["imgA"], [4])
        manager.check_and_update_cache(r, 0)

    meta = manager.get_entry_meta("imgA")
    assert meta is not None
    assert meta.num_hits == 2


def test_stats_repr():
    """EncoderCacheStats has a meaningful string representation."""
    stats = EncoderCacheStats(
        num_cache_hits=10,
        num_cache_misses=5,
        total_time_saved=0.5,
        total_compute_time=0.3,
    )
    s = repr(stats)
    assert "hits=10" in s
    assert "misses=5" in s
    assert "hit_rate=66.67%" in s
