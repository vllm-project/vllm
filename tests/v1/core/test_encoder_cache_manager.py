# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderCacheEntryMeta,
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


# ---------- Lagrangian replacement algorithm tests ----------


def test_step_updates_advantage_for_non_arriving_entries():
    """step() adds m_i * lambda to A_i for entries that did not arrive."""
    manager = EncoderCacheManager(cache_size=100, eta=0.1,
                                  capacity_target_ratio=0.5)
    req = MockRequest("r1", ["a", "b"], [10, 20])

    # Allocate both items
    manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)
    manager.record_compute_time("a", 0.5)  # A_a = -0.5
    manager.can_allocate(req, 1, int(1e9), 0)
    manager.allocate(req, 1)
    manager.record_compute_time("b", 1.0)  # A_b = -1.0

    # First step: arrived items are skipped, lambda is updated
    manager.step()

    # After step(): arrived set is cleared, lambda updated
    # usage = 100 - 70 = 30, target = 50
    # lambda = 0 + 0.1 * (30 - 50) = -2.0
    assert manager.lambda_val == pytest.approx(-2.0)

    # A_a and A_b should be unchanged (they arrived this step)
    assert manager.get_entry_meta("a").advantage == pytest.approx(-0.5)
    assert manager.get_entry_meta("b").advantage == pytest.approx(-1.0)

    # Second step: now they are NOT in arrived set, so advantage accumulates
    manager.step()

    # lambda = -2.0 + 0.1 * (30 - 50) = -4.0
    assert manager.lambda_val == pytest.approx(-4.0)

    # A_a = -0.5 + 10 * (-2.0) = -20.5
    # A_b = -1.0 + 20 * (-2.0) = -41.0
    assert manager.get_entry_meta("a").advantage == pytest.approx(-20.5)
    assert manager.get_entry_meta("b").advantage == pytest.approx(-41.0)


def test_step_lambda_increases_when_over_capacity():
    """lambda increases when cache usage exceeds the capacity target."""
    # cache_size=10, target=5 (50%), eta=0.1
    manager = EncoderCacheManager(cache_size=10, eta=0.1,
                                  capacity_target_ratio=0.5)
    req = MockRequest("r1", ["a"], [8])

    manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)
    manager.record_compute_time("a", 0.1)

    # usage = 8, target = 5 → lambda += 0.1 * (8 - 5) = 0.3
    manager.step()
    assert manager.lambda_val == pytest.approx(0.3)

    # Next step: lambda += 0.1 * (8 - 5) = 0.6
    # A_a = -0.1 + 8 * 0.3 = 2.3 > 0 → is_evictable
    manager.step()
    assert manager.lambda_val == pytest.approx(0.6)
    meta = manager.get_entry_meta("a")
    assert meta.advantage == pytest.approx(2.3)
    assert meta.is_evictable


def test_step_lambda_decreases_when_under_capacity():
    """lambda decreases when cache usage is below the capacity target."""
    manager = EncoderCacheManager(cache_size=100, eta=0.1,
                                  capacity_target_ratio=0.9)
    # Empty cache: usage = 0, target = 90
    manager.step()
    # lambda = 0 + 0.1 * (0 - 90) = -9.0
    assert manager.lambda_val == pytest.approx(-9.0)


def test_eviction_prefers_highest_advantage():
    """When eviction is needed, the entry with the highest A_i is evicted."""
    manager = EncoderCacheManager(cache_size=10, eta=0.1,
                                  capacity_target_ratio=0.5)

    # Allocate two entries, give them different advantages
    req1 = MockRequest("r1", ["cheap"], [4])
    req2 = MockRequest("r2", ["expensive"], [4])

    manager.can_allocate(req1, 0, int(1e9), 0)
    manager.allocate(req1, 0)
    manager.record_compute_time("cheap", 0.01)  # A = -0.01 (cheap)

    manager.can_allocate(req2, 0, int(1e9), 0)
    manager.allocate(req2, 0)
    manager.record_compute_time("expensive", 10.0)  # A = -10.0 (expensive)

    manager.step()  # Clear arrived set

    # Free both so they become evictable
    manager.free_encoder_input(req1, 0)
    manager.free_encoder_input(req2, 0)

    assert "cheap" in manager.freeable
    assert "expensive" in manager.freeable

    # "cheap" has advantage -0.01, "expensive" has advantage -10.0
    # highest advantage is "cheap" → it should be evicted first
    req3 = MockRequest("r3", ["new"], [5])
    manager.can_allocate(req3, 0, int(1e9), 0)
    manager.allocate(req3, 0)

    # "cheap" should have been evicted (higher advantage)
    assert "cheap" not in manager.cached
    # "expensive" should survive (lower advantage = more valuable to keep)
    assert "expensive" in manager.cached


def test_large_memory_items_evicted_before_small():
    """Items with larger m_i accumulate advantage faster and get evicted
    sooner (given the same compute cost and enough cache pressure)."""
    # cache_size=20, target=10 (50%), eta=0.5
    manager = EncoderCacheManager(cache_size=20, eta=0.5,
                                  capacity_target_ratio=0.5)

    req1 = MockRequest("r1", ["small"], [2])    # m=2
    req2 = MockRequest("r2", ["large"], [10])   # m=10

    # Both have the same compute cost
    manager.can_allocate(req1, 0, int(1e9), 0)
    manager.allocate(req1, 0)
    manager.record_compute_time("small", 0.5)

    manager.can_allocate(req2, 0, int(1e9), 0)
    manager.allocate(req2, 0)
    manager.record_compute_time("large", 0.5)

    # usage = 12, target = 10 → lambda += 0.5 * (12 - 10) = 1.0
    manager.step()
    assert manager.lambda_val == pytest.approx(1.0)

    # Now accumulate: A_small = -0.5 + 2*1.0 = 1.5
    #                 A_large = -0.5 + 10*1.0 = 9.5
    manager.step()

    assert manager.get_entry_meta("small").advantage == pytest.approx(1.5)
    assert manager.get_entry_meta("large").advantage == pytest.approx(9.5)
    # "large" has higher advantage → evicted first
    assert manager.get_entry_meta("large").advantage > \
           manager.get_entry_meta("small").advantage


def test_expensive_items_survive_longer():
    """Items with higher c_i start with a more negative advantage and
    take more steps to become evictable."""
    manager = EncoderCacheManager(cache_size=20, eta=0.5,
                                  capacity_target_ratio=0.5)

    req1 = MockRequest("r1", ["cheap"], [4])
    req2 = MockRequest("r2", ["expensive"], [4])

    manager.can_allocate(req1, 0, int(1e9), 0)
    manager.allocate(req1, 0)
    manager.record_compute_time("cheap", 0.1)      # A = -0.1

    manager.can_allocate(req2, 0, int(1e9), 0)
    manager.allocate(req2, 0)
    manager.record_compute_time("expensive", 100.0) # A = -100.0

    # usage = 12, target = 10 → strong pressure
    manager.step()  # Clear arrived; lambda = 0.5*(12-10) = 1.0

    # Simulate many steps of advantage accumulation
    for _ in range(10):
        manager.step()

    meta_cheap = manager.get_entry_meta("cheap")
    meta_expensive = manager.get_entry_meta("expensive")

    # cheap should become evictable much sooner than expensive
    assert meta_cheap.advantage > meta_expensive.advantage
    assert meta_cheap.is_evictable
    assert not meta_expensive.is_evictable


def test_is_evictable_property():
    """EncoderCacheEntryMeta.is_evictable reflects A_i > 0."""
    meta_pos = EncoderCacheEntryMeta(advantage=0.1)
    meta_neg = EncoderCacheEntryMeta(advantage=-0.5)
    meta_zero = EncoderCacheEntryMeta(advantage=0.0)

    assert meta_pos.is_evictable
    assert not meta_neg.is_evictable
    assert not meta_zero.is_evictable


def test_arrival_resets_advantage():
    """re-recorded compute time resets A_i to -c_i (arrival event)."""
    manager = EncoderCacheManager(cache_size=20, eta=0.1,
                                  capacity_target_ratio=0.5)
    req = MockRequest("r1", ["a"], [5])

    manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)
    manager.record_compute_time("a", 0.2)  # A = -0.2

    manager.step()
    # Simulate a few steps so advantage drifts
    for _ in range(5):
        manager.step()

    meta = manager.get_entry_meta("a")
    old_advantage = meta.advantage

    # Now the item is "re-computed" (e.g., after eviction and re-allocation)
    # Simulate by calling record_compute_time again
    manager.record_compute_time("a", 0.3)
    assert meta.advantage == pytest.approx(-0.3)
    assert meta.advantage != old_advantage


def test_step_on_empty_cache_is_safe():
    """step() on an empty cache does not crash and updates lambda."""
    manager = EncoderCacheManager(cache_size=10, eta=0.1,
                                  capacity_target_ratio=0.9)
    # target = 9, usage = 0
    manager.step()
    # lambda = 0 + 0.1 * (0 - 9) = -0.9
    assert manager.lambda_val == pytest.approx(-0.9)
    manager.step()
    # lambda = -0.9 + 0.1 * (0 - 9) = -1.8
    assert manager.lambda_val == pytest.approx(-1.8)
