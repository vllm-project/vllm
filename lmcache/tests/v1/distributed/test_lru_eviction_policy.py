# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for LRU eviction policy.

These tests verify the basic functionality of the LRUEvictionPolicy:
1. Key tracking (create, delete)
2. LRU ordering (least recently used first for eviction)
3. Touch updates access order
4. Expected ratio controls eviction count
5. Eviction destinations
"""

# Third Party

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy import LRUEvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionDestination,
)

# =============================================================================
# Helper Functions
# =============================================================================


def make_key(chunk_hash: int, model: str = "model", kv_rank: int = 0) -> ObjectKey:
    """Create an ObjectKey for testing."""
    hash_bytes = ObjectKey.IntHash2Bytes(chunk_hash)
    return ObjectKey(chunk_hash=hash_bytes, model_name=model, kv_rank=kv_rank)


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestLRUEvictionPolicyBasic:
    """Tests for basic LRU eviction policy functionality."""

    def test_empty_policy_has_no_tracked_keys(self):
        """A new policy should have no tracked keys."""
        policy = LRUEvictionPolicy()
        assert policy.get_num_tracked_keys() == 0

    def test_empty_policy_returns_no_eviction_actions(self):
        """An empty policy should return no eviction actions."""
        policy = LRUEvictionPolicy()
        actions = policy.get_eviction_actions(1.0)
        assert actions == []

    def test_create_keys_increases_tracked_count(self):
        """Creating keys should increase the tracked key count."""
        policy = LRUEvictionPolicy()
        keys = [make_key(1), make_key(2), make_key(3)]
        policy.on_keys_created(keys)
        assert policy.get_num_tracked_keys() == 3

    def test_delete_keys_decreases_tracked_count(self):
        """Deleting keys should decrease the tracked key count."""
        policy = LRUEvictionPolicy()
        keys = [make_key(1), make_key(2), make_key(3)]
        policy.on_keys_created(keys)
        policy.on_keys_removed([make_key(2)])
        assert policy.get_num_tracked_keys() == 2

    def test_delete_nonexistent_key_is_safe(self):
        """Deleting a nonexistent key should not raise an error."""
        policy = LRUEvictionPolicy()
        policy.on_keys_removed([make_key(999)])
        assert policy.get_num_tracked_keys() == 0

    def test_create_duplicate_key_does_not_increase_count(self):
        """Creating the same key twice should not duplicate it."""
        policy = LRUEvictionPolicy()
        key = make_key(1)
        policy.on_keys_created([key])
        policy.on_keys_created([key])
        assert policy.get_num_tracked_keys() == 1


# =============================================================================
# LRU Ordering Tests
# =============================================================================


class TestLRUEvictionPolicyOrdering:
    """Tests for LRU ordering behavior."""

    def test_eviction_order_is_lru(self):
        """Keys should be evicted in LRU order (oldest first)."""
        policy = LRUEvictionPolicy()
        # Create keys in order: 1, 2, 3
        policy.on_keys_created([make_key(1)])
        policy.on_keys_created([make_key(2)])
        policy.on_keys_created([make_key(3)])

        # Get candidates - should be in LRU order: 1, 2, 3
        candidates = policy.get_eviction_candidates(3)
        assert len(candidates) == 3
        assert ObjectKey.Bytes2IntHash(candidates[0].chunk_hash) == 1
        assert ObjectKey.Bytes2IntHash(candidates[1].chunk_hash) == 2
        assert ObjectKey.Bytes2IntHash(candidates[2].chunk_hash) == 3

    def test_touch_moves_key_to_most_recent(self):
        """Touching a key should move it to the most recently used position."""
        policy = LRUEvictionPolicy()
        # Create keys in order: 1, 2, 3, evict order is 3, 2, 1
        policy.on_keys_created([make_key(1), make_key(2), make_key(3)])

        # Touch key 3 - should move it to most recent
        policy.on_keys_touched([make_key(3)])

        # Now order should be: 2, 1, 3
        candidates = policy.get_eviction_candidates(3)
        assert ObjectKey.Bytes2IntHash(candidates[0].chunk_hash) == 2
        assert ObjectKey.Bytes2IntHash(candidates[1].chunk_hash) == 1
        assert ObjectKey.Bytes2IntHash(candidates[2].chunk_hash) == 3

    def test_touch_nonexistent_key_is_safe(self):
        """Touching a nonexistent key should not raise an error."""
        policy = LRUEvictionPolicy()
        policy.on_keys_created([make_key(1)])
        policy.on_keys_touched([make_key(999)])
        assert policy.get_num_tracked_keys() == 1

    def test_create_existing_key_moves_to_most_recent(self):
        """Creating an existing key should move it to most recent position."""
        policy = LRUEvictionPolicy()
        # Create keys in order: 1, 2, 3, evict order is 3, 2, 1
        policy.on_keys_created([make_key(1), make_key(2), make_key(3)])

        # Create key 3 again - should move to end
        policy.on_keys_created([make_key(3)])

        # Now order should be: 2, 1, 3
        candidates = policy.get_eviction_candidates(3)
        assert ObjectKey.Bytes2IntHash(candidates[0].chunk_hash) == 2
        assert ObjectKey.Bytes2IntHash(candidates[1].chunk_hash) == 1
        assert ObjectKey.Bytes2IntHash(candidates[2].chunk_hash) == 3


# =============================================================================
# Expected Ratio Tests
# =============================================================================


class TestLRUEvictionPolicyRatio:
    """Tests for expected_ratio behavior in get_eviction_actions."""

    def test_ratio_zero_returns_no_actions(self):
        """A ratio of 0 should return no eviction actions."""
        policy = LRUEvictionPolicy()
        policy.on_keys_created([make_key(i) for i in range(10)])
        actions = policy.get_eviction_actions(0.0)
        assert actions == []

    def test_ratio_one_returns_all_keys(self):
        """A ratio of 1.0 should return all keys (up to batch size)."""
        policy = LRUEvictionPolicy()
        policy.on_keys_created([make_key(i) for i in range(10)])
        actions = policy.get_eviction_actions(1.0)
        assert len(actions) == 1
        assert len(actions[0].keys) == 10

    def test_ratio_half_returns_half_keys(self):
        """A ratio of 0.5 should return approximately half the keys."""
        policy = LRUEvictionPolicy()
        policy.on_keys_created([make_key(i) for i in range(10)])
        actions = policy.get_eviction_actions(0.5)
        assert len(actions) == 1
        assert len(actions[0].keys) == 5

    def test_small_ratio_returns_at_least_one_key(self):
        """A small ratio > 0 should return at least 1 key."""
        policy = LRUEvictionPolicy()
        policy.on_keys_created([make_key(i) for i in range(10)])
        actions = policy.get_eviction_actions(0.01)
        assert len(actions) == 1
        assert len(actions[0].keys) >= 1

    def test_ratio_clamped_to_valid_range(self):
        """Ratios outside [0, 1] should be clamped."""
        policy = LRUEvictionPolicy()
        policy.on_keys_created([make_key(i) for i in range(10)])

        # Negative ratio should be treated as 0
        actions = policy.get_eviction_actions(-0.5)
        assert actions == []

        # Ratio > 1 should be treated as 1
        actions = policy.get_eviction_actions(1.5)
        assert len(actions[0].keys) == 10


# =============================================================================
# Eviction Destination Tests
# =============================================================================


class TestLRUEvictionPolicyDestination:
    """Tests for eviction destination behavior."""

    def test_default_destination_is_discard(self):
        """Default destination should be DISCARD."""
        policy = LRUEvictionPolicy()
        policy.on_keys_created([make_key(1)])
        actions = policy.get_eviction_actions(1.0)
        assert actions[0].destination == EvictionDestination.DISCARD

    def test_custom_default_destination(self):
        """Custom default destination should be used."""
        policy = LRUEvictionPolicy(default_destination=EvictionDestination.L2_CACHE)
        policy.on_keys_created([make_key(1)])
        actions = policy.get_eviction_actions(1.0)
        assert actions[0].destination == EvictionDestination.L2_CACHE

    def test_registered_destination_takes_precedence(self):
        """Registered destination should take precedence over default."""
        policy = LRUEvictionPolicy(default_destination=EvictionDestination.DISCARD)
        policy.register_eviction_destination(EvictionDestination.L2_CACHE)
        policy.on_keys_created([make_key(1)])
        actions = policy.get_eviction_actions(1.0)
        assert actions[0].destination == EvictionDestination.L2_CACHE


# =============================================================================
# Get Eviction Candidates Tests
# =============================================================================


class TestLRUEvictionPolicyCandidates:
    """Tests for get_eviction_candidates method."""

    def test_get_candidates_returns_lru_order(self):
        """Candidates should be returned in LRU order."""
        policy = LRUEvictionPolicy()
        # Create keys in order: 1, 2, 3, evict order is 3, 2, 1
        policy.on_keys_created([make_key(1), make_key(2), make_key(3)])
        candidates = policy.get_eviction_candidates(2)
        assert len(candidates) == 2
        assert ObjectKey.Bytes2IntHash(candidates[0].chunk_hash) == 3
        assert ObjectKey.Bytes2IntHash(candidates[1].chunk_hash) == 2

    def test_get_candidates_respects_count_limit(self):
        """Candidates should be limited by count parameter."""
        policy = LRUEvictionPolicy()
        policy.on_keys_created([make_key(i) for i in range(10)])
        candidates = policy.get_eviction_candidates(3)
        assert len(candidates) == 3

    def test_get_candidates_returns_all_if_count_exceeds_tracked(self):
        """If count exceeds tracked keys, return all keys."""
        policy = LRUEvictionPolicy()
        policy.on_keys_created([make_key(1), make_key(2)])
        candidates = policy.get_eviction_candidates(10)
        assert len(candidates) == 2


# =============================================================================
# Key Eligible Filter Tests
# =============================================================================


class TestLRUEvictionPolicyKeyEligibleFilter:
    """Tests for key_eligible_filter parameter in get_eviction_actions."""

    def test_key_eligible_filter_skips_filtered_keys(self):
        """Keys that fail the filter should be skipped during eviction."""
        policy = LRUEvictionPolicy()
        keys = [make_key(i) for i in range(5)]
        policy.on_keys_created(keys)

        # Filter out keys with even chunk_hash
        def key_eligible_filter(key: ObjectKey) -> bool:
            return ObjectKey.Bytes2IntHash(key.chunk_hash) % 2 != 0

        actions = policy.get_eviction_actions(
            1.0, key_eligible_filter=key_eligible_filter
        )
        assert len(actions) == 1
        evicted_hashes = {
            ObjectKey.Bytes2IntHash(k.chunk_hash) for k in actions[0].keys
        }
        # Only odd keys should be evicted: 1, 3
        assert evicted_hashes == {1, 3}

    def test_key_eligible_filter_none_evicts_all(self):
        """When key_eligible_filter is None, all keys should be eligible."""
        policy = LRUEvictionPolicy()
        keys = [make_key(i) for i in range(5)]
        policy.on_keys_created(keys)

        actions = policy.get_eviction_actions(1.0, key_eligible_filter=None)
        assert len(actions) == 1
        assert len(actions[0].keys) == 5

    def test_key_eligible_filter_rejects_all_returns_empty(self):
        """When filter rejects all keys, no eviction actions should be returned."""
        policy = LRUEvictionPolicy()
        keys = [make_key(i) for i in range(5)]
        policy.on_keys_created(keys)

        actions = policy.get_eviction_actions(1.0, key_eligible_filter=lambda _: False)
        assert actions == []

    def test_key_eligible_filter_respects_target_count(self):
        """Filter should stop collecting once target_count is reached."""
        policy = LRUEvictionPolicy()
        # Create 10 keys, all pass filter
        keys = [make_key(i) for i in range(10)]
        policy.on_keys_created(keys)

        # Request 50% eviction with a filter that accepts all
        actions = policy.get_eviction_actions(0.5, key_eligible_filter=lambda _: True)
        assert len(actions) == 1
        assert len(actions[0].keys) == 5

    def test_key_eligible_filter_with_partial_acceptance(self):
        """When filter accepts some keys, only accepted keys up to
        target_count should be evicted."""
        policy = LRUEvictionPolicy()
        # Create 10 keys: 0..9
        keys = [make_key(i) for i in range(10)]
        policy.on_keys_created(keys)

        # Filter: only accept keys with hash >= 5
        def key_eligible_filter(key: ObjectKey) -> bool:
            return ObjectKey.Bytes2IntHash(key.chunk_hash) >= 5

        # Request 100% eviction -> target_count = 10, but only 5 pass filter
        actions = policy.get_eviction_actions(
            1.0, key_eligible_filter=key_eligible_filter
        )
        assert len(actions) == 1
        assert len(actions[0].keys) == 5
        for k in actions[0].keys:
            assert ObjectKey.Bytes2IntHash(k.chunk_hash) >= 5

    def test_key_eligible_filter_preserves_lru_order(self):
        """Filtered eviction should still respect LRU order."""
        policy = LRUEvictionPolicy()
        # Create keys in order: 1, 2, 3, 4, 5
        for i in range(1, 6):
            policy.on_keys_created([make_key(i)])

        # Touch key 1 to make it most recently used
        policy.on_keys_touched([make_key(1)])

        # Filter: accept all keys
        actions = policy.get_eviction_actions(0.6, key_eligible_filter=lambda _: True)
        assert len(actions) == 1
        evicted_hashes = [
            ObjectKey.Bytes2IntHash(k.chunk_hash) for k in actions[0].keys
        ]
        # LRU order after touch: 2, 3, 4, 5, 1
        # 60% of 5 = 3 keys, should be 2, 3, 4
        assert evicted_hashes == [2, 3, 4]

    def test_key_eligible_filter_skips_locked_simulated(self):
        """Simulate the real use case: filter skips 'locked' keys."""
        policy = LRUEvictionPolicy()
        keys = [make_key(i) for i in range(1, 6)]
        policy.on_keys_created(keys)

        # Simulate: keys 1 and 3 are "locked" (not evictable)
        locked_hashes = {1, 3}

        def key_eligible_filter(key: ObjectKey) -> bool:
            return ObjectKey.Bytes2IntHash(key.chunk_hash) not in locked_hashes

        # Request 100% eviction
        actions = policy.get_eviction_actions(
            1.0, key_eligible_filter=key_eligible_filter
        )
        assert len(actions) == 1
        evicted_hashes = {
            ObjectKey.Bytes2IntHash(k.chunk_hash) for k in actions[0].keys
        }
        # Keys 1 and 3 should be skipped
        assert evicted_hashes == {2, 4, 5}

    def test_key_eligible_filter_with_small_ratio_and_many_filtered(self):
        """When many keys are filtered, a small ratio should still
        collect enough eligible keys."""
        policy = LRUEvictionPolicy()
        # Create 20 keys: 0..19
        keys = [make_key(i) for i in range(20)]
        policy.on_keys_created(keys)

        # Filter: only accept keys with hash divisible by 5
        # Eligible: 0, 5, 10, 15 (4 keys)
        def key_eligible_filter(key: ObjectKey) -> bool:
            return ObjectKey.Bytes2IntHash(key.chunk_hash) % 5 == 0

        # Request 10% eviction -> target_count = 2
        actions = policy.get_eviction_actions(
            0.1, key_eligible_filter=key_eligible_filter
        )
        assert len(actions) == 1
        assert len(actions[0].keys) == 2
        for k in actions[0].keys:
            assert ObjectKey.Bytes2IntHash(k.chunk_hash) % 5 == 0
