# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for :class:`IsolatedLRUEvictionPolicy`.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.internal_api import EvictionDestination


def _key(chunk_id: int, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="m",
        kv_rank=0,
        cache_salt=cache_salt,
    )


class TestSupportIsolation:
    def test_support_isolation_is_true(self):
        """``IsolatedLRUEvictionPolicy`` reports isolation support so
        the controller routes to the by-cache_salt eviction branch."""
        assert IsolatedLRUEvictionPolicy().support_isolation is True


class TestPerSaltTracking:
    def test_keys_bucketed_by_salt(self):
        """Each ``cache_salt`` lives in its own LRU list; buckets do
        not bleed into each other."""
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(1, "alice"), _key(2, "bob"), _key(3, "alice")])
        assert p.get_num_tracked_keys("alice") == 2
        assert p.get_num_tracked_keys("bob") == 1
        assert p.get_num_tracked_keys("charlie") == 0
        assert sorted(p.get_tracked_salts()) == ["alice", "bob"]

    def test_empty_bucket_dropped_after_removal(self):
        """When a bucket empties the policy forgets the salt entirely
        — keeps status / list snapshots compact."""
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(1, "alice")])
        p.on_keys_removed([_key(1, "alice")])
        assert "alice" not in p.get_tracked_salts()

    def test_touch_updates_order(self):
        """Touching a key moves it to MRU. After ``on_keys_created``
        the reversed-insertion rule leaves ``k2`` as LRU (first out).
        Touching ``k2`` must promote it so ``k1`` becomes the LRU
        victim — this specifically exercises ``move_to_end`` on an
        existing entry (a no-op touch on the already-MRU key would
        pass the test trivially and hide bugs in the touch path)."""
        p = IsolatedLRUEvictionPolicy()
        k1 = _key(1, "alice")
        k2 = _key(2, "alice")
        p.on_keys_created([k1, k2])
        # Default LRU order now: [k2 (oldest), k1 (newest)].
        p.on_keys_touched([k2])
        # Expected after touch: [k1 (oldest), k2 (newest)].
        actions = p.get_eviction_actions(expected_ratio=0.5, cache_salt="alice")
        evicted = actions[0].keys
        assert evicted == [k1], (
            f"k1 should be LRU after touching k2; got {[k.chunk_hash for k in evicted]}"
        )


class TestScopedEviction:
    def test_scoped_eviction_stays_in_bucket(self):
        """Eviction scoped to a salt only returns keys from that
        bucket — other salts' data is untouched."""
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(1, "alice"), _key(2, "bob"), _key(3, "alice")])
        actions = p.get_eviction_actions(expected_ratio=1.0, cache_salt="alice")
        assert len(actions) == 1
        evicted_salts = {k.cache_salt for k in actions[0].keys}
        assert evicted_salts == {"alice"}
        assert len(actions[0].keys) == 2

    def test_scoped_eviction_empty_bucket_returns_empty(self):
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(1, "alice")])
        # No bob bucket yet.
        actions = p.get_eviction_actions(expected_ratio=1.0, cache_salt="bob")
        assert actions == []

    def test_missing_cache_salt_raises(self):
        """``IsolatedLRU`` is per-bucket only — calling without an
        explicit ``cache_salt`` raises ``ValueError`` rather than
        silently falling back to a global pool."""
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(1, "alice")])
        with pytest.raises(ValueError, match="cache_salt"):
            p.get_eviction_actions(expected_ratio=1.0, cache_salt=None)


class TestEvictionAmount:
    def test_at_least_one_when_ratio_positive(self):
        """Matches ``LRUEvictionPolicy`` — a positive ratio that rounds
        to zero still yields one eviction so the caller makes forward
        progress."""
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(i, "alice") for i in range(3)])
        actions = p.get_eviction_actions(expected_ratio=0.01, cache_salt="alice")
        assert len(actions[0].keys) == 1

    def test_zero_ratio_evicts_nothing(self):
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(i, "alice") for i in range(3)])
        assert p.get_eviction_actions(expected_ratio=0.0, cache_salt="alice") == []

    def test_eviction_destination_default_is_discard(self):
        p = IsolatedLRUEvictionPolicy()
        p.on_keys_created([_key(1, "alice")])
        actions = p.get_eviction_actions(expected_ratio=1.0, cache_salt="alice")
        assert actions[0].destination == EvictionDestination.DISCARD

    def test_key_eligible_filter_skips_locked_keys(self):
        """The filter is useful for skipping pinned / locked keys. Any
        key for which the filter returns False should be bypassed when
        choosing victims."""
        p = IsolatedLRUEvictionPolicy()
        k1 = _key(1, "alice")
        k2 = _key(2, "alice")
        p.on_keys_created([k1, k2])
        # Only allow k2 to be evicted.
        actions = p.get_eviction_actions(
            expected_ratio=1.0,
            cache_salt="alice",
            key_eligible_filter=lambda k: k == k2,
        )
        assert actions[0].keys == [k2]
