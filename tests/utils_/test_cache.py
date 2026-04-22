# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.utils.cache import CacheInfo, LRUCache


class _TrackingLRUCache(LRUCache):
    def _on_remove(self, key, value):
        if not hasattr(self, "_remove_counter"):
            self._remove_counter = 0
        self._remove_counter += 1


def test_lru_cache():
    cache = _TrackingLRUCache(3)
    assert cache.stat() == CacheInfo(hits=0, total=0)
    assert cache.stat(delta=True) == CacheInfo(hits=0, total=0)

    cache.put(1, 1)
    assert len(cache) == 1

    cache.put(1, 1)
    assert len(cache) == 1

    cache.put(2, 2)
    assert len(cache) == 2

    cache.put(3, 3)
    assert len(cache) == 3
    assert set(cache.cache) == {1, 2, 3}

    cache.put(4, 4)
    assert len(cache) == 3
    assert set(cache.cache) == {2, 3, 4}
    assert cache._remove_counter == 1

    assert cache.get(2) == 2
    assert cache.stat() == CacheInfo(hits=1, total=1)
    assert cache.stat(delta=True) == CacheInfo(hits=1, total=1)

    assert cache[2] == 2
    assert cache.stat() == CacheInfo(hits=2, total=2)
    assert cache.stat(delta=True) == CacheInfo(hits=1, total=1)

    cache.put(5, 5)
    assert set(cache.cache) == {2, 4, 5}
    assert cache._remove_counter == 2

    assert cache.pop(5) == 5
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3

    assert cache.get(-1) is None
    assert cache.stat() == CacheInfo(hits=2, total=3)
    assert cache.stat(delta=True) == CacheInfo(hits=0, total=1)

    cache.pop(10)
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3

    cache.get(10)
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3

    cache.put(6, 6)
    assert len(cache) == 3
    assert set(cache.cache) == {2, 4, 6}
    assert 2 in cache
    assert 4 in cache
    assert 6 in cache

    cache.remove_oldest()
    assert len(cache) == 2
    assert set(cache.cache) == {2, 6}
    assert cache._remove_counter == 4

    cache.clear()
    assert len(cache) == 0
    assert cache._remove_counter == 6
    assert cache.stat() == CacheInfo(hits=0, total=0)
    assert cache.stat(delta=True) == CacheInfo(hits=0, total=0)

    cache._remove_counter = 0

    cache[1] = 1
    assert len(cache) == 1

    cache[1] = 1
    assert len(cache) == 1

    cache[2] = 2
    assert len(cache) == 2

    cache[3] = 3
    assert len(cache) == 3
    assert set(cache.cache) == {1, 2, 3}

    cache[4] = 4
    assert len(cache) == 3
    assert set(cache.cache) == {2, 3, 4}
    assert cache._remove_counter == 1
    assert cache[2] == 2

    cache[5] = 5
    assert set(cache.cache) == {2, 4, 5}
    assert cache._remove_counter == 2

    del cache[5]
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3

    cache.pop(10)
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3

    cache[6] = 6
    assert len(cache) == 3
    assert set(cache.cache) == {2, 4, 6}
    assert 2 in cache
    assert 4 in cache
    assert 6 in cache


# ---- Tests for pin / unpin ----


def test_lru_cache_pin_prevents_eviction():
    cache = _TrackingLRUCache(3)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(3, "c")

    cache.pin(2)

    # Adding a 4th item evicts key 1 (oldest non-pinned), not key 2
    cache.put(4, "d")
    assert 1 not in cache
    assert 2 in cache
    assert set(cache.cache) == {2, 3, 4}


def test_lru_cache_pin_not_found():
    cache = _TrackingLRUCache(3)
    with pytest.raises(ValueError, match="Cannot pin key"):
        cache.pin(99)


def test_lru_cache_all_pinned_remove_oldest_raises():
    cache = _TrackingLRUCache(3)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(3, "c")
    cache.pin(1)
    cache.pin(2)
    cache.pin(3)

    with pytest.raises(RuntimeError, match="All items are pinned"):
        cache.remove_oldest()


def test_lru_cache_remove_oldest_skips_pinned():
    cache = _TrackingLRUCache(4)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(3, "c")
    cache.put(4, "d")

    cache.pin(1)  # oldest but pinned
    cache.remove_oldest()
    assert 1 in cache
    assert 2 not in cache


def test_lru_cache_popitem_remove_pinned():
    cache = _TrackingLRUCache(3)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(3, "c")
    cache.pin(1)

    key, value = cache.popitem(remove_pinned=True)
    assert key == 1  # oldest, evicted despite being pinned
    assert value == "a"


# ---- Tests for touch() ----


def test_lru_cache_touch_reorders():
    cache = _TrackingLRUCache(3)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(3, "c")

    # Touch key 1 so it becomes most recently used
    cache.touch(1)

    # Adding 4th item evicts key 2 (now oldest)
    cache.put(4, "d")
    assert 1 in cache
    assert 2 not in cache
    assert set(cache.cache) == {1, 3, 4}


def test_lru_cache_touch_missing_key():
    cache = _TrackingLRUCache(3)
    cache.touch(99)
    assert len(cache) == 0
    # Ensure the missing key was not added to the internal order tracking
    assert 99 not in cache.order


# ---- Tests for capacity / usage properties ----


def test_lru_cache_capacity_and_usage():
    cache = _TrackingLRUCache(10)
    assert cache.capacity == 10
    assert cache.usage == 0.0

    cache.put(1, "a")
    assert cache.usage == pytest.approx(0.1)

    cache.put(2, "b")
    assert cache.usage == pytest.approx(0.2)

    cache.clear()
    assert cache.usage == 0.0


def test_lru_cache_capacity_zero():
    cache = _TrackingLRUCache(0)
    assert cache.capacity == 0
    assert cache.usage == 0.0


# ---- Tests for CacheInfo.hit_ratio ----


def test_cache_info_hit_ratio():
    info = CacheInfo(hits=3, total=10)
    assert info.hit_ratio == pytest.approx(0.3)

    empty = CacheInfo(hits=0, total=0)
    assert empty.hit_ratio == 0


def test_cache_info_subtraction():
    a = CacheInfo(hits=5, total=10)
    b = CacheInfo(hits=2, total=4)
    diff = a - b
    assert diff == CacheInfo(hits=3, total=6)


# ---- Tests for weighted cache (getsizeof) ----


def test_lru_cache_weighted_eviction():
    cache = _TrackingLRUCache(10, getsizeof=lambda v: v)

    cache.put("a", 3)  # size 3
    cache.put("b", 4)  # size 4, total 7
    assert cache.usage == pytest.approx(0.7)

    cache.put("c", 5)  # size 5, total 12 > 10, triggers eviction
    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache


# ---- Tests for stat(delta=True) reset behavior ----


def test_lru_cache_stat_delta_resets():
    cache = _TrackingLRUCache(10)

    cache.put(1, "a")
    cache.get(1)  # hit
    cache.get(2)  # miss

    delta1 = cache.stat(delta=True)
    assert delta1 == CacheInfo(hits=1, total=2)

    cache.get(1)  # another hit
    cache.get(3)  # another miss

    delta2 = cache.stat(delta=True)
    assert delta2 == CacheInfo(hits=1, total=2)

    # Non-delta stat should reflect cumulative totals
    cumulative = cache.stat()
    assert cumulative == CacheInfo(hits=2, total=4)
