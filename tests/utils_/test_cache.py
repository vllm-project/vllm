# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.utils.cache import CacheInfo, LRUCache


class TestLRUCache(LRUCache):
    def _on_remove(self, key, value):
        if not hasattr(self, "_remove_counter"):
            self._remove_counter = 0
        self._remove_counter += 1


def test_lru_cache():
    cache = TestLRUCache(3)
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


def test_lru_cache_peek_preserves_eviction_order_and_stats():
    cache = LRUCache[str, object | None](2)
    payload = object()
    cache.put("oldest", payload)
    cache.put("newest", None)
    initial_stats = cache.stat()

    assert cache.peek("oldest") is payload
    assert cache.peek("newest", payload) is None
    assert cache.peek("missing") is None
    assert cache.peek("missing", payload) is payload
    assert cache.stat() == initial_stats

    cache.put("replacement", object())
    assert "oldest" not in cache
    assert "newest" in cache


def test_lru_cache_put_if_fits():
    cache = LRUCache[str, int](10, getsizeof=lambda x: x)

    assert cache.put_if_fits("ok", 4) is True
    assert cache["ok"] == 4

    assert cache.put_if_fits("big", 11) is False
    assert "big" not in cache

    with pytest.raises(ValueError, match="value too large"):
        cache.put("big", 11)
