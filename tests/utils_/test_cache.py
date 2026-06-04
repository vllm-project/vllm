# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


def test_touch_nonexistent_key_does_not_create_orphan():
    """Regression test for https://github.com/vllm-project/vllm/issues/43941

    Calling touch() on a key that doesn't exist in the cache must NOT add it
    to __order, because that creates an "orphan" entry that causes popitem()
    to loop infinitely during eviction.
    """
    cache = LRUCache(3)
    cache.put("a", 1)
    cache.put("b", 2)

    # Touch a key that was never inserted
    cache.touch("ghost")

    # "ghost" must NOT appear in the order
    assert "ghost" not in cache.order
    assert "ghost" not in cache
    assert len(cache) == 2

    # Eviction should still work normally
    cache.put("c", 3)
    cache.put("d", 4)  # triggers eviction of "a"
    assert len(cache) == 3
    assert "a" not in cache
    assert set(cache.cache) == {"b", "c", "d"}


def test_popitem_cleans_orphan_keys_no_infinite_loop():
    """Regression test for https://github.com/vllm-project/vllm/issues/43941

    If an orphan key somehow ends up in __order (e.g. from a prior bug),
    popitem() must clean it up rather than looping forever.
    """
    cache = LRUCache(3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    # Manually inject an orphan key into __order (simulating the old bug)
    cache._LRUCache__order["orphan"] = None  # type: ignore
    # Move orphan to front (oldest position)
    cache._LRUCache__order.move_to_end("orphan", last=False)  # type: ignore

    # Now insert a new item, which requires eviction
    # Without the fix, this would loop forever on the orphan
    cache.put("d", 4)

    # The orphan should have been cleaned, and "a" evicted as the real LRU
    assert "orphan" not in cache.order
    assert "a" not in cache
    assert len(cache) == 3
    assert set(cache.cache) == {"b", "c", "d"}


def test_touch_existing_key_updates_lru_order():
    """touch() on an existing key should move it to the end (most recent)."""
    cache = LRUCache(3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    # Touch "a" to make it most recently used
    cache.touch("a")

    # Now "b" should be the oldest - evicted next
    cache.put("d", 4)
    assert "b" not in cache
    assert "a" in cache
    assert set(cache.cache) == {"a", "c", "d"}
