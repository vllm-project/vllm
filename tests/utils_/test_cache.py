# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

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


def _desync_order_from_data(cache: LRUCache, key) -> None:
    """Remove *key* from the data/size bookkeeping but leave it in the
    order dict — the torn state unsynchronized concurrent mutation can
    produce (see vllm-project/vllm#47958)."""
    del cache._Cache__data[key]  # type: ignore[attr-defined]
    cache._Cache__currsize -= (  # type: ignore[attr-defined]
        cache._Cache__size.pop(key)  # type: ignore[attr-defined]
    )


def test_popitem_skips_stale_order_entry():
    """A stale order entry must be dropped, not silently no-op popped.

    pop() returns the default when the key is missing from the data
    dict, so a stale order entry made popitem() remove nothing; the
    eviction loop in cachetools' Cache.__setitem__ then spun forever
    (vllm-project/vllm#47958).
    """
    cache = LRUCache(10, getsizeof=len)
    cache.put("a", "xxxx")
    cache.put("b", "yyyy")
    _desync_order_from_data(cache, "a")

    key, value = cache.popitem()

    assert (key, value) == ("b", "yyyy")
    assert "a" not in cache.order
    assert len(cache) == 0


def test_eviction_terminates_with_stale_order_entry():
    """Inserting past capacity must terminate despite a stale order
    entry (regression test for the vllm-project/vllm#47958 livelock)."""
    cache = LRUCache(10, getsizeof=len)
    cache.put("a", "xxxx")
    cache.put("b", "yyyy")
    _desync_order_from_data(cache, "a")

    # currsize is 4 ("b"); size 8 forces the eviction loop, which must
    # heal the stale "a" entry and evict "b" instead of spinning.
    # Run in a worker thread so a regression fails fast instead of
    # hanging the test session.
    done = threading.Event()

    def insert() -> None:
        cache.put("c", "z" * 8)
        done.set()

    threading.Thread(target=insert, daemon=True).start()
    assert done.wait(timeout=10), (
        "eviction loop livelocked on a stale order entry (vllm-project/vllm#47958)"
    )

    assert set(cache.cache) == {"c"}
    assert set(cache.order) == {"c"}
    assert cache.currsize == 8
