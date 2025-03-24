from vllm.utils import LRUCache

class TestLRUCache(LRUCache):

    def _on_remove(self, key, value):
        if not hasattr(self, "_remove_counter"):
            self._remove_counter = 0
        self._remove_counter += 1


def unittest_lru_cache():
    cache = TestLRUCache(3)
    print('[LRU Test]')
    cache.put(1, 1)
    assert len(cache) == 1

    cache.put(1, 1)
    assert len(cache) == 1

    cache.put(2, 2)
    assert len(cache) == 2

    cache.put(3, 3)
    assert len(cache) == 3
    
    assert set(cache._Cache__data) == {1, 2, 3}
    cache.put(4, 4)
    assert len(cache) == 3
    assert set(cache._Cache__data) == {2, 3, 4}
    assert cache._remove_counter == 1
    
    assert cache.get(2) == 2
    cache.put(5, 5)
    assert set(cache._Cache__data) == {2, 4, 5}
    assert cache._remove_counter == 2

    assert cache.pop(5) == 5
    assert len(cache) == 2
    assert set(cache._Cache__data) == {2, 4}
    assert cache._remove_counter == 3

    cache.pop(10, None)
    assert len(cache) == 2
    assert set(cache._Cache__data) == {2, 4}
    assert cache._remove_counter == 3

    cache.get(10, None)
    assert len(cache) == 2
    assert set(cache._Cache__data) == {2, 4}
    assert cache._remove_counter == 3

    cache.put(6, 6)
    assert len(cache) == 3
    assert set(cache._Cache__data) == {2, 4, 6}
    assert 2 in cache
    assert 4 in cache
    assert 6 in cache

    cache.remove_oldest()
    assert len(cache) == 2
    assert set(cache._Cache__data) == {2, 6}
    assert cache._remove_counter == 4

    cache.clear()
    assert len(cache) == 0
    assert cache._remove_counter == 6

    cache._remove_counter = 0

    cache[1] = 1
    assert len(cache) == 1

    cache[1] = 1
    assert len(cache) == 1

    cache[2] = 2
    assert len(cache) == 2

    cache[3] = 3
    assert len(cache) == 3
    assert set(cache._Cache__data) == {1, 2, 3}

    cache[4] = 4
    assert len(cache) == 3
    assert set(cache._Cache__data) == {2, 3, 4}
    assert cache._remove_counter == 1
    assert cache[2] == 2

    cache[5] = 5
    assert set(cache._Cache__data) == {2, 4, 5}
    assert cache._remove_counter == 2

    del cache[5]
    assert len(cache) == 2
    assert set(cache._Cache__data) == {2, 4}
    assert cache._remove_counter == 3

    cache.pop(10, None)
    assert len(cache) == 2
    assert set(cache._Cache__data) == {2, 4}
    assert cache._remove_counter == 3

    cache[6] = 6
    assert len(cache) == 3
    assert set(cache._Cache__data) == {2, 4, 6}
    assert 2 in cache
    assert 4 in cache
    assert 6 in cache
    
    cache.clear()
    cache.remove_oldest()
    print('[LRU Test Pass]')
    print('---------------')
    print('[Pin Item Test]')
    cache.clear()
    cache.put(1, 1)
    cache.pin(1)
    cache.put(2, 2)
    cache.put(3, 3)
    assert set(cache._Cache__data) == {1, 2, 3}
    cache.put(4, 4)
    assert set(cache._Cache__data) == {1, 3, 4}
    cache._unpin(1)
    cache.put(5, 5)
    assert set(cache._Cache__data) == {3, 4, 5}
    
    # If all items are pinned, should raise an RuntimeError
    cache.pin(3)
    cache.pin(4)
    cache.pin(5)
    assert set(cache._Cache__data) == {3, 4, 5}
    try:
        cache.put(6, 6)
    except RuntimeError as e:
        pass
        # print(f"Error: {e}")
        
    assert set(cache._Cache__data) == {3, 4, 5}
    print('[Pin Item Pass]')
    print('---------------')    
    print('[Get Sizeof test]')
    def get_item_size(value):
        return value
    
    cache = TestLRUCache(4, getsizeof=get_item_size)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.getsizeof(10) == 10
    cache.put(3, 3)
    assert set(cache._Cache__data) == {3}
    # If the size of the item is greater than the maxsize, should raise an ValueError
    try:
        cache.put(5, 5)
    except ValueError as e:
        pass
        # print(f"Error: {e}")  
    print('[Get Sizeof Pass]')      

if __name__ == '__main__':
    unittest_lru_cache()