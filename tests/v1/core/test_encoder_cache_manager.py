# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.encoder_cache_manager import EncoderCacheManager


# ------------------ Mock Classes ------------------ #
class MockRequest:

    def __init__(self, request_id, mm_hashes, token_counts):
        self.request_id = request_id
        self.mm_hashes = mm_hashes
        self._token_counts = token_counts

    def get_num_encoder_tokens(self, input_id: int) -> int:
        return self._token_counts[input_id]


# ------------------ Unit Tests ------------------ #
def test_basic_allocate_and_reuse():
    cache = EncoderCacheManager(cache_size=10)
    req = MockRequest("r1", ["imgA"], [4])

    assert not cache.check_and_update_cache(req, 0)
    assert cache.try_allocate(req, 0, int(1e9))

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

    assert manager.try_allocate(req, 0, int(1e9))
    manager.allocate(req, 0)

    assert len(manager.cached["img3"]) == 1

    manager.free_encoder_input(req, 0)

    assert not manager.cached["img3"]
    assert "img3" in manager.freeable
    assert manager.num_freeable_slots == 10


def test_free_request_frees_all_inputs():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("req3", ["a", "b"], [2, 3])

    assert manager.try_allocate(req, 0, int(1e9))
    manager.allocate(req, 0)

    assert manager.try_allocate(req, 1, int(1e9))
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

    assert manager.try_allocate(req1, 0, int(1e9))
    manager.allocate(req1, 0)
    manager.free_encoder_input(req1, 0)

    assert manager.try_allocate(req2, 0, int(1e9))
    manager.allocate(req2, 0)

    # 'x' should have been evicted.
    assert "x" not in manager.cached
    assert "x" in manager.get_freed_mm_hashes()


def test_get_cached_input_ids():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("reqX", ["m", "n", "o"], [2, 4, 3])

    assert manager.try_allocate(req, 0, int(1e9))
    manager.allocate(req, 0)

    assert manager.try_allocate(req, 2, int(1e9))
    manager.allocate(req, 2)

    cached_ids = manager.get_cached_input_ids(req)
    assert cached_ids == {0, 2}


def test_has_cache_restores_from_freeable():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("reqY", ["imgZ"], [4])

    assert manager.try_allocate(req, 0, int(1e9))
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

    assert manager.try_allocate(req1, 0, int(1e9))
    manager.allocate(req1, 0)
    manager.free_encoder_input(req1, 0)

    # Should trigger eviction of 'a'.
    assert manager.try_allocate(req2, 0, int(1e9))
    manager.allocate(req2, 0)

    freed = manager.get_freed_mm_hashes()
    assert "a" in freed
    assert manager.get_freed_mm_hashes() == []
