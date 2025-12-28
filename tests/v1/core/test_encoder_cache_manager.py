# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

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

    def get_num_encoder_embeds(self, input_id: int) -> int:
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
    num_tokens_to_schedule += req.get_num_encoder_embeds(0)
    compute_budget -= req.get_num_encoder_embeds(0)

    assert not manager.can_allocate(req, 1, compute_budget, num_tokens_to_schedule)


def test_schedule_request_multi_images_respect_compute_limit():
    manager = EncoderCacheManager(cache_size=100)
    req = MockRequest("reqA", ["a", "b"], [5, 6])
    compute_budget = 10
    num_tokens_to_schedule = 0
    assert manager.can_allocate(req, 0, compute_budget, num_tokens_to_schedule)
    num_tokens_to_schedule += req.get_num_encoder_embeds(0)
    compute_budget -= req.get_num_encoder_embeds(0)

    assert not manager.can_allocate(req, 1, compute_budget, num_tokens_to_schedule)


def test_encoder_cache_with_is_embed_mask():
    class MockRequestWithMask(MockRequest):
        def get_num_encoder_embeds(self, input_id: int) -> int:
            return self.mm_features[input_id].mm_position.get_num_embeds

    is_embed = torch.zeros(100, dtype=torch.bool)
    is_embed[torch.tensor([5, 15, 25, 35, 45, 55, 65, 75])] = True

    request = MockRequestWithMask("r1", ["img1"], [100])
    request.mm_features[0] = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="img1",
        mm_position=PlaceholderRange(offset=0, length=100, is_embed=is_embed),
    )

    manager = EncoderCacheManager(cache_size=100)
    manager.allocate(request, 0)

    assert manager.num_free_slots == 92
    assert "img1" in manager.cached

    old_size = 100
    new_size = request.mm_features[0].mm_position.get_num_embeds
    assert new_size == 8
    savings_ratio = old_size / new_size
    assert savings_ratio == 12.5


def test_encoder_cache_mask_based_retrieval():
    class MockRequestWithMask(MockRequest):
        def get_num_encoder_embeds(self, input_id: int) -> int:
            return self.mm_features[input_id].mm_position.get_num_embeds

    is_embed = torch.tensor(
        [False, False, True, True, False, True, True, True, False, False]
    )

    request = MockRequestWithMask("r1", ["img1"], [10])
    request.mm_features[0] = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="img1",
        mm_position=PlaceholderRange(offset=0, length=10, is_embed=is_embed),
    )

    manager = EncoderCacheManager(cache_size=50)
    manager.allocate(request, 0)

    assert request.mm_features[0].mm_position.get_num_embeds == 5

    start_idx = 2
    end_idx = 8
    num_embeds_before = is_embed[:start_idx].sum().item()
    num_embeds_in_range = is_embed[start_idx:end_idx].sum().item()

    assert num_embeds_before == 0
    assert num_embeds_in_range == 5

    start_idx = 0
    end_idx = 5
    num_embeds_before = is_embed[:start_idx].sum().item() if start_idx > 0 else 0
    num_embeds_in_range = is_embed[start_idx:end_idx].sum().item()

    assert num_embeds_before == 0
    assert num_embeds_in_range == 2
