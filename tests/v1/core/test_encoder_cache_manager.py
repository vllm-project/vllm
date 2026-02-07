# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
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

    def get_num_encoder_embeds(self, input_id: int) -> int:
        return self._token_counts[input_id]


# ------------------ Unit Tests ------------------ #
def test_basic_allocate_and_reuse():
    cache = EncoderCacheManager(cache_size=10)

    req = MockRequest("r1", ["imgA"], [4])
    req_id = req.request_id
    mm_feats = req.mm_features

    assert not cache.check_and_update_cache(req_id, mm_feats[0])
    assert cache.can_allocate(mm_feats[0], int(1e9), 0)

    cache.allocate(req_id, mm_feats[0])

    assert cache.check_and_update_cache(req_id, mm_feats[0])
    assert "r1" in cache.cached["imgA"]
    assert cache.num_free_slots == 6

    # Free twice to bring refcount to 0.
    cache.free_encoder_input(req_id, mm_feats[0])
    cache.free_encoder_input(req_id, mm_feats[0])

    assert not cache.cached["imgA"]
    assert "imgA" in cache.freeable
    assert cache.num_freeable_slots == 10
    assert cache.num_free_slots == 6


def test_freeing_decreases_refcount_and_moves_to_freeable():
    manager = EncoderCacheManager(cache_size=10)

    req = MockRequest("req2", ["img3"], [5])
    req_id = req.request_id
    mm_feats = req.mm_features

    assert manager.can_allocate(mm_feats[0], int(1e9), 0)
    manager.allocate(req_id, mm_feats[0])

    assert len(manager.cached["img3"]) == 1

    manager.free_encoder_input(req_id, mm_feats[0])

    assert not manager.cached["img3"]
    assert "img3" in manager.freeable
    assert manager.num_freeable_slots == 10


def test_free_request_frees_all_inputs():
    manager = EncoderCacheManager(cache_size=10)

    req = MockRequest("req3", ["a", "b"], [2, 3])
    req_id = req.request_id
    mm_feats = req.mm_features

    assert manager.can_allocate(mm_feats[0], int(1e9), 0)
    manager.allocate(req_id, mm_feats[0])

    assert manager.can_allocate(mm_feats[1], int(1e9), 0)
    manager.allocate(req_id, mm_feats[1])

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
    req_id1 = req1.request_id
    mm_feats1 = req1.mm_features

    req2 = MockRequest("req2", ["y"], [5])
    req_id2 = req2.request_id
    mm_feats2 = req2.mm_features

    assert manager.can_allocate(mm_feats1[0], int(1e9), 0)
    manager.allocate(req_id1, mm_feats1[0])
    manager.free_encoder_input(req_id1, mm_feats1[0])

    assert manager.can_allocate(mm_feats2[0], int(1e9), 0)
    manager.allocate(req_id2, mm_feats2[0])

    # 'x' should have been evicted.
    assert "x" not in manager.cached
    assert "x" in manager.get_freed_mm_hashes()


def test_get_cached_features():
    manager = EncoderCacheManager(cache_size=10)

    req = MockRequest("reqX", ["m", "n", "o"], [2, 4, 3])
    req_id = req.request_id
    mm_feats = req.mm_features

    assert manager.can_allocate(mm_feats[0], int(1e9), 0)
    manager.allocate(req_id, mm_feats[0])

    assert manager.can_allocate(mm_feats[2], int(1e9), 0)
    manager.allocate(req_id, mm_feats[2])

    cached_ids = manager.get_cached_features(req)
    assert cached_ids == [mm_feats[0], mm_feats[2]]


def test_has_cache_restores_from_freeable():
    manager = EncoderCacheManager(cache_size=10)

    req = MockRequest("reqY", ["imgZ"], [4])
    req_id = req.request_id
    mm_feats = req.mm_features

    assert manager.can_allocate(mm_feats[0], int(1e9), 0)
    manager.allocate(req_id, mm_feats[0])
    manager.free_encoder_input(req_id, mm_feats[0])

    # Should restore from freeable.
    assert manager.check_and_update_cache(req_id, mm_feats[0])
    assert len(manager.cached["imgZ"]) == 1
    assert "imgZ" not in manager.freeable
    assert manager.num_freeable_slots == 6


def test_get_freed_mm_hashes_clears_freed_list():
    manager = EncoderCacheManager(cache_size=10)

    req1 = MockRequest("reqA", ["a"], [5])
    req_id1 = req1.request_id
    mm_feats1 = req1.mm_features

    req2 = MockRequest("reqB", ["b"], [6])
    req_id2 = req2.request_id
    mm_feats2 = req2.mm_features

    assert manager.can_allocate(mm_feats1[0], int(1e9), 0)
    manager.allocate(req_id1, mm_feats1[0])
    manager.free_encoder_input(req_id1, mm_feats1[0])

    # Should trigger eviction of 'a'.
    assert manager.can_allocate(mm_feats2[0], int(1e9), 0)
    manager.allocate(req_id2, mm_feats2[0])

    freed = manager.get_freed_mm_hashes()
    assert "a" in freed
    assert manager.get_freed_mm_hashes() == []


def test_schedule_request_multi_images_respect_space_limit():
    manager = EncoderCacheManager(cache_size=10)

    req = MockRequest("reqA", ["a", "b"], [5, 6])
    mm_feats = req.mm_features

    compute_budget = 100
    num_tokens_to_schedule = 0

    assert manager.can_allocate(mm_feats[0], compute_budget, num_tokens_to_schedule)
    num_tokens_to_schedule += req.get_num_encoder_embeds(0)
    compute_budget -= req.get_num_encoder_embeds(0)

    assert not manager.can_allocate(mm_feats[1], compute_budget, num_tokens_to_schedule)


def test_schedule_request_multi_images_respect_compute_limit():
    manager = EncoderCacheManager(cache_size=100)

    req = MockRequest("reqA", ["a", "b"], [5, 6])
    mm_feats = req.mm_features

    compute_budget = 10
    num_tokens_to_schedule = 0

    assert manager.can_allocate(mm_feats[0], compute_budget, num_tokens_to_schedule)
    num_tokens_to_schedule += req.get_num_encoder_embeds(0)
    compute_budget -= req.get_num_encoder_embeds(0)

    assert not manager.can_allocate(mm_feats[1], compute_budget, num_tokens_to_schedule)


def test_encoder_cache_with_is_embed_mask():
    is_embed = torch.zeros(100, dtype=torch.bool)
    is_embed[torch.tensor([5, 15, 25, 35, 45, 55, 65, 75])] = True

    mm_feature = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="img1",
        mm_position=PlaceholderRange(offset=0, length=100, is_embed=is_embed),
    )

    manager = EncoderCacheManager(cache_size=100)
    manager.allocate("r1", mm_feature)

    assert manager.num_free_slots == 92
    assert "img1" in manager.cached

    old_size = 100
    new_size = mm_feature.mm_position.get_num_embeds()
    assert new_size == 8
    savings_ratio = old_size / new_size
    assert savings_ratio == 12.5


def test_encoder_cache_mask_based_retrieval():
    is_embed = torch.tensor(
        [False, False, True, True, False, True, True, True, False, False]
    )

    mm_feature = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="img1",
        mm_position=PlaceholderRange(offset=0, length=10, is_embed=is_embed),
    )

    manager = EncoderCacheManager(cache_size=50)
    manager.allocate("r1", mm_feature)

    assert mm_feature.mm_position.get_num_embeds() == 5

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


def test_reset_clears_all_state():
    """Test that reset() clears all cached entries and restores capacity."""
    manager = EncoderCacheManager(cache_size=20)

    req1 = MockRequest("req1", ["img1", "img2"], [5, 3])
    req_id1 = req1.request_id
    mm_feats1 = req1.mm_features

    req2 = MockRequest("req2", ["img3"], [4])
    req_id2 = req2.request_id
    mm_feats2 = req2.mm_features

    manager.allocate(req_id1, mm_feats1[0])
    manager.allocate(req_id1, mm_feats1[1])
    manager.allocate(req_id2, mm_feats2[0])
    manager.free_encoder_input(req_id1, mm_feats1[0])

    req3 = MockRequest("req3", ["img4"], [10])
    req_id3 = req3.request_id
    mm_feats3 = req3.mm_features

    manager.free_encoder_input(req_id1, mm_feats1[1])
    manager.free_encoder_input(req_id2, mm_feats2[0])
    manager.can_allocate(mm_feats3[0], int(1e9), 0)
    manager.allocate(req_id3, mm_feats3[0])

    assert len(manager.cached) > 0
    assert manager.num_free_slots < 20

    manager.reset()

    assert len(manager.cached) == 0
    assert len(manager.freeable) == 0
    assert len(manager.freed) == 0
    assert manager.num_free_slots == 20
    assert manager.num_freeable_slots == 20


def test_reset_allows_fresh_allocations():
    manager = EncoderCacheManager(cache_size=10)

    req1 = MockRequest("req1", ["img1"], [10])
    req_id1 = req1.request_id
    mm_feats1 = req1.mm_features

    manager.allocate(req_id1, mm_feats1[0])
    assert manager.num_free_slots == 0

    manager.reset()

    req2 = MockRequest("req2", ["img2"], [8])
    req_id2 = req2.request_id
    mm_feats2 = req2.mm_features

    assert manager.can_allocate(mm_feats2[0], int(1e9), 0)
    manager.allocate(req_id2, mm_feats2[0])

    assert manager.num_free_slots == 2
    assert "img2" in manager.cached
    assert "img1" not in manager.cached


def test_encoder_decoder_cache_manager_reset():
    manager = EncoderDecoderCacheManager(cache_size=20)

    req1 = MockRequest("req1", ["img1"], [5])
    req_id1 = req1.request_id
    mm_feats1 = req1.mm_features

    req2 = MockRequest("req2", ["img2"], [3])
    req_id2 = req2.request_id
    mm_feats2 = req2.mm_features

    manager.allocate(req_id1, mm_feats1[0])
    manager.allocate(req_id2, mm_feats2[0])
    manager.free(req1)
    manager.get_freed_mm_hashes()

    assert manager.num_free_slots < 20

    manager.reset()

    assert len(manager.allocated) == 0
    assert len(manager.to_free) == 0
    assert manager.num_free_slots == 20


def test_encoder_decoder_cache_manager_reset_allows_fresh_allocations():
    manager = EncoderDecoderCacheManager(cache_size=10)

    req1 = MockRequest("req1", ["img1"], [10])
    req_id1 = req1.request_id
    mm_feats1 = req1.mm_features

    manager.allocate(req_id1, mm_feats1[0])
    assert manager.num_free_slots == 0

    manager.reset()

    req2 = MockRequest("req2", ["img2"], [8])
    req_id2 = req2.request_id
    mm_feats2 = req2.mm_features

    assert manager.can_allocate(mm_feats2[0], int(1e9), 0)
    manager.allocate(req_id2, mm_feats2[0])

    assert manager.num_free_slots == 2
    assert "img2" in manager.allocated
