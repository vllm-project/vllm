# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.cache_policy import LRUWithHotCache


def test_lru_with_hot_cache_hit_and_slot_reuse():
    cache = LRUWithHotCache(num_layers=1, capacity_per_layer=2)

    slot0, hit0 = cache.get(0, 100, hot_score=0.1)
    assert not hit0

    slot1, hit1 = cache.get(0, 100, hot_score=0.9)
    assert hit1
    assert slot1 == slot0


def test_lru_with_hot_cache_respects_pinned_blocks():
    cache = LRUWithHotCache(num_layers=1, capacity_per_layer=2)

    cache.get(0, 1, hot_score=0.1)
    cache.get(0, 2, hot_score=0.2)
    cache.pin_block(0, 1)

    # When full, eviction should avoid pinned block 1.
    cache.add_timer()
    cache.get(0, 3, hot_score=0.3)
    _, hit = cache.get(0, 1, hot_score=0.1)
    assert hit


def test_lru_with_hot_cache_all_pinned_raises():
    cache = LRUWithHotCache(num_layers=1, capacity_per_layer=1)
    cache.get(0, 10, hot_score=0.1)
    cache.pin_block(0, 10)

    with pytest.raises(RuntimeError):
        cache.get(0, 11, hot_score=0.2)
