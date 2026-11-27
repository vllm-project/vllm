# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for prefetch reservations: pin by hash, publish, release.

A reservation is how KV that is already offloaded gets back into the GPU pool
before the request that needs it exists. The blocks are ordinary allocations
(ref_cnt == 1) that belong to no request and carry no hash, so nothing can hit
them until they are published.
"""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash

HASHES = [BlockHash(bytes([i]) * 32) for i in range(8)]


def make_pool(num_blocks: int = 32, events: bool = False) -> BlockPool:
    return BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=True,
        hash_block_size=16,
        enable_kv_cache_events=events,
    )


def test_pin_reserves_blocks_without_hashes():
    pool = make_pool()
    free_before = pool.get_num_free_blocks()
    sets = pool.pin_blocks_by_hash(HASHES[:4], [0], 16)
    assert sets is not None
    assert len(sets) == 1
    pinned = sets[0]
    assert pinned.block_hashes == HASHES[:4]
    assert len(pinned.blocks) == 4
    assert pool.get_num_free_blocks() == free_before - 4
    for block in pinned.blocks:
        assert block.ref_cnt == 1
        assert block.block_hash is None
        assert block.prev_free_block is None
    # Invisible to prefix caching until published.
    assert pool.get_cached_block(HASHES[0], [0]) is None


def test_publish_makes_blocks_hittable_and_release_keeps_them_readable():
    pool = make_pool()
    sets = pool.pin_blocks_by_hash(HASHES[:3], [0], 16)
    assert sets
    pool.publish_pinned_hashes(sets[0])
    hit = pool.get_cached_block(HASHES[0], [0])
    assert hit is not None
    assert hit[0] is sets[0].blocks[0]
    pool.release_pinned(sets[0])
    for block in sets[0].blocks:
        assert block.ref_cnt == 0
    # Still hittable after release: the data stays readable until it is reused.
    assert pool.get_cached_block(HASHES[0], [0]) is not None


def test_reuse_watcher_fires_when_a_released_block_is_handed_out():
    pool = make_pool(num_blocks=16)  # 15 usable blocks, so a quarter is 3
    sets = pool.pin_blocks_by_hash(HASHES[:2], [0], 16)
    assert sets
    seen = []
    pool.release_pinned(sets[0], on_reuse=seen.append)
    handed = pool.get_new_blocks(15)
    assert sets[0].blocks[0] in handed
    assert seen


def test_covered_hashes_cost_nothing():
    pool = make_pool()
    sets = pool.pin_blocks_by_hash(HASHES[:2], [0], 16)
    assert sets
    pool.publish_pinned_hashes(sets[0])
    assert pool.pin_blocks_by_hash(HASHES[:2], [0], 16) == []


def test_reservation_is_declined_over_the_fraction_limit():
    pool = make_pool(num_blocks=32)
    # 31 usable blocks, so a quarter is 7.
    assert pool.pin_blocks_by_hash(HASHES[:8], [0], 16) is None
    sets = pool.pin_blocks_by_hash(HASHES[:7], [0], 16)
    assert sets is not None
    assert len(sets[0].blocks) == 7


def test_every_group_gets_its_own_set():
    pool = make_pool()
    sets = pool.pin_blocks_by_hash(HASHES[:2], [0, 1], 16)
    assert sets is not None
    assert [s.kv_cache_group_id for s in sets] == [0, 1]
    for pinned_set in sets:
        assert len(pinned_set.blocks) == 2
    # A hash only counts as covered when every group serves it.
    pool.publish_pinned_hashes(sets[0])
    again = pool.pin_blocks_by_hash(HASHES[:2], [0, 1], 16)
    assert again is not None
    assert len(again) == 2


def test_events_on_declines_instead_of_diverging():
    pool = make_pool(events=True)
    assert pool.pin_blocks_by_hash(HASHES[:2], [0], 16) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
