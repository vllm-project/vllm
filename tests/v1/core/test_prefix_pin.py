# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVCacheManager.lookup_and_pin: pin a cached prefix by block hash."""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager, PrefixPinMiss
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheGroup
from vllm.v1.request import Request

from .utils import new_kv_cache_group, new_mla_spec, sliding_window_layer

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
NUM_BLOCKS = 11  # 10 usable: block 0 is the null block.


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _full_spec() -> AttentionSpec:
    return AttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )


def make_manager(
    groups: list[KVCacheGroup] | None = None,
    enable_caching: bool = True,
) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=groups or [new_kv_cache_group(["layer"], _full_spec())],
    )
    return KVCacheManager(
        config,
        max_model_len=8192,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=enable_caching,
    )


def make_request(request_id: str, token_ids: list[int]) -> Request:
    params = SamplingParams(max_tokens=1)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def tokens(num_blocks: int, start: int = 0) -> list[int]:
    return [start + i for i in range(num_blocks) for _ in range(BLOCK_SIZE)]


def compute_and_free(manager: KVCacheManager, request: Request) -> None:
    """Prefill `request` fully (caching its full blocks), then free it."""
    computed, num_computed, _ = manager.get_computed_blocks(request)
    blocks = manager.allocate_slots(
        request, request.num_tokens - num_computed, num_computed, computed
    )
    assert blocks is not None
    manager.free(request)


def test_full_hit_pins_without_last_token_cap():
    manager = make_manager()
    cached = make_request("a", tokens(3))
    compute_and_free(manager, cached)
    # A request with this block-aligned prompt would recompute the last block.
    assert manager.get_computed_blocks(make_request("b", tokens(3)))[1] == 32
    free_before = manager.block_pool.get_num_free_blocks()

    pin = manager.lookup_and_pin(cached.block_hashes)

    assert pin.num_tokens == 3 * BLOCK_SIZE
    assert pin.miss is PrefixPinMiss.NONE
    assert len(pin.get_block_ids()[0]) == 3
    assert all(block.ref_cnt == 1 for block in pin.blocks[0])
    assert manager.block_pool.get_num_free_blocks() == free_before - 3
    assert manager.num_pinned_blocks == 3

    manager.release_pin(pin)
    assert all(block.ref_cnt == 0 for block in pin.blocks[0])
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert manager.num_pinned_blocks == 0
    # Released blocks stay cached for later hits.
    again = manager.lookup_and_pin(cached.block_hashes)
    assert again.num_tokens == 48
    manager.release_pin(again)


def test_partial_hit_stops_at_first_uncached_hash():
    manager = make_manager()
    compute_and_free(manager, make_request("a", tokens(2)))
    longer = make_request("b", tokens(4))

    pin = manager.lookup_and_pin(longer.block_hashes)

    assert pin.num_tokens == 2 * BLOCK_SIZE
    assert pin.miss is PrefixPinMiss.NOT_CACHED
    manager.release_pin(pin)


def test_nothing_cached():
    manager = make_manager()
    pin = manager.lookup_and_pin(make_request("a", tokens(2)).block_hashes)
    assert pin.num_tokens == 0
    assert pin.blocks == ([],)
    assert pin.miss is PrefixPinMiss.NOT_CACHED
    manager.release_pin(pin)
    assert manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1


def test_empty_lookup_is_a_full_hit_of_nothing():
    manager = make_manager()
    pin = manager.lookup_and_pin([])
    assert (pin.num_tokens, pin.miss) == (0, PrefixPinMiss.NONE)


@pytest.mark.parametrize("track", [True, False])
def test_evicted_needs_tracking(track: bool):
    manager = make_manager()
    if track:
        manager.block_pool.enable_evicted_hash_tracking(max_entries=64)
    old = make_request("a", tokens(3))
    compute_and_free(manager, old)
    # Ten fresh blocks reuse every free block, evicting the old prefix.
    compute_and_free(manager, make_request("b", tokens(10, start=100)))

    pin = manager.lookup_and_pin(old.block_hashes)

    assert pin.num_tokens == 0
    expected = PrefixPinMiss.EVICTED if track else PrefixPinMiss.NOT_CACHED
    assert pin.miss is expected
    manager.release_pin(pin)


def test_recached_hash_leaves_eviction_record():
    manager = make_manager()
    pool = manager.block_pool
    pool.enable_evicted_hash_tracking(max_entries=64)
    old = make_request("a", tokens(1))
    compute_and_free(manager, old)
    compute_and_free(manager, make_request("b", tokens(10, start=100)))
    assert pool.was_evicted(old.block_hashes[0], kv_cache_group_id=0)

    # Same tokens as "a": caching them again (into a block "b" freed) clears
    # the record.
    compute_and_free(manager, make_request("c", tokens(1)))

    assert not pool.was_evicted(old.block_hashes[0], kv_cache_group_id=0)
    pin = manager.lookup_and_pin(old.block_hashes)
    assert (pin.num_tokens, pin.miss) == (BLOCK_SIZE, PrefixPinMiss.NONE)
    manager.release_pin(pin)


def test_eviction_record_is_bounded():
    manager = make_manager()
    pool = manager.block_pool
    pool.enable_evicted_hash_tracking(max_entries=2)
    old = make_request("a", tokens(3))
    compute_and_free(manager, old)
    compute_and_free(manager, make_request("b", tokens(10, start=100)))

    # Freed tail first, the last block is reused (evicted) first, so the
    # record keeps the two most recent evictions: blocks 1 and 0.
    h0, h1, h2 = old.block_hashes
    assert pool.was_evicted(h0, kv_cache_group_id=0)
    assert pool.was_evicted(h1, kv_cache_group_id=0)
    assert not pool.was_evicted(h2, kv_cache_group_id=0)


def test_invalidated_blocks_are_not_recorded_as_evicted():
    manager = make_manager()
    pool = manager.block_pool
    pool.enable_evicted_hash_tracking(max_entries=64)
    cached = make_request("a", tokens(1))
    compute_and_free(manager, cached)
    block_id = pool.get_cached_block(cached.block_hashes[0], [0])[0].block_id

    # As for blocks whose KV load failed.
    manager.evict_blocks({block_id})

    assert pool.get_cached_block(cached.block_hashes[0], [0]) is None
    assert not pool.was_evicted(cached.block_hashes[0], kv_cache_group_id=0)


def test_pinned_blocks_are_not_reused():
    manager = make_manager()
    cached = make_request("a", tokens(3))
    compute_and_free(manager, cached)
    pin = manager.lookup_and_pin(cached.block_hashes)
    pinned_ids = set(pin.get_block_ids()[0])

    # Exactly the unpinned blocks are left to allocate.
    other = make_request("b", tokens(7, start=100))
    blocks = manager.allocate_slots(other, other.num_tokens)
    assert blocks is not None
    assert pinned_ids.isdisjoint(blocks.get_block_ids()[0])
    assert manager.allocate_slots(make_request("c", tokens(1, start=200)), 16) is None

    # The pinned prefix is still cached under its hashes.
    for block, block_hash in zip(pin.blocks[0], cached.block_hashes):
        assert manager.block_pool.get_cached_block(block_hash, [0]) == [block]
    manager.release_pin(pin)


def test_pin_and_request_share_blocks():
    manager = make_manager()
    first = make_request("a", tokens(3))
    compute_and_free(manager, first)
    pin = manager.lookup_and_pin(first.block_hashes)

    running = make_request("b", tokens(3) + [7])
    computed, num_computed, _ = manager.get_computed_blocks(running)
    assert num_computed == 48
    manager.allocate_slots(running, 1, num_computed, computed)
    assert all(block.ref_cnt == 2 for block in pin.blocks[0])

    manager.free(running)
    assert all(block.ref_cnt == 1 for block in pin.blocks[0])
    manager.release_pin(pin)
    assert all(block.ref_cnt == 0 for block in pin.blocks[0])


def test_release_is_idempotent():
    manager = make_manager()
    cached = make_request("a", tokens(2))
    compute_and_free(manager, cached)
    pin = manager.lookup_and_pin(cached.block_hashes)

    manager.release_pin(pin)
    manager.release_pin(pin)

    assert pin.released
    assert all(block.ref_cnt == 0 for block in pin.blocks[0])
    assert manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1


def test_reset_prefix_cache_waits_for_pins_and_forgets_evictions():
    manager = make_manager()
    pool = manager.block_pool
    pool.enable_evicted_hash_tracking(max_entries=64)
    old = make_request("a", tokens(1))
    compute_and_free(manager, old)
    compute_and_free(manager, make_request("b", tokens(10, start=100)))
    kept = make_request("c", tokens(1, start=300))
    compute_and_free(manager, kept)
    pin = manager.lookup_and_pin(kept.block_hashes)

    assert not manager.reset_prefix_cache()
    # A failed reset keeps the record.
    assert pool.was_evicted(old.block_hashes[0], kv_cache_group_id=0)
    assert manager.release_all_pins() == [pin]
    assert pin.released
    assert manager.reset_prefix_cache()

    assert not pool.was_evicted(old.block_hashes[0], kv_cache_group_id=0)
    after = manager.lookup_and_pin(kept.block_hashes)
    assert (after.num_tokens, after.miss) == (0, PrefixPinMiss.NOT_CACHED)
    manager.release_pin(after)


def test_unsupported_layouts_pin_nothing():
    sliding = sliding_window_layer(_full_spec(), 2 * BLOCK_SIZE)
    hybrid = make_manager(
        [
            new_kv_cache_group(["full"], _full_spec()),
            new_kv_cache_group(["swa"], sliding),
        ]
    )
    window_only = make_manager([new_kv_cache_group(["swa"], sliding)])
    uncached = make_manager(enable_caching=False)
    hashes = make_request("a", tokens(2)).block_hashes

    for manager in (hybrid, window_only, uncached):
        assert not manager.supports_prefix_pinning()
        free_before = manager.block_pool.get_num_free_blocks()
        pin = manager.lookup_and_pin(hashes)
        assert (pin.num_tokens, pin.miss) == (0, PrefixPinMiss.UNSUPPORTED)
        assert all(group == [] for group in pin.blocks)
        manager.release_pin(pin)
        assert manager.block_pool.get_num_free_blocks() == free_before


def test_mla_is_supported():
    mla = new_mla_spec(block_size=BLOCK_SIZE)
    manager = make_manager([new_kv_cache_group(["mla"], mla)])
    assert manager.supports_prefix_pinning()
    cached = make_request("a", tokens(2))
    compute_and_free(manager, cached)
    pin = manager.lookup_and_pin(cached.block_hashes)
    assert (pin.num_tokens, pin.miss) == (32, PrefixPinMiss.NONE)
    manager.release_pin(pin)


def test_max_blocks_caps_the_pin():
    manager = make_manager()
    cached = make_request("a", tokens(3))
    compute_and_free(manager, cached)

    pin = manager.lookup_and_pin(cached.block_hashes, max_blocks=2)

    assert (pin.num_tokens, pin.miss) == (32, PrefixPinMiss.REFUSED)
    manager.release_pin(pin)


def test_min_free_blocks_refuses_without_touching():
    manager = make_manager()
    cached = make_request("a", tokens(3))
    compute_and_free(manager, cached)
    free_before = manager.block_pool.get_num_free_blocks()  # 10

    refused = manager.lookup_and_pin(cached.block_hashes, min_free_blocks=8)
    assert (refused.num_tokens, refused.miss) == (0, PrefixPinMiss.REFUSED)
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert manager.num_pinned_blocks == 0
    manager.release_pin(refused)

    pin = manager.lookup_and_pin(cached.block_hashes, min_free_blocks=7)
    assert pin.num_tokens == 48
    manager.release_pin(pin)


def test_blocks_already_held_do_not_count_against_free_blocks():
    manager = make_manager()
    cached = make_request("a", tokens(3))
    compute_and_free(manager, cached)
    first = manager.lookup_and_pin(cached.block_hashes)

    # Pinning held blocks takes nothing from the free queue.
    free = manager.block_pool.get_num_free_blocks()
    second = manager.lookup_and_pin(cached.block_hashes, min_free_blocks=free)

    assert second.num_tokens == 48
    assert all(block.ref_cnt == 2 for block in second.blocks[0])
    assert manager.num_pinned_blocks == 6
    manager.release_pin(first)
    manager.release_pin(second)
    assert manager.num_pinned_blocks == 0


def test_pins_holding_finds_pins_by_block_id():
    manager = make_manager()
    a = make_request("a", tokens(2))
    b = make_request("b", tokens(2, start=50))
    compute_and_free(manager, a)
    compute_and_free(manager, b)
    pin_a = manager.lookup_and_pin(a.block_hashes)
    pin_b = manager.lookup_and_pin(b.block_hashes)

    hit = manager.pins_holding({pin_a.get_block_ids()[0][1]})

    assert hit == [pin_a]
    manager.release_pin(pin_a)
    assert manager.pins_holding({pin_a.get_block_ids()[0][1]}) == []
    manager.release_pin(pin_b)
