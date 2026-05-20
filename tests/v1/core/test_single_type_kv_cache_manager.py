# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test


def get_sliding_window_manager(sliding_window_spec, block_pool, enable_caching=True):
    # Tests don't exercise admission gating; pass a large cap that is a no-op.
    return SlidingWindowManager(
        sliding_window_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        max_admission_blocks_per_request=10**9,
    )


def get_chunked_local_attention_manager(
    chunked_local_attention_spec, block_pool, enable_caching=True
):
    return ChunkedLocalAttentionManager(
        chunked_local_attention_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        max_admission_blocks_per_request=10**9,
    )


def test_chunked_local_attention_possible_cached_prefix():
    block_size = 2
    chunked_local_attention_spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(
        chunked_local_attention_spec, block_pool
    )

    def run_one_case(block_is_cached, tail_token, expect_length):
        block_hash_list = [
            BlockHash(str(i).encode()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block._cache.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash, is_cached) in enumerate(
            zip(block_hash_list, block_is_cached)
        ):
            if is_cached:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(block_hash, 0),
                    block_pool.blocks[i + 10],
                )

        computed_blocks = manager.find_longest_cache_hit(
            block_hashes=block_hash_list,
            max_length=len(block_hash_list) * block_size + tail_token,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=chunked_local_attention_spec,
            use_eagle=False,
            alignment_tokens=block_size,
        )[0]
        assert len(computed_blocks) == expect_length

        assert all(
            block == block_pool.null_block
            for block in computed_blocks[: (expect_length - 1) // 2]
        )

    run_one_case([True], 0, 1)
    run_one_case([True], 1, 1)
    run_one_case([True, False], 0, 2)
    run_one_case([True, False], 1, 2)
    run_one_case([True, True], 0, 2)
    run_one_case([True, True], 1, 2)
    run_one_case([True, True, False], 0, 2)
    run_one_case([True, True, False], 1, 2)
    run_one_case([True, True, True], 0, 3)
    run_one_case([True, True, True], 1, 3)
    run_one_case([True, True, True, False], 0, 4)
    run_one_case([True, True, True, False], 1, 4)
    run_one_case([random.choice([True, False])] * 8 + [True], 1, 9)
    run_one_case([random.choice([True, False])] * 8 + [False], 1, 8)
    run_one_case([random.choice([True, False])] * 8 + [True, True], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [True, False], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [True, False], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, True], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, True], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, False], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, False], 1, 10)


def test_sliding_window_possible_cached_prefix():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    def run_one_case(block_is_cached, expect_length):
        block_hash_list = [
            BlockHash(str(i).encode()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block._cache.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash, is_cached) in enumerate(
            zip(block_hash_list, block_is_cached)
        ):
            if is_cached:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(block_hash, 0),
                    block_pool.blocks[i + 10],
                )

        computed_blocks = manager.find_longest_cache_hit(
            block_hashes=block_hash_list,
            max_length=len(block_hash_list) * block_size,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=sliding_window_spec,
            use_eagle=False,
            alignment_tokens=block_size,
        )[0]
        assert len(computed_blocks) == expect_length

        assert all(
            block == block_pool.null_block
            for block in computed_blocks[: expect_length - 2]
        )
        for i in range(2):
            if i < expect_length:
                block_index = expect_length - i - 1
                assert computed_blocks[block_index].block_id == block_index + 10

    run_one_case([False] * 10, 0)
    run_one_case([True], 1)
    run_one_case([True, False], 1)
    run_one_case([True, True], 2)
    run_one_case([True, True, False], 2)
    run_one_case([True, True, True], 3)
    run_one_case([True, True, True, False], 3)
    run_one_case(
        [True, True, False, True, False, False, True, True, False, True, True, True], 12
    )
    run_one_case(
        [True, True, False, True, False, False, True, True, False, False, False], 8
    )
    run_one_case(
        [True, True, False, True, False, False, True, True, False, False, False, True],
        8,
    )


def test_chunked_local_attention_remove_skipped_blocks():
    attention_spec = ChunkedLocalAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)

    manager = get_chunked_local_attention_manager(attention_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1010,
    ]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    manager.remove_skipped_blocks("test", 0)
    assert_block_id(block_table, original_block_ids)

    # For 4th token (0-indexed), token 0-3 is out of the local attention window.
    manager.remove_skipped_blocks("test", 4)
    assert_block_id(block_table, [null_block_id] * 2)

    # For 6th token (0-indexed), token 4 - 6 are in local attention window,
    # token 0 - 3 are out, 2 blocks can be removed.
    manager.remove_skipped_blocks("test", 6)
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])
    # For 12th token (0-indexed),
    # token 0-11 are out, 6 block can be removed.
    manager.remove_skipped_blocks("test", 12)
    assert_block_id(block_table, [null_block_id] * 6)


def test_sliding_window_remove_skipped_blocks():
    sliding_window_spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)

    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1010,
    ]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    manager.remove_skipped_blocks("test", 0)
    assert_block_id(block_table, original_block_ids)

    # 4 tokens are computed. Only token 0 is out of the sliding window. As
    # block 1000 also contains token 1 that is in the sliding window, block 1000
    # cannot be removed.
    manager.remove_skipped_blocks("test", 4)
    assert_block_id(block_table, original_block_ids)

    # 5 tokens are computed. Token 0 & 1 are out of the sliding window.
    # Block 1000 can be removed.
    manager.remove_skipped_blocks("test", 5)
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 6 tokens are computed. Token 0-2 are out of the sliding window.
    # Cannot remove new block as the block 1001 is still used by token 3.
    manager.remove_skipped_blocks("test", 6)
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 7 tokens are computed. Token 0-3 are out of the sliding window.
    # Block 1001 can be removed and block 1000 is already removed.
    manager.remove_skipped_blocks("test", 7)
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])

    # 11 tokens are computed. Token 0-7 are out of the sliding window.
    # Block 1002 & 1003 can be removed now. Block 1003 represents a longer
    # sequence, and is expected to be evicted earlier than 1002, so the order
    # of removed blocks should be [1003, 1002].
    manager.remove_skipped_blocks("test", 11)
    assert_block_id(block_table, [null_block_id] * 4 + original_block_ids[4:])


def test_get_num_blocks_to_allocate():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,  # Placeholder value, not related to test result
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)
    cached_blocks_1 = [KVCacheBlock(i + 1) for i in range(10)]
    cached_blocks_2 = [block_pool.null_block for _ in range(5)] + [
        KVCacheBlock(i + 1) for i in range(5)
    ]

    assert (
        manager.get_num_blocks_to_allocate(
            "1", 20 * block_size, cached_blocks_1, 0, 20 * block_size
        )
        == 20
    )
    assert (
        manager.get_num_blocks_to_allocate(
            "2", 20 * block_size, cached_blocks_2, 0, 20 * block_size
        )
        == 15
    )


def test_evictable_cached_blocks_not_double_allocated():
    block_size = 2
    sliding_window_length = 2 * block_size
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window_length,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    request_id = "req"
    evictable_block = block_pool.blocks[1]  # ref_cnt == 0, eviction candidate

    num_blocks_to_allocate = manager.get_num_blocks_to_allocate(
        request_id=request_id,
        num_tokens=2 * block_size,
        new_computed_blocks=[evictable_block],
        total_computed_tokens=block_size,
        num_tokens_main_model=2 * block_size,
    )
    # Free capacity check should count evictable cached blocks, but allocation
    # should only allocate the truly new block.
    assert num_blocks_to_allocate == 2

    manager.allocate_new_computed_blocks(
        request_id,
        [evictable_block],
        num_local_computed_tokens=block_size,
        num_external_computed_tokens=0,
    )
    new_blocks = manager.allocate_new_blocks(
        request_id, num_tokens=4, num_tokens_main_model=4
    )
    assert len(new_blocks) == 1
    assert len(manager.req_to_blocks[request_id]) == 2


def test_chunked_local_attention_get_num_blocks_to_allocate():
    block_size = 2
    attention_spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,  # Placeholder value, not related to test result
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(attention_spec, block_pool)
    cached_blocks_1 = [KVCacheBlock(i + 1) for i in range(10)]
    cached_blocks_2 = [block_pool.null_block for _ in range(5)] + [
        KVCacheBlock(i + 1) for i in range(5)
    ]

    assert (
        manager.get_num_blocks_to_allocate(
            "1", 20 * block_size, cached_blocks_1, 0, 20 * block_size
        )
        == 20
    )
    assert (
        manager.get_num_blocks_to_allocate(
            "2", 20 * block_size, cached_blocks_2, 0, 20 * block_size
        )
        == 15
    )


def test_predictor_matches_allocator_blocks_calculation_with_admission_cap():
    """In forward steps, `get_num_blocks_to_allocate` must return exactly what
    `allocate_new_blocks` will pull; otherwise `block_pool.get_new_blocks`
    raises `ValueError: Cannot get N free blocks from the pool`.
    """
    block_size = 2
    sliding_window = 8  # 4-block live window
    cap = sliding_window // block_size

    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = SlidingWindowManager(
        spec,
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        max_admission_blocks_per_request=cap,
    )

    request_id = "req"
    total_computed = 0
    # Walk through request forward steps. Check num_blocks returned by
    # `get_num_blocks_to_allocate` matches what `allocate_new_blocks` pulls
    for num_tokens in (4, 8, 12, 16):
        predicted = manager.get_num_blocks_to_allocate(
            request_id=request_id,
            num_tokens=num_tokens,
            new_computed_blocks=[],
            total_computed_tokens=total_computed,
            num_tokens_main_model=num_tokens,
        )
        new_blocks = manager.allocate_new_blocks(
            request_id, num_tokens=num_tokens, num_tokens_main_model=num_tokens
        )
        assert predicted == len(new_blocks), (
            f"num_tokens={num_tokens}: predictor returned {predicted} "
            f"but allocator pulled {len(new_blocks)}"
        )


## Mamba prefix caching with PD


def _make_mamba_align_manager(block_size, block_pool):
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    return MambaManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        max_admission_blocks_per_request=10**9,
    )


class _FakeRequest:
    """Minimal stand-in for vllm.v1.request.Request used by cache_blocks."""

    def __init__(self, request_id, block_hashes):
        self.request_id = request_id
        self.block_hashes = block_hashes


def test_mamba_align_num_cached_block_excludes_null_blocks_pd():
    """PD path: all tokens are external, no local prefix hit.

    allocate_new_computed_blocks must set num_cached_block to 0
    (not to the number of null padding blocks), otherwise cache_blocks()
    will early-return and never hash the real state block.
    """
    block_size = 128
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = _make_mamba_align_manager(block_size, block_pool)

    request_id = "req_pd"
    num_external = 400

    manager.allocate_new_computed_blocks(
        request_id,
        new_computed_blocks=[],
        num_local_computed_tokens=0,
        num_external_computed_tokens=num_external,
    )

    # get_num_skipped_tokens(400)=399 → num_skipped_blocks=3
    # req_blocks = [null, null, null, fresh]
    req_blocks = manager.req_to_blocks[request_id]
    assert len(req_blocks) == 4
    assert all(req_blocks[i].is_null for i in range(3))
    assert not req_blocks[3].is_null

    # The fix: num_cached_block must be 0, not 3.
    assert manager.num_cached_block[request_id] == 0


def test_mamba_align_num_cached_block_with_local_prefix_hit():
    """PD path with a partial local prefix cache hit.

    If 2 computed blocks come from a local prefix hit and 1 is skipped,
    num_cached_block should be 2 (all original computed blocks).
    """
    block_size = 128
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = _make_mamba_align_manager(block_size, block_pool)

    # Simulate 2 local prefix-hit blocks.
    cached_block_0 = block_pool.blocks[10]
    cached_block_0.block_hash = make_block_hash_with_group_id(BlockHash(b"h0"), 0)
    cached_block_1 = block_pool.blocks[11]
    cached_block_1.block_hash = make_block_hash_with_group_id(BlockHash(b"h1"), 0)

    request_id = "req_partial"
    # 256 tokens → get_num_skipped_tokens(256)=255 → num_skipped_blocks=1
    # new_computed_blocks[1:] = [cached_block_1] survives.
    manager.allocate_new_computed_blocks(
        request_id,
        new_computed_blocks=[cached_block_0, cached_block_1],
        num_local_computed_tokens=2 * block_size,
        num_external_computed_tokens=0,
    )

    req_blocks = manager.req_to_blocks[request_id]
    # [null, cached_block_1]
    assert len(req_blocks) == 2
    assert req_blocks[0].is_null
    assert req_blocks[1] is cached_block_1

    # num_cached_block = len(original new_computed_blocks) = 2
    assert manager.num_cached_block[request_id] == 2


def test_mamba_align_cache_blocks_registers_null_hashes():
    """After cache_blocks(), null-block-position hashes must be in the
    hash map so that find_longest_cache_hit can discover them.
    """
    block_size = 128
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = _make_mamba_align_manager(block_size, block_pool)

    request_id = "req_nullhash"
    num_external = 400  # 3 full blocks + 16-token partial

    manager.allocate_new_computed_blocks(
        request_id,
        new_computed_blocks=[],
        num_local_computed_tokens=0,
        num_external_computed_tokens=num_external,
    )

    # Build a fake request whose block_hashes cover at least num_full_blocks.
    num_full_blocks = num_external // block_size  # 3
    block_hashes = [BlockHash(f"h{i}".encode()) for i in range(num_full_blocks + 1)]
    fake_req = _FakeRequest(request_id, block_hashes)

    # cache_blocks with num_tokens covering the 3 full blocks.
    manager.cache_blocks(fake_req, num_full_blocks * block_size)

    # All 3 null-block hashes should now be discoverable in the hash map.
    for i in range(num_full_blocks):
        key = make_block_hash_with_group_id(block_hashes[i], 0)
        cached = block_pool.cached_block_hash_to_block.get_one_block(key)
        assert cached is not None, f"hash[{i}] not found after cache_blocks"
        assert cached.is_null, f"hash[{i}] should map to null_block"


def test_mamba_align_cache_blocks_does_not_early_return_pd():
    """End-to-end: cache_blocks must NOT early-return when all blocks are
    null (the original bug).  After cache_blocks, num_cached_block should
    advance so subsequent calls with more tokens can cache the state block.
    """
    block_size = 128
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = _make_mamba_align_manager(block_size, block_pool)

    request_id = "req_noreturn"
    num_external = 400
    num_full_blocks = num_external // block_size  # 3

    manager.allocate_new_computed_blocks(
        request_id,
        new_computed_blocks=[],
        num_local_computed_tokens=0,
        num_external_computed_tokens=num_external,
    )

    block_hashes = [BlockHash(f"h{i}".encode()) for i in range(num_full_blocks + 1)]
    fake_req = _FakeRequest(request_id, block_hashes)

    # First call — should process blocks 0..2 (all null) and advance.
    manager.cache_blocks(fake_req, num_full_blocks * block_size)
    assert manager.num_cached_block[request_id] == num_full_blocks

    # Second call with more tokens — block 3 (the real state block) gets cached.
    manager.cache_blocks(fake_req, (num_full_blocks + 1) * block_size)
    assert manager.num_cached_block[request_id] == num_full_blocks + 1

    # The real block should now be hashed (non-null).
    state_block = manager.req_to_blocks[request_id][num_full_blocks]
    assert not state_block.is_null
    assert state_block.block_hash is not None


def test_mamba_align_find_longest_cache_hit_after_pd_caching():
    """Full round-trip: after caching a PD request's blocks,
    a second request with the same prefix should get a cache hit.
    """
    block_size = 128
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = _make_mamba_align_manager(block_size, block_pool)
    spec = manager.kv_cache_spec

    request_id = "req_roundtrip"
    num_tokens = 3 * block_size  # 384 — exactly block-aligned
    num_full_blocks = num_tokens // block_size  # 3

    # --- First request (PD): allocate + cache ---
    manager.allocate_new_computed_blocks(
        request_id,
        new_computed_blocks=[],
        num_local_computed_tokens=0,
        num_external_computed_tokens=num_tokens,
    )

    block_hashes = [BlockHash(f"h{i}".encode()) for i in range(num_full_blocks + 1)]
    fake_req = _FakeRequest(request_id, block_hashes)

    # Cache the full blocks (null positions registered by MambaManager).
    manager.cache_blocks(fake_req, num_full_blocks * block_size)

    # Simulate decode advancing to fill block 3, then cache it too.
    # allocate_new_blocks would normally do this; we add a fresh block manually.
    req_blocks = manager.req_to_blocks[request_id]
    if len(req_blocks) <= num_full_blocks:
        fresh = block_pool.get_new_blocks(1)[0]
        req_blocks.append(fresh)
    manager.cache_blocks(fake_req, (num_full_blocks + 1) * block_size)

    # --- Second request: find_longest_cache_hit ---
    hit = MambaManager.find_longest_cache_hit(
        block_hashes=block_hashes,
        max_length=num_tokens,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=block_size,
    )

    # Mamba searches right-to-left. With 384 tokens → max_num_blocks=3,
    # it checks block_hashes[2]. The null-block hash was registered,
    # so this should be a hit of length 3 blocks.
    hit_blocks = hit[0]
    assert len(hit_blocks) == num_full_blocks, (
        f"Expected hit length {num_full_blocks}, got {len(hit_blocks)}"
    )


def test_mamba_align_swa_unchanged_by_num_cached_block_fix():
    """Verify the num_cached_block fix does not change SWA behavior.

    For SWA, new_computed_blocks from find_longest_cache_hit includes
    null padding.  len(new_computed_blocks) before skipping should equal
    len(req_blocks) — same as the old len(req_blocks) code.
    """
    block_size = 2
    sliding_window = 4
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    # Simulate SWA find_longest_cache_hit returning [NULL, NULL, B7, B8]
    # for 4-block prefix with 2-block sliding window.
    computed_blocks = [
        block_pool.null_block,
        block_pool.null_block,
        block_pool.blocks[7],
        block_pool.blocks[8],
    ]
    # Mark the real blocks as having hashes (as they would from a cache hit).
    block_pool.blocks[7].block_hash = make_block_hash_with_group_id(BlockHash(b"b7"), 0)
    block_pool.blocks[8].block_hash = make_block_hash_with_group_id(BlockHash(b"b8"), 0)

    num_local_computed_tokens = 4 * block_size  # 8 tokens
    manager.allocate_new_computed_blocks(
        "swa_req",
        new_computed_blocks=computed_blocks,
        num_local_computed_tokens=num_local_computed_tokens,
        num_external_computed_tokens=0,
    )

    # SWA: get_num_skipped_tokens(8) = max(0, 8-4+1) = 5 → num_skipped_blocks=2
    # new_computed_blocks[2:] = [B7, B8]
    # req_blocks = [null, null, B7, B8]
    # num_cached_block should be 4 (= len(original computed_blocks))
    # which is the same as len(req_blocks) — matching old behavior.
    assert manager.num_cached_block["swa_req"] == 4
    assert len(manager.req_to_blocks["swa_req"]) == 4
