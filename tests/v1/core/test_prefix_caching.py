# SPDX-License-Identifier: Apache-2.0
"""Compare the with and without prefix caching."""

from typing import Optional

import pytest
import torch

from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager, Request
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlock,
                                         hash_block_tokens)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)


def make_request(request_id,
                 prompt_token_ids,
                 mm_positions=None,
                 mm_hashes=None,
                 prompt_logprobs: Optional[int] = None):
    if mm_positions is None:
        multi_modal_inputs = None
    else:
        multi_modal_inputs = [MultiModalKwargs({})] * len(mm_positions)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        multi_modal_inputs=multi_modal_inputs,
        multi_modal_hashes=mm_hashes,
        multi_modal_placeholders=mm_positions,
        sampling_params=SamplingParams(max_tokens=17,
                                       prompt_logprobs=prompt_logprobs),
        eos_token_id=100,
        arrival_time=0,
        lora_request=None,
    )


def make_kv_cache_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        tensors={},
        kv_cache_groups=[
            KVCacheGroupSpec(['layer'],
                             FullAttentionSpec(block_size, 1, 1, torch.float32,
                                               False))
        ],
    )


@pytest.mark.parametrize("hash_algo", ["sha256", "hash"])
def test_prefill(hash_algo):
    manager = KVCacheManager(
        make_kv_cache_config(16, 11),
        max_model_len=8192,
        enable_caching=True,
        caching_hash_algo=hash_algo,
    )

    # choose the hash function according to the parameter
    hash_fn = sha256 if hash_algo == "sha256" else hash

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request("0", all_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert len(manager.req_to_block_hashes[req0.request_id]) == 3
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, 55, computed_blocks)
    assert [b.block_id for b in blocks] == [1, 2, 3, 4]

    # Check full block metadata
    parent_block_hash = None
    for block_id in (1, 2, 3):
        block_tokens = tuple(all_token_ids[(block_id - 1) * 16:block_id * 16])
        block_hash = hash_block_tokens(hash_fn, parent_block_hash,
                                       block_tokens)
        assert manager.block_pool.blocks[block_id].block_hash == block_hash
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
        parent_block_hash = block_hash.hash_value

    # Check partial block metadata
    for block_id in (4, ):
        assert manager.block_pool.blocks[block_id].block_hash is None
        assert manager.block_pool.blocks[block_id].ref_cnt == 1

    # Cache hit in the common prefix when the original block is still in use.
    # Incomplete 1 block (5 tokens)
    unique_token_ids = [3] * 5
    req1 = make_request("1", common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(manager.req_to_block_hashes[req1.request_id]) == 3
    assert [b.block_id for b in computed_blocks] == [1, 2, 3]
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(req1, num_new_tokens, computed_blocks)
    assert [b.block_id for b in blocks] == [5]
    for block in computed_blocks:
        assert block.ref_cnt == 2

    # At this point, we should have 5 free blocks left.
    assert manager.block_pool.free_block_queue.num_free_blocks == 5

    manager.free(req0)
    manager.free(req1)

    # All blocks should be available.
    assert manager.block_pool.free_block_queue.num_free_blocks == 10
    # The order should be
    # [unallocated (6, 7, 8, 9, 10)]
    # [unique_req0 (4)]
    # [unique_req1 (5)]
    # [common (3, 2, 1)]
    assert [
        b.block_id
        for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ] == [6, 7, 8, 9, 10, 4, 5, 3, 2, 1]

    # Cache hit in the common prefix when the original block is already free.
    # Incomplete 1 block (6 tokens)
    unique_token_ids = [3] * 6
    req2 = make_request("2", common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(manager.req_to_block_hashes[req2.request_id]) == 3
    assert [b.block_id for b in computed_blocks] == [1, 2, 3]
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(req2, num_new_tokens, computed_blocks)
    assert [b.block_id for b in blocks] == [6]

    # Although we only have 6 free blocks, we have 8 blocks in
    # the free block queue due to lazy removal.
    assert manager.block_pool.free_block_queue.num_free_blocks == 6
    assert all([
        b.ref_cnt == 0
        for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ])
    assert len([
        b for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ]) == 6

    manager.free(req2)

    # Cache miss and eviction.
    req3 = make_request("3", [99] * (16 * 10))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req3, 16 * 10, computed_blocks)
    # This block ID order also checks the eviction order.
    assert [b.block_id for b in blocks] == [7, 8, 9, 10, 4, 5, 6, 3, 2, 1]
    assert manager.block_pool.free_block_queue.num_free_blocks == 0
    assert manager.block_pool.free_block_queue.free_list_head is None
    assert manager.block_pool.free_block_queue.free_list_tail is None


def test_prefill_plp():
    '''Test prefill with APC and some prompt logprobs (plp) requests.

    1. Schedule plp request and validate APC block allocation
    2. Schedule non-plp request and validate blocks
    3. Schedule plp request; no hit should occur; validate blocks
    '''
    manager = KVCacheManager(
        make_kv_cache_config(16, 11),
        max_model_len=8192,
        enable_caching=True,
    )
    # the default hash function is hash
    hash_fn = hash

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Request #0 is a prompt logprobs request
    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request("0", all_token_ids, prompt_logprobs=5)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert len(manager.req_to_block_hashes[req0.request_id]) == 3
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, 55, computed_blocks)
    assert [b.block_id for b in blocks] == [1, 2, 3, 4]
    req0_block_hashes = [b.block_hash for b in blocks]

    # Check full block metadata
    parent_block_hash = None
    for block_id in (1, 2, 3):
        block_tokens = tuple(all_token_ids[(block_id - 1) * 16:block_id * 16])
        block_hash = hash_block_tokens(hash_fn, parent_block_hash,
                                       block_tokens)
        assert manager.block_pool.blocks[block_id].block_hash == block_hash
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
        parent_block_hash = block_hash.hash_value

    # Check partial block metadata
    for block_id in (4, ):
        assert manager.block_pool.blocks[block_id].block_hash is None
        assert manager.block_pool.blocks[block_id].ref_cnt == 1

    # Request #1 is a non-prompt-logprobs request:
    # Cache hit in the common prefix when the original block is still in use.
    # Incomplete 1 block (5 tokens)
    unique_token_ids = [3] * 5
    req1 = make_request("1", common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(manager.req_to_block_hashes[req1.request_id]) == 3
    assert [b.block_id for b in computed_blocks] == [1, 2, 3]
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(req1, num_new_tokens, computed_blocks)
    assert [b.block_id for b in blocks] == [5]
    for block in computed_blocks:
        assert block.ref_cnt == 2

    # At this point, we should have 5 free blocks left.
    assert manager.block_pool.free_block_queue.num_free_blocks == 5

    manager.free(req0)
    manager.free(req1)

    # All blocks should be available.
    assert manager.block_pool.free_block_queue.num_free_blocks == 10
    # The order should be
    # [unallocated (6, 7, 8, 9, 10)]
    # [unique_req0 (4)]
    # [unique_req1 (5)]
    # [common (3, 2, 1)]
    assert [
        b.block_id
        for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ] == [6, 7, 8, 9, 10, 4, 5, 3, 2, 1]

    # Request #2 is a prompt-logprobs request:
    # NO cache hit in the common prefix; duplicates request #0 cached blocks
    unique_token_ids = [3] * 6
    req2 = make_request("2",
                        common_token_ids + unique_token_ids,
                        prompt_logprobs=5)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(manager.req_to_block_hashes[req2.request_id]) == 3
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req2, 55, computed_blocks)
    block_ids = [b.block_id for b in blocks]
    # Duplicate cached blocks have different ids but same hashes vs request #0
    assert [b.block_hash for b in blocks] == req0_block_hashes
    assert block_ids != [1, 2, 3, 4]

    # Request #2 block hashes are valid since request #0 hashes are.
    # Check block reference counts.
    for block_id in block_ids:
        assert manager.block_pool.blocks[block_id].ref_cnt == 1

    manager.free(req2)


def test_decode():
    manager = KVCacheManager(
        make_kv_cache_config(16, 11),
        max_model_len=8192,
        enable_caching=True,
    )

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    req0 = make_request("0", common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, 55, computed_blocks)
    assert [b.block_id for b in blocks] == [1, 2, 3, 4]

    # Append slots without allocating a new block.
    req0.num_computed_tokens = 55
    for _ in range(4):
        req0.append_output_token_ids(8)
    new_blocks = manager.allocate_slots(req0, 4)
    assert new_blocks is not None and len(new_blocks) == 0
    assert manager.req_to_blocks[req0.request_id][-1].block_hash is None

    # Append slots with allocating a new block.
    req0.num_computed_tokens = 59
    # 9 tokens to fill the previous block, and 10 tokens to fill
    # the preallocated block.
    for _ in range(9 + 10):
        req0.append_output_token_ids(7)
    new_blocks = manager.allocate_slots(req0, 19)
    assert new_blocks is not None and len(new_blocks) == 1
    assert manager.req_to_blocks[req0.request_id][-2].block_hash is not None
    assert manager.req_to_blocks[req0.request_id][-1].block_hash is None


def test_evict():
    manager = KVCacheManager(
        make_kv_cache_config(16, 11),
        max_model_len=8192,
        enable_caching=True,
    )

    last_token_id = 5 * 16 + 7
    req0 = make_request("0", list(range(last_token_id)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, 5 * 16 + 7, computed_blocks)
    assert len(blocks) == 6  # 5 full + 1 partial

    # 3 blocks.
    req1 = make_request("1", list(range(last_token_id,
                                        last_token_id + 3 * 16)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req1, 3 * 16, computed_blocks)
    assert len(blocks) == 3  # 3 full blocks
    last_token_id += 3 * 16

    # 10 - (6 + 3) == 1
    assert manager.block_pool.free_block_queue.num_free_blocks == 1

    manager.free(req0)
    manager.free(req1)
    assert manager.block_pool.free_block_queue.num_free_blocks == 10
    assert [
        b.block_id
        for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ] == [10, 6, 5, 4, 3, 2, 1, 9, 8, 7]

    # Touch the first 2 blocks.
    req2 = make_request("2", list(range(2 * 16 + 3)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert [b.block_id for b in computed_blocks] == [1, 2]
    assert num_computed_tokens == 2 * 16
    blocks = manager.allocate_slots(req2, 3, computed_blocks)
    assert [b.block_id for b in blocks] == [10]
    assert manager.block_pool.free_block_queue.num_free_blocks == 7


def test_hash_block_correct_reuse():
    """
    This tests when a previously cached block is reused as a new block,
    its hash metadata should be correctly reset.
    """
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(16, 2),
        max_model_len=8192,
        enable_caching=True,
    )

    # Allocate 1 block and cache it.
    num_tokens = block_size * 1
    req = make_request("0", list(range(num_tokens)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req, num_tokens, computed_blocks)
    assert len(blocks) == 1

    # Deallocate the block.
    manager.free(req)

    # Allocate a new block that's not full, make sure hash info on the
    # block is cleared.
    req = make_request("1", list(range(num_tokens - 1)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req, num_tokens - 1, computed_blocks)
    assert len(blocks) == 1

    assert manager.block_pool.blocks[blocks[0].block_id].block_hash is None


def test_computed_blocks_not_evicted():
    """
    Test that the computed blocks are not evicted when getting new blocks
    for a request if there are any other free blocks.
    """
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 3),
        max_model_len=8192,
        enable_caching=True,
    )

    # Allocate a block and cache it.
    num_tokens = block_size * 1
    req0 = make_request("0", list(range(num_tokens)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, num_tokens, computed_blocks)
    assert len(blocks) == 1
    assert blocks[0].block_id == 1

    # Allocate another block.
    req1 = make_request("1", list(range(num_tokens, num_tokens * 2)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req1, num_tokens, computed_blocks)
    assert len(blocks) == 1
    assert blocks[0].block_id == 2

    # Free the blocks.
    manager.free(req0)
    manager.free(req1)

    # Now if we have a cache hit on the first block, we should evict the second
    # cached block rather than the first one.
    req2 = make_request("2", list(range(num_tokens * 2)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(computed_blocks) == 1
    assert computed_blocks[0].block_id == 1
    assert num_computed_tokens == block_size

    blocks = manager.allocate_slots(req2, num_tokens * 2 - num_tokens,
                                    computed_blocks)
    assert len(blocks) == 1
    assert blocks[0].block_id == 2


def test_basic_prefix_caching_disabled():
    """
    This tests that the prefix caching is disabled.
    """
    block_size = 4
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 5),
        max_model_len=8192,
        enable_caching=False,
    )

    req1 = make_request("1", list(range(10)))  # 2 blocks and some more

    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req1, 10, computed_blocks)
    assert len(blocks) == 3

    # Free the blocks.
    manager.free(req1)

    # No caching.
    req2 = make_request("2", list(range(16)))  # shared prefix
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req2, 16, computed_blocks)
    assert len(blocks) == 4

    # New requests should not have any blocks.
    req3 = make_request("3", list(range(4)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)
    assert not computed_blocks
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req3, 4, computed_blocks)
    assert not blocks


@pytest.mark.parametrize("hash_fn", [sha256, hash])
def test_cache_blocks(hash_fn):
    """
    This is a unit test that tests the correctness of the _cache_full_blocks
    function of KVCacheManager.
    """
    block_size = 4
    block_pool = BlockPool(
        num_gpu_blocks=5,
        enable_caching=True,
    )
    # Req:
    #  Block 0: [0, 1, 2, 3]
    #  Block 1: [4, 5, 6, 7]
    #  Block 2: [8, 9, 10, 11]
    #  Block 3: [12, 13]
    req = make_request("0", list(range(14)))

    # Test that blocks are cached correctly for 2 full blocks from the start.
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    block_hashes: list[BlockHashType] = []

    block_pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        block_hashes=block_hashes,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        hash_fn=hash_fn,
    )

    assert len(block_pool.cached_block_hash_to_block) == 2
    assert all([block.block_hash is not None for block in blocks])

    # Test that blocks that don't start from the beginning are cached correctly.
    blocks += [KVCacheBlock(block_id=2)]
    block_pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        block_hashes=block_hashes,
        num_cached_blocks=2,
        num_full_blocks=3,
        block_size=block_size,
        hash_fn=hash_fn,
    )
    assert len(block_pool.cached_block_hash_to_block) == 3
    assert blocks[0].block_hash is not None


def test_mm_prefix_caching():
    """
    This tests that the multi-modal prefix caching is correct.
    """
    manager = KVCacheManager(
        make_kv_cache_config(16, 11),
        max_model_len=8192,
        enable_caching=True,
    )

    # Common prompt tokens (T is text tokens and P is image placeholder tokens)
    # [T,...,T, P0,...,P0], [P0,...,P0,T,...,T,P1,...,P1], [P1,...,P1]
    common_token_ids = list(range(10)) + [-1] * 6
    common_token_ids += [-1] * 4 + list(range(10, 20)) + [-1] * 2
    common_token_ids += [-1] * 16

    common_mm_positions = [
        PlaceholderRange(offset=11, length=10),
        PlaceholderRange(offset=30, length=18),
    ]
    common_mm_hashes = ["aaa", "bbb"]

    # A unique image plus some text tokens.
    unique_token_ids = [-1] * 7 + [100] * 4
    all_token_ids = common_token_ids + unique_token_ids
    mm_positions = common_mm_positions + [
        PlaceholderRange(offset=48, length=7)
    ]
    mm_hashes = common_mm_hashes + ["ccc"]
    req0 = make_request("0",
                        all_token_ids,
                        mm_positions=mm_positions,
                        mm_hashes=mm_hashes)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)

    # Completed block should have hashes with extra keys.
    assert not computed_blocks
    assert num_computed_tokens == 0
    block_hashes = manager.req_to_block_hashes[req0.request_id]
    assert len(block_hashes) == 3
    assert block_hashes[0].extra_keys == ("aaa", )
    assert block_hashes[1].extra_keys == ("aaa", "bbb")
    assert block_hashes[2].extra_keys == ("bbb", )

    blocks = manager.allocate_slots(req0, 59, computed_blocks)
    assert [b.block_id for b in blocks] == [1, 2, 3, 4]
    req0.num_computed_tokens = 59

    # Append slots without allocating a new block.
    for _ in range(5):
        req0.append_output_token_ids(8)
    new_blocks = manager.allocate_slots(req0, 5)
    assert new_blocks is not None and len(new_blocks) == 0

    # The just completed block should have hashes with extra keys.
    assert len(block_hashes) == 4
    assert block_hashes[3].extra_keys == ("ccc", )

    # Cache hit.
    unique_token_ids = [-1] * 7 + [200] * 5
    all_token_ids = common_token_ids + unique_token_ids
    mm_positions = common_mm_positions + [
        PlaceholderRange(offset=48, length=7)
    ]
    mm_hashes = common_mm_hashes + ["ccc"]
    req1 = make_request("1",
                        all_token_ids,
                        mm_positions=mm_positions,
                        mm_hashes=mm_hashes)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(computed_blocks) == 3
    assert num_computed_tokens == 3 * 16


def test_prefill_not_enough_free_blocks_with_computed_blocks():
    """
    This is a unit test that tests the correctness of the allocate_slots
    when there is not enough free blocks. Specifically, when a request
    has computed blocks but cannot be allocated due to not enough free blocks,
    the computed blocks should not be touched.
    """
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
    )
    # Complete 3 blocks (48 tokens)
    # | Common-0 | Common-1 | Common-2 | ... |
    common_token_ids = [i for i in range(3) for _ in range(16)]
    req0 = make_request("0", common_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks
    assert num_computed_tokens == 0
    manager.allocate_slots(req0, 48, computed_blocks)
    block_part0 = manager.req_to_blocks[req0.request_id]

    # | Common-0 | Common-1 | Common-2 | Req1-3 | Req1-4 | Req1-5 | ... |
    req1 = make_request("1", common_token_ids * 2)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert computed_blocks == block_part0
    assert num_computed_tokens == 3 * 16
    manager.allocate_slots(req1, 48, computed_blocks)
    block_part1 = manager.req_to_blocks[req1.request_id]
    # | Common-0 | Common-1 | Common-2 | Req1-3 (F) | Req1-4 (F) |
    # | Req1-5(F)| ... |
    manager.free(req1)
    assert {block.ref_cnt for block in block_part1[:3]} == {1}
    assert {block.ref_cnt for block in block_part1[3:]} == {0}

    # | Common-0 | Common-1 | Common-2 | Req1-3 (F) | Req1-4 (F) |
    # | Req1-5(F)| Req2-0   | Req2-1   | ... |
    req2 = make_request("2", [7] * block_size * 2)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert not computed_blocks
    assert num_computed_tokens == 0
    manager.allocate_slots(req2, block_size * 2, computed_blocks)

    # Req3 is Req2 + 3 new blocks, so the first 6 blocks are computed,
    # but it cannot be allocated due to insufficient free blocks (2).
    # In this case, the ref_cnt of the computed blocks should not be changed.
    assert manager.block_pool.free_block_queue.num_free_blocks == 5
    req3 = make_request("3", common_token_ids * 3)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)
    assert computed_blocks == block_part1
    assert num_computed_tokens == 6 * 16
    # Req3 cannot be allocated.
    assert manager.allocate_slots(req3, 48, computed_blocks) is None
    # Block 0-2 are used by Req 1.
    assert {block.ref_cnt for block in block_part1[:3]} == {1}
    # Block 3-5 are free.
    assert {block.ref_cnt for block in block_part1[3:]} == {0}


def test_reset_prefix_cache():
    manager = KVCacheManager(
        make_kv_cache_config(16, 11),
        max_model_len=8192,
        enable_caching=True,
    )

    full_block_token_ids = [i for i in range(3) for _ in range(16)]
    unique_token_ids = [3] * 7
    all_token_ids = full_block_token_ids + unique_token_ids
    req0 = make_request("0", all_token_ids)
    blocks = manager.allocate_slots(req0, 55)
    assert [b.block_id for b in blocks] == [1, 2, 3, 4]

    unique_token_ids = [4] * 7
    all_token_ids = full_block_token_ids + unique_token_ids
    req1 = make_request("1", all_token_ids)
    computed_blocks, _ = manager.get_computed_blocks(req1)
    assert len(manager.req_to_block_hashes[req1.request_id]) == 3
    assert len(computed_blocks) == 3
    blocks = manager.allocate_slots(req1, 7, computed_blocks)
    assert [b.block_id for b in blocks] == [5]

    # Failed to reset prefix cache because some blocks are not freed yet.
    assert not manager.reset_prefix_cache()
    assert manager.block_pool.cached_block_hash_to_block

    # Free the blocks.
    manager.free(req0)
    manager.free(req1)

    assert manager.reset_prefix_cache()
    assert not manager.block_pool.cached_block_hash_to_block
    assert all([blk.block_hash is None for blk in manager.block_pool.blocks])


def test_prefix_cache_stats_disabled():
    """Test that prefix_cache_stats is None when log_stats is False."""
    manager = KVCacheManager(
        make_kv_cache_config(16, 11),
        max_model_len=8192,
        enable_caching=True,
        log_stats=False,  # Disable logging stats
    )
    assert manager.prefix_cache_stats is None

    # Call all functions that check whether log_stats is disabled.
    req = make_request("0", list(range(16)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
    assert not computed_blocks
    assert num_computed_tokens == 0
    manager.allocate_slots(req, 16, computed_blocks)
    manager.reset_prefix_cache()

    # Ensure prefix_cache_stats remains None
    assert manager.prefix_cache_stats is None


def test_eagle_enabled_removes_last_block():
    """Verify Eagle does NOT remove blocks when request 
    length is divisible by block size."""
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, num_blocks=10),
        max_model_len=8192,
        enable_caching=True,
        use_eagle=True,
    )

    # Request with 3 full blocks (48 tokens)
    token_ids = [0] * (3 * block_size)
    req = make_request("divisible_request", token_ids)

    # Prime the cache
    computed_blocks, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(req, len(token_ids), computed_blocks)
    manager.free(req)

    # New request with same tokens + Eagle enabled
    req_eagle = make_request("eagle_divisible", token_ids)
    computed_blocks, num_tokens = manager.get_computed_blocks(req_eagle)

    # Should retain 2 blocks:
    # 1. Original 3 blocks → pop last hash → 2 matched blocks
    # 2. last_block_hash is not None → Eagle pop is not SKIPPED
    assert len(computed_blocks) == 1
    assert num_tokens == 1 * block_size  # 32 tokens


def test_eagle_with_partial_blocks():
    """Test Eagle behavior with requests containing partial blocks."""
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, num_blocks=10),
        max_model_len=8192,
        enable_caching=True,
        use_eagle=True,
    )
    # 2 full blocks + 5 tokens (non-divisible length)
    token_ids = [0] * (2 * block_size + 5)
    req = make_request("partial_block_test", token_ids)

    # Prime the cache
    computed_blocks, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(req, len(token_ids), computed_blocks)
    manager.free(req)

    # New request with Eagle enabled
    req_eagle = make_request("partial_eagle", token_ids)
    computed_blocks, num_tokens = manager.get_computed_blocks(req_eagle)
    # Original match: 2 full blocks → Eagle removes 1 → 1 remaining
    assert len(computed_blocks) == 1
    assert num_tokens == 1 * block_size
