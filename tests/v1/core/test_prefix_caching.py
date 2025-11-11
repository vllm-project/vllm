# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the with and without prefix caching."""

import copy
from collections.abc import Callable

import pytest
import torch

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, BlockStored
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256, sha256_cbor
from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager, Request
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    get_group_id,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn(request):
    hash_fn: Callable
    if "hash_fn" in request.fixturenames:
        hash_fn = init_none_hash(request.getfixturevalue("hash_fn"))
    else:
        hash_fn = sha256
    init_none_hash(hash_fn)


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    hash_fn: Callable,
    mm_positions: list[PlaceholderRange] | None = None,
    mm_hashes: list[str] | None = None,
    prompt_logprobs: int | None = None,
    cache_salt: str | None = None,
    lora_request: LoRARequest | None = None,
):
    mm_features = []
    if mm_positions is not None:
        for j, position in enumerate(mm_positions):
            identifier = mm_hashes[j] if mm_hashes else f"hash_{j}"
            mm_feature = MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy("dummy_m"),
                mm_position=position,
                identifier=identifier,
                modality="image",
            )
            mm_features.append(mm_feature)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features if mm_features else None,
        sampling_params=SamplingParams(max_tokens=17, prompt_logprobs=prompt_logprobs),
        pooling_params=None,
        eos_token_id=100,
        lora_request=lora_request,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
    )


def make_kv_cache_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size, 1, 1, torch.float32),
            )
        ],
    )


def make_kv_cache_config_hybrid_model(
    block_size: int, num_blocks: int
) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(block_size, 1, 1, torch.float32),
            ),
            KVCacheGroupSpec(
                ["layer2"],
                SlidingWindowSpec(
                    block_size, 1, 1, torch.float32, sliding_window=2 * block_size
                ),
            ),
            KVCacheGroupSpec(
                ["layer3"],
                SlidingWindowSpec(
                    block_size, 1, 1, torch.float32, sliding_window=2 * block_size
                ),
            ),
        ],
    )


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_prefill(hash_fn):
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
    )

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request("0", all_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert len(req0.block_hashes) == 3
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, 55, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == ([1, 2, 3, 4],)

    # Check full block metadata
    parent_block_hash = None
    for block_id in (1, 2, 3):
        block_tokens = tuple(all_token_ids[(block_id - 1) * 16 : block_id * 16])
        block_hash = hash_block_tokens(hash_fn, parent_block_hash, block_tokens)
        blk_hash = manager.block_pool.blocks[block_id].block_hash
        assert blk_hash is not None
        assert get_block_hash(blk_hash) == block_hash
        assert get_group_id(blk_hash) == 0
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
        parent_block_hash = block_hash

    # Check partial block metadata
    for block_id in (4,):
        assert manager.block_pool.blocks[block_id].block_hash is None
        assert manager.block_pool.blocks[block_id].ref_cnt == 1

    # Cache hit in the common prefix when the original block is still in use.
    # Incomplete 1 block (5 tokens)
    unique_token_ids = [3] * 5
    req1 = make_request("1", common_token_ids + unique_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(req1.block_hashes) == 3
    assert computed_blocks.get_block_ids() == ([1, 2, 3],)
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(
        req1, num_new_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == ([5],)
    for block in computed_blocks.blocks[0]:
        assert block.ref_cnt == 2

    # At this point, we should have 5 free blocks left.
    free_block_queue = manager.block_pool.free_block_queue
    assert free_block_queue.num_free_blocks == 5

    manager.free(req0)
    manager.free(req1)

    # All blocks should be available.
    assert free_block_queue.num_free_blocks == 10
    # The order should be
    # [unallocated (6, 7, 8, 9, 10)]
    # [unique_req0 (4)]
    # [unique_req1 (5)]
    # [common (3, 2, 1)]
    assert [
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ] == [6, 7, 8, 9, 10, 4, 5, 3, 2, 1]

    # Cache hit in the common prefix when the original block is already free.
    # Incomplete 1 block (6 tokens)
    unique_token_ids = [3] * 6
    req2 = make_request("2", common_token_ids + unique_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(req2.block_hashes) == 3
    assert computed_blocks.get_block_ids() == ([1, 2, 3],)
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(
        req2, num_new_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == ([6],)

    # Although we only have 6 free blocks, we have 8 blocks in
    # the free block queue due to lazy removal.
    assert free_block_queue.num_free_blocks == 6
    assert all([b.ref_cnt == 0 for b in free_block_queue.get_all_free_blocks()])
    assert len([b for b in free_block_queue.get_all_free_blocks()]) == 6

    manager.free(req2)

    # Cache miss and eviction.
    req3 = make_request("3", [99] * (16 * 10), block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req3, 16 * 10, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    # This block ID order also checks the eviction order.
    assert blocks is not None and blocks.get_block_ids() == (
        [7, 8, 9, 10, 4, 5, 6, 3, 2, 1],
    )

    assert free_block_queue.num_free_blocks == 0
    assert (
        free_block_queue.fake_free_list_head.next_free_block
        is free_block_queue.fake_free_list_tail
    )
    assert (
        free_block_queue.fake_free_list_tail.prev_free_block
        is free_block_queue.fake_free_list_head
    )


def test_prefill_hybrid_model():
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config_hybrid_model(block_size, 21),
        max_model_len=8192,
        enable_caching=True,
    )

    hash_fn = sha256

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(block_size)]

    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request("0", all_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert len(req0.block_hashes) == 3
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, 55, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == (
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    )

    # Check full block metadata
    parent_block_hash = None
    for length, block_ids in zip((1, 2, 3), ((1, 5, 9), (2, 6, 10), (3, 7, 11))):
        block_tokens = tuple(all_token_ids[(length - 1) * 16 : length * 16])
        block_hash = hash_block_tokens(hash_fn, parent_block_hash, block_tokens)
        for group_id, block_id in enumerate(block_ids):
            blk_hash = manager.block_pool.blocks[block_id].block_hash
            assert blk_hash is not None
            assert get_block_hash(blk_hash) == block_hash
            assert get_group_id(blk_hash) == group_id
            assert manager.block_pool.blocks[block_id].ref_cnt == 1
        parent_block_hash = block_hash

    # Check partial block metadata
    for block_id in (4, 8, 12):
        assert manager.block_pool.blocks[block_id].block_hash is None
        assert manager.block_pool.blocks[block_id].ref_cnt == 1

    # Cache hit in the common prefix
    # Incomplete 1 block (5 tokens)
    unique_token_ids = [3] * 5
    req1 = make_request("1", common_token_ids + unique_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(req1.block_hashes) == 3
    assert computed_blocks.get_block_ids() == ([1, 2, 3], [0, 6, 7], [0, 10, 11])
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(
        req1, num_new_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == ([13], [14], [15])
    for block_per_group in computed_blocks.blocks:
        for block in block_per_group:
            if block != manager.block_pool.null_block:
                assert block.ref_cnt == 2

    block_hashes = req1.block_hashes
    manager.free(req0)
    manager.free(req1)

    cached_block_hash_to_block_bak = copy.copy(
        manager.block_pool.cached_block_hash_to_block._cache
    )

    def test_partial_request_hit(
        request_id: str,
        hash_to_evict: list[BlockHashWithGroupId],
        expect_hit_length: int,
    ):
        req = make_request(
            request_id, common_token_ids + unique_token_ids, block_size, sha256
        )
        for hash_with_group_id in hash_to_evict:
            manager.block_pool.cached_block_hash_to_block._cache.pop(hash_with_group_id)
        computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
        assert len(req.block_hashes) == 3
        assert num_computed_tokens == expect_hit_length * block_size
        for block_per_group in computed_blocks.blocks:
            assert len(block_per_group) == num_computed_tokens // block_size
        for hash_with_group_id in hash_to_evict:
            manager.block_pool.cached_block_hash_to_block._cache[hash_with_group_id] = (
                cached_block_hash_to_block_bak[hash_with_group_id]
            )
        manager.free(req)

    # Evict the blocks outside sliding window, does not affect the hit length.
    test_partial_request_hit(
        "2",
        [
            make_block_hash_with_group_id(block_hashes[0], 1),
            make_block_hash_with_group_id(block_hashes[0], 2),
        ],
        3,
    )

    # Evict the first block of full attention, makes total cache miss.
    test_partial_request_hit(
        "3", [make_block_hash_with_group_id(block_hashes[0], 0)], 0
    )

    # Evict the last block of all layers, reduces the hit length to 2.
    test_partial_request_hit(
        "4",
        [
            make_block_hash_with_group_id(block_hashes[2], 0),
            make_block_hash_with_group_id(block_hashes[2], 1),
            make_block_hash_with_group_id(block_hashes[2], 2),
        ],
        2,
    )

    # Evict the last block of full attention, reduces the hit length to 2.
    test_partial_request_hit(
        "5", [make_block_hash_with_group_id(block_hashes[2], 0)], 2
    )

    # Evict the last block of sliding window, reduces the hit length to 2.
    test_partial_request_hit(
        "6", [make_block_hash_with_group_id(block_hashes[2], 1)], 2
    )

    # Evict the last block of sliding window, reduces the hit length to 2.
    test_partial_request_hit(
        "7", [make_block_hash_with_group_id(block_hashes[2], 2)], 2
    )

    # Evict different set of blocks for full attention and sliding window makes
    # total cache miss.
    # The cache hit length of full attention is 1 * block_size.
    # The cache hit length of sliding window is 2 * block_size.
    # Then it is cache miss as the two type of layers
    # have different hit length.
    test_partial_request_hit(
        "8",
        [
            make_block_hash_with_group_id(block_hashes[2], 0),
            make_block_hash_with_group_id(block_hashes[0], 1),
            make_block_hash_with_group_id(block_hashes[0], 2),
        ],
        0,
    )


def test_prefill_plp():
    """Test prefill with APC and some prompt logprobs (plp) requests.

    1. Schedule plp request and validate APC block allocation
    2. Schedule non-plp request and validate blocks
    3. Schedule plp request; no hit should occur; validate blocks
    """
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
    )
    # the default hash function is sha256
    hash_fn = sha256

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Request #0 is a prompt logprobs request
    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request("0", all_token_ids, block_size, hash_fn, prompt_logprobs=5)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert len(req0.block_hashes) == 3
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, 55, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == ([1, 2, 3, 4],)
    req0_block_hashes = [b.block_hash for b in blocks.blocks[0]]

    # Check full block metadata
    parent_block_hash = None
    for block_id in (1, 2, 3):
        block_tokens = tuple(all_token_ids[(block_id - 1) * 16 : block_id * 16])
        block_hash = hash_block_tokens(hash_fn, parent_block_hash, block_tokens)
        blk_hash = manager.block_pool.blocks[block_id].block_hash
        assert blk_hash is not None
        assert get_block_hash(blk_hash) == block_hash
        assert get_group_id(blk_hash) == 0
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
        parent_block_hash = block_hash

    # Check partial block metadata
    for block_id in (4,):
        assert manager.block_pool.blocks[block_id].block_hash is None
        assert manager.block_pool.blocks[block_id].ref_cnt == 1

    # Request #1 is a non-prompt-logprobs request:
    # Cache hit in the common prefix when the original block is still in use.
    # Incomplete 1 block (5 tokens)
    unique_token_ids = [3] * 5
    req1 = make_request("1", common_token_ids + unique_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(req1.block_hashes) == 3
    assert computed_blocks.get_block_ids() == ([1, 2, 3],)
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(
        req1, num_new_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == ([5],)
    for block in computed_blocks.blocks[0]:
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
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ] == [6, 7, 8, 9, 10, 4, 5, 3, 2, 1]

    # Request #2 is a prompt-logprobs request:
    # NO cache hit in the common prefix; duplicates request #0 cached blocks
    unique_token_ids = [3] * 6
    req2 = make_request(
        "2", common_token_ids + unique_token_ids, block_size, hash_fn, prompt_logprobs=5
    )
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(req2.block_hashes) == 3
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req2, 55, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None
    block_ids = blocks.get_block_ids()
    # Duplicate cached blocks have different ids but same hashes vs request #0
    assert [b.block_hash for b in blocks.blocks[0]] == req0_block_hashes
    assert block_ids != ([1, 2, 3, 4],)

    # Request #2 block hashes are valid since request #0 hashes are.
    # Check block reference counts.
    for block_id in block_ids[0]:
        assert manager.block_pool.blocks[block_id].ref_cnt == 1

    manager.free(req2)


def test_decode():
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
    )

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    req0 = make_request("0", common_token_ids + unique_token_ids, block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, 55, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == ([1, 2, 3, 4],)

    # Append slots without allocating a new block.
    req0.num_computed_tokens = 55
    for _ in range(4):
        req0.append_output_token_ids(8)
    new_blocks = manager.allocate_slots(
        req0, 4, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert new_blocks is not None and len(new_blocks.blocks[0]) == 0
    assert (
        manager.coordinator.single_type_managers[0]
        .req_to_blocks[req0.request_id][-1]
        .block_hash
        is None
    )

    # Append slots with allocating a new block.
    req0.num_computed_tokens = 59
    # 9 tokens to fill the previous block, and 10 tokens to fill
    # the preallocated block.
    for _ in range(9 + 10):
        req0.append_output_token_ids(7)
    new_blocks = manager.allocate_slots(
        req0, 19, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert new_blocks is not None and len(new_blocks.blocks[0]) == 1
    assert (
        manager.coordinator.single_type_managers[0]
        .req_to_blocks[req0.request_id][-2]
        .block_hash
        is not None
    )
    assert (
        manager.coordinator.single_type_managers[0]
        .req_to_blocks[req0.request_id][-1]
        .block_hash
        is None
    )


def test_evict():
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
    )

    last_token_id = 5 * 16 + 7
    req0 = make_request("0", list(range(last_token_id)), block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, 5 * 16 + 7, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    # 5 full + 1 partial
    assert blocks is not None and len(blocks.blocks[0]) == 6

    # 3 blocks.
    req1 = make_request(
        "1", list(range(last_token_id, last_token_id + 3 * 16)), block_size, sha256
    )
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req1, 3 * 16, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and len(blocks.blocks[0]) == 3  # 3 full blocks
    last_token_id += 3 * 16

    # 10 - (6 + 3) == 1
    assert manager.block_pool.free_block_queue.num_free_blocks == 1

    manager.free(req0)
    manager.free(req1)
    assert manager.block_pool.free_block_queue.num_free_blocks == 10
    assert [
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ] == [10, 6, 5, 4, 3, 2, 1, 9, 8, 7]

    # Touch the first 2 blocks.
    req2 = make_request("2", list(range(2 * 16 + 3)), block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert computed_blocks.get_block_ids() == ([1, 2],)
    assert num_computed_tokens == 2 * 16
    blocks = manager.allocate_slots(
        req2, 3, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == ([10],)
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
    req = make_request("0", list(range(num_tokens)), block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req, num_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and len(blocks.blocks[0]) == 1

    # Deallocate the block.
    manager.free(req)

    # Allocate a new block that's not full, make sure hash info on the
    # block is cleared.
    req = make_request("1", list(range(num_tokens - 1)), block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req, num_tokens - 1, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and len(blocks.blocks[0]) == 1

    assert manager.block_pool.blocks[blocks.blocks[0][0].block_id].block_hash is None


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
    req0 = make_request("0", list(range(num_tokens)), block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, num_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and len(blocks.blocks[0]) == 1
    assert blocks.blocks[0][0].block_id == 1

    # Allocate another block.
    req1 = make_request(
        "1", list(range(num_tokens, num_tokens * 2)), block_size, sha256
    )
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req1, num_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and len(blocks.blocks[0]) == 1
    assert blocks.blocks[0][0].block_id == 2

    # Free the blocks.
    manager.free(req0)
    manager.free(req1)

    # Now if we have a cache hit on the first block, we should evict the second
    # cached block rather than the first one.
    req2 = make_request("2", list(range(num_tokens * 2)), block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(computed_blocks.blocks[0]) == 1
    assert computed_blocks.blocks[0][0].block_id == 1
    assert num_computed_tokens == block_size

    blocks = manager.allocate_slots(
        req2,
        num_tokens * 2 - num_tokens,
        len(computed_blocks.blocks[0]) * 16,
        computed_blocks,
    )
    assert blocks is not None and len(blocks.blocks[0]) == 1
    assert blocks.blocks[0][0].block_id == 2


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

    req1 = make_request(
        "1", list(range(10)), block_size, sha256
    )  # 2 blocks and some more

    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req1, 10, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and len(blocks.blocks[0]) == 3

    # Free the blocks.
    manager.free(req1)

    # No caching.
    req2 = make_request("2", list(range(16)), block_size, sha256)  # shared prefix
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req2, 16, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and len(blocks.blocks[0]) == 4

    # New requests should not have any blocks.
    req3 = make_request("3", list(range(4)), block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req3, 4, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert not blocks


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
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
    req = make_request("0", list(range(14)), block_size, hash_fn)

    # Test that blocks are cached correctly for 2 full blocks from the start.
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]

    block_pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=0,
    )

    assert len(block_pool.cached_block_hash_to_block) == 2
    assert all([block.block_hash is not None for block in blocks])

    # Test that blocks that don't start from the beginning are cached
    # correctly.
    blocks += [KVCacheBlock(block_id=2)]
    block_pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=2,
        num_full_blocks=3,
        block_size=block_size,
        kv_cache_group_id=0,
    )
    assert len(block_pool.cached_block_hash_to_block) == 3
    assert blocks[0].block_hash is not None


def test_cache_blocks_multi_group():
    """
    This tests that blocks are cached correctly for different kv cache groups.
    """
    block_size = 4
    block_pool = BlockPool(num_gpu_blocks=10, enable_caching=True)

    # Req:
    #  Block 0/4: [0, 1, 2, 3]
    #  Block 1/5: [4, 5, 6, 7]
    #  Block 2/6: [8, 9, 10, 11]
    #  Block 3/7: [12, 13]
    req = make_request("0", list(range(14)), block_size, sha256)

    # Cache the blocks for group 0.
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    block_pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=0,
    )
    assert len(block_pool.cached_block_hash_to_block) == 2
    assert len(req.block_hashes) == 3
    assert all([block.block_hash is not None for block in blocks])

    # Cache the blocks for group 1.
    blocks = [KVCacheBlock(block_id=i) for i in range(3)]
    block_pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=3,
        block_size=block_size,
        kv_cache_group_id=1,
    )
    assert len(block_pool.cached_block_hash_to_block) == 5
    assert len(req.block_hashes) == 3
    assert all([block.block_hash is not None for block in blocks])

    # Block hash 0: hit for group 0 and 1
    # Block hash 1: hit for group 0 and 1
    # Block hash 2: hit for group 1

    assert (
        block_pool.get_cached_block(req.block_hashes[0], kv_cache_group_ids=[0])
        is not None
    )
    assert (
        block_pool.get_cached_block(req.block_hashes[1], kv_cache_group_ids=[0])
        is not None
    )
    assert (
        block_pool.get_cached_block(req.block_hashes[2], kv_cache_group_ids=[0]) is None
    )
    assert (
        block_pool.get_cached_block(req.block_hashes[0], kv_cache_group_ids=[1])
        is not None
    )
    assert (
        block_pool.get_cached_block(req.block_hashes[1], kv_cache_group_ids=[1])
        is not None
    )
    assert (
        block_pool.get_cached_block(req.block_hashes[2], kv_cache_group_ids=[1])
        is not None
    )
    assert (
        block_pool.get_cached_block(req.block_hashes[0], kv_cache_group_ids=[0, 1])
        is not None
    )
    assert (
        block_pool.get_cached_block(req.block_hashes[1], kv_cache_group_ids=[0, 1])
        is not None
    )
    assert (
        block_pool.get_cached_block(req.block_hashes[2], kv_cache_group_ids=[0, 1])
        is None
    )


def test_mm_prefix_caching():
    """
    This tests that the multi-modal prefix caching is correct.
    """

    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
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
    mm_positions = common_mm_positions + [PlaceholderRange(offset=48, length=7)]
    mm_hashes = common_mm_hashes + ["ccc"]
    req0 = make_request(
        "0",
        all_token_ids,
        block_size,
        sha256,
        mm_positions=mm_positions,
        mm_hashes=mm_hashes,
    )
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)

    # Completed block should have hashes
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    block_hashes = req0.block_hashes
    assert len(block_hashes) == 3
    assert block_hashes[0] == sha256(
        (kv_cache_utils.NONE_HASH, tuple(all_token_ids[:block_size]), ("aaa",))
    )
    assert block_hashes[1] == sha256(
        (
            block_hashes[0],
            tuple(all_token_ids[block_size : block_size * 2]),
            ("aaa", "bbb"),
        )
    )
    assert block_hashes[2] == sha256(
        (
            block_hashes[1],
            tuple(all_token_ids[block_size * 2 : block_size * 3]),
            ("bbb",),
        )
    )

    blocks = manager.allocate_slots(
        req0, 59, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None
    assert blocks.get_block_ids() == ([1, 2, 3, 4],)
    req0.num_computed_tokens = 59

    # Append slots without allocating a new block.
    for _ in range(5):
        req0.append_output_token_ids(8)
    new_blocks = manager.allocate_slots(
        req0, 5, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert new_blocks is not None and len(new_blocks.blocks[0]) == 0
    assert len(block_hashes) == 4
    assert block_hashes[3] == sha256(
        (block_hashes[2], tuple(all_token_ids[3 * block_size :] + [8] * 5), ("ccc",))
    )

    # Cache hit.
    unique_token_ids = [-1] * 7 + [200] * 5
    all_token_ids = common_token_ids + unique_token_ids
    mm_positions = common_mm_positions + [PlaceholderRange(offset=48, length=7)]
    mm_hashes = common_mm_hashes + ["ccc"]
    req1 = make_request(
        "1",
        all_token_ids,
        block_size,
        sha256,
        mm_positions=mm_positions,
        mm_hashes=mm_hashes,
    )
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(computed_blocks.blocks[0]) == 3
    assert num_computed_tokens == 3 * 16


def test_cache_key_salting():
    """
    This tests that cache salts are applied during hashing and the cache
    is separated cache as expected.
    """
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
    )

    # 3 complete blocks and an incomplete block with 11 tokens.
    common_token_ids = [i for i in range(3) for _ in range(block_size)]
    token_ids = common_token_ids + [3] * 11
    req0 = make_request("0", token_ids, block_size, sha256, cache_salt="salt1")
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)

    # Completed block should have hashes
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    block_hashes = req0.block_hashes
    assert len(block_hashes) == 3
    assert block_hashes[0] == sha256(
        (kv_cache_utils.NONE_HASH, tuple(token_ids[:block_size]), ("salt1",))
    )
    assert block_hashes[1] == sha256(
        (block_hashes[0], tuple(token_ids[block_size : block_size * 2]), None)
    )
    assert block_hashes[2] == sha256(
        (block_hashes[1], tuple(token_ids[block_size * 2 : block_size * 3]), None)
    )

    blocks = manager.allocate_slots(
        req0, 59, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None
    assert blocks.get_block_ids() == ([1, 2, 3, 4],)
    req0.num_computed_tokens = 59

    # Append slots without allocating a new block.
    for _ in range(5):
        req0.append_output_token_ids(8)
    new_blocks = manager.allocate_slots(
        req0, 5, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert new_blocks is not None and len(new_blocks.blocks[0]) == 0
    assert len(block_hashes) == 4
    assert block_hashes[3] == sha256(
        (block_hashes[2], tuple(token_ids[3 * block_size :] + [8] * 5), None)
    )

    # Test cache hit with a new request that has the same salt.
    token_ids = common_token_ids + [4] * 11
    req1 = make_request("1", token_ids, block_size, sha256, cache_salt="salt1")
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    # Should match only a prefix of 3 blocks.
    assert len(computed_blocks.blocks[0]) == 3
    assert num_computed_tokens == 3 * block_size

    # Test cache miss with same content but different salt.
    token_ids = common_token_ids + [4] * 11
    req2 = make_request("2", token_ids, block_size, sha256, cache_salt="salt2")
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(computed_blocks.blocks[0]) == 0
    assert num_computed_tokens == 0
    block_hashes = req2.block_hashes
    assert len(block_hashes) == 3
    assert block_hashes[0] == sha256(
        (kv_cache_utils.NONE_HASH, tuple(token_ids[:block_size]), ("salt2",))
    )
    assert block_hashes[1] == sha256(
        (block_hashes[0], tuple(token_ids[block_size : block_size * 2]), None)
    )
    assert block_hashes[2] == sha256(
        (block_hashes[1], tuple(token_ids[block_size * 2 : block_size * 3]), None)
    )


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
    req0 = make_request("0", common_token_ids, block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    manager.allocate_slots(
        req0, 48, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    block_part0 = manager.coordinator.single_type_managers[0].req_to_blocks[
        req0.request_id
    ]

    # | Common-0 | Common-1 | Common-2 | Req1-3 | Req1-4 | Req1-5 | ... |
    req1 = make_request("1", common_token_ids * 2, block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert computed_blocks.blocks[0] == block_part0
    assert num_computed_tokens == 3 * 16
    manager.allocate_slots(
        req1, 48, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    block_part1 = manager.coordinator.single_type_managers[0].req_to_blocks[
        req1.request_id
    ]
    # | Common-0 | Common-1 | Common-2 | Req1-3 (F) | Req1-4 (F) |
    # | Req1-5(F)| ... |
    manager.free(req1)
    assert {block.ref_cnt for block in block_part1[:3]} == {1}
    assert {block.ref_cnt for block in block_part1[3:]} == {0}

    # | Common-0 | Common-1 | Common-2 | Req1-3 (F) | Req1-4 (F) |
    # | Req1-5(F)| Req2-0   | Req2-1   | ... |
    req2 = make_request("2", [7] * block_size * 2, block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    manager.allocate_slots(
        req2,
        block_size * 2,
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )

    # Req3 is Req2 + 3 new blocks, so the first 6 blocks are computed,
    # but it cannot be allocated due to insufficient free blocks (2).
    # In this case, the ref_cnt of the computed blocks should not be changed.
    assert manager.block_pool.free_block_queue.num_free_blocks == 5
    req3 = make_request("3", common_token_ids * 3, block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)
    assert computed_blocks.blocks[0] == block_part1
    assert num_computed_tokens == 6 * 16
    # Req3 cannot be allocated.
    assert (
        manager.allocate_slots(
            req3, 48, len(computed_blocks.blocks[0]) * 16, computed_blocks
        )
        is None
    )
    # Block 0-2 are used by Req 1.
    assert {block.ref_cnt for block in block_part1[:3]} == {1}
    # Block 3-5 are free.
    assert {block.ref_cnt for block in block_part1[3:]} == {0}


def test_reset_prefix_cache():
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
    )

    full_block_token_ids = [i for i in range(3) for _ in range(16)]
    unique_token_ids = [3] * 7
    all_token_ids = full_block_token_ids + unique_token_ids
    req0 = make_request("0", all_token_ids, block_size, sha256)
    blocks = manager.allocate_slots(req0, 55)
    assert blocks is not None and blocks.get_block_ids() == ([1, 2, 3, 4],)

    unique_token_ids = [4] * 7
    all_token_ids = full_block_token_ids + unique_token_ids
    req1 = make_request("1", all_token_ids, block_size, sha256)
    computed_blocks, _ = manager.get_computed_blocks(req1)
    assert len(req1.block_hashes) == 3
    assert len(computed_blocks.blocks[0]) == 3
    blocks = manager.allocate_slots(
        req1, 7, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == ([5],)

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
    block_size = 16
    manager = KVCacheManager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        log_stats=False,  # Disable logging stats
    )
    assert manager.prefix_cache_stats is None

    # Call all functions that check whether log_stats is disabled.
    req = make_request("0", list(range(16)), block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    manager.allocate_slots(
        req, 16, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    manager.reset_prefix_cache()

    # Ensure prefix_cache_stats remains None
    assert manager.prefix_cache_stats is None


def test_maybe_evict_cached_block():
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True)
    block_hash0 = make_block_hash_with_group_id(BlockHash(b"10"), 1000)
    block_hash1 = make_block_hash_with_group_id(BlockHash(b"20"), 2000)
    block_hash2 = make_block_hash_with_group_id(BlockHash(b"30"), 3000)
    block_hashes = [
        block_hash0,
        block_hash1,
        block_hash2,
        # block3 had the exact same block_hash as the first block
        block_hash0,
    ]
    assert len(pool.blocks) == len(block_hashes)
    # Manually add all blocks to cached_blocks
    for block, block_hash in zip(pool.blocks, block_hashes):
        block.block_hash = block_hash
        pool.cached_block_hash_to_block.insert(block_hash, block)

    block0, block1, block2, block3 = pool.blocks
    assert pool.cached_block_hash_to_block._cache == {
        block_hash0: {
            block0.block_id: block0,
            block3.block_id: block3,
        },
        block_hash1: block1,
        block_hash2: block2,
    }
    # Evict block1
    pool._maybe_evict_cached_block(block1)
    assert pool.cached_block_hash_to_block._cache == {
        block_hash0: {block0.block_id: block0, block3.block_id: block3},
        block_hash2: block2,
    }
    # Evict block0: block_hash0 entry should NOT be removed, as block3
    # also use the same hash
    pool._maybe_evict_cached_block(block0)
    assert pool.cached_block_hash_to_block._cache == {
        block_hash0: {block3.block_id: block3},
        block_hash2: block2,
    }
    # Evict block2
    pool._maybe_evict_cached_block(block2)
    assert pool.cached_block_hash_to_block._cache == {block_hash0: {3: block3}}
    # Evict block3
    pool._maybe_evict_cached_block(block3)
    assert pool.cached_block_hash_to_block._cache == {}


@pytest.mark.parametrize("blocks_to_cache", [2, 3, 10])
def test_kv_cache_events(blocks_to_cache: int):
    block_size = 16
    num_blocks = blocks_to_cache + 1

    # Allocate Blocks
    # Should see a single block stored event with a blocks_to_cache number of
    # block hashes
    # take_events should reset the kv_event_queue
    manager = KVCacheManager(
        make_kv_cache_config(block_size, num_blocks),
        max_model_len=8192,
        enable_caching=True,
        enable_kv_cache_events=True,
    )

    num_tokens = block_size * blocks_to_cache
    req0 = make_request("0", list(range(num_tokens)), block_size, sha256)
    _ = manager.allocate_slots(req0, num_tokens)
    events = manager.take_events()

    block = events[-1]
    assert (
        len(block.block_hashes)
        == blocks_to_cache
        == len(manager.block_pool.cached_block_hash_to_block)
    )
    assert len(block.token_ids) == block.block_size * len(block.block_hashes)
    assert len(manager.block_pool.kv_event_queue) == 0

    stored_block_hash = block.block_hashes

    # Remove blocks and send another request
    # Should see block_to_cache number of removed block events and a new block
    # stored event
    manager.free(req0)
    req1 = make_request("1", list(range(num_tokens)), block_size, sha256)
    _ = manager.allocate_slots(req1, num_tokens)
    events = manager.take_events()

    for blocks in events[:-1]:
        assert blocks.block_hashes[0] in stored_block_hash
    assert len(events) == blocks_to_cache + 1
    assert isinstance(events[-2], BlockRemoved)
    assert (
        len(events[-1].block_hashes)
        == blocks_to_cache
        == len(manager.block_pool.cached_block_hash_to_block)
    )

    # All Blocks Cleared
    # Should see a single all blocks cleared event
    manager.free(req1)
    manager.reset_prefix_cache()
    events = manager.take_events()

    assert isinstance(events[-1], AllBlocksCleared)
    assert len(manager.block_pool.cached_block_hash_to_block) == 0


@pytest.mark.parametrize("blocks_to_cache", [2, 3, 10])
def test_kv_cache_events_with_lora(blocks_to_cache: int):
    """Test BlockStored events contain correct lora_id when using LoRA requests."""
    block_size = 16
    num_blocks = blocks_to_cache + 1

    # Create KVCacheManager with events enabled
    manager = KVCacheManager(
        make_kv_cache_config(block_size, num_blocks),
        max_model_len=8192,
        enable_caching=True,
        enable_kv_cache_events=True,
    )

    # Test with LoRA request
    lora_request = LoRARequest(
        lora_name="test_lora", lora_int_id=42, lora_path="/test/path"
    )

    num_tokens = block_size * blocks_to_cache
    req_with_lora = make_request(
        "lora_req",
        list(range(num_tokens)),
        block_size,
        sha256,
        lora_request=lora_request,
    )

    # Allocate slots and get events
    _ = manager.allocate_slots(req_with_lora, num_tokens)
    events = manager.take_events()

    # Verify BlockStored event contains correct lora_id
    block_stored_event = events[-1]
    assert isinstance(block_stored_event, BlockStored)
    assert block_stored_event.lora_id == 42  # Should match lora_request.adapter_id
    assert len(block_stored_event.block_hashes) == blocks_to_cache
    assert block_stored_event.block_size == block_size

    # Clean up
    manager.free(req_with_lora)

    # Test without LoRA request (should have lora_id=None)
    req_without_lora = make_request(
        "no_lora_req", list(range(num_tokens)), block_size, sha256
    )

    _ = manager.allocate_slots(req_without_lora, num_tokens)
    events = manager.take_events()

    block_stored_event = events[-1]
    assert isinstance(block_stored_event, BlockStored)
    assert block_stored_event.lora_id is None  # Should be None when no LoRA request
    assert len(block_stored_event.block_hashes) == blocks_to_cache
    assert block_stored_event.block_size == block_size


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
    req = make_request("divisible_request", token_ids, block_size, sha256)

    # Prime the cache
    computed_blocks, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(
        req, len(token_ids), len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    manager.free(req)

    # New request with same tokens + Eagle enabled
    req_eagle = make_request("eagle_divisible", token_ids, block_size, sha256)
    computed_blocks, num_tokens = manager.get_computed_blocks(req_eagle)

    # Should retain 1 block:
    # 1. Original 3 blocks → pop last hash → 2 matched blocks
    # 2. drop last matched block → 1 remaining block
    assert len(computed_blocks.blocks[0]) == 1
    assert num_tokens == 1 * block_size  # 16 tokens


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
    req = make_request("partial_block_test", token_ids, block_size, sha256)

    # Prime the cache
    computed_blocks, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(
        req, len(token_ids), len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    manager.free(req)

    # New request with Eagle enabled
    req_eagle = make_request("partial_eagle", token_ids, block_size, sha256)
    computed_blocks, num_tokens = manager.get_computed_blocks(req_eagle)
    # Original match: 2 full blocks → Eagle removes 1 → 1 remaining
    assert len(computed_blocks.blocks[0]) == 1
    assert num_tokens == 1 * block_size


def test_eagle_with_sliding_window():
    """Test Eagle behavior with sliding window."""
    block_size = 16
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=block_size,
    )
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer"], sliding_window_spec)],
        ),
        max_model_len=8192,
        enable_caching=True,
        use_eagle=True,
    )

    # 2 full blocks + 5 tokens (non-divisible length)
    token_ids = [0] * (2 * block_size + 5)
    req = make_request("partial_block_test", token_ids, block_size, sha256)

    # Prime the cache
    computed_blocks, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(
        req, len(token_ids), len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    # record the block hash of the first block in the request for later use
    block_hash_first_block = req.block_hashes[0]
    assert block_hash_first_block is not None
    manager.free(req)

    # New request with Eagle enabled
    req_eagle = make_request("partial_eagle", token_ids, block_size, sha256)
    computed_blocks, num_tokens = manager.get_computed_blocks(req_eagle)
    # Original match: 2 full blocks → Eagle removes 1 → 1 remaining
    assert len(computed_blocks.blocks[0]) == 1
    assert num_tokens == 1 * block_size

    # Evict the first block in the request
    assert (
        manager.block_pool.get_cached_block(
            block_hash_first_block, kv_cache_group_ids=[0]
        )
        is not None
    )
    manager.block_pool.cached_block_hash_to_block._cache.pop(
        make_block_hash_with_group_id(block_hash_first_block, 0)
    )

    # New request
    req_after_evict = make_request(
        "partial_eagle_after_evict", token_ids, block_size, sha256
    )
    computed_blocks, num_tokens = manager.get_computed_blocks(req_after_evict)
    # Cache miss. The only hit prefix is [NULL_BLOCK, BLOCK_2] if eagle is
    # not considered. But after dropping the last matched block due to eagle,
    # there will be no matched prefix.
    assert len(computed_blocks.blocks[0]) == 0
    assert num_tokens == 0


def test_block_lookup_cache_single_block_per_key():
    cache = BlockHashToBlockMap()
    key0 = BlockHashWithGroupId(b"hash0")
    key1 = BlockHashWithGroupId(b"hash1")
    key2 = BlockHashWithGroupId(b"hash2")
    block0 = KVCacheBlock(0)
    block1 = KVCacheBlock(1)

    assert cache.get_one_block(key0) is None
    assert cache.get_one_block(key1) is None
    assert cache.get_one_block(key2) is None
    # key0 inserted
    cache.insert(key0, block0)
    assert cache.get_one_block(key0) is block0
    assert cache.get_one_block(key1) is None
    assert cache.get_one_block(key2) is None
    # key1 inserted
    cache.insert(key1, block1)
    assert cache.get_one_block(key0) is block0
    assert cache.get_one_block(key1) is block1
    assert cache.get_one_block(key2) is None
    # No block poped due to block_id mismatch
    assert cache.pop(key0, 100) is None
    assert cache.get_one_block(key0) is block0
    assert cache.get_one_block(key1) is block1
    assert cache.get_one_block(key2) is None
    # block poped with (key0, block ID 0)
    assert cache.pop(key0, 0) is block0
    assert cache.get_one_block(key0) is None
    assert cache.get_one_block(key1) is block1
    assert cache.get_one_block(key2) is None
    # No block poped due to block_id mismatch
    assert cache.pop(key0, 1) is None
    assert cache.get_one_block(key0) is None
    assert cache.get_one_block(key1) is block1
    assert cache.get_one_block(key2) is None
    # block poped with (key1, block ID 1)
    assert cache.pop(key1, 1) is block1
    assert cache.get_one_block(key0) is None
    assert cache.get_one_block(key1) is None
    assert cache.get_one_block(key2) is None


def test_block_lookup_cache_multi_blocks_per_key():
    cache = BlockHashToBlockMap()
    key0 = BlockHashWithGroupId(b"hash0")
    key1 = BlockHashWithGroupId(b"hash1")
    block00 = KVCacheBlock(0)
    block01 = KVCacheBlock(1)
    block10 = KVCacheBlock(10)
    block11 = KVCacheBlock(11)

    assert cache.get_one_block(key0) is None
    assert cache.get_one_block(key1) is None

    cache.insert(key0, block00)
    cache.insert(key0, block01)
    cache.insert(key1, block10)
    cache.insert(key1, block11)

    assert cache.get_one_block(key0) is block00
    assert cache.pop(key0, 0) is block00
    assert cache.get_one_block(key0) is block01
    assert cache.pop(key0, 1) is block01
    assert cache.get_one_block(key0) is None
    assert cache.pop(key0, 2) is None

    assert cache.get_one_block(key1) is block10
    assert cache.pop(key1, 10) is block10
    assert cache.get_one_block(key1) is block11
    assert cache.pop(key1, 11) is block11
    assert cache.get_one_block(key1) is None
    assert cache.pop(key1, 12) is None
