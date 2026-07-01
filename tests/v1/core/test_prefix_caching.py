# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the with and without prefix caching."""

import copy
from collections.abc import Callable
from math import lcm
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.core.kv_cache_manager as kv_cache_manager
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
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager, Request
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    KVCacheBlockCopy,
    get_block_hash,
    get_group_id,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpecKind,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn(request):
    hash_fn: Callable
    if "hash_fn" in request.fixturenames:
        hash_fn = request.getfixturevalue("hash_fn")
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
                data=MultiModalKwargsItem.dummy(),
                mm_position=position,
                identifier=identifier,
                modality="image",
            )
            mm_features.append(mm_feature)

    sampling_params = SamplingParams(max_tokens=17, prompt_logprobs=prompt_logprobs)
    sampling_params.update_from_generation_config({}, eos_token_id=100)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features if mm_features else None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=lora_request,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
    )


def make_kv_cache_manager(kv_cache_config: KVCacheConfig, **kwargs) -> KVCacheManager:
    """Build a ``KVCacheManager``, deriving ``scheduler_block_size`` from the
    config (LCM of group block sizes) unless explicitly provided. This mirrors
    ``resolve_kv_cache_block_sizes`` for the non-context-parallel case used by
    these tests, so callers don't have to pass it at every site."""
    kwargs.setdefault(
        "scheduler_block_size",
        lcm(*(g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups)),
    )
    return KVCacheManager(kv_cache_config, **kwargs)


def make_kv_cache_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def make_kv_cache_config_hybrid_model(
    block_size: int,
    num_blocks: int,
    sliding_window_blocks: int,
    second_spec_type: str = "sliding_window",
) -> KVCacheConfig:
    if second_spec_type == "sliding_window":
        second_spec = SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=sliding_window_blocks * block_size,
        )
    elif second_spec_type == "mamba":
        second_spec = MambaSpec(
            block_size=block_size,
            shapes=(1, 1),
            dtypes=(torch.float32,),
        )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["layer2"],
                second_spec,
            ),
            KVCacheGroupSpec(
                ["layer3"],
                second_spec,
            ),
        ],
    )


def make_kv_cache_config_three_types(
    block_size: int, num_blocks: int, third_spec_type: str = "mamba"
) -> KVCacheConfig:
    if third_spec_type == "mamba":
        third_spec = MambaSpec(
            block_size=block_size,
            shapes=(1, 1),
            dtypes=(torch.float32,),
        )
    elif third_spec_type == "sliding_window":
        third_spec = SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=4 * block_size,
        )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["layer2"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=2 * block_size,
                ),
            ),
            KVCacheGroupSpec(
                ["layer3"],
                third_spec,
            ),
        ],
    )


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_prefill(hash_fn):
    block_size = 16
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
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
    # [partial without hashes from req1 and req0 (5, 4) - prepended for immediate reuse]
    # [unallocated (6, 7, 8, 9, 10)]
    # [common (3, 2, 1)]
    assert [
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ] == [5, 4, 6, 7, 8, 9, 10, 3, 2, 1]

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
    assert blocks is not None and blocks.get_block_ids() == ([5],)  # reuse partial [5]

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
        [5, 4, 6, 7, 8, 9, 10, 3, 2, 1],
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
    manager = make_kv_cache_manager(
        make_kv_cache_config_hybrid_model(block_size, 21, 2),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    hash_fn = sha256

    # Complete 3 blocks (48 tokens)
    num_full_blocks = 3
    common_token_ids = [i for i in range(num_full_blocks) for _ in range(block_size)]

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
    all_token_ids = common_token_ids + unique_token_ids
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

    # Evict the blocks outside sliding window, does not affect the hit length.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "2",
        all_token_ids,
        [
            make_block_hash_with_group_id(block_hashes[0], 1),
            make_block_hash_with_group_id(block_hashes[0], 2),
        ],
        3,
    )

    # Evict the first block of full attention, makes total cache miss.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "3",
        all_token_ids,
        [make_block_hash_with_group_id(block_hashes[0], 0)],
        0,
    )

    # Evict the last block of all layers, reduces the hit length to 2.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "4",
        all_token_ids,
        [
            make_block_hash_with_group_id(block_hashes[2], 0),
            make_block_hash_with_group_id(block_hashes[2], 1),
            make_block_hash_with_group_id(block_hashes[2], 2),
        ],
        2,
    )

    # Evict the last block of full attention, reduces the hit length to 2.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "5",
        all_token_ids,
        [make_block_hash_with_group_id(block_hashes[2], 0)],
        2,
    )

    # Evict the last block of sliding window, reduces the hit length to 2.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "6",
        all_token_ids,
        [make_block_hash_with_group_id(block_hashes[2], 1)],
        2,
    )

    # Evict the last block of sliding window, reduces the hit length to 2.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "7",
        all_token_ids,
        [make_block_hash_with_group_id(block_hashes[2], 2)],
        2,
    )

    # Evict different set of blocks for full attention and sliding window makes
    # total cache miss.
    # The cache hit length of full attention is 1 * block_size.
    # The cache hit length of sliding window is 2 * block_size.
    # Then it is cache miss as the two type of layers
    # have different hit length.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "8",
        all_token_ids,
        [
            make_block_hash_with_group_id(block_hashes[2], 0),
            make_block_hash_with_group_id(block_hashes[0], 1),
            make_block_hash_with_group_id(block_hashes[0], 2),
        ],
        0,
    )


def test_prefill_hybrid_model_eagle():
    block_size = 16
    kv_cache_config = make_kv_cache_config_hybrid_model(block_size, 31, 3)
    manager = make_kv_cache_manager(
        kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        use_eagle=True,
    )

    hash_fn = sha256

    # Complete 6 blocks (96 tokens)
    num_full_blocks = 6
    common_token_ids = [i for i in range(num_full_blocks) for _ in range(block_size)]

    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [6] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request("0", all_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert len(req0.block_hashes) == len(all_token_ids) // block_size
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, len(all_token_ids), num_computed_tokens, computed_blocks
    )
    block_ids = (
        [1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19, 20, 21],
    )
    assert blocks is not None and blocks.get_block_ids() == block_ids

    # Check full block metadata
    parent_block_hash = None
    for i, full_block_ids in enumerate(zip(*(row[:-1] for row in block_ids))):
        block_tokens = tuple(all_token_ids[i * block_size : (i + 1) * block_size])
        block_hash = hash_block_tokens(hash_fn, parent_block_hash, block_tokens)
        for group_id, block_id in enumerate(full_block_ids):
            blk_hash = manager.block_pool.blocks[block_id].block_hash
            assert blk_hash is not None
            assert get_block_hash(blk_hash) == block_hash
            assert get_group_id(blk_hash) == group_id
            assert manager.block_pool.blocks[block_id].ref_cnt == 1
        parent_block_hash = block_hash

    # Check partial block metadata
    for partial_block_id in (row[-1] for row in block_ids):
        assert manager.block_pool.blocks[partial_block_id].block_hash is None
        assert manager.block_pool.blocks[partial_block_id].ref_cnt == 1

    # Cache hit in the common prefix
    # Incomplete 1 block (5 tokens)
    unique_token_ids = [6] * 5
    all_token_ids = common_token_ids + unique_token_ids
    req1 = make_request("1", all_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(req1.block_hashes) == num_full_blocks
    assert computed_blocks.get_block_ids() == (
        [1, 2, 3, 4, 5],
        [0, 0, 10, 11, 12],
        [0, 0, 17, 18, 19],
    )
    assert num_computed_tokens == 5 * block_size
    num_new_tokens = len(all_token_ids) - num_computed_tokens
    blocks = manager.allocate_slots(
        req1, num_new_tokens, num_computed_tokens, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == (
        [22, 23],
        [24, 25],
        [26, 27],
    )
    for block_per_group in computed_blocks.blocks:
        for block in block_per_group:
            if block != manager.block_pool.null_block:
                assert block.ref_cnt == 2

    block_hashes = req1.block_hashes
    manager.free(req0)
    manager.free(req1)

    # Evict the blocks outside sliding window, does not affect the hit length.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "2",
        all_token_ids,
        [
            make_block_hash_with_group_id(block_hashes[0], 1),
            make_block_hash_with_group_id(block_hashes[0], 2),
        ],
        5,
    )

    # Evict the first block of full attention, makes total cache miss.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "3",
        all_token_ids,
        [make_block_hash_with_group_id(block_hashes[0], 0)],
        0,
    )

    # Evict the last block of all layers, reduces the hit length to 4.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "4",
        all_token_ids,
        [
            make_block_hash_with_group_id(block_hashes[-1], 0),
            make_block_hash_with_group_id(block_hashes[-1], 1),
            make_block_hash_with_group_id(block_hashes[-1], 2),
        ],
        4,
    )

    # Evict the last block of full attention, reduces the hit length to 4.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "5",
        all_token_ids,
        [make_block_hash_with_group_id(block_hashes[-1], 0)],
        4,
    )

    # Since the last block of full attention is dropped for eagle, evict
    # the second last block of sliding window, reduces the hit length to 3.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "6",
        all_token_ids,
        [make_block_hash_with_group_id(block_hashes[-2], 1)],
        3,
    )

    # Since the last block of full attention is dropped for eagle, evict
    # the second last block of sliding window, reduces the hit length to 3.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "7",
        all_token_ids,
        [make_block_hash_with_group_id(block_hashes[-2], 2)],
        3,
    )

    # Evict different set of blocks for full attention and sliding window.
    # Full loses its last block so it drops to 4 full blocks after the eagle
    # pop; SWA lost block 0 (outside the sliding window of the final hit),
    # which is not required for the K+1 anchor at position 4. Coordinated
    # single-drop aligns both groups at hit=4.
    _test_partial_request_hit(
        manager,
        block_size,
        num_full_blocks,
        "8",
        all_token_ids,
        [
            make_block_hash_with_group_id(block_hashes[-1], 0),
            make_block_hash_with_group_id(block_hashes[0], 1),
            make_block_hash_with_group_id(block_hashes[0], 2),
        ],
        4,
    )


def _test_partial_request_hit(
    manager: KVCacheManager,
    block_size: int,
    num_full_blocks,
    request_id: str,
    prompt_token_ids: list[int],
    hash_to_evict: list[BlockHashWithGroupId],
    expect_hit_length: int,
):
    cached_block_hash_to_block_bak = copy.copy(
        manager.block_pool.cached_block_hash_to_block._cache
    )
    req = make_request(request_id, prompt_token_ids, block_size, sha256)
    for hash_with_group_id in hash_to_evict:
        manager.block_pool.cached_block_hash_to_block._cache.pop(hash_with_group_id)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
    assert len(req.block_hashes) == num_full_blocks
    assert num_computed_tokens == expect_hit_length * block_size
    for block_per_group in computed_blocks.blocks:
        assert len(block_per_group) == num_computed_tokens // block_size
    for hash_with_group_id in hash_to_evict:
        manager.block_pool.cached_block_hash_to_block._cache[hash_with_group_id] = (
            cached_block_hash_to_block_bak[hash_with_group_id]
        )
    manager.free(req)


def _make_hybrid_kv_cache_config(
    block_size: int, num_blocks: int, spec_types: list[str]
) -> KVCacheConfig:
    """
    Create a KVCacheConfig with the specified spec types.

    Args:
        block_size: The block size for KV cache.
        num_blocks: The number of blocks in the KV cache.
        spec_types: List of spec type strings. Supported types:
            - "full": FullAttentionSpec
            - "sliding_window": SlidingWindowSpec with window=2*block_size
            - "sliding_window_large": SlidingWindowSpec with window=4*block_size
            - "mamba": MambaSpec
    """
    spec_map = {
        "full": lambda: FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
        "sliding_window": lambda: SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=2 * block_size,
        ),
        "sliding_window_large": lambda: SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=4 * block_size,
        ),
        "mamba": lambda: MambaSpec(
            block_size=block_size,
            shapes=(1, 1),
            dtypes=(torch.float32,),
        ),
        "mamba_align": lambda: MambaSpec(
            block_size=block_size,
            shapes=(1, 1),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
    }

    kv_cache_groups = [
        KVCacheGroupSpec([f"layer{i}"], spec_map[spec_type]())
        for i, spec_type in enumerate(spec_types)
    ]

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=kv_cache_groups,
    )


# Test cases covering various combinations of KV cache spec types:
# - Varying number of groups (2, 3, or 4)
# - 0, 1, or 2 full attention groups
# - Sliding window with different window sizes
# - Interleaved group IDs (full attn and other types mixed)
# - Mamba spec combinations
_HYBRID_MODEL_TEST_CASES = [
    # 2 groups: 1 full + 1 other
    pytest.param(["full", "sliding_window"], id="2g-full+sw"),
    pytest.param(["full", "mamba"], id="2g-full+mamba"),
    # 2 groups: 0 full (all other types)
    pytest.param(["sliding_window", "mamba"], id="2g-sw+mamba"),
    pytest.param(["sliding_window", "sliding_window_large"], id="2g-sw+sw_large"),
    # 3 groups: 1 full + 2 others (same type)
    pytest.param(["full", "sliding_window", "sliding_window"], id="3g-full+2sw"),
    pytest.param(["full", "mamba", "mamba"], id="3g-full+2mamba"),
    # 3 groups: 1 full + 2 others (different types)
    pytest.param(["full", "sliding_window", "mamba"], id="3g-full+sw+mamba"),
    pytest.param(
        ["full", "sliding_window", "sliding_window_large"],
        id="3g-full+sw+sw_large",
    ),
    # 3 groups: 2 full + 1 other
    pytest.param(["full", "full", "sliding_window"], id="3g-2full+sw"),
    pytest.param(["full", "full", "mamba"], id="3g-2full+mamba"),
    # 4 groups: interleaved (full, other, full, other)
    pytest.param(
        ["full", "sliding_window", "full", "sliding_window_large"],
        id="4g-interleaved-full+sw+sw_large",
    ),
    pytest.param(
        ["full", "mamba", "full", "mamba"],
        id="4g-interleaved-full+mamba",
    ),
    # 4 groups: interleaved with different sliding windows
    pytest.param(
        ["full", "sliding_window", "full", "sliding_window_large"],
        id="4g-interleaved-full+sw_mixed",
    ),
    # 4 groups: 0 full (all other types)
    pytest.param(
        ["sliding_window", "mamba", "sliding_window_large", "mamba"],
        id="4g-sw+mamba+sw_large+mamba",
    ),
    # 4 groups: 2 full + 2 others (grouped)
    pytest.param(
        ["full", "full", "sliding_window", "mamba"],
        id="4g-2full+sw+mamba",
    ),
]


@pytest.mark.parametrize("spec_types", _HYBRID_MODEL_TEST_CASES)
def test_prefill_hybrid_model_combinations(spec_types: list[str]):
    """
    Test prefix caching with hybrid models containing various combinations of
    KV cache spec types.

    This unified test covers:
    - Various combinations (full attn + other attn types)
    - Varying number of groups (2, 3, or 4)
    - 0, 1, or 2 full attention groups in the combination
    - Two sliding_window attn groups with different window sizes
    - Interleaved group IDs (full attn and other types alternating)
    - Mamba spec with other attention types
    """
    block_size = 16
    num_groups = len(spec_types)
    # Allocate enough blocks for all groups
    num_blocks = 10 * num_groups

    kv_cache_config = _make_hybrid_kv_cache_config(block_size, num_blocks, spec_types)
    manager = make_kv_cache_manager(
        kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    hash_fn = sha256

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(block_size)]
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids

    # First request: no cache hit initially
    req0 = make_request("0", all_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)

    assert len(req0.block_hashes) == 3
    assert not computed_blocks.blocks[0]  # No cache hit initially
    assert num_computed_tokens == 0

    blocks = manager.allocate_slots(
        req0, 55, len(computed_blocks.blocks[0]) * block_size, computed_blocks
    )
    assert blocks is not None
    # Should have blocks for all groups
    assert len(blocks.get_block_ids()) == num_groups

    manager.new_step_starts()

    # Second request: should hit cached blocks for common prefix
    req1 = make_request("1", common_token_ids + [4] * 5, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)

    # Should hit cached blocks for all groups
    assert num_computed_tokens == 3 * block_size
    assert len(computed_blocks.blocks) == num_groups

    # Allocate and verify blocks for second request
    blocks = manager.allocate_slots(
        req1,
        len(common_token_ids) + 5 - num_computed_tokens,
        num_computed_tokens,
        computed_blocks,
    )
    assert blocks is not None
    assert len(blocks.get_block_ids()) == num_groups

    manager.free(req0)
    manager.free(req1)


# Test cases with eagle enabled: Only test a single simple case for now.
# - 2 groups: 1 full + 1 other
_EAGLE_HYBRID_MODEL_TEST_CASES = [
    # 2 groups: 1 full + 1 other
    pytest.param(["full", "sliding_window"], 3, id="2g-full+sw"),
]


@pytest.mark.parametrize("spec_types,expect_hit_length", _EAGLE_HYBRID_MODEL_TEST_CASES)
def test_prefill_hybrid_model_combinations_eagle(
    spec_types: list[str], expect_hit_length: int
):
    """
    Test prefix caching with hybrid models (1 full attn + 1 other) with EAGLE.
    More complex hybrid models with EAGLE are not yet supported (see issue #32802).
    """
    block_size = 16
    num_groups = len(spec_types)
    # Allocate enough blocks for all groups
    num_blocks = 10 * num_groups

    kv_cache_config = _make_hybrid_kv_cache_config(block_size, num_blocks, spec_types)
    manager = make_kv_cache_manager(
        kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        use_eagle=True,
    )

    hash_fn = sha256

    # Complete 3 blocks (48 tokens)
    num_full_blocks = 4
    common_token_ids = [i for i in range(num_full_blocks) for _ in range(block_size)]
    unique_token_ids = [4] * 7
    all_token_ids = common_token_ids + unique_token_ids

    # First request: no cache hit initially
    req0 = make_request("0", all_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)

    assert len(req0.block_hashes) == num_full_blocks
    assert not computed_blocks.blocks[0]  # No cache hit initially
    assert num_computed_tokens == 0

    blocks = manager.allocate_slots(
        req0, len(all_token_ids), num_computed_tokens, computed_blocks
    )
    assert blocks is not None
    # Should have blocks for all groups
    assert len(blocks.get_block_ids()) == num_groups

    # Second request: should hit cached blocks for common prefix
    all_token_ids = common_token_ids + [6] * 5
    req1 = make_request("1", all_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)

    # Should hit cached blocks for all groups
    assert num_computed_tokens == expect_hit_length * block_size
    assert len(computed_blocks.blocks) == num_groups
    # Verify each group has the correct number of computed blocks
    for block_per_group in computed_blocks.blocks:
        assert len(block_per_group) == expect_hit_length

    # Allocate and verify blocks for second request
    blocks = manager.allocate_slots(
        req1,
        len(all_token_ids) - num_computed_tokens,
        num_computed_tokens,
        computed_blocks,
    )
    assert blocks is not None
    assert len(blocks.get_block_ids()) == num_groups

    manager.free(req0)
    manager.free(req1)


def test_prefill_hybrid_model_mamba_align():
    """Test that MambaManager.cache_blocks() handles null blocks in align mode.

    Regression test for https://github.com/vllm-project/vllm/issues/34361.
    In mamba_cache_mode="align", allocate_new_blocks() pads req_to_blocks with
    null blocks. cache_full_blocks() correctly skips them, but
    MambaManager.cache_blocks() must also skip null blocks when tracking
    cached_blocks_this_step.
    """
    block_size = 16
    num_blocks = 30

    kv_cache_config = _make_hybrid_kv_cache_config(
        block_size, num_blocks, ["full", "mamba_align"]
    )
    manager = make_kv_cache_manager(
        kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    hash_fn = sha256

    # 3 full blocks (48 tokens) + 7 partial tokens = 55 tokens total
    all_token_ids = [i for i in range(3) for _ in range(block_size)] + [3] * 7

    # First request: allocate_slots should not crash with the assertion error
    # in MambaManager.cache_blocks() when null blocks are present.
    req0 = make_request("0", all_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert num_computed_tokens == 0

    blocks = manager.allocate_slots(req0, 55, num_computed_tokens, computed_blocks)
    assert blocks is not None
    assert len(blocks.get_block_ids()) == 2  # full_attn + mamba groups

    manager.free(req0)


def test_hybrid_cache_mamba_align_shared_prefix_detection():
    """Test shared prefix detection heuristic for mamba align cache mode

    HybridKVCacheCoordinator returns num_uncached_common > 0 when a shared
    uncached prefix is detected. With mamba_align cache, _mamba_block_aligned_split
    enforces scheduling aligned with the common prefix.
    """
    block_size = 16
    manager = make_kv_cache_manager(
        _make_hybrid_kv_cache_config(block_size, 30, ["full", "mamba_align"]),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )
    hash_fn = sha256

    # Request: 3 blocks
    prefix = [i for i in range(3) for _ in range(block_size)]
    req_0 = make_request("0", prefix, block_size, hash_fn)
    computed_blocks, num_computed = manager.get_computed_blocks(req_0)
    num_uncached_common = manager.coordinator.num_uncached_common_prefix_tokens
    assert num_computed == 0  # nothing cached yet
    assert num_uncached_common == 0
    manager.allocate_slots(req_0, 3 * block_size, 0, computed_blocks)

    # Request: 3 blocks (shared with above) + 7 different tokens
    req_1 = make_request("1", prefix + [100] * 7, block_size, hash_fn)
    computed_blocks, num_computed = manager.get_computed_blocks(req_1)
    num_uncached_common = manager.coordinator.num_uncached_common_prefix_tokens
    assert num_computed == 3 * block_size  # we should observe a 3-block cache hit
    assert num_uncached_common == 0
    manager.allocate_slots(req_1, 7, 3 * block_size, computed_blocks)

    # Request: 3 blocks, but only 2 blocks shared (replace the last token in 3rd block):
    req_2 = make_request("2", prefix[:-1] + [101], block_size, hash_fn)
    computed_blocks, num_computed = manager.get_computed_blocks(req_2)
    num_uncached_common = manager.coordinator.num_uncached_common_prefix_tokens
    assert num_computed == 0  # mamba_align doesn't cache intermediate blocks
    assert num_uncached_common == 2 * block_size  # heuristic detects a shared prefix

    # Next, validate scheduler logic for num_uncached_common_prefix_tokens > 0
    # Create minimal mock with just the needed attributes
    mock = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size), use_eagle=False
    )
    num_new_tokens_adjusted = Scheduler._mamba_block_aligned_split(
        self=mock,
        request=req_2,
        num_new_tokens=3 * block_size,
        num_uncached_common_prefix_tokens=num_uncached_common,
    )
    assert num_new_tokens_adjusted == 2 * block_size  # adjust to the common prefix

    manager.allocate_slots(req_2, 3 * block_size, 0, computed_blocks)
    # Cleanup
    manager.free(req_0)
    manager.free(req_1)
    manager.free(req_2)


def test_hybrid_model_mamba_align_with_dynamic_draft_tokens():
    """Regression test for https://github.com/vllm-project/vllm/issues/39271.

    With suffix decoding enabled, the number of proposed draft token may
    change dynamically each round, causing the MambaManager to crash during
    allocate_slots() as it originally assumes the `num_blocks` to increase.
    """
    block_size = 16
    num_blocks = 30

    kv_cache_config = _make_hybrid_kv_cache_config(
        block_size, num_blocks, ["full", "mamba_align"]
    )
    manager = KVCacheManager(
        kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        scheduler_block_size=block_size,
    )

    # the default hash function is sha256
    hash_fn = sha256

    all_token_ids = [i for i in range(3) for _ in range(block_size)] + [3] * 7
    req0 = make_request("0", all_token_ids, block_size, hash_fn)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, len(all_token_ids), num_computed_tokens, computed_blocks
    )
    assert blocks is not None

    # prefill forward finished
    req0.append_output_token_ids([1])
    req0.num_computed_tokens = len(all_token_ids)

    # Round1: propose 16 draft tokens, accept only one
    req0.spec_token_ids = [4] * 16
    blocks = manager.allocate_slots(req0, num_new_tokens=16, num_new_computed_tokens=0)
    assert blocks is not None
    req0.append_output_token_ids([4])
    req0.num_computed_tokens += 1

    # Round2: propose only one token, allocate should not crash
    req0.spec_token_ids = [5] * 1
    blocks = manager.allocate_slots(req0, num_new_tokens=1, num_new_computed_tokens=0)
    assert blocks is not None and all(len(group) == 0 for group in blocks.blocks)

    manager.free(req0)


def test_hybrid_mamba_align_partial_hash_hit():
    hash_block_size = 2
    mamba_block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=20,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed = manager.get_computed_blocks(req0)
    assert num_computed == 0
    blocks = manager.allocate_slots(req0, 6, num_computed, computed_blocks)
    assert blocks is not None
    manager.free(req0)
    manager.new_step_starts()

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert partial_mamba_block is not None
    assert partial_mamba_block[0].block_hash_num_tokens == 6

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)
    computed_blocks, num_computed = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert [len(group) for group in computed_blocks.blocks] == [3, 2]

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    mamba_new_block_ids = new_blocks.get_block_ids()[1]
    assert len(mamba_new_block_ids) == 1
    assert mamba_new_block_ids[0] != partial_mamba_block[0].block_id
    assert manager.get_blocks("1").get_block_ids()[1][1] == mamba_new_block_ids[0]
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_mamba_block[0].block_id,
            dst_block_id=mamba_new_block_ids[0],
        )
        in manager.take_kv_cache_block_copies()
    )
    assert manager.get_blocks("1").blocks[1][1].block_hash_num_tokens == 8


def test_hybrid_mamba_partial_tail_owner_uses_cow_on_continue():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert partial_mamba_block is not None
    partial_mamba_block_id = partial_mamba_block[0].block_id
    assert manager.get_blocks("0").get_block_ids()[1][1] == partial_mamba_block_id

    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    new_blocks = manager.allocate_slots(req0, 1)
    assert new_blocks is not None

    mamba_new_block_ids = new_blocks.get_block_ids()[1]
    assert len(mamba_new_block_ids) == 1
    assert mamba_new_block_ids[0] != partial_mamba_block_id
    assert manager.get_blocks("0").get_block_ids()[1][1] == mamba_new_block_ids[0]
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_mamba_block_id,
            dst_block_id=mamba_new_block_ids[0],
        )
        in manager.take_kv_cache_block_copies()
    )
    assert (
        manager.block_pool.get_cached_block(partial_mamba_hash, kv_cache_group_ids=[1])
        == partial_mamba_block
    )


def test_hybrid_mamba_partial_tail_owner_continue_preserves_later_hit():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None

    partial_mamba_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_mamba_hash, kv_cache_group_ids=[1]
    )
    assert partial_mamba_block is not None
    partial_mamba_block_id = partial_mamba_block[0].block_id

    req0.num_computed_tokens = 6
    req0.append_output_token_ids([3])
    assert manager.allocate_slots(req0, 1) is not None
    manager.take_kv_cache_block_copies()
    manager.new_step_starts()

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 4, 4], hash_block_size, sha256)
    computed_blocks, num_computed = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert computed_blocks.get_block_ids()[1][1] == partial_mamba_block_id

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    mamba_new_block_ids = new_blocks.get_block_ids()[1]
    assert len(mamba_new_block_ids) == 1
    assert mamba_new_block_ids[0] != partial_mamba_block_id
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_mamba_block_id,
            dst_block_id=mamba_new_block_ids[0],
        )
        in manager.take_kv_cache_block_copies()
    )


def test_hybrid_full_attention_partial_hash_hit_uses_cow():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    req0 = make_request("0", [0, 0, 1, 1, 2, 2], hash_block_size, sha256)
    computed_blocks, num_computed = manager.get_computed_blocks(req0)
    assert num_computed == 0
    assert manager.allocate_slots(req0, 6, num_computed, computed_blocks) is not None
    manager.free(req0)
    manager.new_step_starts()

    partial_full_hash = req0.block_hashes[6 // hash_block_size - 1]
    partial_full_block = manager.block_pool.get_cached_block(
        partial_full_hash, kv_cache_group_ids=[0]
    )
    assert partial_full_block is not None

    req1 = make_request("1", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)
    computed_blocks, num_computed = manager.get_computed_blocks(req1)
    assert num_computed == 6
    assert [len(group) for group in computed_blocks.blocks] == [2, 2]

    new_blocks = manager.allocate_slots(req1, 2, num_computed, computed_blocks)
    assert new_blocks is not None
    full_new_block_ids = new_blocks.get_block_ids()[0]
    assert len(full_new_block_ids) == 1
    assert full_new_block_ids[0] != partial_full_block[0].block_id
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_full_block[0].block_id,
            dst_block_id=full_new_block_ids[0],
        )
        in manager.take_kv_cache_block_copies()
    )
    assert partial_full_block[0].ref_cnt == 1
    manager.new_step_starts()
    assert partial_full_block[0].ref_cnt == 0


def test_hybrid_partial_hash_truncates_full_attention_hit_length():
    hash_block_size = 2
    block_size = 2 * hash_block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=24,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    pool = manager.block_pool
    req = make_request(
        "0",
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        hash_block_size,
        sha256,
    )

    full_blocks = pool.get_new_blocks(3)
    pool.cache_full_blocks(
        request=req,
        blocks=full_blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=0,
    )
    pool.cache_partial_block(
        request=req,
        block=full_blocks[2],
        num_tokens=10,
        kv_cache_group_id=0,
        block_size=block_size,
    )

    mamba_block = pool.get_new_blocks(1)[0]
    pool.cache_partial_block(
        request=req,
        block=mamba_block,
        num_tokens=6,
        kv_cache_group_id=1,
        block_size=block_size,
    )

    computed_blocks, num_computed = manager.get_computed_blocks(req)
    assert num_computed == 6
    assert [len(group) for group in computed_blocks.blocks] == [2, 2]


def test_prefill_plp():
    """Test prefill with APC and some prompt logprobs (plp) requests.

    1. Schedule plp request and validate APC block allocation
    2. Schedule non-plp request and validate blocks
    3. Schedule plp request; no hit should occur; validate blocks
    """
    block_size = 16
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
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
    # [partial without hashes from req1 and req0 (5, 4) - prepended for immediate reuse]
    # [unallocated (6, 7, 8, 9, 10)]
    # [common (3, 2, 1)]
    assert [
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ] == [5, 4, 6, 7, 8, 9, 10, 3, 2, 1]

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
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
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
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
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
    # partial blocks (without hash) at head, other at tail (LRU policy):
    assert [
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ] == [6, 10, 5, 4, 3, 2, 1]
    manager.free(req1)
    assert manager.block_pool.free_block_queue.num_free_blocks == 10
    assert [
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ] == [6, 10, 5, 4, 3, 2, 1, 9, 8, 7]

    # Touch the first 2 blocks.
    req2 = make_request("2", list(range(2 * 16 + 3)), block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert computed_blocks.get_block_ids() == ([1, 2],)
    assert num_computed_tokens == 2 * 16
    blocks = manager.allocate_slots(
        req2, 3, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks is not None and blocks.get_block_ids() == ([6],)
    assert manager.block_pool.free_block_queue.num_free_blocks == 7


def test_hash_block_correct_reuse():
    """
    This tests when a previously cached block is reused as a new block,
    its hash metadata should be correctly reset.
    """
    block_size = 16
    manager = make_kv_cache_manager(
        make_kv_cache_config(16, 2),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
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
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 3),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
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
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 5),
        max_model_len=8192,
        enable_caching=False,
        hash_block_size=block_size,
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
        hash_block_size=block_size,
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
    block_pool = BlockPool(
        num_gpu_blocks=10, enable_caching=True, hash_block_size=block_size
    )

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
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
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
        (
            kv_cache_utils.NONE_HASH,
            tuple(all_token_ids[:block_size]),
            (("aaa", 11),),
        )
    )
    assert block_hashes[1] == sha256(
        (
            block_hashes[0],
            tuple(all_token_ids[block_size : block_size * 2]),
            (("aaa", -5), ("bbb", 14)),
        )
    )
    assert block_hashes[2] == sha256(
        (
            block_hashes[1],
            tuple(all_token_ids[block_size * 2 : block_size * 3]),
            (("bbb", -2),),
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
        (
            block_hashes[2],
            tuple(all_token_ids[3 * block_size :] + [8] * 5),
            (("ccc", 0),),
        )
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
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
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
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
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
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
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
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, 11),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
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
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
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
        block.set_block_hash(block_hash)
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
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, num_blocks),
        max_model_len=8192,
        enable_caching=True,
        enable_kv_cache_events=True,
        hash_block_size=block_size,
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
    assert block.kv_cache_spec_kind == KVCacheSpecKind.FULL_ATTENTION.value
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
        assert isinstance(blocks, BlockRemoved)
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


def test_null_parent_block_hash():
    block_size = 1
    num_cached_blocks = 2
    num_full_blocks = 4
    kv_cache_group_id = 0

    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=block_size,
        enable_kv_cache_events=True,
    )

    req = make_request(
        "req_null_parent",
        prompt_token_ids=[10, 11, 12, 13],
        block_size=block_size,
        hash_fn=sha256,
    )
    assert len(req.block_hashes) == num_full_blocks

    # Physical parent is `null_block` (no hash), while the logical parent hash
    # still exists in `request.block_hashes[num_cached_blocks - 1]`.
    assert pool.null_block.block_hash is None
    new_blocks = pool.get_new_blocks(num_full_blocks - 1)
    blocks = [
        new_blocks[: num_cached_blocks - 1],
        pool.null_block,  # physical parent
        *new_blocks[num_cached_blocks - 1 :],
    ]

    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=num_cached_blocks,
        num_full_blocks=num_full_blocks,
        block_size=block_size,
        kv_cache_group_id=kv_cache_group_id,
    )

    events = pool.take_events()
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, BlockStored)

    expected_parent = kv_cache_utils.maybe_convert_block_hash(
        req.block_hashes[num_cached_blocks - 1]
    )
    assert event.parent_block_hash == expected_parent
    assert event.parent_block_hash is not None

    expected_new_hashes = [
        kv_cache_utils.maybe_convert_block_hash(h)
        for h in req.block_hashes[num_cached_blocks:num_full_blocks]
    ]
    assert event.block_hashes == expected_new_hashes
    assert event.group_idx == kv_cache_group_id
    assert event.kv_cache_spec_kind is None
    assert event.kv_cache_spec_sliding_window is None

    # Ensure we didn't accidentally assign a hash to the null block.
    assert pool.null_block.block_hash is None
    # Sanity check: newly cached physical blocks should have hashes assigned.
    assert blocks[num_cached_blocks].block_hash is not None
    assert blocks[num_full_blocks - 1].block_hash is not None


@pytest.mark.parametrize("blocks_to_cache", [2, 3, 10])
def test_kv_cache_events_with_lora(blocks_to_cache: int):
    """Test BlockStored events contain correct lora_id when using LoRA requests."""
    block_size = 16
    num_blocks = blocks_to_cache + 1

    # Create KVCacheManager with events enabled
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, num_blocks),
        max_model_len=8192,
        enable_caching=True,
        enable_kv_cache_events=True,
        hash_block_size=block_size,
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


@pytest.mark.parametrize("group_id", [0, 1, 2])
def test_block_stored_event_group_idx(group_id: int):
    """Test BlockStored events emitted by cache_full_blocks carry the correct
    group_idx."""
    block_size = 4
    num_tokens = block_size * 2

    manager = make_kv_cache_manager(
        make_kv_cache_config_three_types(block_size, num_blocks=5),
        max_model_len=8192,
        enable_caching=True,
        enable_kv_cache_events=True,
        hash_block_size=block_size,
    )
    pool = manager.block_pool

    req = make_request(
        "req_grp_idx",
        prompt_token_ids=list(range(num_tokens)),
        block_size=block_size,
        hash_fn=sha256,
    )

    blocks = pool.get_new_blocks(2)
    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=group_id,
    )

    events = manager.take_events()
    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert events[0].group_idx == group_id
    assert (
        events[0].kv_cache_spec_kind
        == [
            KVCacheSpecKind.FULL_ATTENTION.value,
            KVCacheSpecKind.SLIDING_WINDOW.value,
            KVCacheSpecKind.MAMBA.value,
        ][group_id]
    )
    assert (
        events[0].kv_cache_spec_sliding_window
        == [
            None,
            2 * block_size,
            None,
        ][group_id]
    )


def test_block_stored_event_group_idx_multiple_groups():
    """
    Test BlockStored events for separate HMA groups that each carry the
    correct group_idx.

    Simulates the HMA scenario where full-attention blocks (group 0) and
    sliding-window blocks (group 1) are cached independently and must be
    distinguishable by consumers doing HMA-aware prefix-cache routing.
    """
    block_size = 4
    num_tokens = block_size * 2

    manager = make_kv_cache_manager(
        KVCacheConfig(
            num_blocks=5,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer1"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                ),
                KVCacheGroupSpec(
                    ["layer2"],
                    SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                        sliding_window=128,
                    ),
                ),
            ],
        ),
        max_model_len=8192,
        enable_caching=True,
        enable_kv_cache_events=True,
        hash_block_size=block_size,
    )
    pool = manager.block_pool

    req = make_request(
        "req_multi_grp",
        prompt_token_ids=list(range(num_tokens)),
        block_size=block_size,
        hash_fn=sha256,
    )

    # Cache blocks for group 0 (full-attention)
    blocks_grp0 = pool.get_new_blocks(2)
    pool.cache_full_blocks(
        request=req,
        blocks=blocks_grp0,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=0,
    )

    # Cache blocks for group 1 (sliding-window)
    blocks_grp1 = pool.get_new_blocks(2)
    pool.cache_full_blocks(
        request=req,
        blocks=blocks_grp1,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=1,
    )

    events = manager.take_events()
    assert len(events) == 2
    assert isinstance(events[0], BlockStored)
    assert events[0].group_idx == 0
    assert events[0].kv_cache_spec_kind == KVCacheSpecKind.FULL_ATTENTION.value
    assert events[0].kv_cache_spec_sliding_window is None
    assert isinstance(events[1], BlockStored)
    assert events[1].group_idx == 1
    assert events[1].kv_cache_spec_kind == KVCacheSpecKind.SLIDING_WINDOW.value
    assert events[1].kv_cache_spec_sliding_window == 128


def test_block_stored_event_group_idx_out_of_bounds(monkeypatch):
    """Out-of-range group_idx events are returned without metadata annotation."""
    block_size = 4
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, num_blocks=5),
        max_model_len=8192,
        enable_caching=True,
        enable_kv_cache_events=True,
        hash_block_size=block_size,
    )
    event = BlockStored(
        block_hashes=[1],
        parent_block_hash=None,
        token_ids=list(range(block_size)),
        block_size=block_size,
        lora_id=None,
        medium=None,
        lora_name=None,
        group_idx=1,
    )
    manager.block_pool.kv_event_queue.append(event)
    warnings = []

    def collect_warning(message, *args, **kwargs):
        del kwargs
        warnings.append(message % args if args else message)

    monkeypatch.setattr(kv_cache_manager.logger, "warning", collect_warning)
    events = manager.take_events()

    assert events == [event]
    assert event.kv_cache_spec_kind is None
    assert event.kv_cache_spec_sliding_window is None
    assert warnings == ["Group index `1` not in KV cache metadata"]


@pytest.mark.parametrize("group_id", [0, 1, 2])
def test_block_removed_event_group_idx(group_id: int):
    """
    Test BlockRemoved events emitted on eviction carry the group_idx extracted
    from the evicted block's BlockHashWithGroupId via get_group_id().
    """
    block_size = 4
    num_tokens = block_size * 2

    # null block + 4 usable; allocate all 4, cache 2, free all, re-allocate
    # all 4 so the 2 cached blocks are forced through _maybe_evict_cached_block.
    pool = BlockPool(
        num_gpu_blocks=5,
        enable_caching=True,
        hash_block_size=block_size,
        enable_kv_cache_events=True,
    )

    req = make_request(
        "req_evict_grp",
        prompt_token_ids=list(range(num_tokens)),
        block_size=block_size,
        hash_fn=sha256,
    )

    # Allocate all usable blocks and cache the first two for the target group.
    all_blocks = pool.get_new_blocks(4)
    pool.cache_full_blocks(
        request=req,
        blocks=all_blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=group_id,
    )

    # Drain the BlockStored events so only eviction events remain later.
    pool.take_events()

    # Return all blocks to the free queue so they become eviction candidates.
    pool.free_blocks(all_blocks)

    # Re-allocate all blocks; the two with hashes trigger BlockRemoved events.
    pool.get_new_blocks(4)

    events = pool.take_events()
    removed_events = [e for e in events if isinstance(e, BlockRemoved)]

    assert len(removed_events) == 2
    for event in removed_events:
        assert event.group_idx == group_id


def test_eagle_enabled_removes_last_block():
    """Verify Eagle does NOT remove blocks when request
    length is divisible by block size."""
    block_size = 16
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, num_blocks=10),
        max_model_len=8192,
        enable_caching=True,
        use_eagle=True,
        hash_block_size=block_size,
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
    manager = make_kv_cache_manager(
        make_kv_cache_config(block_size, num_blocks=10),
        max_model_len=8192,
        enable_caching=True,
        use_eagle=True,
        hash_block_size=block_size,
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
    manager = make_kv_cache_manager(
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer"], sliding_window_spec)],
        ),
        max_model_len=8192,
        enable_caching=True,
        use_eagle=True,
        hash_block_size=block_size,
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


def test_eagle_swa_alignment_caches_extra_block():
    """Regression: SWA + EAGLE with `sliding_window <= alignment_tokens`.

    When the cache-hit alignment (lcm of per-group block sizes) is larger than
    the SWA window, the SWA mask only kept the last block of each aligned
    segment. EAGLE/MTP lookup needs ``tail + 1`` contiguous cached blocks and
    that +1 block lives at the next segment's first position, which was left
    uncached. The fix caches that extra block when ``use_eagle=True``.
    """
    block_size = 8
    # Full group uses 4 * block_size, so lcm/alignment is 4 * block_size.
    # SWA group has sliding_window = block_size (i.e., tail = 1 block).
    # Without the fix, the second cached block needed for the EAGLE 2-block
    # match never exists -> EAGLE cache hit fails entirely.
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=4 * block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["swa_mtp"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=block_size,
                ),
                is_eagle_group=True,
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        use_eagle=True,
    )

    # Prime the cache with a long prompt (16 swa blocks = 4 aligned segments).
    token_ids = [i for i in range(16) for _ in range(block_size)]
    req0 = make_request("0", token_ids, block_size, sha256)
    computed_blocks, _ = manager.get_computed_blocks(req0)
    blocks = manager.allocate_slots(
        req0,
        len(token_ids),
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    assert blocks is not None
    manager.free(req0)

    # Second request with identical prompt should find an EAGLE cache hit.
    # Without the fix, ``num_computed_tokens`` is 0; with the fix, it lands at
    # an alignment boundary (multiple of 32 tokens, minus the EAGLE drop).
    req1 = make_request("1", token_ids, block_size, sha256)
    _, num_computed_tokens = manager.get_computed_blocks(req1)
    assert num_computed_tokens > 0, (
        "EAGLE + SWA with sliding_window <= alignment failed to find any "
        "cache hit; the +1 block past each segment boundary must be cached."
    )
    # Each aligned segment contributes 4 * block_size = 32 tokens; EAGLE drops
    # the last block (block_size tokens) from the hit.
    assert num_computed_tokens % (4 * block_size) == 0


def test_eagle_swa_boundary_caches_post_boundary_block():
    """EAGLE + SWA must cache the first block after an alignment boundary.

    A 40-token computed prefix with 8-token SWA blocks and 32-token hybrid
    alignment needs SWA blocks 3 and 4 cached to reuse a 32-token prefix:
    block 3 is the segment tail, and block 4 is the EAGLE lookahead block
    that gets dropped after lookup.
    """
    block_size = 8
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=4 * block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["swa_mtp"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=block_size,
                ),
                is_eagle_group=True,
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        use_eagle=True,
    )

    token_ids = [i for i in range(5) for _ in range(block_size)]
    req0 = make_request("0", token_ids, block_size, sha256)
    computed_blocks, _ = manager.get_computed_blocks(req0)
    blocks = manager.allocate_slots(
        req0,
        len(token_ids),
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    assert blocks is not None

    pool = manager.block_pool
    assert pool.get_cached_block(req0.block_hashes[3], kv_cache_group_ids=[1])
    assert pool.get_cached_block(req0.block_hashes[4], kv_cache_group_ids=[1])
    manager.free(req0)

    req1 = make_request("1", token_ids + [999], block_size, sha256)
    _, num_computed_tokens = manager.get_computed_blocks(req1)
    assert num_computed_tokens == 4 * block_size


def test_eagle_grouped_swa_siblings_use_same_cache_mask():
    """Grouped SWA siblings must cache the EAGLE lookahead block together."""
    block_size = 8
    swa_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=block_size,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=4 * block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(["swa_main"], swa_spec),
            KVCacheGroupSpec(["swa_mtp"], swa_spec, is_eagle_group=True),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        use_eagle=True,
    )

    token_ids = [i for i in range(9) for _ in range(block_size)]
    req0 = make_request("0", token_ids, block_size, sha256)
    computed_blocks, _ = manager.get_computed_blocks(req0)
    blocks = manager.allocate_slots(
        req0,
        len(token_ids),
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    assert blocks is not None

    pool = manager.block_pool
    assert pool.get_cached_block(req0.block_hashes[4], kv_cache_group_ids=[1, 2])
    assert pool.get_cached_block(req0.block_hashes[8], kv_cache_group_ids=[1, 2])
    manager.free(req0)

    req1 = make_request("1", token_ids + [999], block_size, sha256)
    _, num_computed_tokens = manager.get_computed_blocks(req1)
    assert num_computed_tokens == 8 * block_size


def test_different_block_size():
    block_size = 16
    # full attention and sliding window attention layers have the same page size:
    # (32 tokens/block * float16 token, vs. 16 tokens/block * float32 token)
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=block_size * 2,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["layer2"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=2 * block_size,
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    # 10 blocks of 16 tokens each. Token ids are not strictly aligned for each block.
    common_token_ids = [i for i in range(10) for _ in range(block_size)]

    req0 = make_request("0", common_token_ids, block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert not computed_blocks.blocks[1]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0, 7 * block_size, len(computed_blocks.blocks[0]) * 16, computed_blocks
    )
    assert blocks.get_block_ids() == ([1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11])
    req1 = make_request("1", common_token_ids[: 7 * block_size + 1], block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(computed_blocks.blocks[0]) == 3
    assert len(computed_blocks.blocks[1]) == 6
    assert num_computed_tokens == 6 * 16

    req2 = make_request("2", common_token_ids[: 6 * block_size + 1], block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(computed_blocks.blocks[0]) == 3
    assert len(computed_blocks.blocks[1]) == 6
    assert num_computed_tokens == 6 * 16

    # Evict some blocks to make sliding window cache hit length 5*16
    # But should return 4 * 16 because full attention cache hit length must be
    # a multiple of 32
    manager.block_pool.cached_block_hash_to_block.pop(
        make_block_hash_with_group_id(req1.block_hashes[6], 1), 11
    )
    manager.block_pool.cached_block_hash_to_block.pop(
        make_block_hash_with_group_id(req1.block_hashes[5], 1), 10
    )
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(computed_blocks.blocks[0]) == 2
    assert len(computed_blocks.blocks[1]) == 4
    assert num_computed_tokens == 4 * 16


def test_hybrid_cache_blocks_swa_tail_window_only():
    """Within each lcm-aligned segment, SWA's ``find_longest_cache_hit`` only
    returns the trailing ``ceil((sliding_window - 1) / block_size)`` blocks
    (its right-to-left scan stops once a contiguous match is found). Blocks
    earlier in the segment can never serve a hit, so
    ``HybridKVCacheCoordinator.cache_blocks`` should skip them rather than
    polluting the prefix-cache hash map."""
    block_size = 8
    # Full attn block_size=32, SWA block_size=8, sw=8 -> lcm=32.
    # tail = ceil(7/8) = 1; per_segment = 32/8 = 4.
    # Per-segment template = [F, F, F, T]; only the last SWA block in each
    # 32-token segment ends up in the prefix-cache hash map.
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=4 * block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["layer2"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=block_size,
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    # 8 hash-blocks of 8 tokens (64 tokens, two lcm-aligned segments).
    token_ids = [i for i in range(8) for _ in range(block_size)]
    req = make_request("0", token_ids, block_size, sha256)
    computed_blocks, _ = manager.get_computed_blocks(req)
    blocks = manager.allocate_slots(
        req,
        8 * block_size,
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    assert blocks is not None
    assert len(req.block_hashes) == 8

    pool = manager.block_pool
    # SWA group_id=1: only hash 3 and hash 7 (the last block of each
    # 32-token segment) should be cached. Hashes 0,1,2,4,5,6 cannot serve
    # a hit at any lcm-aligned length, so they must NOT be cached.
    expected_cached = {3, 7}
    for i in range(8):
        cached = pool.get_cached_block(req.block_hashes[i], kv_cache_group_ids=[1])
        if i in expected_cached:
            assert cached is not None, f"SWA hash {i} should be cached"
        else:
            assert cached is None, (
                f"SWA hash {i} cannot serve any lcm-aligned hit; should not be cached"
            )


def test_hybrid_cache_blocks_clamped_to_lcm():
    """HybridKVCacheCoordinator.cache_blocks() clamps to scheduler_block_size.
    Chunks past the last lcm-aligned boundary can never participate in a
    cache hit (find_longest_cache_hit always returns lcm-aligned hits), so
    caching them only pollutes the prefix-cache hash map and keeps blocks
    on the LRU list that could otherwise return to the free pool."""
    block_size = 16
    # Full attn block_size=32, SWA block_size=16 -> lcm=32.
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=block_size * 2,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["layer2"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=2 * block_size,
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    # 7 hash-blocks of 16 tokens (112 tokens). With lcm=32 the clamp truncates
    # to 96 tokens — SWA caches 6 hashes, full-attn caches 3.
    token_ids = [i for i in range(7) for _ in range(block_size)]
    req = make_request("0", token_ids, block_size, sha256)
    computed_blocks, _ = manager.get_computed_blocks(req)
    blocks = manager.allocate_slots(
        req,
        7 * block_size,
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    assert blocks is not None
    assert len(req.block_hashes) == 7

    pool = manager.block_pool
    # SWA group_id=1: hashes 0..5 cached (6 blocks * 16 tokens = 96), hash 6
    # spans tokens [96, 112) past the lcm boundary and must NOT be cached.
    for i in range(6):
        assert (
            pool.get_cached_block(req.block_hashes[i], kv_cache_group_ids=[1])
            is not None
        ), f"SWA hash {i} should be cached"
    assert pool.get_cached_block(req.block_hashes[6], kv_cache_group_ids=[1]) is None, (
        "SWA hash 6 spans tokens past the lcm boundary; should not be cached"
    )


def test_hybrid_local_kv_retention_interval_aligns_in_manager(monkeypatch):
    """Verify fixed intervals retain sparse tails plus the latest replay tail."""
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "64")
    block_size = 8
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=4 * block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["layer2"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=block_size,
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    # The SWA manager uses the configured 64-token interval (a multiple of the
    # 32-token lcm_block_size) as its retention segment. For this 128-token
    # prompt, the retained SWA tails are the 64-token interval boundary, the
    # 96-token replay boundary, and the 128-token interval boundary.
    token_ids = [i for i in range(16) for _ in range(block_size)]
    req = make_request("0", token_ids, block_size, sha256)
    computed_blocks, _ = manager.get_computed_blocks(req)
    blocks = manager.allocate_slots(
        req,
        len(token_ids),
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    assert blocks is not None

    pool = manager.block_pool
    expected_swa_cached = {7, 11, 15}
    for i in range(16):
        cached = pool.get_cached_block(req.block_hashes[i], kv_cache_group_ids=[1])
        if i in expected_swa_cached:
            assert cached is not None, f"SWA hash {i} should be cached"
        else:
            assert cached is None, f"SWA hash {i} should not be cached"


@pytest.mark.parametrize(
    "interval, expected_match",
    [
        # scheduler_block_size is 32 (= lcm(4*8, 8)); 33 is not a multiple of it.
        ("33", "multiple of scheduler_block_size"),
        # A negative multiple (-32 % 32 == 0) must still be rejected explicitly,
        # otherwise it would pass the modulo check and silently degrade to dense.
        ("-32", "non-negative"),
    ],
)
def test_hybrid_local_kv_retention_interval_rejects_invalid(
    monkeypatch, interval, expected_match
):
    """A retention interval that is negative or not a multiple of
    scheduler_block_size errors out at construction time."""
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", interval)
    block_size = 8
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=4 * block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["layer2"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=block_size,
                ),
            ),
        ],
    )
    with pytest.raises(ValueError, match=expected_match):
        make_kv_cache_manager(
            kv_cache_config=kv_cache_config,
            max_model_len=8192,
            enable_caching=True,
            hash_block_size=block_size,
        )


def test_hybrid_local_kv_retention_interval_survives_recycling(monkeypatch):
    """Verify retained local checkpoints are reused after block recycling."""
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "1024")
    hash_block_size = 4
    kv_cache_config = KVCacheConfig(
        num_blocks=800,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                MLAAttentionSpec(
                    block_size=64 * hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.uint8,
                    compress_ratio=4,
                ),
            ),
            KVCacheGroupSpec(
                ["swa"],
                SlidingWindowSpec(
                    block_size=16 * hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=512,
                ),
            ),
            KVCacheGroupSpec(
                ["c128"],
                SlidingWindowSpec(
                    block_size=2 * hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=128,
                ),
            ),
            KVCacheGroupSpec(
                ["c4"],
                SlidingWindowSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=8,
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=4096,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )

    def fill_request(request_id: str, token_offset: int) -> list[int]:
        token_ids = [
            token_offset + i for i in range(1024) for _ in range(hash_block_size)
        ]
        fill_req = make_request(request_id, token_ids, hash_block_size, sha256)
        while fill_req.num_computed_tokens < len(token_ids):
            num_new_tokens = min(512, len(token_ids) - fill_req.num_computed_tokens)
            blocks = manager.allocate_slots(fill_req, num_new_tokens)
            assert blocks is not None
            fill_req.num_computed_tokens += num_new_tokens
        manager.free(fill_req)
        return token_ids

    token_ids = fill_request("fill_0", 0)
    replay_req = make_request("replay", token_ids[:1800], hash_block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(replay_req)
    assert num_computed_tokens == 1024
    assert [len(blocks) for blocks in computed_blocks.blocks] == [4, 16, 128, 256]

    fill_request("fill_1", 100_000)
    replay_req = make_request("replay_again", token_ids[:1800], hash_block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(replay_req)
    assert num_computed_tokens == 1024
    assert [len(blocks) for blocks in computed_blocks.blocks] == [4, 16, 128, 256]


def test_hybrid_local_kv_retention_latest_only_reuses_replay_boundary(monkeypatch):
    """Verify latest-only retention reuses only the replayable prompt boundary."""
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "0")
    block_size = 8
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=4 * block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["layer2"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=block_size,
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    token_ids = [i for i in range(16) for _ in range(block_size)]
    req0 = make_request("0", token_ids, block_size, sha256)
    computed_blocks, _ = manager.get_computed_blocks(req0)
    blocks = manager.allocate_slots(
        req0,
        len(token_ids),
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    assert blocks is not None

    pool = manager.block_pool
    expected_swa_cached = {11}
    for i in range(16):
        cached = pool.get_cached_block(req0.block_hashes[i], kv_cache_group_ids=[1])
        if i in expected_swa_cached:
            assert cached is not None, f"SWA hash {i} should be cached"
        else:
            assert cached is None, f"SWA hash {i} should not be cached"

    manager.free(req0)
    retained_swa_block = pool.get_cached_block(req0.block_hashes[11], [1])
    assert retained_swa_block is not None
    assert retained_swa_block[0].ref_cnt == 0

    req1 = make_request("1", token_ids, block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    # Full prompt hits intentionally recompute the final block for logits, so
    # the longest usable hit is the previous LCM boundary: 96 tokens.
    assert num_computed_tokens == 12 * block_size
    assert len(computed_blocks.blocks[1]) == 12

    shorter_req = make_request("2", token_ids[: 12 * block_size], block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(shorter_req)
    assert num_computed_tokens == 0
    assert len(computed_blocks.blocks[1]) == 0


def test_hybrid_local_kv_retention_mtp_reuses_latest_boundary(monkeypatch):
    """Verify MTP/EAGLE SWA retention keeps the extra proof block.

    EAGLE/MTP lookup matches one additional local block after the returned
    prefix and then drops it. Sparse retention must therefore cache the normal
    local tail at the latest replay boundary plus one extra SWA block.
    """
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "0")
    block_size = 8
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=4 * block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["swa_mtp"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=block_size,
                ),
                is_eagle_group=True,
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        use_eagle=True,
    )

    # 127 tokens: latest replay boundary is floor((127 - 1) / 32) * 32 = 96.
    # The EAGLE/MTP SWA lookup group must cache the local tail ending at
    # 104 tokens, and that tail is two 8-token blocks wide: hashes 11 and 12.
    token_ids = [i for i in range(15) for _ in range(block_size)] + [15] * 7
    req0 = make_request("0", token_ids, block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        req0,
        len(token_ids),
        num_computed_tokens,
        computed_blocks,
    )
    assert blocks is not None

    pool = manager.block_pool
    expected_swa_cached = {11, 12}
    for i in range(15):
        cached = pool.get_cached_block(req0.block_hashes[i], kv_cache_group_ids=[1])
        if i in expected_swa_cached:
            assert cached is not None, f"SWA hash {i} should be cached"
        else:
            assert cached is None, f"SWA hash {i} should not be cached"

    manager.free(req0)

    req1 = make_request("1", token_ids, block_size, sha256)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert num_computed_tokens == 12 * block_size
    assert [len(blocks) for blocks in computed_blocks.blocks] == [3, 12]


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
    # No block popped due to block_id mismatch
    assert cache.pop(key0, 100) is None
    assert cache.get_one_block(key0) is block0
    assert cache.get_one_block(key1) is block1
    assert cache.get_one_block(key2) is None
    # block popped with (key0, block ID 0)
    assert cache.pop(key0, 0) is block0
    assert cache.get_one_block(key0) is None
    assert cache.get_one_block(key1) is block1
    assert cache.get_one_block(key2) is None
    # No block popped due to block_id mismatch
    assert cache.pop(key0, 1) is None
    assert cache.get_one_block(key0) is None
    assert cache.get_one_block(key1) is block1
    assert cache.get_one_block(key2) is None
    # block popped with (key1, block ID 1)
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


def test_can_fit_full_sequence_swa_cap_admits_long_prompt():
    """Hybrid full+SWA model with a pool sized at the startup minimum should
    admit a prompt longer than the SWA cap, because SlidingWindowManager
    recycles blocks during chunked prefill (issue #39734)."""
    block_size = 16
    sliding_window = 4 * block_size  # 64 tokens
    max_num_batched_tokens = 8 * block_size  # 128 tokens
    max_model_len = 64 * block_size  # 1024 tokens — much larger than the SWA cap
    # Startup pool sizing: full demands cdiv(max_model_len, bs) = 64 blocks,
    # SWA demands cdiv(SW-1+max_batched, bs) + 1 = cdiv(191, 16) + 1 = 13.
    # Pool minimum = 64 + 13 = 77; +1 for the null block.
    num_blocks = 64 + 13 + 1

    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["layer_swa"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=sliding_window,
                ),
            ),
        ],
    )

    manager = make_kv_cache_manager(
        config,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_caching=True,
        hash_block_size=block_size,
    )

    # A prompt that is shorter than max_model_len but longer than SW + chunk:
    # cdiv(prompt_len, bs) = 32 blocks. Without the cap, admission would
    # demand 32 (full) + 32 (SWA) = 64 blocks. With the cap, SWA contributes
    # only 13, so total = 32 + 13 = 45 ≤ pool size.
    prompt_len = 32 * block_size
    req = make_request("long", list(range(prompt_len)), block_size, sha256)

    assert (
        manager.allocate_slots(req, block_size, full_sequence_must_fit=True) is not None
    )


def test_can_fit_full_sequence_full_attention_still_gates_oversized():
    """The cap only loosens the SWA group; a prompt that exceeds the
    full-attention pool capacity must still be rejected."""
    block_size = 16
    sliding_window = 4 * block_size
    max_num_batched_tokens = 8 * block_size
    max_model_len = 64 * block_size
    # Provide a tiny pool — even a small prompt should be rejected.
    num_blocks = 5

    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["layer_swa"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=sliding_window,
                ),
            ),
        ],
    )

    manager = make_kv_cache_manager(
        config,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_caching=True,
        hash_block_size=block_size,
    )

    # 16 blocks of full attention demand alone exceeds the 5-block pool.
    prompt_len = 16 * block_size
    req = make_request("oversized", list(range(prompt_len)), block_size, sha256)

    assert manager.allocate_slots(req, block_size, full_sequence_must_fit=True) is None


def test_cache_hit_local_and_external():
    # Regression test for #33775: when a request hits the local prefix cache
    # in one KV cache group and needs external (connector) blocks in another,
    # the external allocation of an earlier group must not evict the local
    # cache-hit blocks of a later group. Otherwise the same physical block can
    # be handed out twice, producing duplicate block IDs / ref_cnt corruption.
    block_size = 16
    kv_cache_config = make_kv_cache_config_hybrid_model(block_size, 31, 100)
    del kv_cache_config.kv_cache_groups[2:]
    req_id = "test"
    manager = make_kv_cache_manager(
        kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        use_eagle=True,
    )

    top_blocks = []
    head = manager.block_pool.free_block_queue.fake_free_list_head
    for _ in range(10):
        top_blocks.append(head.next_free_block)
        head = head.next_free_block
    cache_hit = KVCacheBlocks((top_blocks[:5], top_blocks[5:]))

    manager.allocate_slots(
        make_request(req_id, [0] * (8 * block_size), block_size, sha256),
        16,
        5 * block_size,
        cache_hit,
        0,
        2 * block_size,
    )

    req_blocks = manager.get_blocks(req_id)
    req_block_ids = req_blocks.get_block_ids()
    all_block_ids = req_block_ids[0] + req_block_ids[1]
    assert len(set(all_block_ids)) == len(all_block_ids), "Block IDs are not unique"


def _take_free_blocks(manager: KVCacheManager, num_blocks: int) -> list[KVCacheBlock]:
    """Grab the first ``num_blocks`` blocks at the head of the free queue
    without removing them. These ref_cnt==0 blocks stand in for evictable
    cache-hit blocks left behind by a previous (e.g. preempted) request, and
    sitting at the head guarantees a later group's external ``get_new_blocks``
    would contend for them on unpatched code (issue #33775)."""
    blocks: list[KVCacheBlock] = []
    head = manager.block_pool.free_block_queue.fake_free_list_head
    for _ in range(num_blocks):
        head = head.next_free_block
        blocks.append(head)
    return blocks


def _assert_no_double_allocation(manager: KVCacheManager, req_id: str) -> None:
    """No physical block may be handed out twice across groups, and every
    block referenced by the request must have a live ref_cnt."""
    block_ids = manager.get_blocks(req_id).get_block_ids()
    flat = [block_id for group in block_ids for block_id in group]
    assert len(set(flat)) == len(flat), "Block IDs are not unique across groups"
    null_id = manager.block_pool.null_block.block_id
    for block_id in flat:
        if block_id == null_id:
            continue
        assert manager.block_pool.blocks[block_id].ref_cnt >= 1, (
            f"block {block_id} referenced by the request has ref_cnt 0"
        )


def _two_phase_block_size(manager: KVCacheManager) -> int:
    return manager.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size


def _cross_group_cache_hit(
    manager: KVCacheManager,
    req_id: str,
    num_groups: int,
    local_blocks_per_group: int = 5,
    num_external_blocks: int = 2,
    num_new_blocks: int = 1,
) -> Request:
    """Allocate ``req_id`` with a per-group local prefix hit plus external
    (connector) computed tokens, driving the coordinator's two-phase path.
    Returns the allocated request so callers can free it (e.g. to preempt)."""
    block_size = _two_phase_block_size(manager)
    hit_blocks = _take_free_blocks(manager, num_groups * local_blocks_per_group)
    cache_hit = KVCacheBlocks(
        tuple(
            hit_blocks[i * local_blocks_per_group : (i + 1) * local_blocks_per_group]
            for i in range(num_groups)
        )
    )
    prompt_blocks = local_blocks_per_group + num_external_blocks + num_new_blocks
    request = make_request(
        req_id, [0] * (prompt_blocks * block_size), block_size, sha256
    )
    manager.allocate_slots(
        request,
        num_new_blocks * block_size,
        local_blocks_per_group * block_size,
        cache_hit,
        0,
        num_external_blocks * block_size,
    )
    return request


def _make_two_phase_manager(num_groups: int) -> KVCacheManager:
    assert num_groups in (2, 3)
    block_size = 16
    kv_cache_config = make_kv_cache_config_hybrid_model(block_size, 31, 100)
    del kv_cache_config.kv_cache_groups[num_groups:]
    return make_kv_cache_manager(
        kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        use_eagle=True,
    )


def test_cache_hit_local_and_external_three_groups():
    # Scenario 1 (issue #33775): SWA + full attention with *three* KV cache
    # groups (1 full + 2 sliding-window). A local prefix hit in some groups
    # combined with external (connector) blocks in others must not let one
    # group's external `get_new_blocks` evict another group's not-yet-touched
    # cache-hit blocks, which would hand the same physical block out twice.
    manager = _make_two_phase_manager(num_groups=3)
    _cross_group_cache_hit(manager, "test", num_groups=3)
    _assert_no_double_allocation(manager, "test")


def test_cache_hit_local_and_external_three_groups_preempt_and_reallocate():
    # Scenario 2: the same 3-group hybrid config, but the request is preempted
    # (freed) and then reallocated. After the free, the coordinator must treat
    # the request as new again so external blocks are re-allocated, and the
    # two-phase ordering must still prevent cross-group double allocation when
    # reallocating against the now-evictable cache-hit blocks.
    manager = _make_two_phase_manager(num_groups=3)

    request = _cross_group_cache_hit(manager, "test", num_groups=3)
    _assert_no_double_allocation(manager, "test")

    # Preempt: free the request; its blocks return to the pool (full ones stay
    # cached/evictable) and the coordinator forgets it.
    manager.free(request)
    assert manager.get_blocks("test").get_block_ids() == ([], [], [])

    # Reallocate the same request id against fresh cache-hit blocks taken from
    # the current free-queue head, mirroring a preempted request being
    # scheduled again. Because the request is no longer known, the coordinator
    # re-arms `is_new_request` and re-runs external allocation, which must still
    # not double-allocate across groups.
    _cross_group_cache_hit(manager, "test", num_groups=3)
    _assert_no_double_allocation(manager, "test")
    assert manager.get_blocks("test").get_block_ids() != ([], [], [])


def test_cache_hit_local_and_external_two_groups_preempt_and_reallocate():
    # Scenario 3: the minimal 2-group hybrid config (1 full + 1 sliding-window)
    # exercised through the same preempt -> reallocate cycle as scenario 2.
    manager = _make_two_phase_manager(num_groups=2)

    request = _cross_group_cache_hit(manager, "test", num_groups=2)
    _assert_no_double_allocation(manager, "test")

    manager.free(request)
    assert manager.get_blocks("test").get_block_ids() == ([], [])

    _cross_group_cache_hit(manager, "test", num_groups=2)
    _assert_no_double_allocation(manager, "test")
    assert manager.get_blocks("test").get_block_ids() != ([], [])


def test_swa_free_split_keeps_cached_tail_ahead_of_scratch(monkeypatch):
    """Default path (no retention): freeing an SWA request must place its
    uncached scratch blocks at the front of the free queue (recycled first)
    and keep its cached checkpoint blocks at the back (retained for prefix
    hits). This split is always-on, independent of the retention interval."""
    monkeypatch.delenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", raising=False)
    block_size = 8
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=4 * block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["layer2"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=block_size,
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
    )

    token_ids = [i for i in range(16) for _ in range(block_size)]
    req = make_request("0", token_ids, block_size, sha256)
    computed_blocks, _ = manager.get_computed_blocks(req)
    blocks = manager.allocate_slots(
        req,
        len(token_ids),
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    assert blocks is not None

    swa_manager = manager.coordinator.single_type_managers[1]
    null_block = manager.block_pool.null_block
    cached_ids: set[int] = set()
    uncached_ids: set[int] = set()
    cached_hash_indices: list[int] = []
    for i, block in enumerate(swa_manager.req_to_blocks[req.request_id]):
        if block is null_block:
            continue
        if block.block_hash is None:
            uncached_ids.add(block.block_id)
        else:
            cached_ids.add(block.block_id)
            cached_hash_indices.append(i)
    # The dense default mask caches only the per-segment tails, so a 16-block
    # SWA prompt must produce a mix of retained and scratch blocks.
    assert cached_ids, "expected some retained (cached) SWA tail blocks"
    assert uncached_ids, "expected some scratch (uncached) SWA blocks"

    manager.free(req)

    order = [
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ]
    pos = {bid: i for i, bid in enumerate(order)}
    # Every scratch block is recycled before every retained block.
    assert max(pos[bid] for bid in uncached_ids) < min(pos[bid] for bid in cached_ids)
    # The retained tails survive the free and still serve a prefix-cache hit.
    for i in cached_hash_indices:
        assert (
            manager.block_pool.get_cached_block(
                req.block_hashes[i], kv_cache_group_ids=[1]
            )
            is not None
        )


def _make_pure_swa_manager(block_size, sliding_window, num_blocks=100, **kwargs):
    """Single sliding-window group (UnitaryKVCacheCoordinator)."""
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=sliding_window,
                ),
            ),
        ],
    )
    return make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        **kwargs,
    )


def test_pure_swa_retention_interval_caches_sparse_tails(monkeypatch):
    """Sparse retention must work for a pure-SWA single-group model, not just
    hybrid models: only the per-interval tails plus the latest replay tail are
    cached, and a replay still hits the latest replayable boundary."""
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "64")
    block_size = 16
    manager = _make_pure_swa_manager(block_size, sliding_window=block_size)
    assert type(manager.coordinator).__name__ == "UnitaryKVCacheCoordinator"

    token_ids = [i for i in range(16) for _ in range(block_size)]
    req = make_request("0", token_ids, block_size, sha256)
    computed_blocks, _ = manager.get_computed_blocks(req)
    blocks = manager.allocate_slots(
        req,
        len(token_ids),
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    assert blocks is not None

    pool = manager.block_pool
    cached = {
        i
        for i in range(16)
        if pool.get_cached_block(req.block_hashes[i], kv_cache_group_ids=[0])
        is not None
    }
    # per_segment = 64 / 16 = 4, need = cdiv(16-1, 16) = 1 -> segment tails at
    # i%4==3 -> {3,7,11,15}; latest replay boundary (255//16*16 = 240) -> tail
    # block 14. Crucially this is a strict subset of all 16 blocks: retention
    # is actually sparse for pure SWA (not silently dense).
    assert cached == {3, 7, 11, 14, 15}

    # A replay of the same prompt hits the latest replayable boundary (240).
    replay = make_request("1", token_ids, block_size, sha256)
    _, num_computed = manager.get_computed_blocks(replay)
    assert num_computed == 240


def test_pure_swa_retention_latest_only(monkeypatch):
    """`=0` on a pure-SWA model keeps only the latest replay tail."""
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "0")
    block_size = 16
    manager = _make_pure_swa_manager(block_size, sliding_window=block_size)

    token_ids = [i for i in range(16) for _ in range(block_size)]
    req = make_request("0", token_ids, block_size, sha256)
    computed_blocks, _ = manager.get_computed_blocks(req)
    blocks = manager.allocate_slots(
        req,
        len(token_ids),
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    assert blocks is not None

    pool = manager.block_pool
    cached = {
        i
        for i in range(16)
        if pool.get_cached_block(req.block_hashes[i], kv_cache_group_ids=[0])
        is not None
    }
    # No segment tails (interval 0); only the latest replay tail (block 14).
    assert cached == {14}

    replay = make_request("1", token_ids, block_size, sha256)
    _, num_computed = manager.get_computed_blocks(replay)
    assert num_computed == 240


def test_pure_swa_retention_dense_default_caches_all(monkeypatch):
    """With retention unset, a pure-SWA model must keep the dense behavior:
    every block boundary is a potential hit, so all blocks are cached."""
    monkeypatch.delenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", raising=False)
    block_size = 16
    manager = _make_pure_swa_manager(block_size, sliding_window=block_size)

    token_ids = [i for i in range(16) for _ in range(block_size)]
    req = make_request("0", token_ids, block_size, sha256)
    computed_blocks, _ = manager.get_computed_blocks(req)
    blocks = manager.allocate_slots(
        req,
        len(token_ids),
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    assert blocks is not None

    pool = manager.block_pool
    cached = {
        i
        for i in range(16)
        if pool.get_cached_block(req.block_hashes[i], kv_cache_group_ids=[0])
        is not None
    }
    assert cached == set(range(16))
