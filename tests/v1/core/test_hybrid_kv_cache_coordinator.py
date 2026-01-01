# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test HybridKVCacheCoordinator with multiple sliding window groups."""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
):
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=17),
        pooling_params=None,
        eos_token_id=100,
        lora_request=None,
        cache_salt=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def make_kv_cache_config_multi_sliding_window(
    block_size: int,
    num_blocks: int,
    sliding_windows: list[int],
) -> KVCacheConfig:
    """
    Create a KVCacheConfig with one full attention group and multiple
    sliding window groups with different window sizes.
    """
    groups = [
        KVCacheGroupSpec(
            ["full_attn_layer"],
            FullAttentionSpec(block_size, 1, 1, torch.float32),
        )
    ]
    for i, sw in enumerate(sliding_windows):
        groups.append(
            KVCacheGroupSpec(
                [f"sw_layer_{i}"],
                SlidingWindowSpec(block_size, 1, 1, torch.float32, sliding_window=sw),
            )
        )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )


class TestHybridKVCacheCoordinatorMultipleSlidingWindows:
    def test_verify_and_sort_multiple_sliding_windows(self):
        block_size = 16
        kv_cache_config = make_kv_cache_config_multi_sliding_window(
            block_size=block_size,
            num_blocks=100,
            sliding_windows=[256, 512, 128],
        )

        manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=8192,
            enable_caching=True,
            hash_block_size=block_size,
        )

        coordinator = manager.coordinator

        assert coordinator.full_attention_group_ids == [0]

        # Verify sliding window groups are sorted by window size (descending)
        # Original: [1, 2, 3] with windows [256, 512, 128]
        # Sorted:   [2, 1, 3] with windows [512, 256, 128]
        assert coordinator.sliding_window_group_ids == [2, 1, 3]

        window_sizes = [
            spec.sliding_window for spec in coordinator.sliding_window_specs
        ]
        assert window_sizes == [512, 256, 128], (
            "Specs should be sorted by window size descending"
        )

    def test_cache_hit_multiple_sliding_windows(self):
        block_size = 16
        kv_cache_config = make_kv_cache_config_multi_sliding_window(
            block_size=block_size,
            num_blocks=100,
            sliding_windows=[512, 256],
        )

        manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=8192,
            enable_caching=True,
            hash_block_size=block_size,
        )
        common_token_ids = [i for i in range(6) for _ in range(block_size)]
        req0 = make_request("0", common_token_ids, block_size)
        computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
        assert not computed_blocks.blocks[0]
        assert num_computed_tokens == 0

        blocks = manager.allocate_slots(
            req0,
            len(common_token_ids),
            len(computed_blocks.blocks[0]) * block_size,
            computed_blocks,
        )
        assert blocks is not None

        manager.free(req0)
        req1 = make_request("1", common_token_ids, block_size)
        computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)

        assert num_computed_tokens == 80
        assert len(computed_blocks.blocks[0]) == 5
        assert len(computed_blocks.blocks[1]) == 5
        assert len(computed_blocks.blocks[2]) == 5

    def test_partial_cache_hit_different_sliding_windows(self):
        block_size = 16
        kv_cache_config = make_kv_cache_config_multi_sliding_window(
            block_size=block_size,
            num_blocks=50,
            sliding_windows=[64, 32],
        )

        manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=8192,
            enable_caching=True,
            hash_block_size=block_size,
        )

        common_token_ids = [i for i in range(5) for _ in range(block_size)]

        req0 = make_request("0", common_token_ids, block_size)
        computed_blocks, _ = manager.get_computed_blocks(req0)
        manager.allocate_slots(
            req0,
            len(common_token_ids),
            len(computed_blocks.blocks[0]) * block_size,
            computed_blocks,
        )

        block_hashes = req0.block_hashes
        assert len(block_hashes) == 5

        manager.free(req0)

        req1 = make_request("1", common_token_ids, block_size)
        computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)

        assert num_computed_tokens == 64
        assert len(computed_blocks.blocks[0]) == 4
        assert len(computed_blocks.blocks[1]) == 4
        assert len(computed_blocks.blocks[2]) == 4
        manager.free(req1)

        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

        # Evict block[1] from SW-32 (group 2)
        # group 1 = SW-64, group 2 = SW-32
        hash_to_evict = make_block_hash_with_group_id(block_hashes[2], 2)
        manager.block_pool.cached_block_hash_to_block._cache.pop(hash_to_evict, None)

        req2 = make_request("2", common_token_ids, block_size)
        computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)

        assert num_computed_tokens == 32, f"Expected 16, got {num_computed_tokens}"
        assert len(computed_blocks.blocks[0]) == 2, (
            "Full Attention should have 3 blocks"
        )
        assert len(computed_blocks.blocks[1]) == 2, "SW-64 should have 3 blocks"
        assert len(computed_blocks.blocks[2]) == 2, "SW-32 should have 3 blocks"
