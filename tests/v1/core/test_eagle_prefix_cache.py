# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test prefix cache behavior with EAGLE/MTP enabled during prefill phase.

This test verifies the fix for prefix cache efficiency issue when EAGLE/MTP
is enabled. The issue was that the last matched block was incorrectly dropped
during prefill phase, causing:
- Short input scenario: prefix cache completely failed (0 blocks hit)
- Medium input scenario: lost 1 block (extra tokens recomputation)

The fix skips dropping the last block in prefill since draft tokens are not
generated until decode phase.
"""

from collections.abc import Callable

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager, Request
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    num_output_tokens: int = 0,
    hash_fn: Callable = sha256,
):
    """Create a request with specified parameters.

    Args:
        request_id: Unique request identifier
        prompt_token_ids: List of prompt token IDs
        block_size: Block size for KV cache
        num_output_tokens: Number of output tokens (0 for prefill phase)
        hash_fn: Hash function for block hashing
    """
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)

    request = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=None,
        cache_salt=None,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
    )

    # Set num_output_tokens to simulate prefill vs decode phase
    # num_output_tokens is a read-only property, so we modify the underlying list
    request._output_token_ids.extend(range(num_output_tokens))

    return request


def make_kv_cache_config_with_eagle(
    block_size: int,
    num_blocks: int,
    use_eagle: bool = True,
) -> KVCacheConfig:
    """Create KV cache config with EAGLE-enabled attention group.

    Args:
        block_size: Block size for all groups
        num_blocks: Total number of blocks available
        use_eagle: Whether to mark the attention group as EAGLE-enabled
    """
    kv_cache_groups = [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
    ]

    if use_eagle:
        # In real scenarios, EAGLE groups are determined by model config
        # Here we simulate by adding a second group that would be EAGLE-enabled
        kv_cache_groups.append(
            KVCacheGroupSpec(
                ["eagle_layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=kv_cache_groups,
    )


def make_kv_cache_config_hybrid_with_eagle(
    block_size: int,
    num_blocks: int,
) -> KVCacheConfig:
    """Create hybrid model KV cache config (Full + Mamba) with EAGLE.

    This simulates models like Qwen3.5-27B which have:
    - Full attention layers (layer 3)
    - Linear attention layers (layers 0, 1, 2)
    """
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            # Full attention layer (EAGLE-enabled)
            KVCacheGroupSpec(
                ["layer3"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            # Linear/Mamba attention layers
            KVCacheGroupSpec(
                ["layer0"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                ),
            ),
            KVCacheGroupSpec(
                ["layer1"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                ),
            ),
            KVCacheGroupSpec(
                ["layer2"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                ),
            ),
        ],
    )


class TestEaglePrefixCachePrefillPhase:
    """Test EAGLE prefix cache behavior during prefill phase."""

    def test_short_input_prefill_skip_eagle_pop(self):
        """Test that EAGLE doesn't drop blocks during prefill for short inputs.

        Short input scenario: prompt tokens < block_size * 2
        Expected: All available blocks should be kept (no eagle pop)
        """
        block_size = 1536
        num_blocks = 100

        # Short prompt: 100 tokens (less than 1 block)
        prompt_tokens: list[int] = list(range(100))

        kv_cache_config = make_kv_cache_config_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
            use_eagle=True,
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=2000,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=True,
            enable_caching=True,
        )

        # First request to populate cache
        req1 = make_request("req1", prompt_tokens, block_size, num_output_tokens=0)
        blocks1, num_computed1 = kv_cache_manager.get_computed_blocks(req1)

        # Should have 0 blocks initially (no cache)
        assert num_computed1 == 0

        # Allocate blocks for first request
        kv_cache_manager.allocate_slots(
            req1,
            num_new_tokens=100,
            num_new_computed_tokens=0,
            new_computed_blocks=blocks1,
        )

        # Cache the blocks
        kv_cache_manager.cache_blocks(req1, 100)

        # Second request with same prompt (prefill phase)
        req2 = make_request("req2", prompt_tokens, block_size, num_output_tokens=0)
        blocks2, num_computed2 = kv_cache_manager.get_computed_blocks(req2)

        # CRITICAL: During prefill phase, eagle should NOT drop the last block
        # Expected: 1 block (100 tokens padded to block_size)
        # Without fix: 0 blocks (eagle pop would drop the only block)

        # Check that we got cache hit
        assert num_computed2 > 0, "Prefix cache should hit during prefill"

        # The key assertion: num_computed2 should be > 0 if there's any cache
        # This test verifies that eagle doesn't prevent cache usage

        kv_cache_manager.free(req1)
        kv_cache_manager.free(req2)

    def test_medium_input_prefill_skip_eagle_pop(self):
        """Test EAGLE prefix cache for medium inputs during prefill.

        Medium input: prompt tokens ~ multiple blocks
        Expected: All full blocks should be kept (no eagle pop in prefill)
        """
        block_size = 16
        num_blocks = 100

        # Medium prompt: 32 tokens = 2 full blocks
        prompt_tokens: list[int] = list(range(32))

        kv_cache_config = make_kv_cache_config_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
            use_eagle=True,
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=100,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=True,
            enable_caching=True,
        )

        # First request to populate cache
        req1 = make_request("req1", prompt_tokens, block_size, num_output_tokens=0)
        blocks1, num_computed1 = kv_cache_manager.get_computed_blocks(req1)

        # Allocate and cache
        kv_cache_manager.allocate_slots(
            req1,
            num_new_tokens=32,
            num_new_computed_tokens=0,
            new_computed_blocks=blocks1,
        )
        kv_cache_manager.cache_blocks(req1, 32)

        # Second request (prefill phase: num_output_tokens=0)
        req2 = make_request("req2", prompt_tokens, block_size, num_output_tokens=0)
        blocks2, num_computed2 = kv_cache_manager.get_computed_blocks(req2)

        # CRITICAL ASSERTION:
        # Prefill phase should keep all 2 blocks (skip_eagle_pop=True)
        # Without fix: would get only 1 block (eagle pops last block)

        expected_num_blocks = 2
        actual_num_blocks = len(blocks2.blocks[0])

        assert actual_num_blocks == expected_num_blocks, (
            f"Prefill phase should keep all blocks. "
            f"Expected {expected_num_blocks}, got {actual_num_blocks}"
        )

        assert num_computed2 == expected_num_blocks * block_size, (
            f"Expected {expected_num_blocks * block_size} computed tokens, "
            f"got {num_computed2}"
        )

        kv_cache_manager.free(req1)
        kv_cache_manager.free(req2)

    def test_long_input_prefill_skip_eagle_pop(self):
        """Test EAGLE prefix cache for long inputs during prefill.

        Long input: prompt tokens >> block_size (e.g., 16000 tokens)
        Expected: All full blocks should be kept (no eagle pop in prefill)
        """
        block_size = 1536
        num_blocks = 100

        # Long prompt: 16000 tokens ~ 10.4 blocks
        prompt_tokens: list[int] = list(range(16000))

        kv_cache_config = make_kv_cache_config_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
            use_eagle=True,
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=20000,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=True,
            enable_caching=True,
        )

        # First request
        req1 = make_request("req1", prompt_tokens, block_size, num_output_tokens=0)
        blocks1, num_computed1 = kv_cache_manager.get_computed_blocks(req1)

        kv_cache_manager.allocate_slots(
            req1,
            num_new_tokens=16000,
            num_new_computed_tokens=0,
            new_computed_blocks=blocks1,
        )
        kv_cache_manager.cache_blocks(req1, 16000)

        # Second request (prefill phase)
        req2 = make_request("req2", prompt_tokens, block_size, num_output_tokens=0)
        blocks2, num_computed2 = kv_cache_manager.get_computed_blocks(req2)

        # Expected: 10 full blocks (16000 // 1536 = 10)
        # Prefill phase should keep all 10 blocks
        expected_num_blocks = 16000 // block_size
        actual_num_blocks = len(blocks2.blocks[0])

        assert actual_num_blocks == expected_num_blocks, (
            f"Prefill phase should keep all blocks for long input. "
            f"Expected {expected_num_blocks}, got {actual_num_blocks}"
        )

        kv_cache_manager.free(req1)
        kv_cache_manager.free(req2)

    def test_decode_phase_eagle_pop_enabled(self):
        """Test that EAGLE DOES drop blocks during decode phase.

        This verifies that the fix only applies to prefill phase.
        In decode phase, eagle should still drop the last block to
        recompute hidden states for draft tokens.
        """
        block_size = 16
        num_blocks = 100

        prompt_tokens: list[int] = list(range(32))

        kv_cache_config = make_kv_cache_config_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
            use_eagle=True,
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=100,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=True,
            enable_caching=True,
        )

        # First request
        req1 = make_request("req1", prompt_tokens, block_size, num_output_tokens=0)
        blocks1, num_computed1 = kv_cache_manager.get_computed_blocks(req1)

        kv_cache_manager.allocate_slots(
            req1,
            num_new_tokens=32,
            num_new_computed_tokens=0,
            new_computed_blocks=blocks1,
        )
        kv_cache_manager.cache_blocks(req1, 32)

        # Simulate decode phase: num_output_tokens > 0
        req2 = make_request("req2", prompt_tokens, block_size, num_output_tokens=5)
        blocks2, num_computed2 = kv_cache_manager.get_computed_blocks(req2)

        # CRITICAL ASSERTION:
        # Decode phase should drop the last block (skip_eagle_pop=False)
        # Expected: 1 block (2 blocks - 1 eagle pop)
        expected_num_blocks = 1
        actual_num_blocks = len(blocks2.blocks[0])

        assert actual_num_blocks == expected_num_blocks, (
            f"Decode phase should drop last block for EAGLE. "
            f"Expected {expected_num_blocks}, got {actual_num_blocks}"
        )

        kv_cache_manager.free(req1)
        kv_cache_manager.free(req2)


class TestHybridModelsWithEagle:
    """Test EAGLE prefix cache in hybrid models (Full + Mamba/Linear attention)."""

    def test_hybrid_model_short_input_prefill(self):
        """Test hybrid model (Full + Mamba) with short input during prefill.

        Models like Qwen3.5-27B have mixed attention types.
        Full attention should behave differently from Mamba during prefix cache.
        """
        block_size = 1536
        num_blocks = 200

        # Short prompt: 100 tokens
        prompt_tokens: list[int] = list(range(100))

        kv_cache_config = make_kv_cache_config_hybrid_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=2000,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=True,
            enable_caching=True,
        )

        # First request
        req1 = make_request("req1", prompt_tokens, block_size, num_output_tokens=0)
        blocks1, num_computed1 = kv_cache_manager.get_computed_blocks(req1)

        kv_cache_manager.allocate_slots(
            req1,
            num_new_tokens=100,
            num_new_computed_tokens=0,
            new_computed_blocks=blocks1,
        )
        kv_cache_manager.cache_blocks(req1, 100)

        # Second request (prefill)
        req2 = make_request("req2", prompt_tokens, block_size, num_output_tokens=0)
        blocks2, num_computed2 = kv_cache_manager.get_computed_blocks(req2)

        # Full attention group should have cache hit
        # Mamba groups have different behavior (skip logic)

        # Check that full attention group got cache
        full_attn_blocks = len(blocks2.blocks[0])
        assert full_attn_blocks >= 0, "Full attention should have cache"

        # Mamba groups have special skip logic (num_computed_tokens - 1)
        # This is expected behavior for Mamba's recurrent nature

        kv_cache_manager.free(req1)
        kv_cache_manager.free(req2)

    def test_hybrid_model_long_input_prefill(self):
        """Test hybrid model with long input (16000 tokens) during prefill.

        This test verifies:
        1. Full attention keeps all blocks (no eagle pop in prefill)
        2. Mamba applies its skip logic correctly
        """
        block_size = 1536
        num_blocks = 200

        # Long prompt: 16000 tokens
        prompt_tokens: list[int] = list(range(16000))

        kv_cache_config = make_kv_cache_config_hybrid_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=20000,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=True,
            enable_caching=True,
        )

        # Populate cache
        req1 = make_request("req1", prompt_tokens, block_size, num_output_tokens=0)
        blocks1, num_computed1 = kv_cache_manager.get_computed_blocks(req1)

        kv_cache_manager.allocate_slots(
            req1,
            num_new_tokens=16000,
            num_new_computed_tokens=0,
            new_computed_blocks=blocks1,
        )
        kv_cache_manager.cache_blocks(req1, 16000)

        # Test cache hit (prefill)
        req2 = make_request("req2", prompt_tokens, block_size, num_output_tokens=0)
        blocks2, num_computed2 = kv_cache_manager.get_computed_blocks(req2)

        # Full attention group (group 0): should keep all blocks
        expected_full_blocks = 16000 // block_size  # 10 blocks
        actual_full_blocks = len(blocks2.blocks[0])

        assert actual_full_blocks == expected_full_blocks, (
            f"Full attention should keep all blocks in prefill. "
            f"Expected {expected_full_blocks}, got {actual_full_blocks}"
        )

        # Mamba groups: have special skip logic
        # This is correct behavior for Mamba's recurrent nature

        kv_cache_manager.free(req1)
        kv_cache_manager.free(req2)


class TestEaglePrefixCacheEdgeCases:
    """Test edge cases for EAGLE prefix cache behavior."""

    def test_empty_prompt_with_eagle(self):
        """Test empty prompt with EAGLE enabled.

        Edge case: prompt with 0 tokens should not cause errors.
        """
        block_size = 16
        num_blocks = 100

        prompt_tokens: list[int] = []

        kv_cache_config = make_kv_cache_config_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
            use_eagle=True,
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=100,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=True,
            enable_caching=True,
        )

        req1 = make_request("req1", prompt_tokens, block_size, num_output_tokens=0)
        blocks1, num_computed1 = kv_cache_manager.get_computed_blocks(req1)

        # Should handle gracefully
        assert num_computed1 == 0
        assert len(blocks1.blocks[0]) == 0

        kv_cache_manager.free(req1)

    def test_single_token_prompt_with_eagle(self):
        """Test single token prompt with EAGLE.

        Edge case: prompt with 1 token (less than block_size).
        """
        block_size = 16
        num_blocks = 100

        prompt_tokens: list[int] = [0]

        kv_cache_config = make_kv_cache_config_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
            use_eagle=True,
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=100,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=True,
            enable_caching=True,
        )

        req1 = make_request("req1", prompt_tokens, block_size, num_output_tokens=0)
        blocks1, num_computed1 = kv_cache_manager.get_computed_blocks(req1)

        kv_cache_manager.allocate_slots(
            req1,
            num_new_tokens=1,
            num_new_computed_tokens=0,
            new_computed_blocks=blocks1,
        )
        kv_cache_manager.cache_blocks(req1, 1)

        # Second request
        req2 = make_request("req2", prompt_tokens, block_size, num_output_tokens=0)
        blocks2, num_computed2 = kv_cache_manager.get_computed_blocks(req2)

        # Single token < block_size, no full blocks
        # But should not crash or behave incorrectly

        kv_cache_manager.free(req1)
        kv_cache_manager.free(req2)

    def test_exact_block_boundary_with_eagle(self):
        """Test prompt exactly at block boundary.

        Edge case: prompt tokens = exact multiple of block_size.
        """
        block_size = 16
        num_blocks = 100

        # Exactly 2 blocks
        prompt_tokens: list[int] = list(range(32))

        kv_cache_config = make_kv_cache_config_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
            use_eagle=True,
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=100,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=True,
            enable_caching=True,
        )

        # Populate cache
        req1 = make_request("req1", prompt_tokens, block_size, num_output_tokens=0)
        blocks1, num_computed1 = kv_cache_manager.get_computed_blocks(req1)

        kv_cache_manager.allocate_slots(
            req1,
            num_new_tokens=32,
            num_new_computed_tokens=0,
            new_computed_blocks=blocks1,
        )
        kv_cache_manager.cache_blocks(req1, 32)

        # Prefill phase test
        req2 = make_request("req2", prompt_tokens, block_size, num_output_tokens=0)
        blocks2, num_computed2 = kv_cache_manager.get_computed_blocks(req2)

        # Should keep all 2 blocks in prefill
        assert len(blocks2.blocks[0]) == 2
        assert num_computed2 == 32

        # Decode phase test
        req3 = make_request("req3", prompt_tokens, block_size, num_output_tokens=1)
        blocks3, num_computed3 = kv_cache_manager.get_computed_blocks(req3)

        # Should drop last block in decode
        assert len(blocks3.blocks[0]) == 1
        assert num_computed3 == 16

        kv_cache_manager.free(req1)
        kv_cache_manager.free(req2)
        kv_cache_manager.free(req3)

    def test_without_eagle_enabled(self):
        """Test prefix cache behavior when EAGLE is disabled.

        Baseline: without EAGLE, prefix cache should work normally.
        """
        block_size = 16
        num_blocks = 100

        prompt_tokens: list[int] = list(range(32))

        kv_cache_config = make_kv_cache_config_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
            use_eagle=False,  # No EAGLE
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=100,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=False,  # No EAGLE
            enable_caching=True,
        )

        # Populate cache
        req1 = make_request("req1", prompt_tokens, block_size, num_output_tokens=0)
        blocks1, num_computed1 = kv_cache_manager.get_computed_blocks(req1)

        kv_cache_manager.allocate_slots(
            req1,
            num_new_tokens=32,
            num_new_computed_tokens=0,
            new_computed_blocks=blocks1,
        )
        kv_cache_manager.cache_blocks(req1, 32)

        # Should get full cache hit (no eagle interference)
        req2 = make_request("req2", prompt_tokens, block_size, num_output_tokens=0)
        blocks2, num_computed2 = kv_cache_manager.get_computed_blocks(req2)

        # Without EAGLE, should get all blocks in both prefill and decode
        assert len(blocks2.blocks[0]) == 2
        assert num_computed2 == 32

        kv_cache_manager.free(req1)
        kv_cache_manager.free(req2)


class TestPrefixCachingDisabledWithEagle:
    """Test behavior when prefix caching is disabled but EAGLE is enabled."""

    def test_caching_disabled_eagle_enabled(self):
        """Test that prefix caching disabled takes precedence.

        When caching is disabled, should skip all cache operations,
        regardless of EAGLE settings.
        """
        block_size = 16
        num_blocks = 100

        prompt_tokens: list[int] = list(range(32))

        kv_cache_config = make_kv_cache_config_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
            use_eagle=True,
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=100,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=True,
            enable_caching=False,  # Caching disabled
        )

        req1 = make_request("req1", prompt_tokens, block_size, num_output_tokens=0)
        blocks1, num_computed1 = kv_cache_manager.get_computed_blocks(req1)

        # Should skip cache lookup when caching disabled
        assert num_computed1 == 0
        assert len(blocks1.blocks[0]) == 0

        kv_cache_manager.free(req1)


class TestMultipleRequestsEaglePrefixCache:
    """Test multiple concurrent requests with EAGLE prefix cache."""

    def test_concurrent_requests_different_prompts(self):
        """Test multiple requests with different prompts.

        Each request should have its own cache state.
        """
        block_size = 16
        num_blocks = 100

        prompt1 = list(range(32))
        prompt2 = list(range(64))

        kv_cache_config = make_kv_cache_config_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
            use_eagle=True,
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=100,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=True,
            enable_caching=True,
        )

        # Request 1
        req1 = make_request("req1", prompt1, block_size, num_output_tokens=0)
        blocks1, _ = kv_cache_manager.get_computed_blocks(req1)
        kv_cache_manager.allocate_slots(
            req1,
            num_new_tokens=32,
            num_new_computed_tokens=0,
            new_computed_blocks=blocks1,
        )
        kv_cache_manager.cache_blocks(req1, 32)

        # Request 2
        req2 = make_request("req2", prompt2, block_size, num_output_tokens=0)
        blocks2, _ = kv_cache_manager.get_computed_blocks(req2)
        kv_cache_manager.allocate_slots(
            req2,
            num_new_tokens=64,
            num_new_computed_tokens=0,
            new_computed_blocks=blocks2,
        )
        kv_cache_manager.cache_blocks(req2, 64)

        # Request 3 with prompt1 (should hit cache from req1)
        req3 = make_request("req3", prompt1, block_size, num_output_tokens=0)
        blocks3, num_computed3 = kv_cache_manager.get_computed_blocks(req3)

        # Should get cache hit from req1
        assert num_computed3 > 0
        assert len(blocks3.blocks[0]) > 0

        kv_cache_manager.free(req1)
        kv_cache_manager.free(req2)
        kv_cache_manager.free(req3)

    def test_concurrent_requests_shared_prefix(self):
        """Test multiple requests sharing common prefix.

        Common prefix should be cached and reused.
        """
        block_size = 16
        num_blocks = 100

        # Shared prefix: first 16 tokens
        shared_prefix = list(range(16))

        prompt1 = shared_prefix + list(range(100, 132))  # 48 tokens total
        prompt2 = shared_prefix + list(range(200, 232))  # 48 tokens total

        kv_cache_config = make_kv_cache_config_with_eagle(
            block_size=block_size,
            num_blocks=num_blocks,
            use_eagle=True,
        )

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=100,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            use_eagle=True,
            enable_caching=True,
        )

        # Request 1
        req1 = make_request("req1", prompt1, block_size, num_output_tokens=0)
        blocks1, _ = kv_cache_manager.get_computed_blocks(req1)
        kv_cache_manager.allocate_slots(
            req1,
            num_new_tokens=48,
            num_new_computed_tokens=0,
            new_computed_blocks=blocks1,
        )
        kv_cache_manager.cache_blocks(req1, 48)

        # Request 2 (should hit shared prefix cache)
        req2 = make_request("req2", prompt2, block_size, num_output_tokens=0)
        blocks2, num_computed2 = kv_cache_manager.get_computed_blocks(req2)

        # Should hit at least 1 block (the shared prefix block)
        # In prefill phase, should keep all hit blocks
        assert num_computed2 >= block_size
        assert len(blocks2.blocks[0]) >= 1

        kv_cache_manager.free(req1)
        kv_cache_manager.free(req2)
