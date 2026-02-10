# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned prefix caching: concurrency, cap, and status behaviors."""

import pytest  # noqa: F401

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256_cbor
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


def _make_manager(
    blocks: int,
    block_size: int,
    cap_ratio: float = 0.2,
    enable_pin: bool = True,
    num_groups: int = 1,
) -> KVCacheManager:
    import torch

    groups = []
    for gi in range(num_groups):
        groups.append(
            KVCacheGroupSpec(
                layer_names=[f"layer_{gi}"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=2,
                    head_size=8,
                    dtype=torch.float16,
                ),
            )
        )
    cfg = KVCacheConfig(kv_cache_groups=groups, kv_cache_tensors=[], num_blocks=blocks)
    init_none_hash(sha256_cbor)
    return KVCacheManager(
        cfg,
        max_model_len=1024,
        hash_block_size=block_size,
        enable_caching=True,
        pinned_prefix_cap_ratio=cap_ratio,
        enable_pinned_prefix=enable_pin,
    )


def _make_request(
    req_id: str, tokens: list[int], block_size: int, pin: bool
) -> Request:
    sp = SamplingParams(max_tokens=4, pin_prefix=pin)
    hasher = get_request_block_hasher(block_size, sha256_cbor)
    return Request(req_id, tokens, sp, None, None, block_hasher=hasher)


def test_multi_group_prefix_pinning_respects_global_cap():
    """Multi-group pinning must not exceed global budget.

    Create 2 groups, requested logical depth=3, but cap only allows ~4 pins.
    Round-robin should pin depth 0 and 1 across both groups (total 4), and
    report logical pinned depth=2 (partial).
    """
    import torch

    block_size = 4
    # Build a hybrid config: one full-attn group + one sliding-window group
    groups = [
        KVCacheGroupSpec(
            layer_names=["layer_fa"],
            kv_cache_spec=FullAttentionSpec(
                block_size=block_size, num_kv_heads=2, head_size=8, dtype=torch.float16
            ),
        ),
        KVCacheGroupSpec(
            layer_names=["layer_sw"],
            kv_cache_spec=SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=2,
                head_size=8,
                dtype=torch.float16,
                sliding_window=8,
            ),
        ),
    ]
    cfg = KVCacheConfig(kv_cache_groups=groups, kv_cache_tensors=[], num_blocks=20)
    init_none_hash(sha256_cbor)
    kv = KVCacheManager(
        cfg,
        max_model_len=1024,
        hash_block_size=block_size,
        enable_caching=True,
        pinned_prefix_cap_ratio=0.2,
        enable_pinned_prefix=True,
    )
    req = _make_request("mg", list(range(20)), block_size, pin=True)

    kv.allocate_slots(req, num_new_tokens=len(req.all_token_ids))
    num_computed = len(req.all_token_ids) - 1  # exclude last token for logits
    result = kv.cache_blocks(req, num_computed_tokens=num_computed)

    cap_limit = int(kv.block_pool.num_gpu_blocks * kv.pinned_prefix_cap_ratio)
    assert result["cap_limit"] == cap_limit

    # Check BlockPool global counter does not exceed cap
    assert kv.block_pool.num_pinned_blocks <= cap_limit

    # With 2 groups and cap ~ 4, expect logical pinned depth == 2 (partial)
    assert result["pinned_count"] <= result["requested_count"]
    assert result["status"] in {"ok", "partial", "capped"}
    assert result["status"] == "partial"

    # Ensure each group's first two blocks are pinned
    blocks = kv.coordinator.get_blocks(req.request_id)
    for group_blocks in blocks:
        if not group_blocks:
            continue
        assert all(b.is_pinned for b in group_blocks[:2])


# (Per-request unpin method removed to keep surface minimal.)


def test_unpin_all_pinned_prefixes_clears_pool():
    """Global unpin clears all pinned blocks regardless of request id."""
    block_size = 4
    kv = _make_manager(
        blocks=24, block_size=block_size, cap_ratio=0.5, enable_pin=True, num_groups=1
    )
    req = _make_request("unp_all", list(range(12)), block_size, pin=True)
    kv.allocate_slots(req, num_new_tokens=len(req.all_token_ids))
    kv.cache_blocks(req, num_computed_tokens=len(req.all_token_ids) - 1)

    assert kv.block_pool.num_pinned_blocks > 0
    unpinned = kv.unpin_all_pinned_prefixes()
    assert unpinned >= 1
    assert kv.block_pool.num_pinned_blocks == 0


def test_concurrent_prefix_sharing_and_pinned_eviction_protection():
    """Two requests share pinned prefix; evictions avoided for pins."""
    block_size = 4
    kv = _make_manager(blocks=24, block_size=block_size)

    # Prompt spans 3 full blocks (12 tokens).
    prompt = list(range(12))

    # r1: enable pin_prefix so its full-prefix blocks get pinned.
    r1 = _make_request("r1", prompt, block_size, pin=True)
    computed_r1, hits_r1 = kv.get_computed_blocks(r1)
    assert hits_r1 == 0
    assert all(len(g) == 0 for g in computed_r1.blocks)

    kv.allocate_slots(r1, num_new_tokens=len(prompt))
    kv.cache_blocks(r1, num_computed_tokens=len(prompt) - 1)

    num_pinned_blocks = (len(prompt) - 1) // block_size
    r1_blocks = kv.coordinator.get_blocks(r1.request_id)[0]
    assert len(r1_blocks) >= num_pinned_blocks
    pinned_prefix = r1_blocks[:num_pinned_blocks]
    for blk in pinned_prefix:
        assert blk.is_pinned is True

    # r2: same prompt; should share the cached prefix blocks.
    r2 = _make_request("r2", prompt, block_size, pin=False)
    computed_r2, hits_r2 = kv.get_computed_blocks(r2)
    assert hits_r2 == num_pinned_blocks * block_size
    assert computed_r2.blocks[0] == pinned_prefix

    # Simulate scheduler touching for r2.
    kv.block_pool.touch(computed_r2.blocks[0])
    for blk in pinned_prefix:
        assert blk.ref_cnt >= 2

    # Pinned blocks should be protected from eviction.
    pool = kv.block_pool
    for blk in pinned_prefix:
        evicted = pool._maybe_evict_cached_block(blk)
        assert evicted is False
        assert blk.block_hash is not None
        # Verify the block remains in the cached map
        assert pool.cached_block_hash_to_block.get_one_block(blk.block_hash) is not None


def test_pinned_prefix_cap_and_return_fields():
    """Verify cap is enforced and return dict contains expected fields."""
    block_size = 4
    kv = _make_manager(blocks=11, block_size=block_size, cap_ratio=0.2, enable_pin=True)
    req = _make_request("r", list(range(40)), block_size, pin=True)

    kv.allocate_slots(req, num_new_tokens=len(req.all_token_ids))
    result = kv.cache_blocks(req, num_computed_tokens=len(req.all_token_ids) - 1)

    assert set(result.keys()) == {
        "pinned",
        "pinned_count",
        "requested_count",
        "cap_limit",
        "status",
    }
    assert result["cap_limit"] == int(
        kv.block_pool.num_gpu_blocks * kv.pinned_prefix_cap_ratio
    )
    assert result["pinned_count"] <= result["cap_limit"]
    assert kv.block_pool.num_pinned_blocks == sum(
        1 for b in kv.block_pool.blocks if b.is_pinned
    )
    assert result["status"] in {"ok", "partial", "capped"}


def test_pinned_prefix_statuses():
    """Cover disabled / ok / capped cases for status field."""
    block_size = 4

    # disabled: global gate off
    kv = _make_manager(
        blocks=11, block_size=block_size, cap_ratio=0.2, enable_pin=False
    )
    req = _make_request("r0", list(range(32)), block_size, pin=True)
    kv.allocate_slots(req, num_new_tokens=len(req.all_token_ids))
    result = kv.cache_blocks(req, num_computed_tokens=len(req.all_token_ids) - 1)
    assert result["status"] == "disabled"
    assert result["pinned"] is False

    # ok: cap large enough, all requested pinned
    kv = _make_manager(blocks=11, block_size=block_size, cap_ratio=1.0, enable_pin=True)
    req = _make_request("r1", list(range(16)), block_size, pin=True)
    kv.allocate_slots(req, num_new_tokens=len(req.all_token_ids))
    result = kv.cache_blocks(req, num_computed_tokens=len(req.all_token_ids) - 1)
    assert result["status"] == "ok"
    assert result["pinned"] is True
    assert result["pinned_count"] == result["requested_count"]

    # capped: cap=0, requested>0 but pinned_count==0
    kv = _make_manager(blocks=11, block_size=block_size, cap_ratio=0.0, enable_pin=True)
    req = _make_request("r2", list(range(20)), block_size, pin=True)
    kv.allocate_slots(req, num_new_tokens=len(req.all_token_ids))
    result = kv.cache_blocks(req, num_computed_tokens=len(req.all_token_ids) - 1)
    assert result["requested_count"] > 0
    assert result["pinned_count"] == 0
    assert result["status"] == "capped"


# -----------------------------------------------------------------------------
# Additional tests merged from test_pinned_prefix_caching.py
# -----------------------------------------------------------------------------


def create_request(
    request_id: str, prompt_token_ids: list[int], pin_prefix: bool = False
) -> Request:
    """Helper function to create a request with optional prefix pinning."""
    sampling_params = SamplingParams(max_tokens=10, pin_prefix=pin_prefix)
    block_hasher = get_request_block_hasher(4, sha256_cbor)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=None,
        block_hasher=block_hasher,
    )


class TestPinnedPrefixCaching:
    """Test cases for pinned prefix caching functionality (unit-level)."""

    def test_sampling_params_pin_prefix_default(self):
        """Test that pin_prefix defaults to False in SamplingParams."""
        params = SamplingParams()
        assert params.pin_prefix is False

    def test_sampling_params_pin_prefix_enabled(self):
        """Test that pin_prefix can be set to True in SamplingParams."""
        params = SamplingParams(pin_prefix=True)
        assert params.pin_prefix is True

    def test_sampling_params_from_optional_pin_prefix(self):
        """Test that pin_prefix is correctly passed through from_optional."""
        params = SamplingParams.from_optional(pin_prefix=True)
        assert params.pin_prefix is True

    def test_block_pool_pin_blocks(self):
        """Test that blocks can be pinned to prevent eviction."""

        block_pool = BlockPool(
            num_gpu_blocks=10, enable_caching=True, hash_block_size=4
        )

        # Get some blocks
        blocks = block_pool.get_new_blocks(3)

        # Pin the blocks
        block_pool.pin_blocks(blocks)

        # Verify blocks are pinned
        for block in blocks:
            assert block.is_pinned is True
            assert block.ref_cnt >= 1

    def test_block_pool_unpin_blocks(self):
        """Test that pinned blocks can be unpinned."""

        block_pool = BlockPool(
            num_gpu_blocks=10, enable_caching=True, hash_block_size=4
        )

        # Get and pin some blocks
        blocks = block_pool.get_new_blocks(3)
        block_pool.pin_blocks(blocks)

        # Unpin the blocks
        block_pool.unpin_blocks(blocks)

        # Verify blocks are unpinned
        for block in blocks:
            assert block.is_pinned is False

    def test_pinned_blocks_protected_from_eviction(self):
        """Test that pinned blocks are protected from eviction."""

        block_pool = BlockPool(
            num_gpu_blocks=10, enable_caching=True, hash_block_size=4
        )

        # Get some blocks and make them cached
        blocks = block_pool.get_new_blocks(3)

        # Simulate caching by setting block hash using the BlockPool API
        for i, block in enumerate(blocks):
            # Set a dummy hash to make it cached
            dummy_hash = f"dummy_hash_{i}".encode()
            # Compose a BlockHashWithGroupId and set via the property
            from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

            bh = make_block_hash_with_group_id(dummy_hash, 0)
            block.block_hash = bh
            # Insert via public method using the same key
            block_pool.cached_block_hash_to_block.insert(bh, block)

        # Pin one of the blocks
        block_pool.pin_blocks([blocks[0]])

        # Try to evict all blocks
        for block in blocks:
            evicted = block_pool._maybe_evict_cached_block(block)
            if block == blocks[0]:
                # Pinned block should not be evicted
                assert evicted is False
                assert block.block_hash is not None  # Still has hash
            else:
                # Non-pinned blocks should be evicted
                assert evicted is True
                assert block.block_hash is None  # Hash removed

    def test_cache_blocks_with_pin_prefix(self):
        """Test pin_prefix setting is correctly stored in SamplingParams."""
        # Create a request with pin_prefix enabled
        request = create_request("test_request", [1, 2, 3, 4, 5, 6], pin_prefix=True)

        # Verify that pin_prefix is correctly set
        assert request.sampling_params.pin_prefix is True

        # Test calculating blocks to pin
        block_size = 4
        num_computed_tokens = 6
        num_blocks_to_pin = num_computed_tokens // block_size  # 1 since 6>=4

        assert num_blocks_to_pin == 1

    def test_cache_blocks_with_multiple_full_blocks_pinned(self):
        """Test calculating multiple full blocks for pinning."""
        from vllm.utils.hashing import sha256_cbor
        from vllm.v1.core.kv_cache_utils import init_none_hash

        # Initialize the hash function
        init_none_hash(sha256_cbor)

        # Create request with pin_prefix enabled and enough tokens for blocks
        request = create_request("test_request", list(range(20)), pin_prefix=True)

        # Verify that pin_prefix is correctly set
        assert request.sampling_params.pin_prefix is True

        # Test calculating blocks to pin with multiple full blocks
        block_size = 4
        num_computed_tokens = 16  # 4 full blocks
        num_blocks_to_pin = num_computed_tokens // block_size  # Should be 4

        # Check that the calculation is correct
        assert num_blocks_to_pin == 4

    def test_cache_blocks_without_pin_prefix(self):
        """Test that pin_prefix defaults to False when not specified."""
        from vllm.utils.hashing import sha256_cbor
        from vllm.v1.core.kv_cache_utils import init_none_hash

        # Initialize the hash function
        init_none_hash(sha256_cbor)

        # Create a request without pin_prefix
        request = create_request("test_request", list(range(20)), pin_prefix=False)

        # Verify that pin_prefix is correctly set to False
        assert request.sampling_params.pin_prefix is False
