# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused MLA index convert + upconvert kernel."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_index_convert_basic():
    """Test basic index conversion without upconvert."""

    # Create simple test case
    _num_tokens = 2
    _num_requests = 2
    _max_blocks = 4
    _topk = 4
    block_size = 64

    req_id = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    block_table = torch.arange(8, dtype=torch.int32, device="cuda").reshape(2, 4)
    token_indices = torch.tensor(
        [[0, 64, -1, 128], [0, 63, -1, 191]], dtype=torch.int32, device="cuda"
    )

    # Call kernel without prefill tracking
    result = torch.ops._C.convert_req_index_to_global_index(
        req_id, block_table, token_indices, block_size, None, None, None, None
    )

    # Verify output shape
    assert result.shape == token_indices.shape

    # Verify conversions (block_table[req][block_id] * block_size + offset)
    # Token 0, index 0: block_table[0][0] * 64 + 0 = 0 * 64 + 0 = 0
    assert result[0, 0].item() == 0
    # Token 0, index 1: block_table[0][1] * 64 + 0 = 1 * 64 + 0 = 64
    assert result[0, 1].item() == 64
    # Token 0, index 2: -1 (invalid)
    assert result[0, 2].item() == -1
    # Token 1, index 0: block_table[1][0] * 64 + 0 = 4 * 64 + 0 = 256
    assert result[1, 0].item() == 256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_index_convert_with_upconvert():
    """Test fused index convert + upconvert for prefill tokens."""

    # Setup
    num_tokens = 4
    _num_requests = 2
    _max_blocks = 8
    _topk = 8
    block_size = 64
    head_dim = 576
    num_kv_slots = _max_blocks * _num_requests * block_size

    req_id = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
    block_table = torch.arange(16, dtype=torch.int32, device="cuda").reshape(2, 8)

    # Create token indices with some duplicates to test deduplication
    token_indices = torch.randint(
        0, 100, (num_tokens, _topk), dtype=torch.int32, device="cuda"
    )

    # Create prefill mask (last 2 tokens are prefill)
    prefill_mask = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
    prefill_mask[2:] = 1

    # Allocate buffers
    prefill_seen = torch.zeros(num_kv_slots, dtype=torch.int32, device="cuda")
    prefill_bf16_workspace = torch.zeros(
        (num_kv_slots, head_dim), dtype=torch.bfloat16, device="cuda"
    )

    # Create mock FP8 cache (656 bytes per slot: 512 fp8 + 16 scales + 128 rope)
    kv_cache = torch.randint(
        0,
        255,
        (_max_blocks * _num_requests, block_size, 656),
        dtype=torch.uint8,
        device="cuda",
    )

    # Call fused kernel
    result = torch.ops._C.convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        block_size,
        prefill_mask,
        prefill_seen,
        prefill_bf16_workspace,
        kv_cache,
    )

    # Verify output shape
    assert result.shape == token_indices.shape

    # Check that some tokens were marked as seen (upconverted)
    assert prefill_seen.sum().item() > 0, "Expected some tokens to be upconverted"

    # Verify workspace has non-zero values where tokens were upconverted
    seen_indices = torch.nonzero(prefill_seen).squeeze()
    if seen_indices.numel() > 0:
        # Check first seen index has non-zero workspace data
        first_seen = seen_indices[0].item()
        workspace_data = prefill_bf16_workspace[first_seen]
        # Should have some non-zero values after upconversion
        assert workspace_data.abs().sum().item() > 0, "Workspace should be populated"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_deduplication():
    """Test that duplicate indices are deduplicated correctly."""

    # Simple case: 1 token with duplicate indices
    num_tokens = 1
    topk = 4
    block_size = 64
    head_dim = 576
    num_kv_slots = 1024

    req_id = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
    block_table = torch.arange(16, dtype=torch.int32, device="cuda").reshape(1, 16)

    # All indices point to same slot (0)
    token_indices = torch.zeros((num_tokens, topk), dtype=torch.int32, device="cuda")

    prefill_mask = torch.ones(num_tokens, dtype=torch.int32, device="cuda")
    prefill_seen = torch.zeros(num_kv_slots, dtype=torch.int32, device="cuda")
    prefill_bf16_workspace = torch.zeros(
        (num_kv_slots, head_dim), dtype=torch.bfloat16, device="cuda"
    )
    kv_cache = torch.zeros((16, block_size, 656), dtype=torch.uint8, device="cuda")

    # Call kernel
    result = torch.ops._C.convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        block_size,
        prefill_mask,
        prefill_seen,
        prefill_bf16_workspace,
        kv_cache,
    )

    # All outputs should be same (global index 0)
    assert (result == 0).all(), "All indices should map to slot 0"

    # Should only upconvert once (deduplication)
    assert prefill_seen[0].item() == 1, "Slot 0 should be marked as seen"
    assert prefill_seen.sum().item() == 1, "Only 1 unique slot should be upconverted"
