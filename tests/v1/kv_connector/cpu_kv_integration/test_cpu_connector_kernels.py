# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.cpu_connector import (
    d2h_page_copy, h2d_copy_leading_tokens, h2d_copy_trailing_tokens,
    h2d_page_copy)


@pytest.fixture
def device_tensors():
    """Create sample device tensors for testing."""
    # Create tensors with shape (2, num_blocks, page_size, head_size,
    # hidden_size)
    num_blocks = 4
    page_size = 16
    head_size = 8
    hidden_size = 128

    # Initialize with unique values for each position
    k_tensor = torch.arange(num_blocks * page_size * head_size * hidden_size,
                            dtype=torch.float32,
                            device='cuda')
    k_tensor = k_tensor.reshape(num_blocks, page_size, head_size, hidden_size)

    v_tensor = k_tensor + 1000  # Different values for v

    # Stack k and v tensors
    kv_tensor = torch.stack([k_tensor, v_tensor], dim=0)
    return kv_tensor


@pytest.fixture
def host_buffer():
    """Create host buffer for testing."""
    # Create buffer with same dimensions as device tensor but fewer blocks
    num_blocks = 2  # Smaller than device tensor
    page_size = 16
    head_size = 8
    hidden_size = 128

    k_buffer = torch.zeros(num_blocks * page_size * head_size * hidden_size,
                           dtype=torch.float32)
    k_buffer = k_buffer.reshape(num_blocks, page_size, head_size, hidden_size)

    v_buffer = torch.zeros_like(k_buffer)

    # Stack k and v buffers
    kv_buffer = torch.stack([k_buffer, v_buffer], dim=0)
    return kv_buffer


def test_d2h_page_copy(device_tensors, host_buffer):
    """Test device to host copy operation."""
    # Copy blocks 1 and 3 from device to host
    block_ids = [1, 3]

    d2h_page_copy(device_tensors, host_buffer, block_ids)

    # Verify copied data
    for i, block_id in enumerate(block_ids):
        # Check key tensor
        assert torch.allclose(host_buffer[0, i].cpu(),
                              device_tensors[0, block_id].cpu())
        # Check value tensor
        assert torch.allclose(host_buffer[1, i].cpu(),
                              device_tensors[1, block_id].cpu())


def test_h2d_copy_leading_tokens():
    """Test copying leading tokens from host to device."""
    # Create sample tensors
    page_size = 16
    head_size = 8
    hidden_size = 128

    src_buffer = torch.ones((2, 1, page_size, head_size, hidden_size),
                            dtype=torch.float32)
    # Initialize destination with a known pattern
    dst_layer = torch.full((2, 1, page_size, head_size, hidden_size),
                           fill_value=2.0,
                           dtype=torch.float32,
                           device='cuda')

    # Copy first 8 tokens (half of page_size)
    end_position = 8
    h2d_copy_leading_tokens(src_buffer,
                            dst_layer,
                            src_block_id=0,
                            dst_block_id=0,
                            end_position_in_block=end_position)

    # Verify first 8 tokens were copied
    assert torch.allclose(dst_layer[0, 0, :end_position].cpu(),
                          src_buffer[0, 0, :end_position])
    assert torch.allclose(dst_layer[1, 0, :end_position].cpu(),
                          src_buffer[1, 0, :end_position])

    # Verify remaining tokens are unchanged (should still be 2.0)
    expected_unchanged = torch.full(
        (page_size - end_position, head_size, hidden_size),
        fill_value=2.0,
        dtype=torch.float32)
    assert torch.allclose(dst_layer[0, 0, end_position:].cpu(),
                          expected_unchanged)
    assert torch.allclose(dst_layer[1, 0, end_position:].cpu(),
                          expected_unchanged)


def test_h2d_copy_trailing_tokens():
    """Test copying trailing tokens from host to device."""
    # Create sample tensors
    page_size = 16
    head_size = 8
    hidden_size = 128

    src_buffer = torch.ones((2, 1, page_size, head_size, hidden_size),
                            dtype=torch.float32)
    # Initialize destination with a known pattern
    dst_layer = torch.full((2, 1, page_size, head_size, hidden_size),
                           fill_value=2.0,
                           dtype=torch.float32,
                           device='cuda')

    # Copy last 8 tokens (half of page_size)
    start_position = 8
    h2d_copy_trailing_tokens(src_buffer,
                             dst_layer,
                             src_block_id=0,
                             dst_block_id=0,
                             start_position_in_block=start_position)

    # Verify last 8 tokens were copied
    assert torch.allclose(dst_layer[0, 0, start_position:].cpu(),
                          src_buffer[0, 0, start_position:])
    assert torch.allclose(dst_layer[1, 0, start_position:].cpu(),
                          src_buffer[1, 0, start_position:])

    # Verify leading tokens are unchanged (should still be 2.0)
    expected_unchanged = torch.full((start_position, head_size, hidden_size),
                                    fill_value=2.0,
                                    dtype=torch.float32)
    assert torch.allclose(dst_layer[0, 0, :start_position].cpu(),
                          expected_unchanged)
    assert torch.allclose(dst_layer[1, 0, :start_position].cpu(),
                          expected_unchanged)


def test_h2d_page_copy():
    """Test host to device page copy operation."""
    # Create sample tensors
    num_blocks = 4
    page_size = 16
    head_size = 8
    hidden_size = 128
    block_size = page_size

    src_buffer = torch.ones((2, num_blocks, page_size, head_size, hidden_size),
                            dtype=torch.float32)
    # Initialize destination with a known pattern
    dst_layer = torch.full((2, num_blocks, page_size, head_size, hidden_size),
                           fill_value=2.0,
                           dtype=torch.float32,
                           device='cuda')

    # Test copying a range of tokens that spans multiple blocks
    block_ids = [0, 1, 2, 3]
    start_token_idx = 8
    stop_token_idx = 56

    h2d_page_copy(src_buffer, dst_layer, block_ids, start_token_idx,
                  stop_token_idx, block_size)

    # Calculate which blocks should be fully/partially copied
    start_block = start_token_idx // block_size
    end_block = (stop_token_idx + block_size - 1) // block_size
    start_pos = start_token_idx % block_size
    end_pos = stop_token_idx % block_size

    # Expected unchanged value
    expected_unchanged = torch.full((page_size, head_size, hidden_size),
                                    fill_value=2.0,
                                    dtype=torch.float32)

    # Verify copied and unchanged data for each block
    for i in range(num_blocks):
        if i < start_block or i >= end_block:
            # Blocks outside the copy range should be unchanged
            assert torch.allclose(dst_layer[:, block_ids[i]].cpu(),
                                  expected_unchanged)
        elif i == start_block:
            # First block - verify both copied and unchanged parts
            # Leading part should be unchanged
            assert torch.allclose(dst_layer[:, block_ids[i], :start_pos].cpu(),
                                  expected_unchanged[:start_pos])
            # Trailing part should be copied
            assert torch.allclose(dst_layer[:, block_ids[i], start_pos:].cpu(),
                                  src_buffer[:, i, start_pos:])
        elif i == end_block - 1:
            # Last block - verify both copied and unchanged parts
            # Leading part should be copied
            assert torch.allclose(dst_layer[:, block_ids[i], :end_pos].cpu(),
                                  src_buffer[:, i, :end_pos])
            # Trailing part should be unchanged
            assert torch.allclose(dst_layer[:, block_ids[i], end_pos:].cpu(),
                                  expected_unchanged[end_pos:])
        else:
            # Middle blocks - verify full copy
            assert torch.allclose(dst_layer[:, block_ids[i]].cpu(),
                                  src_buffer[:, i])


def test_h2d_page_copy_edge_cases():
    """Test edge cases for host to device page copy."""
    # Create sample tensors
    num_blocks = 2
    page_size = 16
    head_size = 8
    hidden_size = 128
    block_size = page_size

    src_buffer = torch.ones((2, num_blocks, page_size, head_size, hidden_size),
                            dtype=torch.float32)
    dst_layer = torch.zeros((2, num_blocks, page_size, head_size, hidden_size),
                            dtype=torch.float32,
                            device='cuda')

    # Test case 1: Copy exactly one block
    block_ids = [0, 1]
    start_token_idx = 0
    stop_token_idx = block_size

    h2d_page_copy(src_buffer, dst_layer, block_ids, start_token_idx,
                  stop_token_idx, block_size)

    assert torch.allclose(dst_layer[:, block_ids[0]].cpu(), src_buffer[:, 0])

    # Test case 2: Copy partial block
    dst_layer.zero_()
    block_ids = [0, 1]
    start_token_idx = block_size + 2
    stop_token_idx = block_size + 6

    h2d_page_copy(src_buffer, dst_layer, block_ids, start_token_idx,
                  stop_token_idx, block_size)

    start_pos = start_token_idx % block_size
    end_pos = stop_token_idx % block_size

    assert torch.allclose(dst_layer[:, block_ids[1], start_pos:end_pos].cpu(),
                          src_buffer[:, 1, start_pos:end_pos])


def test_h2d_page_copy_aligned():
    """Test host to device page copy operation with block-aligned boundaries."""
    # Create sample tensors
    num_blocks = 4
    page_size = 16
    head_size = 8
    hidden_size = 128
    block_size = page_size

    src_buffer = torch.ones((2, num_blocks, page_size, head_size, hidden_size),
                            dtype=torch.float32)
    # Initialize destination with a known pattern
    dst_layer = torch.full((2, num_blocks, page_size, head_size, hidden_size),
                           fill_value=2.0,
                           dtype=torch.float32,
                           device='cuda')

    # Test copying exactly 2 blocks (from block 1 to block 3)
    block_ids = [0, 1, 2, 3]
    start_token_idx = block_size  # Start at beginning of block 1
    stop_token_idx = block_size * 3  # End at end of block 2

    h2d_page_copy(src_buffer, dst_layer, block_ids, start_token_idx,
                  stop_token_idx, block_size)

    # Expected unchanged value
    expected_unchanged = torch.full((page_size, head_size, hidden_size),
                                    fill_value=2.0,
                                    dtype=torch.float32)

    # Verify copied and unchanged data for each block
    for i in range(num_blocks):
        if i == 0 or i == 3:
            # First and last blocks should be unchanged
            assert torch.allclose(
                dst_layer[:, block_ids[i]].cpu(),
                expected_unchanged), f"Block {i} should be unchanged"
        else:
            # Middle blocks (1 and 2) should be fully copied
            assert torch.allclose(
                dst_layer[:, block_ids[i]].cpu(),
                src_buffer[:, i]), f"Block {i} should be fully copied"

    # Test copying a single block-aligned region
    dst_layer.fill_(2.0)  # Reset destination
    start_token_idx = block_size * 2  # Start at beginning of block 2
    stop_token_idx = block_size * 3  # End at end of block 2

    h2d_page_copy(src_buffer, dst_layer, block_ids, start_token_idx,
                  stop_token_idx, block_size)

    # Verify only block 2 was copied, others unchanged
    for i in range(num_blocks):
        if i == 2:
            # Block 2 should be fully copied
            assert torch.allclose(
                dst_layer[:, block_ids[i]].cpu(),
                src_buffer[:, i]), "Block 2 should be fully copied"
        else:
            # All other blocks should be unchanged
            assert torch.allclose(
                dst_layer[:, block_ids[i]].cpu(),
                expected_unchanged), f"Block {i} should be unchanged"
