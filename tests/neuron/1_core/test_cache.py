# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.attention.ops.nki_flash_attn import reshape_and_cache


@pytest.mark.parametrize(
    "num_tokens, n_kv_head, d_head, num_blocks, block_size",
    [
        # Small model configuration (e.g., GPT-2 small)
        (32, 12, 64, 4, 128),  # Typical sequence processing
        (1, 12, 64, 4, 128),  # Single token update
        (128, 12, 64, 4, 128),  # Longer sequence

        # Medium model configuration (e.g., GPT-2 medium)
        (64, 16, 96, 8, 256),  # Standard batch
        (256, 16, 96, 8, 256),  # Large batch

        # Large model configuration (e.g., GPT-3 style)
        (48, 32, 128, 16, 512),  # Typical processing window
        (512, 32, 128, 16, 512),  # Full context window

        # Edge cases and stress tests
        (1024, 8, 32, 32, 32),  # Many tokens, small heads
        (16, 64, 256, 4, 64),  # Few tokens, many heads
        (2048, 24, 128, 64, 128),  # Large scale test

        # Minimal configurations for debugging
        (4, 2, 16, 2, 16),  # Tiny test case
        (1, 1, 8, 1, 8),  # Minimal possible
    ])
def test_reshape_and_cache(num_tokens, n_kv_head, d_head, num_blocks,
                           block_size):
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create CPU tensors for reference implementation
    key_cpu = torch.randn(num_tokens, n_kv_head, d_head) / torch.sqrt(
        torch.tensor(d_head))
    value_cpu = torch.randn(num_tokens, n_kv_head, d_head) / torch.sqrt(
        torch.tensor(d_head))
    key_cache_cpu = torch.zeros(num_blocks, n_kv_head, block_size, d_head)
    value_cache_cpu = torch.zeros(num_blocks, n_kv_head, block_size, d_head)
    slot_mapping_cpu = torch.randperm(num_blocks * block_size)[:num_tokens]

    # Run reference implementation on CPU
    block_indices = torch.div(slot_mapping_cpu,
                              block_size,
                              rounding_mode="floor")
    block_offsets = slot_mapping_cpu % block_size

    for i in range(num_tokens):
        block_idx = block_indices[i]
        block_offset = block_offsets[i]
        key_cache_cpu[block_idx, :, block_offset, :] = key_cpu[i]
        value_cache_cpu[block_idx, :, block_offset, :] = value_cpu[i]

    # Create XLA device tensors
    device = torch.device('xla')
    key = key_cpu.to(device)
    value = value_cpu.to(device)
    key_cache = torch.zeros_like(key_cache_cpu, device=device)
    value_cache = torch.zeros_like(value_cache_cpu, device=device)
    slot_mapping = slot_mapping_cpu.to(device)
    kv_cache = torch.stack([key_cache, value_cache])

    # Run vectorized implementation on XLA device
    reshape_and_cache(key, value, kv_cache, slot_mapping)
    key_cache, value_cache = torch.unbind(kv_cache, dim=0)

    # Move results back to CPU for comparison
    key_cache_result = key_cache.cpu()
    value_cache_result = value_cache.cpu()

    # Assert results match
    torch.testing.assert_close(key_cache_result,
                               key_cache_cpu,
                               rtol=1e-5,
                               atol=1e-5)
    torch.testing.assert_close(value_cache_result,
                               value_cache_cpu,
                               rtol=1e-5,
                               atol=1e-5)
