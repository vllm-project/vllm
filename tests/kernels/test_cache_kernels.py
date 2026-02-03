# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CUDA kernels in cache_kernels.cu."""

import pytest
import torch

try:
    from vllm import _custom_ops as ops
except ImportError:
    pytest.skip(
        "Could not import vllm._custom_ops. (pip install -e .)", allow_module_level=True
    )


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_oob():
    """
    Tests for OOB read in gather_and_maybe_dequant_cache (Issue #27909).
    This test constructs a boundary case identified in the issue where
    seq_starts causes the block_table offset to read out of bounds.
    """

    batch_size = 1
    block_size = 64
    entry_size = 128

    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")

    # This will result in offset = 128 / block_size = 128 / 64 = 2
    # This will cause the kernel to try to read from
    # block_table[0, 2], but its size is only 2.
    seq_starts = torch.tensor([128], dtype=torch.int32, device="cuda")

    seq_len = 65
    cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    # src_cache: [num_blocks, block_size, entry_size]
    num_blocks = 5
    src_cache = torch.randn(
        (num_blocks, block_size, entry_size), dtype=torch.float16, device="cuda"
    )

    dst = torch.empty((seq_len, entry_size), dtype=torch.float16, device="cuda")

    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    # Calling the C++ function gather_and_maybe_dequant_cache
    ops.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        batch_size,
        "auto",  # kv_cache_dtype
        scale,
        seq_starts,
    )

    torch.cuda.synchronize()
    assert True


if __name__ == "__main__":
    pytest.main([__file__])
