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
    Tests for OOB read in gather_cache token-major mode (Issue #27909).
    This test constructs a boundary case identified in the issue where
    seq_starts causes the block_table offset to read out of bounds.
    """

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

    token_to_seq = torch.zeros((seq_len,), dtype=torch.int32, device="cuda")

    # Calling the C++ unified gather_cache function in token-major mode.
    ops.gather_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        seq_len,
        -1,
        "auto",  # kv_cache_dtype
        scale,
        seq_starts,
    )

    torch.cuda.synchronize()
    assert True


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_token_major_zero_tokens_noop():
    block_size = 16
    entry_size = 576
    src_cache = torch.randn(
        (4, block_size, entry_size), dtype=torch.float16, device="cuda"
    )
    dst = torch.empty((0, entry_size), dtype=torch.float16, device="cuda")
    block_table = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu_seq_lens = torch.zeros((2,), dtype=torch.int32, device="cuda")
    token_to_seq = torch.empty((0,), dtype=torch.int32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    ops.gather_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        0,
        -1,
        "auto",
        scale,
        None,
    )
    torch.cuda.synchronize()


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_requires_mode_metadata():
    block_size = 16
    entry_size = 576
    src_cache = torch.randn(
        (4, block_size, entry_size), dtype=torch.float16, device="cuda"
    )
    dst = torch.empty((1, entry_size), dtype=torch.float16, device="cuda")
    block_table = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="requires either token-major metadata"):
        ops.gather_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            None,
            -1,
            -1,
            "auto",
            None,
            None,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_batch_major_zero_batches_noop():
    src = torch.randn((4, 16, 576), dtype=torch.float16, device="cuda")
    dst = torch.empty((0, 576), dtype=torch.float16, device="cuda")
    bt = torch.zeros((0, 4), dtype=torch.int32, device="cuda")
    cu = torch.zeros((1,), dtype=torch.int32, device="cuda")

    ops.gather_cache(src, dst, bt, cu, None, 0, 0, "auto", None, None)
    torch.cuda.synchronize()


if __name__ == "__main__":
    pytest.main([__file__])
