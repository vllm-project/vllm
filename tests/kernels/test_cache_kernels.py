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


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="Need CUDA device")
def test_gather_cache_oob():
    """Tests for OOB read in gather_and_maybe_dequant_cache (Issue #27909).
    This test constructs a boundary case identified in the issue where
    seq_starts causes the block_table offset to read out of bounds.
    """
    block_size = 64
    # The kernel only supports the MLA entry sizes.
    entry_size = 576

    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")

    # This will result in offset = 128 / block_size = 128 / 64 = 2
    # This will cause the kernel to try to read from
    # block_table[0, 2], but its size is only 2.
    seq_starts = torch.tensor([128], dtype=torch.int32, device="cuda")

    seq_len = 65
    cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    token_to_seq = torch.zeros(seq_len, dtype=torch.int32, device="cuda")

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
        token_to_seq,
        seq_len,
        "auto",  # kv_cache_dtype
        scale,
        seq_starts,
    )

    torch.accelerator.synchronize()
    assert True


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="Need CUDA device")
def test_convert_fp8_strided():
    """convert_fp8 must honor row strides and leave padding untouched (#70382)."""
    dtype = torch.bfloat16
    src_base = torch.full((4, 96), 99.0, dtype=dtype, device="cuda")
    dst_base = torch.full((4, 96), 77, dtype=torch.uint8, device="cuda")
    src = src_base[:, :64]
    dst = dst_base[:, :64]
    torch.manual_seed(0)
    src.copy_(torch.randn(4, 64, dtype=dtype, device="cuda"))

    ops.convert_fp8(dst, src, 1.0, "auto")
    torch.accelerator.synchronize()

    ref = torch.empty((4, 64), dtype=torch.uint8, device="cuda")
    ops.convert_fp8(ref, src.contiguous(), 1.0, "auto")

    assert torch.equal(dst, ref)
    assert torch.equal(dst_base[:, 64:], torch.full_like(dst_base[:, 64:], 77))


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="Need CUDA device")
def test_convert_fp8_empty():
    """An empty cache must return instead of launching a zero-block grid (#70382)."""
    src = torch.empty((0, 64), dtype=torch.bfloat16, device="cuda")
    dst = torch.empty((0, 64), dtype=torch.uint8, device="cuda")
    ops.convert_fp8(dst, src, 1.0, "auto")
    torch.accelerator.synchronize()


if __name__ == "__main__":
    pytest.main([__file__])
