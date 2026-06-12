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
    """
    Tests for OOB read in gather_and_maybe_dequant_cache (issues #27909 and
    #45380). Constructs the boundary case where seq_starts pushes the
    block-table index past the table width (block_table_id == 2 for a 2-wide
    table). Run under compute-sanitizer to catch the OOB read; without the
    in-kernel bound the read is silently absorbed by allocator slack.
    """

    block_size = 64
    entry_size = 576

    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")

    # offset = 128 / block_size = 128 / 64 = 2 -> one past the table width.
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


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="Need CUDA device")
def test_cp_gather_upconvert_negative_offset():
    """
    Tests for OOB read in cp_gather_and_upconvert_fp8_kv_cache (issue #45377).
    When workspace_starts[0] != 0, the binary search clamps req_id to 0 and
    token_offset goes negative, underflowing the block-table read
    (block_table[-1]). Run under compute-sanitizer to catch the OOB read.
    """

    num_blocks = 4
    num_reqs = 1
    total_tokens = 8

    # FP8 KV cache layout expected by the kernel: [num_blocks, block_size, 656]
    src_cache = torch.zeros((num_blocks, 64, 656), dtype=torch.uint8, device="cuda")
    dst = torch.empty((total_tokens, 576), dtype=torch.bfloat16, device="cuda")
    block_table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([total_tokens], dtype=torch.int32, device="cuda")
    # Nonzero first entry: tokens 0..3 get token_offset < 0.
    workspace_starts = torch.tensor([4], dtype=torch.int32, device="cuda")

    ops.cp_gather_and_upconvert_fp8_kv_cache(
        src_cache,
        dst,
        block_table,
        seq_lens,
        workspace_starts,
        num_reqs,
    )

    torch.accelerator.synchronize()


if __name__ == "__main__":
    pytest.main([__file__])
