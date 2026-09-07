# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression and unit tests for DeepSeek-V4 cache utils Triton kernels."""

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    compute_global_topk_indices_and_lens,
)


def _ref_compute_global_topk(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
    num_reqs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_reqs is None:
        num_reqs = block_table.shape[0]
    num_tokens, topk = topk_indices.shape
    stride = block_table.stride(0)
    ref_indices = torch.full_like(topk_indices, fill_value=-1)
    ref_lens = torch.zeros(num_tokens, dtype=torch.int32, device=topk_indices.device)

    for t in range(num_tokens):
        req = token_to_req_indices[t].item()
        valid_tok = is_valid_token[t].item()
        count = 0
        if 0 <= req < num_reqs:
            for k in range(topk):
                loc = topk_indices[t, k].item()
                if loc >= 0:
                    b_idx = loc // block_size
                    if b_idx < stride:
                        b_num = block_table[req, b_idx].item()
                        slot = b_num * block_size + (loc % block_size)
                        ref_indices[t, k] = slot
                        count += 1
        if valid_tok:
            ref_lens[t] = count
    return ref_indices, ref_lens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compute_global_topk_indices_and_lens_basic():
    """Verify normal valid lookup matches reference."""
    device = torch.device("cuda")
    block_size = 64
    num_reqs = 4
    num_blocks = 8
    topk = 16
    num_tokens = 6

    block_table = torch.arange(
        num_reqs * num_blocks, dtype=torch.int32, device=device
    ).view(num_reqs, num_blocks)
    token_to_req = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int32, device=device)
    is_valid_token = torch.tensor(
        [True, True, True, True, True, False], dtype=torch.bool, device=device
    )

    # Valid local indices: block indices in [0, 4)
    topk_indices = torch.randint(
        0, 4 * block_size, (num_tokens, topk), dtype=torch.int32, device=device
    )
    # Some invalid local indices (-1)
    topk_indices[0, 2] = -1
    topk_indices[1, 5] = -1

    global_indices, topk_lens = compute_global_topk_indices_and_lens(
        topk_indices, token_to_req, block_table, block_size, is_valid_token
    )

    ref_indices, ref_lens = _ref_compute_global_topk(
        topk_indices, token_to_req, block_table, block_size, is_valid_token
    )

    torch.testing.assert_close(global_indices, ref_indices)
    torch.testing.assert_close(topk_lens, ref_lens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compute_global_topk_indices_out_of_bounds_req_idx():
    """Verify out-of-bounds req_idx yields -1 and 0 lens without memory fault."""
    device = torch.device("cuda")
    block_size = 64
    num_reqs = 2
    num_blocks = 4
    topk = 32
    num_tokens = 5

    block_table = torch.randint(
        1, 100, (num_reqs, num_blocks), dtype=torch.int32, device=device
    )
    # Token 0, 1: valid req_idx (0, 1)
    # Token 2: req_idx == num_reqs (out of bounds)
    # Token 3: req_idx > num_reqs (out of bounds, e.g. padding request slot)
    # Token 4: req_idx == -1 (negative invalid)
    token_to_req = torch.tensor([0, 1, 2, 100, -1], dtype=torch.int32, device=device)
    is_valid_token = torch.tensor(
        [True, True, True, True, True], dtype=torch.bool, device=device
    )

    topk_indices = torch.randint(
        0, 2 * block_size, (num_tokens, topk), dtype=torch.int32, device=device
    )

    global_indices, topk_lens = compute_global_topk_indices_and_lens(
        topk_indices, token_to_req, block_table, block_size, is_valid_token
    )

    ref_indices, ref_lens = _ref_compute_global_topk(
        topk_indices, token_to_req, block_table, block_size, is_valid_token
    )

    # Verify out-of-bounds tokens produced -1 for all topk entries
    assert (global_indices[2] == -1).all()
    assert (global_indices[3] == -1).all()
    assert (global_indices[4] == -1).all()
    assert topk_lens[2].item() == 0
    assert topk_lens[3].item() == 0
    assert topk_lens[4].item() == 0

    torch.testing.assert_close(global_indices, ref_indices)
    torch.testing.assert_close(topk_lens, ref_lens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compute_global_topk_indices_out_of_bounds_block_indices():
    """Verify block_indices exceeding column stride safely clamp/mask to -1."""
    device = torch.device("cuda")
    block_size = 64
    num_reqs = 3
    num_blocks = 4  # column stride = 4

    block_table = torch.randint(
        1, 50, (num_reqs, num_blocks), dtype=torch.int32, device=device
    )
    token_to_req = torch.tensor([0, 1], dtype=torch.int32, device=device)
    is_valid_token = torch.tensor([True, True], dtype=torch.bool, device=device)

    # Craft local_idx where some block indices are in-bounds (< 4)
    # and some are out-of-bounds (>= 4, i.e. local_idx >= 4 * 64 = 256)
    topk_indices = torch.tensor(
        [
            [0, 63, 64, 128, 256, 320, 500, 10000],
            [10, 20, 255, 256, 1000, -1, 512, 1024],
        ],
        dtype=torch.int32,
        device=device,
    )

    global_indices, topk_lens = compute_global_topk_indices_and_lens(
        topk_indices, token_to_req, block_table, block_size, is_valid_token
    )

    ref_indices, ref_lens = _ref_compute_global_topk(
        topk_indices, token_to_req, block_table, block_size, is_valid_token
    )

    # In token 0:
    # 0, 63 -> block 0 (valid)
    # 64 -> block 1 (valid)
    # 128 -> block 2 (valid)
    # 256, 320, 500, 10000 -> block >= 4 (out of bounds -> must be -1)
    assert (global_indices[0, :4] >= 0).all()
    assert (global_indices[0, 4:] == -1).all()
    assert topk_lens[0].item() == 4

    # In token 1:
    # 10, 20 -> block 0 (valid)
    # 255 -> block 3 (valid, 255 // 64 == 3 < 4)
    # 256, 1000 -> block >= 4 (out of bounds -> -1)
    # -1 -> local_idx < 0 (-1)
    # 512, 1024 -> block >= 4 (out of bounds -> -1)
    assert (global_indices[1, :3] >= 0).all()
    assert (global_indices[1, 3:] == -1).all()
    assert topk_lens[1].item() == 3

    torch.testing.assert_close(global_indices, ref_indices)
    torch.testing.assert_close(topk_lens, ref_lens)
