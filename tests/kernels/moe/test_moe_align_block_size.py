# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size_triton)


@pytest.mark.parametrize(
    "block_size,num_tokens,topk,num_experts",
    list(
        itertools.product(
            [32, 64, 128, 256],  # block_size
            [
                1,
                3,
                7,
                16,
                256,
                2256,
                4096,
            ],  # num_tokens
            [1, 4, 16, 64],  # topk
            [64, 160, 256, 257, 260, 264],  #  num_experts
        )),
)
def test_moe_align_block_size_compare_implementations(block_size, num_tokens,
                                                      topk, num_experts):
    topk_ids = torch.stack([
        torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:topk]
        for _ in range(num_tokens)
    ])

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)

    sorted_ids_cuda = torch.empty((max_num_tokens_padded, ),
                                  dtype=torch.int32,
                                  device=topk_ids.device)
    sorted_ids_cuda.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids_cuda = torch.zeros((max_num_m_blocks, ),
                                  dtype=torch.int32,
                                  device=topk_ids.device)
    num_tokens_post_pad_cuda = torch.empty((1),
                                           dtype=torch.int32,
                                           device=topk_ids.device)

    sorted_ids_triton = torch.empty_like(sorted_ids_cuda)
    sorted_ids_triton.fill_(topk_ids.numel())
    expert_ids_triton = torch.zeros_like(expert_ids_cuda)
    num_tokens_post_pad_triton = torch.empty_like(num_tokens_post_pad_cuda)

    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_cuda,
        expert_ids_cuda,
        num_tokens_post_pad_cuda,
    )

    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_triton,
        expert_ids_triton,
        num_tokens_post_pad_triton,
    )

    assert torch.allclose(expert_ids_cuda, expert_ids_triton), (
        f"Expert IDs mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA expert_ids: {expert_ids_cuda}\n"
        f"Triton expert_ids: {expert_ids_triton}")

    assert torch.allclose(
        num_tokens_post_pad_cuda, num_tokens_post_pad_triton), (
            f"Num tokens post pad mismatch for block_size={block_size}, "
            f"num_tokens={num_tokens}, topk={topk}\n"
            f"CUDA num_tokens_post_pad: {num_tokens_post_pad_cuda}\n"
            f"Triton num_tokens_post_pad: {num_tokens_post_pad_triton}")


if __name__ == "__main__":
    pytest.main([__file__])
