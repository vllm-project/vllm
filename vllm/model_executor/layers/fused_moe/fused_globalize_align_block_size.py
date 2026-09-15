# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton


def fused_globalize_align_block_size(
    recv_topk_idx: torch.Tensor,
    psum_recv_per_rank: torch.Tensor,
    rank_expert_offset: int,
    global_num_experts: int,
    local_num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the DeepEP-v2 INDEXED decode metadata chain into one launch.

    Replaces ``_globalize_recv_topk_idx`` + ``moe_align_block_size`` +
    ``count_and_sort_expert_tokens``. ``recv_topk_idx`` holds LOCAL expert ids
    and is globalized in place; valid-token count is read on-device from
    ``psum_recv_per_rank[-1]`` (CUDA-graph safe).

    Returns the globalized ``recv_topk_idx`` and the ``moe_align_block_size``
    outputs ``(sorted_ids, expert_ids, num_tokens_post_pad)``.
    """
    assert recv_topk_idx.dtype == torch.int64
    _, topk = recv_topk_idx.shape
    numel = recv_topk_idx.numel()
    device = recv_topk_idx.device

    max_num_tokens_padded = numel + local_num_experts * (block_size - 1)
    if numel < local_num_experts:
        max_num_tokens_padded = min(numel * block_size, max_num_tokens_padded)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)

    ops.fused_globalize_align_block_size(
        recv_topk_idx,
        psum_recv_per_rank,
        rank_expert_offset,
        global_num_experts,
        local_num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return recv_topk_idx, sorted_ids, expert_ids, num_tokens_post_pad
