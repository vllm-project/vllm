# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from typing import Optional

import torch

from vllm.model_executor.layers.fused_moe.modular_kernel import (
    ExpertTokensMetadata)
from vllm.utils import round_up


@functools.cache
def deep_gemm_block_shape() -> list[int]:
    # Lazy import to avoid CUDA initialization problems.
    import deep_gemm as dg
    block = dg.get_m_alignment_for_contiguous_layout()
    return [block, block]


def expert_num_tokens_round_up_and_sum(expert_num_tokens: torch.Tensor,
                                       alignment: int) -> int:
    # Round up each element in expert_num_tokens to the nearest multiple of
    # alignment.
    ent = (expert_num_tokens.to(torch.int64) +
           (alignment - 1)) // alignment * alignment
    return torch.sum(ent).item()


def compute_aligned_M(
        M: int, num_topk: int, local_num_experts: int, alignment: int,
        expert_tokens_meta: Optional[ExpertTokensMetadata]) -> int:

    if ((expert_tokens_meta is not None)
            and (expert_tokens_meta.expert_num_tokens_cpu is not None)):
        return expert_num_tokens_round_up_and_sum(
            expert_tokens_meta.expert_num_tokens_cpu, alignment)

    # expert_num_tokens information is not available on the cpu.
    # compute the max required size.
    M_sum = (M * num_topk) + local_num_experts * (alignment - 1)
    M_sum = round_up(M_sum, alignment)
    return M_sum
