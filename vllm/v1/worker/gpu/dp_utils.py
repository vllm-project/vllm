# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""数据并行工具函数模块。

本模块提供数据并行（DP）相关的辅助函数，负责：
- 创建跨 DP ranks 的 token 数量张量
- 同步 CUDA Graph 模式和 DP padding 信息

主要函数：
- make_num_tokens_across_dp: 创建跨 DP 的 token 数量张量
- sync_cudagraph_and_dp_padding: 同步 CUDA Graph 模式和 DP padding
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)


def make_num_tokens_across_dp(dp_size: int, num_tokens: int) -> torch.Tensor | None:
    """创建跨 DP ranks 的 token 数量张量。

    Args:
        dp_size: DP 组大小
        num_tokens: token 数量

    Returns:
        填充了 num_tokens 的张量，如果 dp_size 为 1 则返回 None
    """
    if dp_size == 1:
        return None
    return torch.full((dp_size,), num_tokens, dtype=torch.int32, device="cpu")


def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """在所有 ranks 之间协调批次描述符和 DP padding。

    此函数执行以下操作：
    1. 收集所有 ranks 的 num_tokens、CUDA Graph 模式和 uniform token count
    2. 使用 all_reduce 同步这些信息
    3. 确定最保守的 CUDA Graph 模式（如果有任何 rank 使用 eager，则全部使用 eager）
    4. 同步 uniform token count（如果 ranks 之间不一致则为 None）
    5. 返回同步后的批次描述符和跨 DP 的 token 数量

    Args:
        cudagraph_manager: CUDA Graph 管理器
        desired_batch_desc: 期望的批次执行描述符
        num_tokens: 当前 rank 的 token 数量
        num_reqs: 当前 rank 的请求数量
        uniform_token_count: uniform token count（如果有）
        dp_size: DP 组大小
        dp_rank: 当前 DP rank ID

    Returns:
        (同步后的批次描述符，跨 DP 的 token 数量张量) 元组
    """
    assert dp_size > 1, "DP size must be greater than 1"
    group = get_dp_group().cpu_group
    tensor = torch.zeros(3, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode.value
    tensor[2][dp_rank] = uniform_token_count or 0  # 0 表示 None
    dist.all_reduce(tensor, group=group)

    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0
        )
        return synced_desc, None

    synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min().item()))

    # 如果有任何 rank 想要运行 eager，所有 ranks 都运行 eager
    if synced_cg_mode == CUDAGraphMode.NONE:
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
        ), num_tokens_across_dp

    synced_num_tokens = int(num_tokens_across_dp.max().item())
    synced_uniform_token_count = uniform_token_counts_across_dp[0]
    # 如果 ranks 之间 uniform token count 不一致，或者为 0（表示 None），则设置为 None
    if synced_uniform_token_count == 0 or not torch.all(
        uniform_token_counts_across_dp == synced_uniform_token_count
    ):
        synced_uniform_token_count = None

    # 使用同步后的值进行调度，使用 num_reqs 而不是 synced_num_reqs
    # 这样我们就不会为 PIECEWISE graphs 执行请求 padding
    synced_desc = cudagraph_manager.dispatch(
        num_reqs, synced_num_tokens, synced_uniform_token_count
    )

    # 更新 num_tokens_across_dp 以反映 padding 后的大小
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, num_tokens_across_dp
