# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logprobs 工具函数模块。

本模块实现了 logprobs 相关的实用函数，负责：
- 计算批次中大于特定值的元素数量
- 支持 logprob 排名计算
- 使用 torch.compile 优化性能

主要函数：
- batched_count_greater_than: 批次化计数大于特定值的元素
"""

import torch

from vllm.platforms import current_platform


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def batched_count_greater_than(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """计算 x 中每行大于对应 values 值的元素数量。

    使用 torch.compile 生成优化的内核。
    否则，它会创建输入张量的额外副本并导致内存问题。

    此函数用于计算采样 token 的排名：
    - x 是 logprobs 张量
    - values 是采样 token 的 logprobs
    - 返回值是排名（有多少 token 的 logprob 大于等于采样 token）

    Args:
        x: 形状为 (batch_size, n_elements) 的 2D 张量
        values: 形状为 (batch_size, 1) 的 2D 张量

    Returns:
        形状为 (batch_size,) 的 1D 张量，包含计数结果

    示例：
        >>> x = torch.tensor([[0.1, 0.5, 0.3], [0.2, 0.4, 0.6]])
        >>> values = torch.tensor([[0.3], [0.4]])
        >>> batched_count_greater_than(x, values)
        tensor([2, 2])  # 第一行有 2 个>=0.3，第二行有 2 个>=0.4
    """
    return (x >= values).sum(-1)
