# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""流水线并行工具函数模块（V2 Model Runner）。

本模块提供流水线并行相关的辅助函数，负责：
- 在流水线并行组内广播采样结果
- 接收来自最后 ranks 的采样 token

主要函数：
- pp_broadcast: 从最后 rank 广播采样 token
- pp_receive: 非最后 rank 接收采样 token
"""

import torch

from vllm.distributed.parallel_state import get_pp_group


def pp_broadcast(
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> None:
    """从流水线并行最后 rank 广播采样结果。

    在流水线并行中，只有最后 rank 执行采样，需要将结果
    广播给所有其他 ranks。

    Args:
        sampled_token_ids: 采样的 token ID 张量
        num_sampled: 采样数量
        num_rejected: 拒绝数量
    """
    pp = get_pp_group()
    assert pp.is_last_rank

    assert sampled_token_ids.dtype == torch.int64
    torch.distributed.broadcast(
        sampled_token_ids.contiguous(), src=pp.last_rank, group=pp.device_group
    )

    combined = torch.stack((num_sampled, num_rejected), dim=0)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)


def pp_receive(
    num_reqs: int, max_sample_len: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """非最后 rank 接收来自流水线并行最后 rank 的采样结果。

    Args:
        num_reqs: 请求数量
        max_sample_len: 最大采样长度（默认 1）

    Returns:
        (采样的 token，采样数量，拒绝数量) 元组
    """
    pp = get_pp_group()
    assert not pp.is_last_rank

    sampled_tokens = torch.empty(
        num_reqs, max_sample_len, dtype=torch.int64, device=pp.device
    )
    torch.distributed.broadcast(sampled_tokens, src=pp.last_rank, group=pp.device_group)

    combined = torch.empty(2, num_reqs, dtype=torch.int32, device=pp.device)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)
    num_sampled, num_rejected = combined.unbind(dim=0)
    return sampled_tokens, num_sampled, num_rejected
