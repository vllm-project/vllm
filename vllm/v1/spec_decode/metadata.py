# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""推测解码元数据模块。

本模块定义了推测解码过程中使用的元数据数据结构，负责：
- 追踪草稿 token ID 和数量
- 管理 logits 索引映射
- 支持批量验证的元数据组织

主要类：
- SpecDecodeMetadata: 推测解码元数据容器
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    """推测解码元数据容器。

    存储推测解码过程中所需的元数据信息，用于批量验证草稿 token。
    所有张量都存储在 GPU 上以加速计算。

    Attributes:
        draft_token_ids: 草稿 token ID 列表展平后的张量 [num_tokens]
        num_draft_tokens: 每个请求的草稿 token 数量列表 [batch_size]
        cu_num_draft_tokens: 草稿 token 数量的累积和 [batch_size]
        cu_num_sampled_tokens: 采样 token 数量（草稿 +1）的累积和 [batch_size]
        target_logits_indices: 目标 logits 索引映射 [num_tokens]
        bonus_logits_indices: bonus logits 索引（每个请求的最后一个 token）[batch_size]
        logits_indices: 完整的 logits 索引映射 [num_tokens + batch_size]
    """

    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor

    def __post_init__(self):
        """初始化后计算最大草稿长度。

        设置 max_spec_len 为所有请求中草稿 token 数量的最大值。
        """
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        """创建虚拟元数据实例（用于测试或初始化）。

        Args:
            draft_token_ids: 每个请求的草稿 token ID 列表
            device: 张量设备

        Returns:
            SpecDecodeMetadata 实例
        """
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
