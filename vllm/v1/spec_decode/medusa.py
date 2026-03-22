# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Medusa 推测解码 proposer 模块。

本模块实现了基于 Medusa 架构的推测解码 proposer，负责：
- 使用 Medusa 多头生成候选 token
- 加载和管理 Medusa 模型
- 支持多个并行的解码头

Medusa 架构使用多个独立的解码头（heads）并行生成候选 token，
每个头预测序列中的下一个位置。

主要类：
- MedusaProposer: Medusa 推测解码 proposer
"""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.v1.sample.metadata import SamplingMetadata

# 初始化日志记录器
logger = init_logger(__name__)


class MedusaProposer:
    """Medusa 推测解码 proposer。

    使用 Medusa 多头架构生成候选 token 序列。
    Medusa 使用多个独立的解码头并行生成多个候选 token，
    每个头预测序列中的下一个位置。

    Attributes:
        vllm_config: vLLM 配置
        spec_config: 推测解码配置
        device: 设备（CUDA）
        max_num_tokens: 最大 token 数量
        hidden_size: 隐藏层大小
        dtype: 数据类型
        model: 加载的 Medusa 模型
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """初始化 Medusa proposer。

        Args:
            vllm_config: vLLM 配置
            device: 设备（CUDA）
        """
        # 保存配置参数
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None, (
            "Speculative config must be set"
        )
        self.spec_config = vllm_config.speculative_config
        self.device = device
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.hidden_size = self.spec_config.draft_model_config.get_hidden_size()
        self.dtype = vllm_config.model_config.dtype

    def propose(
        self,
        target_hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> torch.Tensor:
        """生成候选 token。

        使用 Medusa 模型生成多个解码头，每个头预测一个候选 token。

        Args:
            target_hidden_states: 目标模型的隐藏状态
            sampling_metadata: 采样元数据（未使用）
            slot_mappings: KV 缓存槽映射（未使用）

        Returns:
            草稿 token 张量，形状：[batch_size, num_heads]
        """
        # 生成块并计算 logits
        blocks = self.model(target_hidden_states)
        logits = self.model.compute_logits(blocks)

        # 对每个 Medusa 头计算 argmax 并堆叠为单个张量
        # 形状：[batch_size, num_heads]
        draft_tokens = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)

        return draft_tokens

    def load_model(self, target_model: nn.Module) -> None:
        """加载 Medusa 模型。

        Args:
            target_model: 目标模型（未使用）
        """
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("medusa_head"):
            self.model = get_model(
                vllm_config=self.vllm_config,
                model_config=self.spec_config.draft_model_config,
            )
        assert not (
            is_mixture_of_experts(self.model)
            and self.vllm_config.parallel_config.enable_eplb
        ), "EPLB for Medusa is not supported"

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int) -> None:
        """运行虚拟推理以初始化模型。

        Args:
            num_tokens: token 数量
        """
        hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        with set_forward_context(None, self.vllm_config, num_tokens=num_tokens):
            self.model(hidden_states)
