# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""草稿模型 proposer 模块。

本模块实现了基于独立草稿模型的推测解码 proposer，负责：
- 使用小型草稿模型生成候选 token
- 验证草稿模型与目标模型的兼容性
- 加载和管理草稿模型

主要类：
- DraftModelProposer: 基于独立草稿模型的 proposer
"""

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import create_vllm_config_for_draft_model

logger = init_logger(__name__)


class DraftModelProposer(SpecDecodeBaseProposer):
    """基于独立草稿模型的推测解码 proposer。

    使用一个独立的小型草稿模型（如 TinyLlama）生成候选 token，
    然后由大型目标模型验证。草稿模型和目标模型需要有相同的
    词表大小和张量并行度。

    继承自 SpecDecodeBaseProposer，实现特定的模型加载和验证逻辑。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        """初始化草稿模型 proposer。

        Args:
            vllm_config: vLLM 配置
            device: 设备（CUDA）
            runner: 模型运行器（可选）
        """
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def _raise_if_vocab_size_mismatch(self):
        """检查词表大小是否匹配。

        如果草稿模型和目标模型的词表大小不同，则抛出异常。
        """
        self.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self):
        """检查张量并行度是否匹配。

        如果目标模型使用 TP > 1 而草稿模型使用 TP = 1，
        会导致不同 TP  ranks 之间的冲突。具体来说，当所有 ranks
        在 rank 0 上编译草稿模型时（因为 TP=1），torch compile 缓存
        会被覆盖和损坏。

        因此我们要求两者的 TP 大小必须相同。
        """
        spec_cfg = self.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    @override
    def _get_model(self) -> nn.Module:
        """获取草稿模型。

        草稿模型可能被量化或使用不同的并行策略，
        因此我们使用修改后的 vllm 配置加载它们。

        Returns:
            加载的草稿模型
        """
        from vllm.compilation.backends import set_model_tag

        temp_vllm_config = create_vllm_config_for_draft_model(self.vllm_config)
        with set_model_tag("draft_model"):
            model = get_model(
                vllm_config=temp_vllm_config,
                prefix="draft_model",
            )
        return model

    @override
    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        """可能共享嵌入层。

        草稿模型不与目标模型共享嵌入层。
        """
        # 草稿模型不与目标模型共享嵌入层
        pass

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        """可能共享 LM 输出头。

        草稿模型不与目标模型共享 LM 头。
        """
        # 草稿模型不与目标模型共享 lm_head
        pass
