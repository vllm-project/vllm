# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.models.utils import maybe_fuse_shared_experts
from vllm.models.hy_v4.nvidia.mtp import HYV4MTP as BaseMTP
from vllm.models.hy_v4.nvidia.mtp import (
    HYV4MultiTokenPredictor as BaseMultiTokenPredictor,
)
from vllm.models.hy_v4.nvidia.mtp import (
    HYV4MultiTokenPredictorLayer as BaseMultiTokenPredictorLayer,
)

from .model import HYV4DecoderLayer
from .moe import HYV4MoEFused


class HYV4MultiTokenPredictorLayer(BaseMultiTokenPredictorLayer):
    decoder_layer_cls = HYV4DecoderLayer


class HYV4MultiTokenPredictor(BaseMultiTokenPredictor):
    predictor_layer_cls = HYV4MultiTokenPredictorLayer

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers.values(),
            HYV4MoEFused,
            "mtp_block.mlp",
        )
        first_layer = next(iter(self.layers.values()), None)
        self.num_fused_shared_experts = (
            first_layer.mtp_block.config.num_shared_experts
            if self.is_fused_shared_expert_enabled and first_layer is not None
            else 0
        )

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        current_step_idx = self.spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        proj_input = mtp_layer.shared_head(hidden_states)
        return self.logits_processor.get_top_tokens(
            mtp_layer.shared_head.head, proj_input
        )


@support_torch_compile
class HYV4MTP(BaseMTP):
    predictor_cls = HYV4MultiTokenPredictor

    def load_weights(self, weights):
        if self.model.is_fused_shared_expert_enabled:
            weights = maybe_fuse_shared_experts(
                weights,
                n_routed_experts=self.config.n_routed_experts,
                n_shared_experts=self.config.n_shared_experts,
                ckpt_prefix="mlp.shared_experts",
                enabled=True,
            )
        return super().load_weights(weights)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states)
