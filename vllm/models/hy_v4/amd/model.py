# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.models.utils import maybe_fuse_shared_experts
from vllm.models.hy_v4.nvidia.model import (
    HYV4DecoderLayer as BaseDecoderLayer,
)
from vllm.models.hy_v4.nvidia.model import HYV4ForCausalLM as BaseForCausalLM
from vllm.models.hy_v4.nvidia.model import HYV4Model as BaseModel

from .attention import HYV4MLAAttention
from .ihc import HYV4HCLayer
from .moe import HYV4MoEFused


class HYV4DecoderLayer(BaseDecoderLayer):
    attention_cls = HYV4MLAAttention
    hc_layer_cls = HYV4HCLayer
    moe_cls = HYV4MoEFused


@support_torch_compile
class HYV4Model(BaseModel):
    decoder_layer_cls = HYV4DecoderLayer

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            HYV4MoEFused,
            "mlp",
        )
        self.num_fused_shared_experts = (
            self.config.num_shared_experts if self.is_fused_shared_expert_enabled else 0
        )


class HYV4ForCausalLM(BaseForCausalLM):
    model_cls = HYV4Model

    def load_weights(self, weights):
        if self.model.is_fused_shared_expert_enabled:
            weights = maybe_fuse_shared_experts(
                weights,
                n_routed_experts=self.config.n_routed_experts,
                n_shared_experts=self.config.n_shared_experts,
                ckpt_prefix="mlp.shared_experts",
                enabled=True,
            )
        loaded_params = super().load_weights(weights)

        from vllm.model_executor.layers.quantization.kv_cache import (
            KVCacheScaleParameter,
        )

        unassigned = sorted(
            name
            for name, param in self.named_parameters()
            if name not in loaded_params
            and not isinstance(param, KVCacheScaleParameter)
        )
        if unassigned:
            raise ValueError(
                "HYV4 ROCm target model has parameters with no checkpoint value: "
                + ", ".join(unassigned)
            )
        return loaded_params

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(
            self.lm_head,
            hidden_states,
            skip_gather=True,
        )
        if getattr(self.config, "soft_logits_capping", False):
            soft_cap = self.config.soft_logits_capping_logits
            logits = soft_cap * torch.nn.functional.tanh(logits / soft_cap)
        return logits
