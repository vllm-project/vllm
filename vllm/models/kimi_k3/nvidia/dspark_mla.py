# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA binding for the Kimi-K3 DSpark MLA draft."""

from vllm.config import VllmConfig
from vllm.models.kimi_k3.common.dspark_mla import (
    K3DSparkDecoderLayer as _K3DSparkDecoderLayer,
)
from vllm.models.kimi_k3.common.dspark_mla import (
    K3DSparkForCausalLM as _K3DSparkForCausalLM,
)
from vllm.models.kimi_k3.common.dspark_mla import (
    K3DSparkModel as _K3DSparkModel,
)
from vllm.models.kimi_k3.common.dspark_mla import _duplicate_context_kv_weights
from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.models.kimi_k3.nvidia.model import KimiMLP

__all__ = [
    "K3DSparkDecoderLayer",
    "K3DSparkForCausalLM",
    "K3DSparkModel",
    "_duplicate_context_kv_weights",
]


class K3DSparkDecoderLayer(_K3DSparkDecoderLayer):
    def build_self_attn(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        quant_config,
        prefix: str,
    ):
        return MultiHeadLatentAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
            use_rope=True,
            non_causal_multi_token_decode=True,
        )

    def build_mlp(self, *, config, quant_config, prefix: str):
        return KimiMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=prefix,
        )


class K3DSparkModel(_K3DSparkModel):
    decoder_layer_cls = K3DSparkDecoderLayer


class K3DSparkForCausalLM(_K3DSparkForCausalLM):
    model_cls = K3DSparkModel
