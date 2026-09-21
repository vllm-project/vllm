# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only AliceAI MTP model."""

from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.alice_ai import (
    AliceAIModel,
    _make_alice_ai_full_attention_config,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextMLP,
    Qwen3NextRMSNorm,
    Qwen3NextSparseMoeBlock,
    _should_use_sequence_parallel,
)
from vllm.model_executor.models.qwen3_next_mtp import (
    Qwen3NextMTP,
    Qwen3NextMultiTokenPredictor,
)
from vllm.model_executor.models.utils import extract_layer_index


class AliceAIMTPDecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        if layer_type != "full_attention":
            raise ValueError("AliceAI MTP requires full_attention")
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        is_moe_layer = (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        )
        self.use_attn_reduce_scatter_for_moe = _should_use_sequence_parallel(
            vllm_config
        )

        self.self_attn = Qwen3NextAttention(
            _make_alice_ai_full_attention_config(config),
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            reduce_results=not self.use_attn_reduce_scatter_for_moe,
            prefix=f"{prefix}.self_attn",
        )

        if is_moe_layer:
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = False


class AliceAIMultiTokenPredictor(Qwen3NextMultiTokenPredictor):
    hf_to_vllm_mapper = AliceAIModel.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=AliceAIMTPDecoderLayer,
        )


class AliceAIMTP(Qwen3NextMTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, model_cls=AliceAIMultiTokenPredictor
        )
