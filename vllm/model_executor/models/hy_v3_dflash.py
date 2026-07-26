# SPDX-License-Identifier: Apache-2.0
"""DFlash draft model whose decoder FFNs match corresponding HY3 layers."""

import torch
from torch import nn
from torch.profiler import record_function

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from .hy_v3 import HYV3FeedForward, HYV3Model, HYV3MoEFused
from .qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import maybe_prefix


class DFlashHYV3DecoderLayer(nn.Module):
    """DFlash context attention plus the mapped HY3 dense/MoE FFN."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=False,
            head_dim=config.head_dim,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        dflash_config = getattr(config, "dflash_config", None) or {}
        target_layer_ids = dflash_config.get("target_layer_ids") or getattr(
            config, "target_layer_ids", None
        )
        if not target_layer_ids or len(target_layer_ids) != config.num_hidden_layers:
            raise ValueError("DFlashHYV3 requires one target_layer_id per draft layer.")
        target_layer_id = int(target_layer_ids[layer_idx])
        first_dense = int(getattr(config, "first_k_dense_replace", 1))
        if target_layer_id < first_dense:
            self.mlp = HYV3FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.block_type = "feedforward"
        else:
            self.mlp = HYV3MoEFused(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.block_type = "moe"

    def forward(
        self,
        positions,
        hidden_states,
        residual,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        with record_function("dflash_hyv3_attention"):
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        with record_function("dflash_hyv3_ffn"):
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DFlashHYV3Model(DFlashQwen3Model):
    """DFlash runtime with HY3-A3B mapped decoder blocks."""

    decoder_layer_class = DFlashHYV3DecoderLayer

    def get_expert_mapping(self):
        return HYV3Model.get_expert_mapping(self)

    def load_weights(self, weights):
        loaded = HYV3Model.load_weights(self, weights)
        self._build_fused_kv_buffers()
        return loaded


class DFlashHYV3ForCausalLM(DFlashQwen3ForCausalLM):
    """Top-level speculative draft wrapper for HY3-A3B layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = self.config.vocab_size
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlashHYV3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None


__all__ = [
    "DFlashHYV3DecoderLayer",
    "DFlashHYV3Model",
    "DFlashHYV3ForCausalLM",
]
