# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Monolithic decoder layer for DeepSeek V3.2 on SM100 (Blackwell).
Direct kernel calls, no module wrappers for norms.
Gate weight inlined, FusedMoE kept for quantized expert kernels.
"""

import torch
from torch import nn

import vllm._custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size

from .attention import MonolithicMLAAttention


def _fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ops.fused_add_rms_norm(x, residual, weight, eps)
    return x, residual


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, eps)
    return out


class MonolithicDecoderLayer(nn.Module):
    """
    Single decoder layer: norm -> attn -> norm -> MoE/MLP.
    Norms are raw weight + direct kernel call.
    Gate inlined as raw weight, experts kept as FusedMoE for quantization.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        topk_indices_buffer: torch.Tensor,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.tp_size = get_tensor_model_parallel_world_size()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        # LayerNorm weights (raw)
        dtype = torch.get_default_dtype()
        self.input_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=dtype)
        )
        self.post_attention_layernorm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=dtype)
        )

        # Fused QKV A-projection lives inside self_attn namespace
        # for weight loading compatibility with original checkpoint paths
        from vllm.model_executor.models.deepseek_v2 import (
            DeepSeekV2FusedQkvAProjLinear,
        )

        self.self_attn = nn.Module()
        self.self_attn.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            config.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.fused_qkv_a_proj",
        )

        # MLA Attention
        self.attn = MonolithicMLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            cache_config=cache_config,
            quant_config=quant_config,
            topk_indices_buffer=topk_indices_buffer,
            prefix=f"{prefix}.self_attn",
        )

        # MoE or Dense MLP
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        self.is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        from vllm.model_executor.models.deepseek_v2 import (
            DeepseekV2MLP,
            DeepseekV2MoE,
        )

        if self.is_moe:
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def _forward_fused_qkv_a_proj(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.self_attn.fused_qkv_a_proj(x)
        if isinstance(out, tuple):
            out = out[0]
        return out.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Input norm + residual
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = _rms_norm(
                hidden_states, self.input_layernorm_weight, self.rms_norm_eps
            )
        else:
            hidden_states, residual = _fused_add_rms_norm(
                hidden_states,
                residual,
                self.input_layernorm_weight,
                self.rms_norm_eps,
            )

        # Fused QKV A-proj + Q norm
        q_c, kv_lora = self._forward_fused_qkv_a_proj(hidden_states)
        q_c = _rms_norm(q_c, self.attn.q_a_layernorm_weight, self.rms_norm_eps)

        # Attention
        hidden_states = self.attn(positions, hidden_states, q_c, kv_lora)

        # Post-attn norm + residual
        hidden_states, residual = _fused_add_rms_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm_weight,
            self.rms_norm_eps,
        )

        # MLP / MoE
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
