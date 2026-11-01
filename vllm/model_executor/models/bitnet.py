# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only BitNet b1.58 model (https://arxiv.org/abs/2504.12285).

BitNet is Llama with ReLU² gated MLPs and an extra RMSNorm ("SubLN") in front
of `o_proj` and `down_proj`. Ternary linears are handled by `BitNetConfig`.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BitNetConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.transformers.fusers.rms_norm import TPAwareRMSNorm

from .llama import LlamaAttention, LlamaForCausalLM


class BitNetMLP(nn.Module):
    def __init__(
        self,
        config: BitNetConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "relu2":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only relu2 is supported for now."
            )
        self.ffn_sub_norm = TPAwareRMSNorm(
            config.intermediate_size, eps=config.rms_norm_eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        gate, up = x.chunk(2, dim=-1)
        x, _ = self.down_proj(self.ffn_sub_norm(F.relu(gate).square() * up))
        return x


class BitNetAttention(LlamaAttention):
    def __init__(self, config: BitNetConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.attn_sub_norm = TPAwareRMSNorm(
            self.total_num_heads * self.head_dim, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn_sub_norm(self.attn(q, k, v))
        output, _ = self.o_proj(attn_output)
        return output


class BitNetDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.self_attn = BitNetAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            cache_config=vllm_config.cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = BitNetMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class BitNetForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, layer_type=BitNetDecoderLayer
        )
