# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/Motif-Technologies/Motif-2.6B/blob/main/modeling_motif.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Motif-Technologies/Motif-2.6B/blob/main/LICENSE
"""Inference-only Motif model compatible with HuggingFace weights."""
import math
from typing import Any, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionType
from vllm.attention.selector import _Backend
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import PolyNorm, RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.llama import LlamaForCausalLM

from .adapters import as_seq_cls_model
from .interfaces import SupportsV0Only
from .utils import extract_layer_index


class MotifMLP(nn.Module):
    """MLP for the language component of the Motif model, which contains a
    MergedColumnParallelLinear merging 2 outputs via PolyNorm activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "poly_norm",
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "poly_norm":
            raise NotImplementedError(f"Unsupported activation: {hidden_act}. "
                                      "Only poly_norm is supported for now.")
        self.act_fn = PolyNorm()
        self.intermediate_size = intermediate_size
        tp_size = get_tensor_model_parallel_world_size()
        if hidden_act == "poly_norm" and tp_size > 1:
            raise NotImplementedError(
                "Tensor parallelism for poly_norm is not supported yet. "
                "Support will be added in the future.")

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(
            x[..., :self.intermediate_size]) * x[..., self.intermediate_size:]
        x, _ = self.down_proj(x)
        return x


class MotifAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        # Phi models introduced a partial_rotary_factor parameter in the config
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor",
                                             1)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        assert self.num_heads % 2 == 0, 'num_heads should be even'
        assert self.num_kv_heads % 2 == 0, 'num_heads should be even'

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb(config,
                              rope_scaling=rope_scaling,
                              quant_config=quant_config)
        sliding_window = None

        self.lambda_init = self.lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,
                                                                    std=0.1))
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,
                                                                    std=0.1))
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,
                                                                    std=0.1))
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,
                                                                    std=0.1))
        self.subln = RMSNorm(2 * self.head_dim, eps=config.attn_rms_norm_eps)

        params = {
            'differential_flash_attention_config': {
                'lambda_init': self.lambda_init,
                'lambda_q1': self.lambda_q1,
                'lambda_k1': self.lambda_k1,
                'lambda_q2': self.lambda_q2,
                'lambda_k2': self.lambda_k2,
                "subln": self.subln,
            }
        }

        diff_attn_err_msg = (
            'Set VLLM_ATTENTION_BACKEND="DIFFERENTIAL_FLASH_ATTN" '
            'to enable Differential Flash Attention.')
        try:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                per_layer_sliding_window=sliding_window,
                attn_type=attn_type,
                prefix=f"{prefix}.attn",
                **params,
            )
        except TypeError as e:
            raise ValueError(diff_attn_err_msg) from e
        assert (self.attn.backend == _Backend.DIFFERENTIAL_FLASH_ATTN
                ), diff_attn_err_msg

    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _init_rotary_emb(self, config: PretrainedConfig,
                         rope_scaling: Optional[dict[str, Any]],
                         quant_config: Optional[QuantizationConfig]) -> None:
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            partial_rotary_factor=self.partial_rotary_factor,
        )


class MotifDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "use_bias", False)
        bias_o_proj = attention_bias
        if hasattr(config, 'qkv_bias'):
            attention_bias = config.qkv_bias

        # By default, Motif uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. parasail-ai/GritLM-7B-vllm)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = MotifAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = MotifMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "use_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# Motif model uses differential attention
# Only supported in v0 (no chunked prefill support)
class MotifForCausalLM(LlamaForCausalLM, SupportsV0Only):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = MotifDecoderLayer):

        # Prefix caching and chunked prefill is not supported for this model.
        assert not vllm_config.cache_config.enable_prefix_caching, \
            "Motif currently does not support prefix caching"
        assert not vllm_config.scheduler_config.chunked_prefill_enabled, \
            "Motif currently does not support chunked prefill"

        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         layer_type=layer_type)


MotifForSequenceClassification = as_seq_cls_model(MotifForCausalLM)
