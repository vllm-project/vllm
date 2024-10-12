# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2024 The ModelBest team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MiniCPM3 model compatible with HuggingFace weights."""
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.minicpm import (MiniCPMDecoderLayer,
                                                MiniCPMForCausalLM,
                                                MiniCPMModel)

from .utils import make_layers


class MiniCPM3Attention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                         self.q_lora_rank,
                                         bias=False,
                                         quant_config=quant_config)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                             self.num_heads * self.qk_head_dim,
                                             bias=False,
                                             quant_config=quant_config)

        self.kv_a_proj_with_mqa = ReplicatedLinear(self.hidden_size,
                                                   self.kv_lora_rank +
                                                   self.qk_rope_head_dim,
                                                   bias=False,
                                                   quant_config=quant_config)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config)
        # O projection.
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config)

        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_local_heads,
                              self.qk_head_dim,
                              self.scaling,
                              num_kv_heads=self.num_local_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        q, _ = self.q_a_proj(hidden_states)
        q = self.q_a_layernorm(q)
        q, _ = self.q_b_proj(q)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                          dim=-1)
        latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
        kv_a, _ = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv, _ = self.kv_b_proj(kv_a)
        kv = kv.view(-1, self.num_local_heads,
                     self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_pe = latent_cache[:, :, self.kv_lora_rank:]

        q_pe, k_pe = self.rotary_emb(
            positions,
            q_pe.reshape(-1, self.num_local_heads * self.qk_rope_head_dim),
            k_pe.reshape(-1, self.qk_rope_head_dim))
        q_pe = q_pe.view(-1, self.num_local_heads, self.qk_rope_head_dim)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)

        q[..., self.qk_nope_head_dim:] = q_pe

        k = torch.empty_like(q)

        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe

        q = q.reshape(-1, self.num_local_heads * self.qk_head_dim)
        k = k.view(-1, self.num_local_heads * self.qk_head_dim)
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim],
            value=0).view(-1, self.num_local_heads * self.qk_head_dim)

        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        attn_output = attn_output.view(
            -1, self.num_local_heads,
            self.qk_head_dim)[..., :self.v_head_dim].reshape(
                -1, self.num_local_heads * self.v_head_dim)

        output, _ = self.o_proj(attn_output)
        return output


class MiniCPM3DecoderLayer(MiniCPMDecoderLayer):

    def _init_attn_block(self):
        self.input_layernorm = RMSNorm(self.config.hidden_size,
                                       eps=self.config.rms_norm_eps)
        self.self_attn = MiniCPM3Attention(
            config=self.config,
            hidden_size=self.hidden_size,
            num_heads=self.config.num_attention_heads,
            qk_nope_head_dim=self.config.qk_nope_head_dim,
            qk_rope_head_dim=self.config.qk_rope_head_dim,
            v_head_dim=self.config.v_head_dim,
            q_lora_rank=self.config.q_lora_rank,
            kv_lora_rank=self.config.kv_lora_rank,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            max_position_embeddings=self.max_position_embeddings,
            cache_config=self.cache_config,
            quant_config=self.quant_config,
        )


class MiniCPM3Model(MiniCPMModel):

    def _init_layers(
        self,
        prefix: str,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig],
        quant_config: Optional[QuantizationConfig],
    ):
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniCPM3DecoderLayer(config, cache_config,
                                                quant_config),
            prefix=f"{prefix}.layers")


class MiniCPM3ForCausalLM(MiniCPMForCausalLM):
    packed_modules_mapping = {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "kv_a_proj_with_mqa",
        "q_a_proj",
        "q_b_proj",
        "kv_b_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]

    # `embedding_modules` and `embedding_padding_modules`
    # are inherited from MiniCPMForCausalLM

    def _init_model(self):
        self.model = MiniCPM3Model(config=self.config,
                                   cache_config=self.cache_config,
                                   quant_config=self.quant_config,
                                   lora_config=self.lora_config)
