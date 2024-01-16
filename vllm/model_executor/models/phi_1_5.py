# coding=utf-8
# Adapted from
# https://huggingface.co/microsoft/phi-1_5/blob/main/modeling_phi.py
# Copyright 2023 The vLLM team.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# BSD 3-Clause License
#
# Copyright (c) 2022, Tri Dao, trid@cs.stanford.edu.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Inference-only Phi-1.5 model compatible with HuggingFace weights."""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class PhiEmbedding(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

    def forward(self, input_ids: torch.LongTensor):
        return self.wte(input_ids)


class PhiAttention(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads

        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)

        # pylint: disable=C0103
        self.Wqkv = QKVParallelLinear(
            self.hidden_size,
            self.head_size,
            self.total_num_heads,
            linear_method=linear_method,
        )
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_size,
            self.total_num_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.out_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            linear_method=linear_method,
        )

        scaling = self.head_size**-0.5
        rotary_dim = config.rotary_dim
        assert rotary_dim % 2 == 0

        # pylint: disable=C0301
        # Refer to:
        # https://huggingface.co/microsoft/phi-1_5/blob/d212a789620c380ff32ca1d1ee9943a777360987/modeling_phi.py#L518
        rope_theta = 10000
        max_position_embeddings = getattr(config, "n_positions", 2048)
        self.rotary_emb = get_rope(
            self.head_size,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
        )
        self.attn = PagedAttention(self.num_heads, self.head_size, scaling)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.Wqkv(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.out_proj(attn_output)
        return output


class PhiMLP(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()

        n_inner = getattr(config, "n_inner", None)
        n_inner = n_inner if n_inner is not None else 4 * config.hidden_size

        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            n_inner,
            linear_method=linear_method,
        )
        self.fc2 = RowParallelLinear(
            n_inner,
            config.hidden_size,
            linear_method=linear_method,
        )
        quant_config = getattr(linear_method, "quant_config", None)
        self.act = get_act_fn(config.activation_function, quant_config,
                              n_inner)

    def forward(self, hidden_states):
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class PhiLayer(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size,
                               eps=config.layer_norm_epsilon)
        self.mixer = PhiAttention(config, linear_method)
        self.mlp = PhiMLP(config, linear_method)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln(hidden_states)
        attn_outputs = self.mixer(
            position_ids=position_ids,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        return hidden_states


class PhiModel(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.embd = PhiEmbedding(config)
        self.h = nn.ModuleList([
            PhiLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embd(input_ids)
        for i in range(self.config.num_hidden_layers):
            layer = self.h[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
            )
        return hidden_states


class PhiCausalLMHead(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size,
                               eps=config.layer_norm_epsilon)
        self.linear = ParallelLMHead(config.vocab_size,
                                     config.hidden_size,
                                     bias=True)


class PhiForCausalLM(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.linear_method = linear_method

        self.transformer = PhiModel(config, linear_method)
        self.lm_head = PhiCausalLMHead(config)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         input_metadata)
        hidden_states = self.lm_head.ln(hidden_states)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        head = self.lm_head.linear
        next_tokens = self.sampler(head.weight, hidden_states,
                                   sampling_metadata, head.bias)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue

            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            # pylint: disable=E1136
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
