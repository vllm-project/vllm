# coding=utf-8
# Adapted from
# https://huggingface.co/microsoft/phi-1_5/blob/main/modeling_mixformer_sequential.py
# Copyright 2023 The vLLM team.
# Copyright 2023 Microsoft and HuggingFace Teams. All rights reserved.
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
"""Inference-only Phi-1.5 model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.layers import (VocabParallelEmbedding,
                                                       ColumnParallelLinear,
                                                       RowParallelLinear)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class PhiAttention(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.n_embd
        self.head_size = self.hidden_size // self.total_num_heads

        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)

        # pylint: disable=C0103
        self.Wqkv = ColumnParallelLinear(
            config.n_embd,
            3 * config.n_embd,
            gather_output=False,
        )
        self.out_proj = RowParallelLinear(
            config.n_embd,
            config.n_embd,
            input_is_parallel=True,
        )

        scaling = self.head_size**-0.5
        rotary_dim = config.rotary_dim
        assert rotary_dim % 2 == 0

        # pylint: disable=C0301
        # See https://huggingface.co/microsoft/phi-1_5/blob/92557d03bb12543040c8bb5f0475cbdd9968f05f/modeling_mixformer_sequential.py#L222
        rope_theta = 10000
        max_position_embeddings = getattr(config, "n_positions", 2048)
        self.attn = PagedAttentionWithRoPE(
            self.num_heads,
            self.head_size,
            scaling,
            rotary_dim,
            base=rope_theta,
            max_position=max_position_embeddings)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.Wqkv(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(position_ids, q, k, v, k_cache, v_cache,
                                input_metadata, cache_event)
        output, _ = self.out_proj(attn_output)
        return output


class PhiMLP(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        n_inner = getattr(config, "n_inner", None)
        n_inner = n_inner if n_inner is not None else 4 * config.n_embd

        self.fc1 = ColumnParallelLinear(
            config.n_embd,
            n_inner,
            gather_output=False,
        )
        self.fc2 = RowParallelLinear(
            n_inner,
            config.n_embd,
            input_is_parallel=True,
        )
        self.act = get_act_fn(config.activation_function)

    def forward(self, hidden_states):
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class PhiLayer(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mixer = PhiAttention(config)
        self.mlp = PhiMLP(config)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln(hidden_states)
        attn_outputs = self.mixer(
            position_ids=position_ids,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        # pylint: disable=C0301
        # Dropout 0.0 https://huggingface.co/microsoft/phi-1_5/blob/92557d03bb12543040c8bb5f0475cbdd9968f05f/modeling_mixformer_sequential.py#L696
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        return hidden_states


class PhiModel(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.n_embd,
        )
        self.layers = nn.ModuleList(
            [PhiLayer(config) for _ in range(config.n_layer)])

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                position_ids,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        return hidden_states


class PhiForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.phi = PhiModel(config)
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.linear = ColumnParallelLinear(
            config.n_embd,
            config.vocab_size,
            gather_output=False,
        )
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.phi(input_ids, positions, kv_caches,
                                 input_metadata, cache_events)
        hidden_states = self.ln(hidden_states)
        next_tokens = self.sampler(self.linear.weight, hidden_states,
                                   input_metadata, self.linear.bias)
        return next_tokens

    _column_parallel_weights = [
        "embed_in.weight", "embed_out.weight", "embed_out.bias", "fc1.weight",
        "fc1.bias"
    ]
    _row_parallel_weights = ["out_proj.weight", "fc2.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                # FIXME: This is a hack. Handle the following by post-initializing RoPE.
                t = torch.arange(self.config.n_positions, dtype=torch.float32)

                freqs = torch.einsum("i,j -> ij", t, loaded_weight)
                cos = freqs.cos()
                sin = freqs.sin()
                cache = torch.cat((cos, sin), dim=-1)

                for i in range(len(self.phi.layers)):
                    self.phi.layers[
                        i].mixer.attn.rotary_emb.cos_sin_cache.copy_(cache)
                continue
            _, layer_idx, *tail = name.split(".")
            tail = ".".join(tail)
            layer_idx = int(layer_idx)

            # First or last layers are Embeddings and CausalLMHead respectively
            if layer_idx == 0:
                key = f"phi.{tail}"
            elif layer_idx == self.config.n_layer + 1:
                key = tail
            else:
                key = f"phi.layers.{layer_idx - 1}.{tail}"

            # pylint: disable=E1136
            param = state_dict[key]
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)
