# coding=utf-8
# Adapted from https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
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
"""Inference-only GPT-J model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import GPTJConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SequenceOutputs

KVCache = Tuple[torch.Tensor, torch.Tensor]

# class GPTJForCausalLM(nn.Module):
#     def __init__(self, config: GPTJConfig):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.total_num_heads = config.num_attention_heads

#         tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
#         assert self.total_num_heads % tensor_model_parallel_world_size == 0
#         self.embed_out = ColumnParallelLinear(config.hidden_size,
#                                               config.vocab_size,
#                                               bias=False, gather_output=False,
#                                               perform_initialization=False)
#         self.sampler = Sampler(config.vocab_size)


class GPTJAttention(nn.Module):

    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads

        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size

        self.query_key_value = ColumnParallelLinear(config.hidden_size,
                                                    3 * config.hidden_size,
                                                    gather_output=False,
                                                    perform_initialization=False)
        self.dense = RowParallelLinear(config.hidden_size, config.hidden_size,
                                       input_is_parallel=True,
                                       perform_initialization=False)

        scaling = self.head_size ** -0.5
        self.attn = PagedAttention(self.num_heads, self.head_size, scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)

        q, k, v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata, cache_event)
        output, _ = self.dense(attn_output)
        return output


class GPTJMLP(nn.Module):

    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(config.hidden_size,
                                                  config.intermediate_size,
                                                  gather_output=False,
                                                  perform_initialization=False)
        self.dense_4h_to_h = RowParallelLinear(config.intermediate_size, config.hidden_size,
                                               input_is_parallel=True,
                                               perform_initialization=False)
        self.act = get_act_fn(config.hidden_act)

    def forward(self, hidden_states):
        hidden_states, _ = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.dense_4h_to_h(hidden_states)
        return hidden_states


class GPTJLayer(nn.Module):

    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = GPTJAttention(config)
        self.mlp = GPTJMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        attn_input = self.input_layernorm(hidden_states)
        attn_output = self.attention(
            hidden_states=attn_input,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )

        if self.use_parallel_residual:
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            attn_output = attn_output + hidden_states
            mlp_input = self.post_attention_layernorm(attn_output)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output + attn_output
        return hidden_states


class GPTJModel(nn.Module):

    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.config = config

        self.embed_in = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                               perform_initialization=False)
        self.layers = nn.ModuleList([GPTJLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.embed_in(input_ids)
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class GPTJForCausalLM(nn.Module):

    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.gpt_j = GPTJModel(config)
        self.embed_out = ColumnParallelLinear(config.hidden_size, config.vocab_size,
                                              bias=False, gather_output=False,
                                              perform_initialization=False)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.gpt_j(
            input_ids, kv_caches, input_metadata, cache_events)
        next_tokens = self.sampler(
            self.embed_out.weight, hidden_states, input_metadata)
        return next_tokens

    _column_parallel_weights = ["embed_in.weight", "embed_out.weight", "dense_h_to_4h.weight", "dense_h_to_4h.bias"]
    _row_parallel_weights = ["dense.weight", "dense_4h_to_h.weight"]

    def load_weights(self, model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, use_np_cache):
            if ("attention.bias" in name or "attention.masked_bias" in name):
                continue
            param = state_dict[name]
            if "query_key_value" in name:
                shard_size = param.shape[0]
                loaded_weight = loaded_weight[shard_size * tensor_model_parallel_rank
                                              :shard_size * (tensor_model_parallel_rank + 1)]

                num_heads = self.config.num_attention_heads
                hidden_size = self.config.hidden_size
                head_size = hidden_size // num_heads
                if 'query_key_value.weight' in name:
                    loaded_weight = loaded_weight.view(-1, 3, head_size, hidden_size)
                    loaded_weight = loaded_weight.transpose(0, 1)
                    loaded_weight = loaded_weight.reshape(-1, hidden_size)
                elif 'query_key_value.bias' in name:
                    loaded_weight = loaded_weight.view(-1, 3, head_size)
                    loaded_weight = loaded_weight.transpose(0, 1)
                    loaded_weight = loaded_weight.reshape(-1)
                else:
                    raise ValueError(f"Unexpected weight name: {name}")
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)