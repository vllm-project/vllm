# coding=utf-8
# Adapted from
# https://huggingface.co/models?filter=glm

# Copyright 2023 The vLLM team.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Inference-only GLM model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.sequence import SequenceOutputs
from vllm.transformers_utils.configs.glm import GLMConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]

class GLMAttention(nn.Module):
    def __init__(self, config: GLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim**-0.5

        self.multi_query_attention = config.multi_query_attention if hasattr(config, "multi_query_attention") else False
        if self.multi_query_attention:
            self.num_kv_heads = config.multi_query_group_num
            self.kv_dim = self.head_dim
            self.c_attn_q = ColumnParallelLinear(self.hidden_size,
                                                 self.hidden_size,
                                                 bias=True,
                                                 gather_output=False,
                                                 perform_initialization=False)
            self.c_attn_kv = nn.Linear(self.hidden_size,
                                       2 * self.num_kv_heads * self.kv_dim,
                                       bias=True)
        else:
            self.num_kv_heads = self.num_heads
            self.kv_dim = self.num_kv_heads * self.head_dim
            self.query_key_value = ColumnParallelLinear(self.hidden_size,
                                               3 * self.hidden_size,
                                               bias=True,
                                               gather_output=False,
                                               perform_initialization=False)

        self.dense = RowParallelLinear(self.hidden_size,
                                        self.hidden_size,
                                        bias=True,
                                        input_is_parallel=True,
                                        perform_initialization=False)
        self.attn = PagedAttention(self.num_heads,
                                   self.head_dim,
                                   scale=self.scale,
                                   num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        if self.multi_query_attention:
            q, _ = self.c_attn_q(hidden_states)
            kv = self.c_attn_kv(hidden_states)
            k, v = kv.split([self.num_kv_heads * self.kv_dim, self.num_kv_heads * self.kv_dim], dim=-1)
        else:
            qkv, _ = self.query_key_value(hidden_states)
            q, k, v = qkv.split([self.hidden_size, self.hidden_size, self.hidden_size],
                                dim=-1)
        key_cache, value_cache = kv_cache
        attn_output = self.attn(q, k, v, key_cache, value_cache,
                                input_metadata, cache_event)
        attn_output, _ = self.dense(attn_output)
        return attn_output


class GLMMLP(nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        config: GLMConfig,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size,
                                         intermediate_size,
                                         bias=True,
                                         gather_output=False,
                                         perform_initialization=False)
        self.dense_4h_to_h = RowParallelLinear(intermediate_size,
                                        hidden_size,
                                        bias=True,
                                        input_is_parallel=True,
                                        perform_initialization=False)
        self.act = get_act_fn("gelu")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.dense_4h_to_h(hidden_states)
        return hidden_states

class GLMBlock(nn.Module):

    def __init__(self, config: GLMConfig):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim =  4 * hidden_size

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1.0e-5)
        self.attention = GLMAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1.0e-5)
        self.mlp = GLMMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.attention(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states

class GLMModel(nn.Module):

    def __init__(self, config: GLMConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.block_position_encoding = config.block_position_encoding
        # Optimization: While the vocab size of GPT-2 is 50257, we extend it
        # to 50304 in order to make it divisible by 64.
        # This improves performance since GPUs are faster if the dimension
        # is divisible by 64. In addition, it allows us to shard the embedding
        # layer across 2, 4, 8, or more GPUs.
        # vocab_size = ((config.vocab_size + 63) // 64) * 64
        vocab_size = config.vocab_size
        self.word_embeddings = VocabParallelEmbedding(vocab_size, self.embed_dim)
        if config.block_position_encoding:
            self.position_embeddings = nn.Embedding(
                config.max_sequence_length + 1, self.embed_dim)
            self.block_position_embeddings = nn.Embedding(
                config.max_sequence_length + 1, self.embed_dim)
        else:
            self.position_embeddings = nn.Embedding(config.max_sequence_length, self.embed_dim)
        self.layers = nn.ModuleList(
            [GLMBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = nn.LayerNorm(self.embed_dim, eps=1.0e-5)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        inputs_embeds = self.word_embeddings(input_ids)
        is_not_warmup = position_ids.ndim > 1
        if self.block_position_encoding and is_not_warmup:
            position_ids = position_ids.permute(*range(position_ids.dim() - 2), -1, -2)
            position_ids, block_position_ids = position_ids[...,0], position_ids[...,1]
            position_ids = position_ids.view(-1)
            block_position_ids = block_position_ids.view(-1)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeds
        if self.block_position_encoding and is_not_warmup:
            block_position_embeddings = self.block_position_embeddings(block_position_ids)
            hidden_states = hidden_states + block_position_embeddings

        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(hidden_states, kv_caches[i], input_metadata,
                                  cache_event)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class GLMForCausalLM(nn.Module):

    def __init__(self, config: GLMConfig):
        super().__init__()
        self.config = config
        self.transformer = GLMModel(config)
        # TODO(zhuohan): create a new weight after implementing pipeline
        #                parallelism
        self.word_embeddings = self.transformer.word_embeddings.weight
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         input_metadata, cache_events)
        next_tokens = self.sampler(self.word_embeddings, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = [
        "word_embeddings.weight", "dense_h_to_4h.weight", "dense_h_to_4h.bias"
    ]
    _row_parallel_weights = ["dense.weight", "dense_4h_to_h.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        tp_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache):
            if not name.startswith("transformer."):
                name = "transformer." + name
            param = state_dict[name]

            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights, tp_rank)
