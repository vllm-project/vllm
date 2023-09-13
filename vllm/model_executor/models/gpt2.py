# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/gpt2/modeling_gpt2.py
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
"""Inference-only GPT-2 model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import GPT2Config

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (
    convert_pyslice_to_tensor, hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab, load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class GPT2Attention(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim**-0.5

        self.c_attn = ColumnParallelLinear(self.hidden_size,
                                           3 * self.hidden_size,
                                           bias=True,
                                           gather_output=False,
                                           perform_initialization=False)
        self.c_proj = RowParallelLinear(self.hidden_size,
                                        self.hidden_size,
                                        bias=True,
                                        input_is_parallel=True,
                                        perform_initialization=False)
        self.attn = PagedAttention(self.num_heads,
                                   self.head_dim,
                                   scale=self.scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        key_cache, value_cache = kv_cache
        attn_output = self.attn(q, k, v, key_cache, value_cache,
                                input_metadata, cache_event)
        attn_output, _ = self.c_proj(attn_output)
        return attn_output


class GPT2MLP(nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        config: GPT2Config,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.c_fc = ColumnParallelLinear(hidden_size,
                                         intermediate_size,
                                         bias=True,
                                         gather_output=False,
                                         perform_initialization=False)
        self.c_proj = RowParallelLinear(intermediate_size,
                                        hidden_size,
                                        bias=True,
                                        input_is_parallel=True,
                                        perform_initialization=False)
        self.act = get_act_fn(config.activation_function)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = (config.n_inner if config.n_inner is not None else 4 *
                     hidden_size)

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


class GPT2Model(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn
        self.embed_dim = config.hidden_size

        # Optimization: While the vocab size of GPT-2 is 50257, we extend it
        # to 50304 in order to make it divisible by 64.
        # This improves performance since GPUs are faster if the dimension
        # is divisible by 64. In addition, it allows us to shard the embedding
        # layer across 2, 4, 8, or more GPUs.
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.wte = VocabParallelEmbedding(vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.h = nn.ModuleList(
            [GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        for i in range(len(self.h)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.h[i]
            hidden_states = layer(hidden_states, kv_caches[i], input_metadata,
                                  cache_event)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPT2LMHeadModel(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        # TODO(zhuohan): create a new weight after implementing pipeline
        #                parallelism
        self.lm_head_weight = self.transformer.wte.weight
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head_weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = ["c_fc.weight", "c_fc.bias"]
    _row_parallel_weights = ["c_proj.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto"):
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format):
            if "lm_head.weight" in name:
                # GPT-2 ties the weights of the embedding layer and the final
                # linear layer.
                continue
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue

            if not name.startswith("transformer."):
                name = "transformer." + name

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)

            # The HF's GPT-2 implementation uses Conv1D instead of Linear.
            # Because of this, we need to transpose the weights.
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in name:
                    continue
                if not name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()
            param = state_dict[name]

            if name == "transformer.wte.weight":
                load_padded_tensor_parallel_vocab(param, loaded_weight,
                                                  tensor_model_parallel_rank)
                continue

            # For the fused QKV linear layer, manually shard the weights.
            if "c_attn" in name:
                # GPT-2's fused QKV has the shape of
                # [3 * num_heads * head_size, hidden_size].
                # When tensor parallelism is used, we shard the weights along
                # the head dimension.
                total_num_heads = self.config.num_attention_heads
                hidden_size = self.config.hidden_size
                head_size = hidden_size // total_num_heads
                num_heads = total_num_heads // tensor_model_parallel_world_size
                head_start = tensor_model_parallel_rank * num_heads
                head_end = (tensor_model_parallel_rank + 1) * num_heads

                if name.endswith(".weight"):
                    loaded_weight = loaded_weight.view(3, total_num_heads,
                                                       head_size, hidden_size)
                    loaded_weight = loaded_weight[:, head_start:head_end, :, :]
                    loaded_weight = loaded_weight.reshape(-1, hidden_size)
                elif name.endswith(".bias"):
                    loaded_weight = loaded_weight.view(3, total_num_heads,
                                                       head_size)
                    loaded_weight = loaded_weight[:, head_start:head_end, :]
                    loaded_weight = loaded_weight.reshape(-1)
                else:
                    raise ValueError(f"Unexpected parameter name {name}")
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)
