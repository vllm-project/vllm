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
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SequenceOutputs

KVCache = Tuple[torch.Tensor, torch.Tensor]

class GPTJAttention(nn.Module):

    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads

        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert self.head_size * tensor_model_parallel_world_size != self.hidden_size

        self.q_proj = ColumnParallelLinear(config.hidden_size, 
                                           config.hidden_size,
                                           bias=False,
                                           gather_output=False,
                                           perform_initialization=False)
        
        self.k_proj = ColumnParallelLinear(config.hidden_size, 
                                           config.hidden_size,
                                           bias=False,
                                           gather_output=False,
                                           perform_initialization=False)
        
        self.v_proj = ColumnParallelLinear(config.hidden_size, 
                                           config.hidden_size,
                                           bias=False,
                                           gather_output=False,
                                           perform_initialization=False)
    
        self.out_projection = RowParallelLinear(config.hidden_size, 
                                           config.hidden_size,
                                           bias=False,
                                           input_is_parallel=False,
                                           perform_initialization=False)
        
        scaling = self.head_size ** -0.5
        rotary_dim = config.rotary_dim 
        self.attn = PagedAttentionWithRoPE(self.num_heads, self.head_size,
                                           scaling, rotary_dim)
        
    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        k_cache, v_cache = kv_cache
        attn_output = self.attn(
            position_ids, q, k, v, k_cache, v_cache, input_metadata, cache_event)
        attn_output, _ = self.out_projection(attn_output)
        return attn_output
    

class GPTJMLP(nn.Module):

    def __init__(self, intermediate_size: int, config: GPTJConfig):
        super().__init__()
        hidden_size = config.n_embd
        self.fc_in = ColumnParallelLinear(hidden_size,
                                          intermediate_size,
                                          gather_output=False,
                                          perform_initialization=False)
        self.fc_out = RowParallelLinear(intermediate_size, 
                                        hidden_size,
                                        input_is_parallel=True,
                                        perform_initialization=False)
        
        self.act = get_act_fn(config.activation_function)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states
    

class GPTJBlock(nn.Module):

    def __init__(self, config: GPTJConfig):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd

        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTJAttention(config)
        self.mlp = GPTJMLP(inner_dim, config)
    
    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + residual + feed_forward_hidden_states
        return hidden_states
    

class GPTJModel(nn.Module):

    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.config = config

        self.embed_dim = config.n_embd

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.wte = VocabParallelEmbedding(vocab_size, 
                                          self.embed_dim, 
                                          perform_initialization=False)
        self.h = nn.ModuleList(
            [GPTJBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds 
        for i in range(len(self.h)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.h[i]
            hidden_states = layer(
                position_ids, 
                hidden_states,
                kv_caches[i], 
                input_metadata, 
                cache_event
            )

        hidden_states = self.ln_f(hidden_states)
        return hidden_states
    
class GPTJForCausalLM(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPTJModel(config)
        self.lm_head = ColumnParallelLinear(config.n_embd,
                                            config.vocab_size,
                                            gather_output=False, 
                                            perform_initialization=False)
        
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.transformer(
            input_ids, token_type_ids, positions, kv_caches, input_metadata, cache_events)
        next_tokens = self.sampler(
            self.lm_head.weight, hidden_states, input_metadata)
        return next_tokens
    
    _column_parallel_weights = ["self.wte.weight", "lm_head.weight", "fc_in.weight", "fc_in.bias"]
    _row_parallel_weights = ["out_projection.weight", "fc_out.weight"]
    
    def load_weights(self, model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, use_np_cache):
            if ("attn.bias" in name or "attn.masked_bias" in name):
                continue
            param = state_dict[name]
            if ("q_proj" in name or "k_proj" in name or "v_proj" in name):
                shard_size = param.shape[0]
                loaded_weight = loaded_weight[shard_size * tensor_model_parallel_rank
                                              :shard_size * (tensor_model_parallel_rank + 1)]

                num_heads = self.config.num_attention_heads
                hidden_size = self.config.hidden_size
                head_size = hidden_size // num_heads
                if ('q_proj.weight' in name or 'k_proj.weight' in name or 'v_proj.weight' in name):
                    loaded_weight = loaded_weight.view(-1, 3, head_size, hidden_size)
                    loaded_weight = loaded_weight.transpose(0, 1)
                    loaded_weight = loaded_weight.reshape(-1, hidden_size)
                elif ('q_proj.bias' in name or 'k_proj.bias' in name or 'v_proj.bias' in name):
                    loaded_weight = loaded_weight.view(-1, 3, head_size)
                    loaded_weight = loaded_weight.transpose(0, 1)
                    loaded_weight = loaded_weight.reshape(-1)
                else:
                    raise ValueError(f"Unexpected weight name: {name}")
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)