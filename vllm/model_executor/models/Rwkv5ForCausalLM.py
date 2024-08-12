# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/RWKV/modeling_RWKV.py
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
"""Inference-only GPT-2 model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import RwkvConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput


class RWKVAttention(nn.Module):

    def __init__(
        self,
        config: RwkvConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim =  self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.receptance = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.time_mix_key = nn.Parameter(torch.zeros(1,1,self.hidden_size))
        self.time_mix_value = nn.Parameter(torch.zeros(1,1,self.hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.zeros(1,1,self.hidden_size))
        self.time_mix_gate = nn.Parameter(torch.zeros(1,1,self.hidden_size))
        self.time_decay = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.time_faaaa = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.ln_x = nn.GroupNorm(self.num_heads,self.hidden_size)
        self.head_size_divisor = 8

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        x = hidden_states
        ott = torch.arange(x.shape[0])
        ott = ott-1
        state = x[ott]
        state[ott==-1] = kv_cache[1,0,0,:,:,0].reshape_as(state[ott==-1]) if kv_cache != None else torch.zeros_like(state[ott==-1])
        # state[position_ids.query_start_loc] = cache
        # cache = state[position_ids.query_start_loc + position_ids.seq_lens]
        
        xx =  state - x
        xk = x + xx * (1-self.time_mix_key[0])
        xv = x + xx * (1-self.time_mix_value[0])
        xr = x + xx * (1-self.time_mix_receptance[0])
        xg = x + xx * (1-self.time_mix_gate[0])

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        g = torch.nn.functional.silu(self.gate(xg))

        k = k.view(-1,self.num_heads, self.head_dim,1)
        v = v.view(-1,self.num_heads, 1, self.head_dim)
        r = r.view(-1,self.num_heads, self.head_dim, 1).transpose(-1,-2)
        
        at = (k*v)
        u = self.time_faaaa.reshape(1,self.num_heads,self.head_dim,1).transpose(-1,-2)
        w = self.time_decay.reshape(self.num_heads,self.head_dim,1).exp().neg().exp()
        # print(at.shape, r.shape, u.shape)
        ur = (u*r)
        # print(ur.shape)
        out = ur@at

        # if(kv_cache != None):
        #     print(attn_metadata.num_prefill_tokens)
        
        T = attn_metadata.num_prefill_tokens
        
        if (T == 0): T = attn_metadata.num_decode_tokens
        
        if(attn_metadata.num_prefill_tokens != 0):
        # print(kv_cache.shape if kv_cache != None else None)
            s = kv_cache[0,0,0] if kv_cache != None else torch.zeros(self.num_heads, self.head_dim, self.head_dim, device=at.device, dtype=at.dtype)
            # print(kv_cache.shape if kv_cache != None else None)
            for t in range(T):
                # print(out[t].shape, r[t].shape, s.shape)
                out[t] += r[t] @ s
                s *= w
                s += at[t]
            
            if(kv_cache != None):
                kv_cache[1,0,0,:,:,0] = x[attn_metadata.num_prefill_tokens-1].reshape_as(kv_cache[1,0,0,:,:,0])
                kv_cache[0,0,0] = s

        else:
            # print(kv_cache.shape if kv_cache != None else None)
       
            
            for t in range(T):
                s = kv_cache[0,0,0] if kv_cache != None else torch.zeros(self.num_heads, self.head_dim, self.head_dim, device=at.device, dtype=at.dtype)
            
                # print(out[t].shape, r[t].shape, s.shape)
                out[t] += r[t] @ s
                s *= w
                s += at[t]
            
                if(kv_cache != None):
                    kv_cache[1,0,0,:,:,0] = x[attn_metadata.num_prefill_tokens-1].reshape_as(kv_cache[1,0,0,:,:,0])
                    kv_cache[0,0,0] = s
        out = out.view(-1, self.hidden_size)
        out = self.ln_x(out/self.head_size_divisor)
        hidden_states = self.output(out*g)

        
        # print(attn_metadata.num_decode_tokens)
        # print(attn_metadata.num_prefill_tokens)
        # for i in range(attn_metadata.slot_mapping.__len__()):
        #     pass
        #     # if(attn_metadata.slot_mapping[i] == -1):
        #     #     attn_metadata.slot_mapping[i] = i
        #     # else:
        #     #     pass
            
        # if(kv_cache != None):
        #     print(kv_cache.shape)
        #     print(kv_cache[0,0,0,0,0])
        #     kv_cache[0,0,0,0,0] += 1
        # exit()
        # state = hidden_states[ott]

        # state[attn_metadata.query_start_loc] = kv_cache
        # kv_cache = state[attn_metadata.query_start_loc + attn_metadata.seq_lens]

        return kv_cache, hidden_states


class RWKVMLP(nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        config: RwkvConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size

    
        self.time_mix_key = nn.Parameter(torch.zeros(1,1,hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.zeros(1,1,hidden_size))

        self.key = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=True,
            # quant_config=quant_config,
        )
        self.value = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=True,
            # quant_config=quant_config,
        )
        self.receptance = nn.Linear(
            hidden_size,
            hidden_size,
            bias=True,
            # quant_config=quant_config,
        )

    def forward(self, x, kv_cache, attn_metadata:AttentionMetadata):
    
        ott = torch.arange(x.shape[0])
        ott = ott-1
        state = x[ott]
        state[ott==-1] = kv_cache[1,0,0,:,:,1].reshape_as(state[ott==-1]) if kv_cache != None else torch.zeros_like(state[ott==-1])
        # state[position_ids.query_start_loc] = cache
        # cache = state[position_ids.query_start_loc + position_ids.seq_lens]

        xx =  state - x
        xk = x + xx * (1-self.time_mix_key[0])
        xr = x + xx * (1-self.time_mix_receptance[0])

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)

        if(kv_cache != None):
            kv_cache[1,0,0,:,:,1] = x[attn_metadata.num_prefill_tokens-1].reshape_as(kv_cache[1,0,0,:,:,1])

        return torch.sigmoid(self.receptance(xr)) * kv


class RWKVBlock(nn.Module):

    def __init__(
        self,
        config: RwkvConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        position = 0
    ):
        super().__init__()
        hidden_size = config.hidden_size

        if position == 0:
            self.pre_ln = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        

        # self.ln0 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.ln1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attention = RWKVAttention(config, cache_config, quant_config)
        self.ln2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.feed_forward = RWKVMLP(int(config.hidden_size*3.5), config, quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_ids: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        kv_cache,attn_output = self.attention(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        feed_forward_hidden_states = self.feed_forward(hidden_states, kv_cache, attn_metadata)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


class RWKV5Model(nn.Module):

    def __init__(
        self,
        config: RwkvConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.embeddings = VocabParallelEmbedding(config.vocab_size, self.embed_dim)
        self.blocks = nn.ModuleList([
            RWKVBlock(config, cache_config, quant_config, _)
            for _ in range(config.num_hidden_layers)
        ])
        self.ln_out = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.head = nn.Linear(self.embed_dim,config.vocab_size, False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # print(position_ids.size(),position_ids)
        # print(attn_metadata)
        inputs_embeds = self.embeddings(input_ids)

        hidden_states = self.blocks[0].pre_ln(inputs_embeds)

        for i in range(len(self.blocks)):
            layer = self.blocks[i]
            hidden_states = layer(hidden_states, kv_caches[i], attn_metadata, position_ids)

        hidden_states = self.ln_out(hidden_states)
        
        return hidden_states


class Rwkv5ForCausalLM(nn.Module):

    def __init__(
        self,
        config: RwkvConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        print(config)
        print(cache_config)
        print(quant_config)
        self.rwkv = RWKV5Model(config, cache_config, quant_config)
        self.lm_head_weight = self.rwkv.head.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.rwkv(input_ids, positions, kv_caches,
                                         attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head_weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        print(params_dict.keys())
            
        for name, loaded_weight in weights:
            if not name.startswith("rwkv."):
                name = "rwkv." + name

            
            param = params_dict[name]
            
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            print(name, param.size(), loaded_weight.size())
            weight_loader(param, loaded_weight)
