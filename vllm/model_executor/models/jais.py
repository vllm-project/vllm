# coding=utf-8
# Adapted from
# https://huggingface.co/core42/jais-13b-chat/blob/main/modeling_jais.py
# Copyright 2024 The Core42 team.
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
"""Inference-only Jais model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import List, Optional, Tuple

import torch
import math
from torch import nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttentionWithALiBi, PagedAttention
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (
    convert_pyslice_to_tensor,
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights
)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, 
    get_tensor_model_parallel_world_size
)
from vllm.model_executor.parallel_utils.layers import (
    VocabParallelEmbedding,
    ColumnParallelLinear,
    RowParallelLinear
)
from transformers.activations import ACT2FN


KVCache = Tuple[torch.Tensor, torch.Tensor]


class AlibiPositionEmbeddingLayer(nn.Module):
    def __init__(self, num_heads):
        super(AlibiPositionEmbeddingLayer, self).__init__()

        self.num_heads = num_heads
        slopes = torch.tensor(
            AlibiPositionEmbeddingLayer._get_alibi_slopes(num_heads)
        ).unsqueeze(-1)
        self.slopes = nn.parameter.Parameter(slopes, requires_grad=False)

    def forward(self, seq_length, key_length, cached_qk_len):
        context_position = torch.arange(
            cached_qk_len, cached_qk_len + seq_length, device=self.slopes.device
        )[:, None]
        memory_position = torch.arange(
            key_length + cached_qk_len, device=self.slopes.device
        )[None, :]
        relative_position = memory_position - context_position
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.num_heads, -1, -1)
        alibi = (self.slopes * -1.0).unsqueeze(1) * relative_position
        return alibi

    @staticmethod
    def _get_alibi_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + AlibiPositionEmbeddingLayer._get_alibi_slopes(
                    2 * closest_power_of_2
                )[0::2][: n - closest_power_of_2]
            )

def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2**math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2**(-(2**-(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2**(-(2**-(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2,
                                  total_num_heads - closest_power_of_2)
        extra_powers = torch.arange(start=1,
                                    end=1 + 2 * num_remaining_heads,
                                    step=2,
                                    dtype=torch.int32)
        slopes = torch.cat(
            [slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes

class SwiGLUActivation(nn.Module):
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 * nn.functional.silu(x2)

class JAISAttention(nn.Module):

    def __init__(
        self,
        config,
        linear_method=None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim**-1.0

        self.c_attn = ColumnParallelLinear(
            self.hidden_size,
            3 * self.hidden_size,
            total_num_heads,
            gather_output=False,
        )
        self.c_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            input_is_parallel=True,
            #linear_method=linear_method,
        )
        # Create the alibi slopes and slice them.
        # if self.postion_embedding == "ALIBI":
        tp_rank = get_tensor_model_parallel_rank()
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        alibi_slopes = _get_alibi_slopes(total_num_heads)
        alibi_slopes = alibi_slopes[head_start:head_end].tolist()

        # scaling = self.head_dim ** -0.5
        self.attn = PagedAttentionWithALiBi(self.num_heads,
                                            self.head_dim,
                                            self.scale, 
                                            alibi_slopes)

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


class JAISMLP(nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        config,
        linear_method=None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.swiglu = config.activation_function == "swiglu"
        self.c_fc = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            #linear_method=linear_method,
        )
        self.c_fc2 = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            #linear_method=linear_method,
        ) if self.swiglu else None
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            #linear_method=linear_method,
        )
        #quant_config = getattr(linear_method, "quant_config", None)

        #TODO: add swiglu to https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py#L102
        # self.act = get_act_fn(config.activation_function, quant_config,
        #                       intermediate_size)
        # self.act = SiluAndMul()
        
        self.act = SwiGLUActivation() if self.swiglu else ACT2FN[config.activation_function]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.swiglu:
            hidden_states2, _ = self.c_fc2(hidden_states)
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states, hidden_states2) if self.swiglu else self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class JAISBlock(nn.Module):

    def __init__(
        self,
        config,
        linear_method=None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = (config.n_inner if config.n_inner is not None else 4 *
                     hidden_size)

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = JAISAttention(config, linear_method)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = JAISMLP(inner_dim, config, linear_method)

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


class JAISModel(nn.Module):

    def __init__(
        self,
        config,
        linear_method=None,
    ):
        super().__init__()
        self.config = config
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn
        self.embed_dim = config.hidden_size
        self.wte = VocabParallelEmbedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim) if config.position_embedding_type != "alibi" else None
        self.h = nn.ModuleList([
            JAISBlock(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # TODO: include position bias
        self.relative_pe = AlibiPositionEmbeddingLayer(
            config.num_attention_heads) if config.position_embedding_type == "alibi" else None

        self.embeddings_scale = config.embeddings_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        if self.wpe is not None:
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds

        hidden_states *= torch.tensor(
            float(self.embeddings_scale), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # if self.relative_pe is not None:
        #     length = input_ids.shape[1]
        #     cached_kv_length = 0
        #     # cached_kv = past_key_values[0]
        #     cached_kv = None
        #     if cached_kv is not None:
        #         cached_kv_length = cached_kv[0].shape[-2]
        #     position_bias = self.relative_pe(length, length, cached_kv_length)
        # else:
        #     position_bias = None

        # TODO: include position bias
        for i in range(len(self.h)):
            cache_event = None if cache_events is None else cache_events[i]
            layer = self.h[i]
            hidden_states = layer(hidden_states, kv_caches[i], input_metadata,
                                  cache_event)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class JAISLMHeadModel(nn.Module):

    def __init__(
        self,
        config,
        linear_method=None,
    ):
        super().__init__()
        self.config = config

        # TODO: use logits_scale in Sampler
        self.output_logits_scale = config.width_scale
        self.linear_method = linear_method
        self.transformer = JAISModel(config, linear_method)
        self.lm_head_weight = self.transformer.wte.weight
        self.sampler = Sampler(config.vocab_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ):
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         input_metadata, cache_events)
        
        # lm_logits = self.lm_head(hidden_states)
        # lm_logits *= torch.tensor(
        #     float(self.output_logits_scale), dtype=lm_logits.dtype, device=lm_logits.device
        # )
        next_tokens = self.sampler(self.lm_head_weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = ["c_fc.weight", "c_fc.bias", "c_fc2.weight", "c_fc2.bias"]
    _row_parallel_weights = ["c_proj.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        tensor_model_parallel_world_size = (get_tensor_model_parallel_world_size())
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        # print(state_dict)

        for name, param in self.named_parameters():
            if param.requires_grad:
                print (name, param.data.shape)        
        print(self)

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "lm_head.weight" in name:
                # Jais ties the weights of the embedding layer and the final
                # linear layer.
                continue
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue

            if not name.startswith("transformer."):
                name = "transformer." + name

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)

            # The HF's Jais implementation uses Conv1D instead of Linear.
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
                                                  tensor_model_parallel_rank
                                                  )
                continue

            # For the fused QKV linear layer, manually shard the weights.
            if "c_attn" in name:
                # Jais's fused QKV has the shape of
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
