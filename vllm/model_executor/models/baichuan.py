# coding=utf-8
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
"""Inference-only BaiChuan model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import (PagedAttentionWithRoPE,
                                                  PagedAttentionWithALiBi)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.quantized_linear import ParallelLinear
from vllm.model_executor.quantization_utils import QuantizationConfig
from vllm.model_executor.weight_utils import (
    convert_pyslice_to_tensor, hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab, load_tensor_parallel_weights,
    get_parallel_weight)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.layers import VocabParallelEmbedding
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.baichuan import BaiChuanConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


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


class BaiChuanMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.gate_up_proj = ParallelLinear.column(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
        )
        self.down_proj = ParallelLinear.row(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class BaiChuanAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        position_embedding: str,
        rope_theta: float = 10000,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size(
        )
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.postion_embedding = position_embedding
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # pylint: disable=invalid-name
        self.W_pack = ParallelLinear.column(
            hidden_size,
            3 * hidden_size,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
        )
        self.o_proj = ParallelLinear.row(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
        )
        # Create the alibi slopes and slice them.
        if self.postion_embedding == "ALIBI":
            tp_rank = get_tensor_model_parallel_rank()
            head_start = tp_rank * self.num_heads
            head_end = (tp_rank + 1) * self.num_heads
            alibi_slopes = _get_alibi_slopes(self.total_num_heads)
            alibi_slopes = alibi_slopes[head_start:head_end].tolist()

            scaling = self.head_dim**-0.5
            self.attn = PagedAttentionWithALiBi(self.num_heads, self.head_dim,
                                                scaling, alibi_slopes)
        else:
            self.scaling = self.head_dim**-0.5
            self.attn = PagedAttentionWithRoPE(
                self.num_heads,
                self.head_dim,
                self.scaling,
                rotary_dim=self.head_dim,
                base=self.rope_theta,
                max_position=self.max_position_embeddings)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.W_pack(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        if self.postion_embedding == "ALIBI":
            attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata,
                                    cache_event)
        else:
            attn_output = self.attn(positions, q, k, v, k_cache, v_cache,
                                    input_metadata, cache_event)

        output, _ = self.o_proj(attn_output)
        return output


class BaiChuanDecoderLayer(nn.Module):

    def __init__(self,
                 config: BaiChuanConfig,
                 position_embedding: str,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = BaiChuanAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            position_embedding=position_embedding,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
        )
        self.mlp = BaiChuanMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class BaiChuanModel(nn.Module):

    def __init__(self,
                 config: BaiChuanConfig,
                 position_embedding: str,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            BaiChuanDecoderLayer(config, position_embedding, quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class BaiChuanBaseForCausalLM(nn.Module):

    def __init__(self,
                 config,
                 position_embedding: str,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = BaiChuanModel(config, position_embedding, quant_config)
        self.lm_head = ParallelLinear.column(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=False,
            quant_config=None,
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
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   input_metadata)
        return next_tokens

    column_parallel_layers = []
    row_parallel_layers = ["o_proj", "down_proj"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        column_parallel_weights, row_parallel_weights = get_parallel_weight(
            self)
        column_weight_suffixes = (
            self.quant_config.get_col_parallel_tensor_names()
        ) if self.quant_config is not None else ["weight", "bias"]

        tp_world_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)

            is_transposed = False
            if self.quant_config is not None:
                is_transposed = self.quant_config.is_transposed(name)
            if is_transposed:
                loaded_weight = loaded_weight.T

            if "W_pack" in name and any(
                    name.endswith(suffix)
                    for suffix in column_weight_suffixes):
                weight_shape = loaded_weight.shape
                total_num_heads = self.config.num_attention_heads
                num_heads = total_num_heads // tp_world_size
                head_start = tp_rank * num_heads
                head_end = (tp_rank + 1) * num_heads

                loaded_weight = loaded_weight.view(3, total_num_heads, -1,
                                                   *weight_shape[1:])
                loaded_weight = loaded_weight[:, head_start:head_end]
                loaded_weight = loaded_weight.reshape(-1, *weight_shape[1:])

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, "gate_up_proj")
                if name not in state_dict:
                    break
                param = state_dict[name]
                if is_transposed:
                    param = param.T
                shard_size = param.shape[0] // 2
                if any(
                        name.endswith(suffix)
                        for suffix in column_weight_suffixes):
                    loaded_weight = loaded_weight[shard_size *
                                                  tp_rank:shard_size *
                                                  (tp_rank + 1)]
                    param_slice = param.data[shard_size *
                                             stride_id:shard_size *
                                             (stride_id + 1)]
                else:
                    param_slice = param.data
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            if name not in state_dict:
                continue
            param = state_dict[name]
            if is_transposed:
                param = param.T

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight,
                                                  tp_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tp_rank,
            )


class BaichuanForCausalLM(BaiChuanBaseForCausalLM):  # baichuan 13b

    def __init__(self, config, quant_config=None):
        super().__init__(config, "ALIBI", quant_config)


class BaiChuanForCausalLM(BaiChuanBaseForCausalLM):  # baichuan 7b

    def __init__(self, config, quant_config=None):
        super().__init__(config, "ROPE", quant_config)
