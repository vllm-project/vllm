# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
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
"""Inference-only Yi model (https://01.ai) compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from vllm.transformers_utils.configs.yi import YiConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.quantized_linear import ParallelLinear
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.layers import VocabParallelEmbedding
from vllm.model_executor.quantization_utils import QuantizationConfig
from vllm.model_executor.weight_utils import (
    convert_pyslice_to_tensor, hf_model_weights_iterator,
    load_tensor_parallel_weights, load_padded_tensor_parallel_vocab)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class YiMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = ParallelLinear.column(hidden_size,
                                                  2 * intermediate_size,
                                                  bias=False,
                                                  gather_output=False,
                                                  quant_config=quant_config)
        self.down_proj = ParallelLinear.row(intermediate_size,
                                            hidden_size,
                                            bias=False,
                                            input_is_parallel=True,
                                            quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class YiAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        num_kv_heads_replicas = max(1, tp_size // self.total_num_kv_heads)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = ParallelLinear.column(
            hidden_size,
            (self.total_num_heads +
             2 * self.total_num_kv_heads * num_kv_heads_replicas) *
            self.head_dim,
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
        self.attn = PagedAttentionWithRoPE(
            self.num_heads,
            self.head_dim,
            self.scaling,
            base=self.rope_theta,
            max_position=self.max_position_embeddings,
            rotary_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            rope_scaling=rope_scaling)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(positions, q, k, v, k_cache, v_cache,
                                input_metadata, cache_event)
        output, _ = self.o_proj(attn_output)
        return output


class YiDecoderLayer(nn.Module):

    def __init__(
        self,
        config: YiConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = YiAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
        )
        self.mlp = YiMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.ln1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ln2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = self.ln1(hidden_states)
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
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class YiModel(nn.Module):

    def __init__(
        self,
        config: YiConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            YiDecoderLayer(config, quant_config)
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


class YiForCausalLM(nn.Module):

    def __init__(
        self,
        config: YiConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = YiModel(config, quant_config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        # NOTE: The LM head is not quantized.
        self.lm_head = ParallelLinear.column(config.hidden_size,
                                             vocab_size,
                                             bias=False,
                                             gather_output=False,
                                             quant_config=None)
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

    _column_parallel_layers = []
    _row_parallel_layers = ["o_proj", "down_proj"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        if self.quant_config is None:
            col_weight_suffixes = ["weight"]
            row_weight_suffixes = ["weight"]
        else:
            col_weight_suffixes = (
                self.quant_config.get_col_parallel_tensor_names())
            row_weight_suffixes = (
                self.quant_config.get_row_parallel_tensor_names())

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in col_weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in row_weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        q_proj_shard_size = (self.config.hidden_size // tp_size)
        num_kv_heads_replicas = max(1,
                                    tp_size // self.config.num_key_value_heads)
        num_kv_heads_per_gpu = max(1,
                                   self.config.num_key_value_heads // tp_size)
        kv_proj_shard_size = (self.config.hidden_size //
                              self.config.num_attention_heads *
                              num_kv_heads_per_gpu)
        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size,
             q_proj_shard_size + kv_proj_shard_size),
        ]
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue

            packed_dim = None
            is_transposed = False
            if self.quant_config is not None:
                packed_dim = self.quant_config.get_packed_dim(name)
                is_transposed = self.quant_config.is_transposed(name)
            if is_transposed:
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                loaded_weight = loaded_weight.T

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "qkv_proj")]
                if is_transposed:
                    param = param.T

                if packed_dim is not None:
                    shard_dim = 0 if not is_transposed else 1
                    if packed_dim == shard_dim:
                        shard_size //= self.quant_config.pack_factor
                        offset //= self.quant_config.pack_factor

                if weight_name in ["k_proj", "v_proj"]:
                    shard_id = tp_rank // num_kv_heads_replicas
                else:
                    shard_id = tp_rank
                loaded_weight = loaded_weight[shard_size *
                                              shard_id:shard_size *
                                              (shard_id + 1)]
                param_slice = param.data[offset:offset + shard_size]
                assert param_slice.shape == loaded_weight.shape

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
                if is_transposed:
                    param = param.T

                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[shard_size * tp_rank:shard_size *
                                              (tp_rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size *
                                         (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]
            if is_transposed:
                param = param.T

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight,
                                                  tp_rank)
                continue

            load_tensor_parallel_weights(param, loaded_weight, name,
                                         column_parallel_weights,
                                         row_parallel_weights, tp_rank)
