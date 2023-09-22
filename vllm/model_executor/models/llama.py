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
"""Inference-only LLaMA model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm, I8RMSNorm
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.quantized_linear import ParallelLinear
from vllm.model_executor.layers.int8_linear.w8a8linear import W8A8BFP32OFP32LinearWithSFactor, W8A8BFP32OFP32Linear
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding)
from vllm.model_executor.quantization_utils import QuantizationConfig
from vllm.model_executor.weight_utils import (
    convert_pyslice_to_tensor, hf_model_weights_iterator,
    load_tensor_parallel_weights, load_padded_tensor_parallel_vocab)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        if quant_config is not None and quant_config.get_name() == "smoothquant":
            self.gate_up_proj = W8A8BFP32OFP32Linear(hidden_size,
                                                  2 * intermediate_size)
            self.down_proj = W8A8BFP32OFP32LinearWithSFactor(intermediate_size,
                                                    hidden_size)
        else:
            self.gate_up_proj = ParallelLinear.column(hidden_size,
                                                    2 * intermediate_size,
                                                    bias=False,
                                                    gather_output=False,
                                                    perform_initialization=False,
                                                    quant_config=quant_config)
            self.down_proj = ParallelLinear.row(intermediate_size,
                                                hidden_size,
                                                bias=False,
                                                input_is_parallel=True,
                                                perform_initialization=False,
                                                quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        gate_up = gate_up.half()
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        x = x.half()
        return x
    
    def trans_int8(self):
        int8_gate_up = Linear8bitLt(
            self.gate_up_proj.in_features,
            self.gate_up_proj.out_features,
            self.gate_up_proj.bias is not None,
            has_fp16_weights=False,
            threshold=6.0,
        )
        int8_gate_up.weight = bnb.nn.Int8Params(
            self.gate_up_proj.weight.data.clone(), requires_grad=False, has_fp16_weights=False
            ).to(self.gate_up_proj.weight.dtype)
        self.gate_up_proj = int8_gate_up

        int8_down = Linear8bitLt(
            self.down_proj.in_features,
            self.down_proj.out_features,
            self.down_proj.bias is not None,
            has_fp16_weights=False,
            threshold=6.0,
        )
        int8_down.weight = bnb.nn.Int8Params(
            self.down_proj.weight.data.clone(), requires_grad=False, has_fp16_weights=False
            ).to(self.down_proj.weight.dtype)
        self.down_proj = int8_down
    
    def trans_fp4(self):
        int8_gate_up = LinearFP4(
            self.gate_up_proj.in_features,
            self.gate_up_proj.out_features,
            self.gate_up_proj.bias is not None
        )
        int8_gate_up.weight = bnb.nn.Params4bit(
            self.gate_up_proj.weight.data.clone(), requires_grad=False).to(self.gate_up_proj.weight.dtype)
        self.gate_up_proj = int8_gate_up

        int8_down = LinearFP4(
            self.down_proj.in_features,
            self.down_proj.out_features,
            self.down_proj.bias is not None
        )
        int8_down.weight = bnb.nn.Params4bit(
            self.down_proj.weight.data.clone(), requires_grad=False).to(self.down_proj.weight.dtype)
        self.down_proj = int8_down


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # tp_size = get_tensor_model_parallel_world_size()
        tp_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        if quant_config is not None and quant_config.get_name() == "smoothquant":
            self.qkv_proj = W8A8BFP32OFP32Linear(
                hidden_size,
                (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim)
            self.o_proj = W8A8BFP32OFP32LinearWithSFactor(
                self.total_num_heads * self.head_dim,
                hidden_size)
        else:
            self.qkv_proj = ParallelLinear.column(
                hidden_size,
                (self.total_num_heads + 2 * self.total_num_kv_heads) *
                self.head_dim,
                bias=False,
                gather_output=False,
                perform_initialization=False,
                quant_config=quant_config,
            )
            self.o_proj = ParallelLinear.row(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=False,
                input_is_parallel=True,
                perform_initialization=False,
                quant_config=quant_config,
            )
        self.attn = PagedAttentionWithRoPE(self.num_heads,
                                           self.head_dim,
                                           self.scaling,
                                           base=self.rope_theta,
                                           rotary_dim=self.head_dim,
                                           num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        qkv = qkv.half()
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(positions, q, k, v, k_cache, v_cache,
                                input_metadata, cache_event)
        output, _ = self.o_proj(attn_output)
        output = output.half()
        return output
    
    def trans_int8(self):
        int8_qkv = Linear8bitLt(
            self.qkv_proj.in_features,
            self.qkv_proj.out_features,
            self.qkv_proj.bias is not None,
            has_fp16_weights=False,
            threshold=6.0,
        )
        int8_qkv.weight = bnb.nn.Int8Params(
            self.qkv_proj.weight.data.clone(), requires_grad=False, has_fp16_weights=False
            ).to(self.qkv_proj.weight.dtype)
        self.qkv_proj = int8_qkv

        int8_o = Linear8bitLt(
            self.o_proj.in_features,
            self.o_proj.out_features,
            self.o_proj.bias is not None,
            has_fp16_weights=False,
            threshold=6.0,
        )
        int8_o.weight = bnb.nn.Int8Params(
            self.o_proj.weight.data.clone(), requires_grad=False, has_fp16_weights=False
            ).to(self.o_proj.weight.dtype)
        self.o_proj = int8_o

    def trans_fp4(self):
        int8_qkv = LinearFP4(
            self.qkv_proj.in_features,
            self.qkv_proj.out_features,
            self.qkv_proj.bias is not None
        )
        int8_qkv.weight = bnb.nn.Params4bit(
            self.qkv_proj.weight.data.clone(), requires_grad=False).to(self.qkv_proj.weight.dtype)
        self.qkv_proj = int8_qkv

        int8_o = LinearFP4(
            self.o_proj.in_features,
            self.o_proj.out_features,
            self.o_proj.bias is not None
        )
        int8_o.weight = bnb.nn.Params4bit(
            self.o_proj.weight.data.clone(), requires_grad=False).to(self.o_proj.weight.dtype)
        self.o_proj = int8_o

class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            quant_config=quant_config,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = I8RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = I8RMSNorm(config.hidden_size,
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


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size, config.hidden_size, perform_initialization=False)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, quant_config)
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


class LlamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = LlamaModel(config, quant_config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        # NOTE: The LM head is not quantized.
        self.lm_head = ParallelLinear.column(config.hidden_size,
                                             vocab_size,
                                             bias=False,
                                             gather_output=False,
                                             perform_initialization=False,
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
        if self.quant_config is not None and self.quant_config.get_name() == "smoothquant":
            return self._load_int8_weights(
                model_name_or_path,
                cache_dir,
                load_format,
                revision
            )
        
        if self.quant_config is None:
            weight_suffixes = ["weight"]
        else:
            weight_suffixes = self.quant_config.get_tp_tensor_names()

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        tp_size = get_tensor_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        q_proj_shard_size = (self.config.hidden_size // tp_size)
        kv_proj_shard_size = (self.config.hidden_size //
                              self.config.num_attention_heads *
                              self.config.num_key_value_heads // tp_size)
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

            is_packed = False
            is_transposed = False
            if self.quant_config is not None:
                is_packed = self.quant_config.is_packed(name)
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

                if is_packed:
                    shard_size //= self.quant_config.pack_factor
                    offset //= self.quant_config.pack_factor

                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
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
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
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
                                                  tensor_model_parallel_rank)
                continue

            load_tensor_parallel_weights(param, loaded_weight, name,
                                         column_parallel_weights,
                                         row_parallel_weights,
                                         tensor_model_parallel_rank)
    
    def _load_int8_weights(self,
                           model_name_or_path: str,
                           cache_dir: Optional[str] = None,
                           load_format: str = "auto",
                           revision: Optional[str] = None):
        # TODO: support tp in intlinear
        tp_size = 1
        tensor_model_parallel_rank = 0
        q_proj_shard_size = (self.config.hidden_size // tp_size)
        kv_proj_shard_size = (self.config.hidden_size //
                              self.config.num_attention_heads *
                              self.config.num_key_value_heads // tp_size)
        
        if self.quant_config is None:
            weight_suffixes = ["weight"]
        else:
            weight_suffixes = self.quant_config.get_tp_tensor_names()

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")
        
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
            print(f"{name} origin weight shape: {loaded_weight.shape}")
            # bias is useless for llama
            if "bias" in name:
                continue

            is_packed = False
            is_transposed = False
            if self.quant_config is not None:
                is_packed = self.quant_config.is_packed(name)
                is_transposed = self.quant_config.is_transposed(name)
            if is_transposed:
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                loaded_weight = loaded_weight.T

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "qkv_proj")]
                # if is_transposed:
                #     param = param.T

                if is_packed:
                    shard_size //= self.quant_config.pack_factor
                    offset //= self.quant_config.pack_factor
                
                if "proj.a" in name:
                    param.copy_(loaded_weight)
                    is_attention_weight = True
                    continue

                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
                param_slice = param.data[offset:offset + shard_size]
                print(f"{name}  param shape: {param.shape}  param_slice shape:{param_slice.shape} weight shape:{loaded_weight.shape}")
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
                # if is_transposed:
                #     loaded_weight = loaded_weight.T

                if "proj.a" in name:
                    param.copy_(loaded_weight)
                    is_gate_up_weight = True
                    continue

                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
                if is_transposed:
                    loaded_weight = loaded_weight.T
                param_slice = param.data[shard_size * stride_id:shard_size *
                                         (stride_id + 1)]
                print(f"{name}  param shape: {param.shape}  param_slice shape:{param_slice.shape} weight shape:{loaded_weight.shape}")
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]
            if is_transposed:
                param = param.T

            #copy down and out pro
            if "proj.a" in name or "bias" in name or "inscale" in name:
                param.copy_(loaded_weight)
                continue

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight,
                                                  tensor_model_parallel_rank)
                continue

            load_tensor_parallel_weights(param, loaded_weight, name,
                                         column_parallel_weights,
                                         row_parallel_weights,
                                         tensor_model_parallel_rank)        
