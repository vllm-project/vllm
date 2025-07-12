# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2023 Stability AI, EleutherAI, and The HuggingFace Inc. team.
# All rights reserved.
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
#
# This code is based off the following work:
# https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/modeling_stablelm_alpha.py
"""Inference-only StableLM-Alpha model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Optional, Union, cast

import torch
from torch import nn

from vllm.attention import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.stablelm_alpha import StableLMAlphaConfig

from .interfaces import SupportsPP
from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


class StableLMAlphaMLP(nn.Module):

    def __init__(self,
                 config: StableLMAlphaConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        # Calculate intermediate size using StableLM-Alpha's specific logic
        multiple_of = 256
        ff_dim = int(8 * hidden_size / 3)
        intermediate_size = multiple_of * (
            (ff_dim + multiple_of - 1) // multiple_of)

        # Gate projection outputs 2 * intermediate_size, then gets chunked
        self.gate_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],  # ff and ff_gate
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj")

        self.out_proj = RowParallelLinear(intermediate_size,
                                          hidden_size,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.out_proj")

        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.out_proj(x)
        return x


class StableLMAlphaAttention(nn.Module):

    def __init__(self,
                 config: StableLMAlphaConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_heads
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_key_value_heads = config.num_heads
        if self.total_num_key_value_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_key_value_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_key_value_heads == 0
        self.num_key_value_heads = max(
            1, self.total_num_key_value_heads // tp_size)

        self.head_dim = self.hidden_size // self.total_num_heads
        self.max_position_embeddings = config.max_position_embeddings

        # Partial rotary embeddings
        self.rotary_ndims = int(self.head_dim * config.rotary_pct)

        self.scaling = self.head_dim**-0.5
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        if (self.head_dim * self.num_heads * tp_size) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads "
                             f"(got `hidden_size`: {self.hidden_size}"
                             f" and `num_heads`: {self.num_heads}).")

        self.qkv_proj = QKVParallelLinear(self.hidden_size,
                                          self.head_dim,
                                          self.total_num_heads,
                                          self.total_num_key_value_heads,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.qkv_proj")

        self.out_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                          self.hidden_size,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.out_proj")

        # Rotary embedding with partial rotary support
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_ndims,
            max_position=self.config.max_position_embeddings,
            base=self.config.rotary_emb_base,
        )

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_key_value_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class StableLMAlphaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: StableLMAlphaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        self.attention = StableLMAlphaAttention(config,
                                                cache_config,
                                                quant_config,
                                                prefix=f"{prefix}.attention")

        self.mlp = StableLMAlphaMLP(config,
                                    quant_config,
                                    prefix=f"{prefix}.mlp")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm: apply normalization once and use for both attention and MLP
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        # Self-attention
        attn_output = self.attention(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Feed-forward (using the same normalized input)
        mlp_output = self.mlp(hidden_states)

        # Residual connection: residual + attn_output + mlp_output
        hidden_states = residual + attn_output + mlp_output

        return hidden_states, residual


class StableLMAlphaModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: StableLMAlphaDecoderLayer(cast(
                StableLMAlphaConfig, config),
                                                     cache_config,
                                                     quant_config,
                                                     prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.final_norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # StableLM-Alpha weight name mappings
        weight_name_mappings = {
            "embed.weight": "embed_tokens.weight",
        }
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Apply weight name mappings
            if name in weight_name_mappings:
                name = weight_name_mappings[name]
            
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class StableLMAlphaForCausalLM(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.transformer = StableLMAlphaModel(vllm_config=vllm_config,
                                              prefix=maybe_prefix(
                                                  prefix, "transformer"))

        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.lm_head")

        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.transformer.embed_tokens.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.transformer(input_ids, positions,
                                         intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
