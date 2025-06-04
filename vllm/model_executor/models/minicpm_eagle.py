# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
"""Inference-only EagleMiniCPM model compatible with HuggingFace weights."""
import math
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .minicpm import MiniCPMAttention as EagleMiniCPMAttention
from .minicpm import MiniCPMMLP as EagleMiniCPMMLP
from .minicpm import MiniCPMMoE as EagleMiniCPMMoE
from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, maybe_prefix)


class EagleMiniCPMDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.hidden_size = config.hidden_size
        self.rope_theta = getattr(config, "rope_theta", 10000)
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.max_position_embeddings = getattr(config,
                                               "max_position_embeddings", 8192)
        self.prefix = prefix
        self._init_attn_block()
        self._init_ffn_block()

    def _init_attn_block(self):
        self.input_layernorm = RMSNorm(self.config.hidden_size,
                                       eps=self.config.rms_norm_eps)
        self.self_attn = EagleMiniCPMAttention(
            hidden_size=self.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            max_position_embeddings=self.max_position_embeddings,
            cache_config=self.cache_config,
            quant_config=self.quant_config,
            prefix=f"{self.prefix}.self_attn",
        )

    def _init_ffn_block(self):
        self.post_attention_layernorm = RMSNorm(self.config.hidden_size,
                                                eps=self.config.rms_norm_eps)
        self.num_experts = getattr(self.config, "num_experts", 0)
        if self.num_experts == 0:
            self.mlp = EagleMiniCPMMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
                hidden_act_param=getattr(self.config, "hidden_act_param", 0.),
                quant_config=self.quant_config,
            )
        else:
            self.mlp = EagleMiniCPMMoE(
                num_experts=self.config.num_experts,
                top_k=self.config.num_experts_per_tok,
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states * \
            (self.config.scale_depth / math.sqrt(self.config.mup_denominator))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * \
            (self.config.scale_depth / math.sqrt(self.config.mup_denominator))

        return hidden_states, None


@support_torch_compile
class EagleMiniCPMModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 start_layer: int = 0):
        super().__init__()

        config = vllm_config.speculative_config.draft_model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.cache_config = cache_config
        self.quant_config = quant_config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.fc = torch.nn.Linear(self.config.hidden_size * 2,
                                  self.config.hidden_size,
                                  bias=False)
        self.input_norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.num_experts = getattr(self.config, "num_experts", 0)
        self._init_layers(prefix, config, cache_config, quant_config,
                          start_layer)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], self.config.hidden_size))

    def _init_layers(
        self,
        prefix: str,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig],
        quant_config: Optional[QuantizationConfig],
        start_layer: int,
    ):
        self.eagle_layers = nn.ModuleList([
            EagleMiniCPMDecoderLayer(
                config,
                cache_config,
                quant_config,
                f"{prefix}.eagle_layers.{i + start_layer}",
            ) for i in range(self.config.num_hidden_layers)
        ])

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(input_ids)
        return embedding * self.config.scale_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        input_embeds = self.get_input_embeddings(input_ids)
        input_embeds = self.input_norm1(input_embeds)
        hidden_states = self.input_norm2(hidden_states)

        hidden_states = self.fc(
            torch.cat((input_embeds, hidden_states), dim=-1))
        residual = None
        for layer in self.eagle_layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        return hidden_states, hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            ("ws" if weight_name in ["w1", "w3"] else "w2s",
             f"experts.{expert_id}.{weight_name}.weight", expert_id)
            for expert_id in range(self.num_experts)
            for weight_name in ["w1", "w2", "w3"]
        ]
        params_dict = dict(self.named_parameters())

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  weight_name,
                                  expert_id=expert_id)
                    break
                else:
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


class EagleMiniCPMForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.prefix = prefix
        self.vllm_config = vllm_config
        self.config = config
        self.lora_config = lora_config
        self.cache_config = cache_config
        self.quant_config = quant_config

        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)

        self.model = self._init_model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"),
                                      start_layer=target_layer_num)

        unpadded_vocab_size = config.vocab_size
        if lora_config:
            unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
            quant_config=quant_config,
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.scale_width = self.config.hidden_size / self.config.dim_model_base

        self.logits_processor = LogitsProcessor(unpadded_vocab_size,
                                                config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def _init_model(self,
                    *,
                    vllm_config: VllmConfig,
                    prefix: str = "",
                    start_layer: int = 0):
        return EagleMiniCPMModel(vllm_config=vllm_config,
                                 prefix=prefix,
                                 start_layer=start_layer)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, hidden_states2 = self.model(input_ids, positions,
                                                   hidden_states)
        hidden_states = hidden_states / self.scale_width
        hidden_states2 = hidden_states2 / self.scale_width
        return hidden_states, hidden_states2

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
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
