# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel

from .llama import LlamaDecoderLayer
from .utils import AutoWeightsLoader, PPMissingLayer, WeightsMapper


class TeleChat2Model(LlamaModel):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".query": (".qkv_proj", "q"),
            ".key": (".qkv_proj", "k"),
            ".value": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        hf_config = vllm_config.model_config.hf_config

        vllm_config.model_config.hf_config.attribute_map = {
            "num_hidden_layers": "n_layer",
            "num_attention_heads": "n_head",
            "intermediate_size": "ffn_hidden_size",
            "rms_norm_eps": "layer_norm_epsilon",
        }
        vllm_config.model_config.hf_config.hidden_act = "silu"

        # 1. Initialize the LlamaModel with bias
        hf_config.bias = True
        hf_config.mlp_bias = True

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # 2. Remove the bias from the qkv_proj and gate_up_proj based on config
        # Telechat2's gate_up_proj and qkv_proj don't have bias
        # see: https://github.com/vllm-project/vllm/pull/10311#issuecomment-2490297566
        for layer in self.layers:
            if not isinstance(layer, PPMissingLayer):
                layer.self_attn.qkv_proj.bias = None
                layer.self_attn.qkv_proj.skip_bias_add = True
                layer.mlp.gate_up_proj.bias = None
                layer.mlp.gate_up_proj.skip_bias_add = True

    def _split_key_value(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        # TeleChat2 stores k/v as a single per-head-interleaved `key_value`
        # tensor. De-interleave it into separate k/v so the qkv_proj mapper
        # can stack them.
        total_num_heads = self.config.n_head
        head_dim = self.config.hidden_size // total_num_heads
        for name, loaded_weight in weights:
            if "self_attn.key_value" in name:
                starts = [i * head_dim * 2 for i in range(total_num_heads)]
                k_weight = torch.cat(
                    [loaded_weight[s : s + head_dim, :] for s in starts], dim=0
                )
                v_weight = torch.cat(
                    [loaded_weight[s + head_dim : s + 2 * head_dim, :] for s in starts],
                    dim=0,
                )
                yield name.replace("key_value", "key"), k_weight
                yield name.replace("key_value", "value"), v_weight
            else:
                yield name, loaded_weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            self._split_key_value(weights), mapper=self.hf_to_vllm_mapper
        )


class TeleChat2ForCausalLM(LlamaForCausalLM):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "transformer.": "model.",
        },
        orig_to_new_substr={
            ".h.": ".layers.",
            ".self_attention.": ".self_attn.",
            ".word_embeddings.": ".embed_tokens.",
            ".dense.": ".o_proj.",
            ".ln_f.": ".norm.",
        },
    )

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        return TeleChat2Model(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
