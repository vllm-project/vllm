# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 The Qwen team.
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
"""Inference-only MiMo model compatible with HuggingFace weights."""
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (Qwen2Attention,
                                                      Qwen2ForCausalLM,
                                                      Qwen2MLP,Qwen2Model,
                                                      Qwen2RMSNorm)


class MiMoConfig(Qwen2Config):
    model_type = "mimo"

    def __init__(self, *args, num_nextn_predict_layers=0, **kwargs):
        self.num_nextn_predict_layers = num_nextn_predict_layers
        super().__init__(
            *args,
            **kwargs,
        )


class MiMoMTPLayers(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)
        self.token_layernorm = Qwen2RMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.hidden_layernorm = Qwen2RMSNorm(config.hidden_size,
                                             eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(config.hidden_size * 2,
                                    config.hidden_size,
                                    bias=False)
        self.final_layernorm = Qwen2RMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx=0)
        self.mlp = Qwen2MLP(config)

    def forward(
        self,
        input_embeds,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embedding: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position=None,
        **kwargs,
    ):
        input_embeds = self.token_layernorm(input_embeds)
        previous_hidden_states = self.hidden_layernorm(hidden_states)
        hidden_states = self.input_proj(
            torch.cat([previous_hidden_states, input_embeds], dim=-1))
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embedding=position_embedding,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class MiMoModel(Qwen2Model):
    config_class = MiMoConfig

    def __init__(self, config: MiMoConfig):
        super().__init__(config)
        self.mtp_layers = nn.ModuleList([
            MiMoMTPLayers(config)
            for _ in range(config.num_nextn_predict_layers)
        ])


class MiMoForCausalLM(Qwen2ForCausalLM):
    config_class = MiMoConfig

    def __init__(self, config: MiMoConfig):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = MiMoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.post_init()
