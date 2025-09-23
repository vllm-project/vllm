# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
"""Inference-only Qwen3 Guard model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig, PoolerConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.pooler import (DispatchPooler, Pooler)

from .interfaces import SupportsPP
from .interfaces_base import default_pooling_type
from .qwen3 import Qwen3Model
from .utils import (AutoWeightsLoader, PPMissingLayer, maybe_prefix)

logger = init_logger(__name__)


@default_pooling_type("ALL")
class Qwen3ForGuardModel(nn.Module, SupportsPP):

    if envs.VLLM_USE_V1:
        is_pooling_model = True

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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen3Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        self.risk_level_category_pre = nn.Linear(config.hidden_size,
                                                 config.guard_inner_size,
                                                 bias=False)
        self.risk_level_category_layernorm = RMSNorm(config.guard_inner_size,
                                                     eps=config.rms_norm_eps)
        self.risk_level_head = nn.Linear(config.guard_inner_size,
                                         config.num_risk_level,
                                         bias=False)
        self.category_head = nn.Linear(config.guard_inner_size,
                                       config.num_category,
                                       bias=False)

        self.query_risk_level_category_pre = nn.Linear(config.hidden_size,
                                                       config.guard_inner_size,
                                                       bias=False)
        self.query_risk_level_category_layernorm = RMSNorm(
            config.guard_inner_size, eps=config.rms_norm_eps)
        self.query_risk_level_head = nn.Linear(config.guard_inner_size,
                                               config.num_query_risk_level,
                                               bias=False)
        self.query_category_head = nn.Linear(config.guard_inner_size,
                                             config.num_query_category,
                                             bias=False)

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.pooler = DispatchPooler({
            "encode":
            Pooler.for_encode(
                PoolerConfig(
                    pooling_type="ALL",
                    normalize=False,
                    dimensions=None,
                    enable_chunked_processing=True,
                    activation=False,
                    softmax=False,
                )),
        })

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)

        hidden_states = hidden_states[:, None, :]

        risk_level_category_x = self.risk_level_category_pre(hidden_states)
        risk_level_category_x = self.risk_level_category_layernorm(
            risk_level_category_x)
        risk_level_logits = self.risk_level_head(risk_level_category_x)
        category_logits = self.category_head(risk_level_category_x)

        query_risk_level_category_x = self.query_risk_level_category_pre(
            hidden_states)
        query_risk_level_category_x = self.query_risk_level_category_layernorm(
            query_risk_level_category_x)
        query_risk_level_logits = self.query_risk_level_head(
            query_risk_level_category_x)
        query_category_logits = self.query_category_head(
            query_risk_level_category_x)

        return torch.cat([
            risk_level_logits, category_logits, query_risk_level_logits,
            query_category_logits, hidden_states
        ],
                         dim=-1)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
