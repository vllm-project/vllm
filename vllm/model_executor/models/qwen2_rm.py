# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/Qwen/Qwen2.5-Math-RM-72B/blob/main/modeling_qwen2_rm.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
"""Inference-only Qwen2-RM model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.pooler import DispatchPooler, Pooler
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .interfaces_base import default_pooling_type
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, maybe_prefix


class Qwen2RewardBaseModel(nn.Module, SupportsLoRA, SupportsPP):

    is_pooling_model = True
    pooler: Pooler

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
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))
        self.head_dtype = vllm_config.model_config.head_dtype

        self.score = nn.Sequential(
            ColumnParallelLinear(config.hidden_size,
                                 config.hidden_size,
                                 quant_config=quant_config,
                                 params_dtype=self.head_dtype,
                                 return_bias=False),
            nn.ReLU(),
            RowParallelLinear(config.hidden_size,
                              config.num_labels,
                              params_dtype=self.head_dtype,
                              quant_config=quant_config,
                              return_bias=False),
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

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
        hidden_states = hidden_states.to(self.head_dtype)
        logits = self.score(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self,
                                   ignore_unexpected_prefixes=["lm_head."])
        return loader.load_weights(weights)


@default_pooling_type("ALL")
class Qwen2ForRewardModel(Qwen2RewardBaseModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        vllm_config.model_config.hf_config.num_labels = 1
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler(
            {"encode": Pooler.for_encode(pooler_config)}, )


@default_pooling_type("STEP")
class Qwen2ForProcessRewardModel(Qwen2RewardBaseModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        vllm_config.model_config.hf_config.num_labels = 2
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler(
            {"encode": Pooler.for_encode(pooler_config)})
