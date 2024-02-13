# coding=utf-8
# Copyright 2023 Stability AI, EleutherAI, and The HuggingFace Inc. team. All rights reserved.
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
# https://huggingface.co/stabilityai/stablelm-3b-4e1t/blob/main/modeling_stablelm_epoch.py
# https://huggingface.co/stabilityai/stablelm-3b-4e1t/blob/main/config.json
"""Inference-only StabeLM (https://github.com/Stability-AI/StableLM) model compatible with HuggingFace weights."""
from typing import Optional

from transformers import PretrainedConfig

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.config import LoRAConfig


class StablelmForCausalLM(LlamaForCausalLM):

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        norm = LayerNorm(config.hidden_size, config.norm_eps)
        super().__init__(config=config,
                         linear_method=linear_method,
                         norm=norm,
                         lora_config=lora_config)
