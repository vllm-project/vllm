# SPDX-License-Identifier: Apache-2.0

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
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Optional, Union

import jax
import jax.numpy as jnp
from flax import nnx

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.models import VllmModelForTextGeneration
from vllm.sequence import IntermediateTensors


class _FusionMeta(type(nnx.Module), type(VllmModelForTextGeneration)):
    pass


class SimpleLayer(nnx.Module):

    def __init__(self, layer_id):
        self.layer_id = id
        compilation_config = get_current_vllm_config().compilation_config
        compilation_config.static_forward_context[f'layers.{layer_id}'] = self

    def __call__(self, x):
        return x * 2


class JAXLlamaForCausalLM(nnx.Module,
                          VllmModelForTextGeneration,
                          metaclass=_FusionMeta):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: str = ""):
        parallel_config = vllm_config.parallel_config
        self.layers = [
            SimpleLayer(layer_id=i) for i in range(
                vllm_config.model_config.get_num_layers(parallel_config))
        ]

    def named_parameters(self):
        return set()

    def load_weights(self, *args, **kwargs):
        return set()

    def named_modules(self, *args, **kwargs):
        return set()

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> Union[jax.Array, IntermediateTensors]:
        x = jnp.ones((2, 4))
        for layer in self.layers:
            x = layer(x)
        return x
