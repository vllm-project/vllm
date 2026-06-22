# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PARD-2 parallel draft model (Llama family).

A stock Llama decoder stack running on embeddings fused with a projection of
target-model hidden states. All the PARD-2 fusion/loading logic is shared in
``pard2_base.py``; this file only supplies the Llama decoder layers. See
``pard2_base.py`` for the fusion math and ``qwen3_pard2.py`` for the Qwen3 twin.
"""

import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM

from .pard2_base import (
    PARD2_COMPILE_DYNAMIC_ARG_DIMS,
    Pard2ForCausalLMMixin,
    Pard2ModelBase,
)
from .utils import maybe_prefix


@support_torch_compile(dynamic_arg_dims=PARD2_COMPILE_DYNAMIC_ARG_DIMS)
class Pard2LlamaModel(Pard2ModelBase):
    def build_layers(
        self, vllm_config: VllmConfig, start_layer_id: int, prefix: str
    ) -> nn.ModuleList:
        current_vllm_config = get_current_vllm_config()
        return nn.ModuleList(
            [
                LlamaDecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
                    config=self.config,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )


class Pard2LlamaForCausalLM(Pard2ForCausalLMMixin, LlamaForCausalLM):
    pard2_model_cls = Pard2LlamaModel
