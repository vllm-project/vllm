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
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM

from .pard2_base import (
    PARD2_COMPILE_DYNAMIC_ARG_DIMS,
    Pard2ForCausalLMMixin,
    Pard2ModelBase,
)
from .utils import get_draft_quant_config


class Pard2LlamaDecoderLayer(LlamaDecoderLayer):
    def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
        """Use drafter's quantization config instead of verifier's."""
        return get_draft_quant_config(vllm_config)


@support_torch_compile(dynamic_arg_dims=PARD2_COMPILE_DYNAMIC_ARG_DIMS)
class Pard2LlamaModel(Pard2ModelBase):
    def _make_decoder_layer(self, vllm_config: VllmConfig, prefix: str) -> nn.Module:
        return Pard2LlamaDecoderLayer(vllm_config, prefix=prefix, config=self.config)


class Pard2LlamaForCausalLM(Pard2ForCausalLMMixin, LlamaForCausalLM):
    pard2_model_cls = Pard2LlamaModel
