# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only HF format GLM-4 model compatible with THUDM weights."""
from vllm.config import VllmConfig
from vllm.model_executor.models.llama import LlamaForCausalLM

from .utils import PPMissingLayer


class GlmForCausalLM(LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        vllm_config.model_config.hf_config.partial_rotary_factor = 0.5
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Hack Llama model to fit HF format GLM implementation
        # Attention difference between GLM and Llama:
        # 1. Half partial rotary_dim and no Neox style.
        # 2. There is no bias for o_proj in attention
        for layer in self.model.layers:
            if not isinstance(layer, PPMissingLayer):
                layer.self_attn.rotary_emb.is_neox_style = False
                layer.self_attn.o_proj.bias = None
                layer.self_attn.o_proj.skip_bias_add = True
