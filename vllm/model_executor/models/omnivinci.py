# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Thin wrapper to support nvidia/omnivinci LLM weights stored under llm/.

This model maps the root architecture (VILAForCausalLM) to the text-only
Qwen2 architecture by reusing vLLM's Qwen2ForCausalLM and ensures the weight
loader searches in the `llm/` subfolder of the repository.
"""

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM


class OmniVinciForCausalLM(Qwen2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # direct the default loader to read weights from the llm/ subfolder
        self.allow_patterns_overrides = [
            "llm/*.safetensors",
            "llm/consolidated*.safetensors",
            "llm/*.pt",
        ]
