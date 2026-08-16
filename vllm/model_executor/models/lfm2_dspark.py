# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for LFM2.5."""

from vllm.config import VllmConfig

from .qwen3_dspark import Qwen3DSparkForCausalLM


class Lfm2DSparkForCausalLM(Qwen3DSparkForCausalLM):
    """LFM2.5's dense DSpark checkpoint using the Qwen3-style backbone."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.speculative_config.draft_model_config.hf_config
        config.is_neox_style = getattr(config, "rope_is_neox_style", False)
        if getattr(config, "enable_confidence_head", False) and not hasattr(
            config, "confidence_head_with_markov"
        ):
            config.confidence_head_with_markov = True
        super().__init__(vllm_config=vllm_config, prefix=prefix)
