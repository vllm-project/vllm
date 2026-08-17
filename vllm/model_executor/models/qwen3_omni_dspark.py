# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-Omni-specific DSpark draft model.

The draft consumes auxiliary hidden states from the Omni thinker's text
backbone. It keeps the dense DSpark decoder and logical 1-D positions while
using a dedicated architecture contract for Qwen3-Omni checkpoints.
"""

from vllm.config import VllmConfig

from .qwen3_dflash import DFlashQwen3Attention, DFlashQwen3DecoderLayer
from .qwen3_dspark import Qwen3DSparkForCausalLM, Qwen3DSparkModel


class Qwen3OmniDSparkAttention(DFlashQwen3Attention):
    """Qwen3-MoE-style GQA used by the Qwen3-Omni DSpark checkpoint."""


class Qwen3OmniDSparkDecoderLayer(DFlashQwen3DecoderLayer):
    """DSpark decoder layer with the dedicated Omni attention type."""

    attention_cls = Qwen3OmniDSparkAttention


class Qwen3OmniDSparkModel(Qwen3DSparkModel):
    """Dense DSpark backbone trained for Qwen3-Omni thinker features."""

    decoder_layer_cls = Qwen3OmniDSparkDecoderLayer


class Qwen3OmniDSparkForCausalLM(Qwen3DSparkForCausalLM):
    """Standalone Qwen3-Omni DSpark checkpoint entry point."""

    model_cls = Qwen3OmniDSparkModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
