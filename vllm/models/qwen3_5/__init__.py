# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 model — hardware-isolated entry point."""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING or not current_platform.is_rocm():
    from .nvidia.model import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5ProcessingInfo,
    )
    from .nvidia.mtp import (
        Qwen3_5MoeMTP,
        Qwen3_5MTP,
        Qwen3_5MultiTokenPredictor,
    )
else:
    from .amd.model import (  # type: ignore[assignment]
        Qwen3_5ForCausalLM,
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5ProcessingInfo,
    )
    from .amd.mtp import (  # type: ignore[assignment]
        Qwen3_5MoeMTP,
        Qwen3_5MTP,
        Qwen3_5MultiTokenPredictor,
    )

__all__ = [
    "Qwen3_5ForCausalLMBase",
    "Qwen3_5Model",
    "Qwen3_5MultiTokenPredictor",
    "Qwen3_5ProcessingInfo",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeMTP",
    "Qwen3_5MTP",
]
