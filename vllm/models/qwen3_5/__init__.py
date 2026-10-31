# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 model — hardware-isolated entry point.

NVIDIA and XPU use hardware-isolated eager implementations.
ROCm and other platforms keep
the previous torch.compile implementation in ``vllm/model_executor/models/``
so their behavior is unchanged.
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING or current_platform.is_cuda():
    from .nvidia.model import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
    )
    from .nvidia.mtp import Qwen3_5MoeMTP, Qwen3_5MTP
elif current_platform.is_xpu():
    from .xpu.model import (  # type: ignore[assignment]
        Qwen3_5ForCausalLM,
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
    )
    from .xpu.mtp import Qwen3_5MoeMTP, Qwen3_5MTP  # type: ignore[assignment]
else:
    from vllm.model_executor.models.qwen3_5 import (  # type: ignore[assignment]
        Qwen3_5ForCausalLM,
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
    )
    from vllm.model_executor.models.qwen3_5_mtp import (  # type: ignore[assignment]
        Qwen3_5MoeMTP,
        Qwen3_5MTP,
    )

__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForCausalLMBase",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5Model",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeMTP",
    "Qwen3_5MTP",
]
