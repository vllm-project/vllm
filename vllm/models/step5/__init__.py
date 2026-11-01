# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 entry point: the NVIDIA or AMD implementation, picked on lookup."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nvidia.model import Step5ForCausalLM
    from .nvidia.mtp import Step5MTP
    from .nvidia.vl import Step5ForConditionalGeneration


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from .amd.model import Step5ForCausalLM
        from .amd.mtp import Step5MTP
        from .amd.vl import Step5ForConditionalGeneration
    else:
        from .nvidia.model import Step5ForCausalLM
        from .nvidia.mtp import Step5MTP
        from .nvidia.vl import Step5ForConditionalGeneration

    return {
        "Step5ForCausalLM": Step5ForCausalLM,
        "Step5ForConditionalGeneration": Step5ForConditionalGeneration,
        "Step5MTP": Step5MTP,
    }[name]


__all__ = ["Step5ForCausalLM", "Step5ForConditionalGeneration", "Step5MTP"]
