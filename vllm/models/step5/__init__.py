# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .amd.model import Step5ForCausalLM, Step5ForConditionalGeneration
    from .amd.mtp import Step5MTP


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        raise NotImplementedError("Step-5 currently supports ROCm only.")
    from .amd.model import Step5ForCausalLM, Step5ForConditionalGeneration
    from .amd.mtp import Step5MTP

    return {
        "Step5ForCausalLM": Step5ForCausalLM,
        "Step5ForConditionalGeneration": Step5ForConditionalGeneration,
        "Step5MTP": Step5MTP,
    }[name]


__all__ = ["Step5ForCausalLM", "Step5ForConditionalGeneration", "Step5MTP"]
