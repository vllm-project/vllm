# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 (``glm_moe_dsa``, architecture ``GlmMoeDsaForCausalLM``) entry point."""

__all__ = ["GlmMoeDsaForCausalLM"]


def __getattr__(name: str) -> type:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from vllm.platforms import current_platform

    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        from . import generic

        return generic.GlmMoeDsaForCausalLM
    from .nvidia import model

    return model.GlmMoeDsaForCausalLM
