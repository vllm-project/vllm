# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime requirements for the Qwen4Exp CPU implementation."""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.triton_utils import HAS_TRITON

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def has_active_triton_cpu_backend() -> bool:
    """Return whether Triton's selected runtime target is CPU."""
    if not HAS_TRITON:
        return False
    try:
        from triton.runtime import driver

        return driver.active.get_current_target().backend == "cpu"
    except Exception:
        return False


def verify_cpu_config(vllm_config: "VllmConfig") -> None:
    """Reject runtime configurations unsupported by Qwen4Exp on CPU."""
    if current_platform.get_cpu_architecture() != CpuArchEnum.X86:
        raise NotImplementedError("Qwen4Exp CPU support currently requires x86-64.")
    if vllm_config.speculative_config is not None:
        raise NotImplementedError(
            "Qwen4Exp CPU support does not yet support speculative decoding."
        )
    if vllm_config.lora_config is not None:
        raise NotImplementedError("Qwen4Exp CPU support does not yet support LoRA.")
    if not vllm_config.use_v2_model_runner:
        raise ValueError(
            "Qwen4Exp on CPU requires Model Runner V2; remove "
            "VLLM_USE_V2_MODEL_RUNNER=0."
        )
    if not has_active_triton_cpu_backend():
        raise ValueError(
            "Qwen4Exp on CPU requires Triton with an active CPU backend. "
            "Use the official x86 CPU build or image containing Triton-CPU."
        )
    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is not None and not multimodal_config.language_model_only:
        raise NotImplementedError(
            "Qwen4Exp CPU support is text-only; pass --language-model-only."
        )
