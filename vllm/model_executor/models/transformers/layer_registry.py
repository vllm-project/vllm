# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer provider resolution for the Transformers modeling backend.

The backend's fusers and `recursive_replace` build vLLM layers through these
getters to replace the HF ones.

It allows to mixin the hw-agnostic implementations based on the environment
variable `VLLM_HW_AGNOSTIC_LAYERS`. In particular, it allows to select
between native vLLM or hw-agnostic implementations.
Selection is driven thorugh the same `+name`/`-name`/`all`/`none` grammar
as `CompilationConfig.custom_ops`. This
picks *which implementation* a layer uses; it is independent of `custom_ops`,
which picks native-vs-device dispatch *within* a chosen hw-agnostic `CustomOp`.

If a hw-agnostic implementation for a selected layer is missing, the getter
falls back to the vLLM class, so partial coverage never breaks a model.
"""

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Selector keys.
RMS_NORM = "rms_norm"
GEMMA_RMS_NORM = "gemma_rms_norm"
SILU_AND_MUL = "silu_and_mul"


def _use_hw_agnostic(layer: str) -> bool:
    """Whether `layer` should use the hw-agnostic implementation or the native vLLM."""
    selector = envs.VLLM_HW_AGNOSTIC_LAYERS
    if f"-{layer}" in selector:
        return False
    return "all" in selector or f"+{layer}" in selector


def get_rms_norm_cls() -> type:
    """The `RMSNorm` class the RMSNorm fuser should build."""
    from vllm.model_executor.layers.layernorm import RMSNorm

    if _use_hw_agnostic(RMS_NORM):
        try:
            from vllm.model_executor.hw_agnostic.layers.layernorm import (
                RMSNorm as HwRMSNorm,
            )

            return HwRMSNorm
        except ImportError:
            logger.warning_once(
                "hw-agnostic %s requested but unavailable; using vLLM's.", RMS_NORM
            )
    return RMSNorm


def get_gemma_rms_norm_cls() -> type:
    """The `GemmaRMSNorm` class the RMSNorm fuser should build."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    if _use_hw_agnostic(GEMMA_RMS_NORM):
        try:
            from vllm.model_executor.hw_agnostic.layers.layernorm import (
                GemmaRMSNorm as HwGemmaRMSNorm,
            )

            return HwGemmaRMSNorm
        except ImportError:
            logger.warning_once(
                "hw-agnostic %s requested but unavailable; using vLLM's.",
                GEMMA_RMS_NORM,
            )
    return GemmaRMSNorm


def get_act_and_mul_fn(act_fn_name: str):
    """The fused activation-and-multiply module the GLU fuser should build."""
    if _use_hw_agnostic(SILU_AND_MUL):
        try:
            from vllm.model_executor.hw_agnostic.layers.activation import (
                get_act_and_mul_fn as hw_get_act_and_mul_fn,
            )

            return hw_get_act_and_mul_fn(act_fn_name)
        except ImportError:
            logger.warning_once(
                "hw-agnostic %s requested but unavailable; using vLLM's.",
                SILU_AND_MUL,
            )
    from vllm.model_executor.layers.activation import get_act_and_mul_fn as vllm_fn

    return vllm_fn(act_fn_name)
