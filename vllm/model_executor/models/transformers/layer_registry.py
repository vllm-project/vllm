# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer provider resolution for the Transformers modeling backend.

The backend's fusers and `recursive_replace` build vLLM layers through these
getters to replace the HF ones.

The boolean environment variable `VLLM_HW_AGNOSTIC_LAYERS` selects the
implementation for every layer at once: when set, the getters return the
hw-agnostic classes; otherwise the native vLLM ones.

If a hw-agnostic implementation for a layer is missing, the getter falls back
to the vLLM class, so partial coverage never breaks a model.
"""

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def get_rms_norm_cls() -> type:
    """The `RMSNorm` class the RMSNorm fuser should build."""
    from vllm.model_executor.layers.layernorm import RMSNorm

    if envs.VLLM_HW_AGNOSTIC_LAYERS:
        try:
            from vllm.model_executor.hw_agnostic.layers.layernorm import (
                RMSNorm as HwRMSNorm,
            )

            return HwRMSNorm
        except ImportError:
            logger.warning_once(
                "hw-agnostic RMSNorm requested but unavailable; using vLLM's."
            )
    return RMSNorm


def get_gemma_rms_norm_cls() -> type:
    """The `GemmaRMSNorm` class the RMSNorm fuser should build."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    if envs.VLLM_HW_AGNOSTIC_LAYERS:
        try:
            from vllm.model_executor.hw_agnostic.layers.layernorm import (
                GemmaRMSNorm as HwGemmaRMSNorm,
            )

            return HwGemmaRMSNorm
        except ImportError:
            logger.warning_once(
                "hw-agnostic GemmaRMSNorm requested but unavailable; using vLLM's."
            )
    return GemmaRMSNorm


def get_act_and_mul_fn(act_fn_name: str):
    """The fused activation-and-multiply module the GLU fuser should build."""
    if envs.VLLM_HW_AGNOSTIC_LAYERS:
        try:
            from vllm.model_executor.hw_agnostic.layers.activation import (
                get_act_and_mul_fn as hw_get_act_and_mul_fn,
            )

            return hw_get_act_and_mul_fn(act_fn_name)
        except ImportError:
            logger.warning_once(
                "hw-agnostic activation-and-mul requested but unavailable; "
                "using vLLM's."
            )
    from vllm.model_executor.layers.activation import get_act_and_mul_fn as vllm_fn

    return vllm_fn(act_fn_name)
