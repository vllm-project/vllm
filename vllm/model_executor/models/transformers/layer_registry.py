# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer provider resolution for the Transformers modeling backend.

When ``VLLM_USE_HW_AGNOSTIC`` is set, layer symbols are imported from
``vllm.model_executor.hw_agnostic.layers.<module>``, falling back to
``vllm.model_executor.layers.<module>`` for anything not yet ported. The
resolved source of every symbol is logged so it is clear which layers run
hw-agnostic and which fell back to vLLM.
"""

import importlib

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_HW_PKG = "vllm.model_executor.hw_agnostic.layers"
_VLLM_PKG = "vllm.model_executor.layers"


def _resolve(module: str, name: str):
    """Return `name` from the hw-agnostic `module` when enabled and available,
    else from vLLM. Logs which source was used."""
    if envs.VLLM_USE_HW_AGNOSTIC:
        try:
            obj = getattr(importlib.import_module(f"{_HW_PKG}.{module}"), name)
            logger.info("Using hw-agnostic layer: %s", name)
            return obj
        except (ImportError, AttributeError):
            logger.warning(
                "hw-agnostic layer %s is not available; falling back to vLLM", name
            )
    return getattr(importlib.import_module(f"{_VLLM_PKG}.{module}"), name)


RMSNorm = _resolve("layernorm", "RMSNorm")
GemmaRMSNorm = _resolve("layernorm", "GemmaRMSNorm")


def get_act_and_mul_fn(act_fn_name: str):
    """Fused activation-and-mul op for `act_fn_name`, preferring hw-agnostic.

    Resolved per call because the op is name-parameterized: an activation with
    no hw-agnostic equivalent falls back to vLLM individually.
    """
    if envs.VLLM_USE_HW_AGNOSTIC:
        try:
            from vllm.model_executor.hw_agnostic.layers.activation import (
                get_act_and_mul_fn as hw_fn,
            )

            fn = hw_fn(act_fn_name)
            logger.info_once("Using hw-agnostic activation: %s", act_fn_name)
            return fn
        except (ImportError, KeyError):
            logger.warning_once(
                "hw-agnostic activation %s is not available; falling back to vLLM",
                act_fn_name,
            )
    from vllm.model_executor.layers.activation import get_act_and_mul_fn as vllm_fn

    return vllm_fn(act_fn_name)
