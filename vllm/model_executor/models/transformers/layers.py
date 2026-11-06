# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer resolution for the Transformers modeling backend.

Layers may come from the in-tree and the hw-agnostic path,
resolved via `vllm.model_executor.hw_agnostic.resolve`.
"""

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor import hw_agnostic

logger = init_logger(__name__)

RMSNorm = hw_agnostic.resolve("layernorm", "RMSNorm")
GemmaRMSNorm = hw_agnostic.resolve("layernorm", "GemmaRMSNorm")


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
