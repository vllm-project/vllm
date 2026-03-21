# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# This file is kept for backward compatibility only.
# The canonical implementation has moved to gptq.py.
from vllm.model_executor.layers.quantization.gptq import (  # noqa: F401
    GPTQConfig,
    GPTQMarlinLinearMethod,
    GPTQMarlinMoEMethod,
    get_moe_quant_method,
)
from vllm.model_executor.layers.quantization.gptq import (
    GPTQConfig as GPTQMarlinConfig,
)

__all__ = [
    "GPTQConfig",
    "GPTQMarlinConfig",
    "GPTQMarlinLinearMethod",
    "GPTQMarlinMoEMethod",
    "get_moe_quant_method",
]
