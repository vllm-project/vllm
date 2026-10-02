# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA entry point for the Kimi-K3 DSpark MLA draft.

The implementation lives in ``vllm.models.kimi_k3.common.dspark_mla``.
"""

from vllm.models.kimi_k3.common.dspark_mla import (
    K3DSparkDecoderLayer,
    K3DSparkForCausalLM,
    K3DSparkModel,
    _duplicate_context_kv_weights,
)

__all__ = [
    "K3DSparkDecoderLayer",
    "K3DSparkForCausalLM",
    "K3DSparkModel",
    "_duplicate_context_kv_weights",
]
