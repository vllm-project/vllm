# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 (dense) model — vendor-specific entry point.

Dispatches to ``nvidia/`` based on the current platform, otherwise falls
back to the upstream model in vllm.model_executor.models.
"""

from vllm.platforms import current_platform

if current_platform.is_cuda():
    from .nvidia.model import Qwen3ForCausalLM  # type: ignore[assignment]
else:
    from vllm.model_executor.models.qwen3 import (  # type: ignore[assignment]
        Qwen3ForCausalLM,
    )

__all__ = [
    "Qwen3ForCausalLM",
]
