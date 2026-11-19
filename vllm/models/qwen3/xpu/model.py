# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU-specific entry point for the dense Qwen3 SP implementation."""

from vllm.models.qwen3.nvidia.model import (
    Qwen3ForCausalLM,
    Qwen3Model,
)

__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3Model",
]
