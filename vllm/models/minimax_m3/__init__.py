# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 model — hardware-isolated entry point.

The implementation lives under ``nvidia/`` and ``amd/``; this module picks the
right one for the current platform and re-exports the public classes used by
the model registry. (Mirrors ``vllm.models.deepseek_v4``.)
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

# The NVIDIA branch is the static default that type-checkers see; the ROCm
# branch overrides it at runtime (kept type-compatible via type: ignore).
if TYPE_CHECKING or not current_platform.is_rocm():
    from .nvidia.model import (
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )
    from .nvidia.mtp import MiniMaxM3MTP
else:
    from .amd.model import (  # type: ignore[assignment]
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )
    from .amd.mtp import MiniMaxM3MTP  # type: ignore[assignment]

__all__ = [
    "MiniMaxM3MTP",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
]
