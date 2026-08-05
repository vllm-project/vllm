# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 model — hardware-isolated entry point.

The implementation lives under ``nvidia/`` and ``amd/``; this module picks the
right one for the current platform and re-exports the public classes used by
the model registry. (Mirrors ``vllm.models.minimax_m3``.)
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

# The NVIDIA branch is the static default that type-checkers see; the ROCm
# branch overrides it at runtime (kept type-compatible via type: ignore).
if TYPE_CHECKING or not current_platform.is_rocm():
    from .nvidia.model import KimiK3ForConditionalGeneration, KimiLinearForCausalLM
    from .nvidia.mtp import KimiK3MTP
else:
    from .amd.linear import KimiLinearForCausalLM  # type: ignore[assignment]
    from .amd.model import KimiK3ForConditionalGeneration  # type: ignore[assignment]
    from .amd.mtp import KimiK3MTP  # type: ignore[assignment]

__all__ = [
    "KimiK3ForConditionalGeneration",
    "KimiK3MTP",
    "KimiLinearForCausalLM",
]
