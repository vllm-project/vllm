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
# TPU plugins import the shared ``common`` modules through this package, but
# register their own model classes. Do not eagerly import a GPU implementation.
if TYPE_CHECKING:
    from .nvidia.model import KimiK3ForConditionalGeneration, KimiLinearForCausalLM
    from .nvidia.mtp import KimiK3MTP
elif current_platform.device_type == "tpu":
    pass
elif current_platform.is_rocm():
    from .amd.linear import KimiLinearForCausalLM  # type: ignore[assignment]
    from .amd.model import KimiK3ForConditionalGeneration  # type: ignore[assignment]
    from .amd.mtp import KimiK3MTP  # type: ignore[assignment]
else:
    from .nvidia.model import KimiK3ForConditionalGeneration, KimiLinearForCausalLM
    from .nvidia.mtp import KimiK3MTP

__all__ = [
    "KimiK3ForConditionalGeneration",
    "KimiK3MTP",
    "KimiLinearForCausalLM",
]
