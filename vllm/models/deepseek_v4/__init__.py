# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 model — hardware-isolated entry point.

The actual implementation lives under ``nvidia/`` and ``amd/``; this module
picks the right one for the current platform and re-exports the public
classes used by the model registry and quantization config lookup.
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

from .quant_config import DeepseekV4FP8Config

# Pick the per-platform implementation. The NVIDIA branch is the static
# default that mypy sees; the ROCm branch overrides it at runtime and is
# kept type-compatible via ``# type: ignore[assignment]``.
if TYPE_CHECKING or current_platform.is_cuda():
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP
elif current_platform.is_rocm():
    from .amd.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .amd.mtp import DeepSeekV4MTP  # type: ignore[assignment]
else:
    DeepseekV4ForCausalLM = object  # type: ignore[assignment]
    DeepSeekV4MTP = object  # type: ignore[assignment]

__all__ = [
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]
