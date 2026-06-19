# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 model — hardware-isolated entry point.

Implementations live under ``nvidia/``, ``amd/``, ``xpu/``, and
``hw_agnostic/``; this module picks one and re-exports the public classes
used by the model registry and quantization config lookup.

Out-of-tree platforms (registered via the ``vllm.platform_plugins`` entry
point group) automatically select the ``hw_agnostic/`` branch.
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

# The hw-agnostic platform uses a vendored DeepseekV4FP8Config whose
# ``get_quant_method`` does isinstance against vendored ``LinearBase`` /
# ``RoutedExperts`` (under ``hw_agnostic/shared/layers/``). Without the
# override the upstream config would return None for our vendored layers
# and crash with "All linear layers should support quant method.".
if current_platform.is_out_of_tree():
    from .hw_agnostic.quantization.quant_config import (  # type: ignore[assignment]
        DeepseekV4FP8Config,
    )
else:
    from .quant_config import DeepseekV4FP8Config

# NVIDIA branch is the static default mypy sees; the others override at
# runtime and carry ``# type: ignore[assignment]``.
if TYPE_CHECKING:
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP

if current_platform.is_out_of_tree():
    from .hw_agnostic.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .hw_agnostic.mtp import DeepSeekV4MTP  # type: ignore[assignment]
elif current_platform.is_rocm():
    from .amd.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .amd.mtp import DeepSeekV4MTP  # type: ignore[assignment]
elif current_platform.is_cuda():
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP
elif current_platform.is_xpu():
    from .xpu.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .xpu.mtp import DeepSeekV4MTP  # type: ignore[assignment]
else:
    # Fallback for non-{cuda,rocm,xpu} platforms (e.g. CPU).
    from .hw_agnostic.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .hw_agnostic.mtp import DeepSeekV4MTP  # type: ignore[assignment]

__all__ = [
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]
