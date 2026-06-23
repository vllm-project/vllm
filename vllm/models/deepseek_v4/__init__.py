# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if current_platform.is_out_of_tree():
    from .hw_agnostic.quantization.quant_config import DeepseekV4FP8Config
else:
    from .quant_config import DeepseekV4FP8Config  # type: ignore[assignment]

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
    from .nvidia.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .nvidia.mtp import DeepSeekV4MTP  # type: ignore[assignment]

__all__ = [
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]
