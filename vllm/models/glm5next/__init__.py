# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms import current_platform

if current_platform.is_rocm() or current_platform.is_xpu():
    raise NotImplementedError(
        "glm 5 next currently supports NVIDIA SM90 and above only."
    )

from .nvidia.model import Glm5NextForCausalLM
from .nvidia.mtp import Glm5NextMTP

__all__ = [
    "Glm5NextForCausalLM",
    "Glm5NextMTP",
]
