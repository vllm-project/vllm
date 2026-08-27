# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms import current_platform

if current_platform.is_xpu():
    raise NotImplementedError("GLM-5.3-Flash does not currently support XPU.")

from .nvidia.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration
from .nvidia.mtp import Glm5NextMTP

__all__ = [
    "Glm5NextForCausalLM",
    "Glm5NextForConditionalGeneration",
    "Glm5NextMTP",
]
