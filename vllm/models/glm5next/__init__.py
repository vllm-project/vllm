# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from .common.model import (
        Glm5NextForCausalLM,
        Glm5NextForConditionalGeneration,
    )
    from .common.mtp import Glm5NextMTP

__all__ = [
    "Glm5NextForCausalLM",
    "Glm5NextForConditionalGeneration",
    "Glm5NextMTP",
]

if current_platform.is_xpu():

    def __getattr__(name: str) -> Any:
        if name not in __all__:
            raise AttributeError(name)
        raise NotImplementedError("GLM-5.3-Flash does not currently support XPU.")
else:
    from .common.model import (
        Glm5NextForCausalLM,
        Glm5NextForConditionalGeneration,
    )
    from .common.mtp import Glm5NextMTP
