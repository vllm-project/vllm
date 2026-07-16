# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING:
    if current_platform.is_rocm():
        from .amd.model import (
            InklingForCausalLM,
            InklingForConditionalGeneration,
        )
        from .amd.mtp import InklingMTP
    else:
        from .nvidia.model import (
            InklingForCausalLM,
            InklingForConditionalGeneration,
        )
        from .nvidia.mtp import InklingMTP


__all__ = [
    "InklingForConditionalGeneration",
    "InklingForCausalLM",
    "InklingMTP",
]


def __getattr__(name: str):
    if name == "InklingMTP":
        if current_platform.is_rocm():
            from .amd import mtp as amd_mtp

            return amd_mtp.InklingMTP

        from .nvidia import mtp as nvidia_mtp

        return nvidia_mtp.InklingMTP

    if name in __all__:
        if current_platform.is_rocm():
            from .amd import model as amd_model

            return getattr(amd_model, name)

        from .nvidia import model as nvidia_model

        return getattr(nvidia_model, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
