# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.platforms import current_platform


def _use_common_impl() -> bool:
    return current_platform.is_rocm() or current_platform.is_xpu()


if TYPE_CHECKING:
    if _use_common_impl():
        from .common.model import (
            InklingForCausalLM,
            InklingForConditionalGeneration,
        )
        from .common.mtp import InklingMTP as InklingMTP
    else:
        from .nvidia.model import (
            InklingForCausalLM,
            InklingForConditionalGeneration,
        )
        from .nvidia.mtp import InklingMTP as InklingMTP

__all__ = [
    "InklingForConditionalGeneration",
    "InklingForCausalLM",
    "InklingMTP",
]


def __getattr__(name: str):
    if name == "InklingMTP":
        if _use_common_impl():
            from .common import mtp as common_mtp

            return common_mtp.InklingMTP

        from .nvidia import mtp as nvidia_mtp

        return nvidia_mtp.InklingMTP

    if name in __all__:
        if _use_common_impl():
            from .common import model as common_model

            return getattr(common_model, name)

        from .nvidia import model as nvidia_model

        return getattr(nvidia_model, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
