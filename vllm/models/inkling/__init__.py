# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        from .nvidia import mtp

        return mtp.InklingMTP
    if name in __all__:
        from .nvidia import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
