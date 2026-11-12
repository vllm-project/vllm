# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Inkling2ForConditionalGeneration

__all__ = ["Inkling2ForConditionalGeneration"]


def __getattr__(name: str):
    if name in __all__:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
