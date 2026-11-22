# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2: platform-specific implementations live under `nvidia/`; configs,
processing, quantization metadata and the DSpark speculator are shared."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nvidia.dspark import Inkling2DSparkModel
    from .nvidia.loop import Inkling2LoopForConditionalGeneration
    from .nvidia.model import Inkling2ForConditionalGeneration

__all__ = [
    "Inkling2DSparkModel",
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
]


def __getattr__(name: str):
    if name == "Inkling2DSparkModel":
        from .nvidia import dspark

        return dspark.Inkling2DSparkModel
    if name == "Inkling2LoopForConditionalGeneration":
        from .nvidia import loop

        return loop.Inkling2LoopForConditionalGeneration
    if name in __all__:
        from .nvidia import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
