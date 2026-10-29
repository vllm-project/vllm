# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam model entry point."""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

from .common import dflash2_conv_ops, dflash2_ops, layernorm

if TYPE_CHECKING:
    from .nvidia import (
        activation,
        attention,
        attention_ops,
        model,
    )
    from .nvidia.dflash2 import DFlash2BeamForCausalLM
    from .nvidia.model import BeamForCausalLM
elif current_platform.is_rocm():
    from .amd import (
        activation,
        attention,
        attention_ops,
        model,
    )
    from .amd.dflash2 import DFlash2BeamForCausalLM
    from .amd.model import BeamForCausalLM
else:
    from .nvidia import (
        activation,
        attention,
        attention_ops,
        model,
    )
    from .nvidia.dflash2 import DFlash2BeamForCausalLM
    from .nvidia.model import BeamForCausalLM

__all__ = [
    "BeamForCausalLM",
    "DFlash2BeamForCausalLM",
    "activation",
    "attention",
    "attention_ops",
    "dflash2_conv_ops",
    "dflash2_ops",
    "layernorm",
    "model",
]
