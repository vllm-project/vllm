# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam model entry point."""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from .nvidia.dflash2 import DFlash2BeamForCausalLM
    from .nvidia.model import BeamForCausalLM
elif current_platform.is_rocm():
    from .amd.dflash2 import DFlash2BeamForCausalLM
    from .amd.model import BeamForCausalLM
else:
    from .nvidia.dflash2 import DFlash2BeamForCausalLM
    from .nvidia.model import BeamForCausalLM

__all__ = ["BeamForCausalLM", "DFlash2BeamForCausalLM"]
