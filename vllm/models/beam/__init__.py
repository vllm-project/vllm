# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam model entry point."""

from .nvidia.dflash2 import DFlash2BeamForCausalLM
from .nvidia.model import BeamForCausalLM

__all__ = ["BeamForCausalLM", "DFlash2BeamForCausalLM"]
