# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU implementation of Kimi-K3."""

from .linear import KimiLinearForCausalLM
from .model import KimiK3ForConditionalGeneration
from .mtp import KimiK3MTP

__all__ = [
    "KimiK3ForConditionalGeneration",
    "KimiK3MTP",
    "KimiLinearForCausalLM",
]