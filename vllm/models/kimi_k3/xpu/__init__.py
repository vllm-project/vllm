# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU implementation of Kimi-K3."""

from .kda import KimiK3DeltaAttention, KimiLinearDeltaAttention
from .linear import (
    KimiDecoderLayer,
    KimiLinearForCausalLM,
    KimiLinearModel,
    KimiMLAAttention,
)
from .model import KimiK3ForConditionalGeneration
from .mtp import KimiK3MTP

__all__ = [
    "KimiDecoderLayer",
    "KimiK3DeltaAttention",
    "KimiK3ForConditionalGeneration",
    "KimiK3MTP",
    "KimiLinearDeltaAttention",
    "KimiLinearForCausalLM",
    "KimiLinearModel",
    "KimiMLAAttention",
]