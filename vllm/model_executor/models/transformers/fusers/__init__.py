# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete fusers for the Transformers modeling backend."""

from vllm.model_executor.models.transformers.fusers.base import BaseFuser, StackedFuser
from vllm.model_executor.models.transformers.fusers.glu import GLUFuser
from vllm.model_executor.models.transformers.fusers.moe import MoEBlockFuser
from vllm.model_executor.models.transformers.fusers.qkv import QKVFuser
from vllm.model_executor.models.transformers.fusers.rms_norm import RMSNormFuser

__all__ = [
    "BaseFuser",
    "StackedFuser",
    "GLUFuser",
    "MoEBlockFuser",
    "QKVFuser",
    "RMSNormFuser",
]
