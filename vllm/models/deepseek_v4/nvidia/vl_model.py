# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility imports for the platform-neutral DeepSeek-V4 vision model."""

from ..common.vl_model import (
    DeepseekV4ForConditionalGeneration,
)
from ..common.vl_model import (
    _make_deepseek_v4_vl_weights_mapper as _make_common_vl_weights_mapper,
)
from .model import _make_deepseek_v4_weights_mapper


def _make_deepseek_v4_vl_weights_mapper(expert_dtype: str, image_enabled: bool):
    """Retain the original NVIDIA helper signature for downstream imports."""
    return _make_common_vl_weights_mapper(
        _make_deepseek_v4_weights_mapper(expert_dtype), image_enabled
    )


__all__ = ["DeepseekV4ForConditionalGeneration"]
