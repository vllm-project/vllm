# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark HF model entry point."""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING or not current_platform.is_rocm():
    from .nvidia.model import MuseSpark1xForCausalLM
    from .nvidia.multimodal import MuseSparkForConditionalGeneration
else:
    from .amd.model import MuseSpark1xForCausalLM  # type: ignore[assignment]
    from .amd.multimodal import (
        MuseSparkForConditionalGeneration,  # type: ignore[assignment]
    )

__all__ = ["MuseSpark1xForCausalLM", "MuseSparkForConditionalGeneration"]
