# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark HF model entry point."""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from .nvidia.model import MuseSpark1xForCausalLM
    from .nvidia.multimodal import MuseSparkForConditionalGeneration
elif current_platform.is_rocm():
    from .amd.model import MuseSpark1xForCausalLM  # type: ignore[assignment]
    from .amd.multimodal import (
        MuseSparkForConditionalGeneration,  # type: ignore[assignment]
    )
elif current_platform.is_cpu():
    from .cpu.model import MuseSpark1xForCausalLM  # type: ignore[assignment]
    from .cpu.multimodal import (  # type: ignore[assignment]
        MuseSparkForConditionalGeneration,
    )
else:
    from .nvidia.model import MuseSpark1xForCausalLM
    from .nvidia.multimodal import MuseSparkForConditionalGeneration

__all__ = ["MuseSpark1xForCausalLM", "MuseSparkForConditionalGeneration"]
