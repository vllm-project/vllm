# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Damork-branded processor aliases for Qwen3-VL-compatible checkpoints."""

from transformers import AutoImageProcessor, AutoProcessor, AutoVideoProcessor
from transformers.models.qwen2_vl import Qwen2VLImageProcessorFast
from transformers.models.qwen3_vl import Qwen3VLProcessor, Qwen3VLVideoProcessor

from vllm.transformers_utils.configs.qwen3_5 import DamorkConfig


class DamorkVLImageProcessorFast(Qwen2VLImageProcessorFast):
    """Damork alias for the Qwen2-VL fast image processor."""


class DamorkVLVideoProcessor(Qwen3VLVideoProcessor):
    """Damork alias for the Qwen3-VL video processor."""


class DamorkVLProcessor(Qwen3VLProcessor):
    """Damork alias for the Qwen3-VL processor."""

    image_processor_class = "DamorkVLImageProcessorFast"
    video_processor_class = "DamorkVLVideoProcessor"


AutoImageProcessor.register(
    DamorkConfig,
    fast_image_processor_class=DamorkVLImageProcessorFast,
    exist_ok=True,
)
AutoVideoProcessor.register(
    DamorkConfig,
    DamorkVLVideoProcessor,
    exist_ok=True,
)
AutoProcessor.register(DamorkConfig, DamorkVLProcessor, exist_ok=True)


__all__ = [
    "DamorkVLImageProcessorFast",
    "DamorkVLProcessor",
    "DamorkVLVideoProcessor",
]
