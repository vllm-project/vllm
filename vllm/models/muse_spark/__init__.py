# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark HF model entry point."""

from .amd.model import MuseSpark1xForCausalLM
from .amd.multimodal import MuseSparkForConditionalGeneration

__all__ = ["MuseSpark1xForCausalLM", "MuseSparkForConditionalGeneration"]
