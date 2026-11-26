# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dots3Note model entry point."""

from .nvidia.mtp import Dots3NoteMTP
from .nvidia.multimodal import Dots3NoteForCausalLM

__all__ = ["Dots3NoteForCausalLM", "Dots3NoteMTP"]
