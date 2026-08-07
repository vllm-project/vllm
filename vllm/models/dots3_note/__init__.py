# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dots3 Note model entry point."""

from .mtp import Dot3NoteMTP
from .omni import Dots3NoteOmniForCausalLM

__all__ = ["Dots3NoteOmniForCausalLM", "Dot3NoteMTP"]
