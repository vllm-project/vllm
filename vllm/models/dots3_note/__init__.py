# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dot3 Note model entry point."""

from .mtp import Dot3NoteMTP
from .omni import Dot3NoteOmniForCausalLM as Dot3NoteForCausalLM

__all__ = ["Dot3NoteForCausalLM", "Dot3NoteMTP"]
