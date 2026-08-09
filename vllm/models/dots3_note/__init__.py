# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dots3 Note model entry point."""

from .mtp import DotsNote3MTP
from .omni import DotsNoteOmni3ForCausalLM

__all__ = ["DotsNoteOmni3ForCausalLM", "DotsNote3MTP"]
