# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility entry point for the Muse Glimmer reasoning parser."""

from vllm.reasoning.muse_reasoning_parser import MuseReasoningParser


class MuseGlimmerReasoningParser(MuseReasoningParser):
    _glimmer_compat = True
