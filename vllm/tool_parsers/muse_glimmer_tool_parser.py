# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility entry point for the Muse Glimmer tool parser."""

from vllm.tool_parsers.muse_tool_parser import MuseToolParser


class MuseGlimmerToolParser(MuseToolParser):
    _reasoning_recipients = ("self",)
    _content_recipients = (None, "user")
    _non_tool_recipients = ("self", "user")
