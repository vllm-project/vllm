# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Longcat Flash tool calls: the Hermes wrapper grammar with its own tags."""

from __future__ import annotations

from vllm.parser.hermes import HermesParser


class LongcatParser(HermesParser):
    TOOL_CALL_START = "<longcat_tool_call>"
    TOOL_CALL_END = "</longcat_tool_call>"
