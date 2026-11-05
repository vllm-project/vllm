# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZAYA1 parser for tool calls and reasoning.

ZAYA1 uses the Qwen3 grammar (``<think>`` reasoning and XML tool calls) but
wraps tool calls in ``<zyphra_tool_call>`` instead of ``<tool_call>``.
"""

from vllm.parser.qwen3 import Qwen3Parser


class ZayaParser(Qwen3Parser):
    CONFIG_NAME = "zaya"
    TOOL_START = "<zyphra_tool_call>"
    TOOL_END = "</zyphra_tool_call>"
