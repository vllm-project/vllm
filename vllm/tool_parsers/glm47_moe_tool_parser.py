# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GLM-4.7 Tool Call Parser.

GLM-4.7 uses a slightly different tool call format compared to GLM-4.5:
  - The function name may appear on the same line as ``<tool_call>`` without
    a newline separator before the first ``<arg_key>``.
  - Tool calls may have zero arguments
    (e.g. ``<tool_call>func</tool_call>``).

This parser overrides the parent regex patterns to handle both formats.
"""

import regex as re

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool
from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser

logger = init_logger(__name__)


class Glm47MoeModelToolParser(Glm4MoeModelToolParser):
    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        # GLM-4.7 format: <tool_call>func_name[<arg_key>...]*</tool_call>
        # The function name can be followed by a newline, whitespace, or
        # directly by <arg_key> tags (no separator).  The arg section is
        # optional so that zero-argument calls are supported.
        self.func_detail_regex = re.compile(
            r"<tool_call>\s*(\S+?)\s*(<arg_key>.*)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
