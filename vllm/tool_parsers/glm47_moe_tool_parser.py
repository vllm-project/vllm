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
from openai.types.responses.tool_choice_function import ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
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

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Skip parent's Hermes JSON-schema injection for required / named.

        Extends ``Glm4MoeModelToolParser.adjust_request`` with the Responses
        API named-function form. The parent recognises only
        ``ChatCompletionNamedToolChoiceParam`` (Chat Completions); on
        ``/v1/responses`` the same ``{"type": "function", "name": ...}`` dict
        is parsed by Pydantic into ``ToolChoiceFunction`` — a different class
        — and would otherwise fall through to the grandparent's
        structured-output injection. GLM-4.7 emits XML ``<tool_call>``
        markers, so that schema would force JSON output and the parser would
        find nothing to extract.
        """
        if request.tools:
            tool_choice = request.tool_choice
            is_named = isinstance(
                tool_choice, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
            )
            if tool_choice == "required" or is_named:
                # Skip parent + grandparent. Keep tool-call XML tokens visible.
                if tool_choice != "none":
                    request.skip_special_tokens = False
                return request
        return super().adjust_request(request)
