# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GLM4 Tool-Only Parser - Optimized for ultra-low latency tool calling.

This parser works exclusively with glm4_tool_only_chat_template.jinja to force
GLM4 models to emit ONLY tool calls (no text/reasoning), significantly reducing
tool calling latency by eliminating all non-essential tokens.

The template ends with "<tool_call>" so the model's first token is the function
name, followed only by XML argument tags. Cannot be used for general-purpose
tool calling - use Glm4MoeModelToolParser for mixed content instead.

Usage: tool_parser="glm4_tool_only"
"""

import json
from collections.abc import Sequence
from typing import Any, Optional

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParserManager)
from vllm.entrypoints.openai.tool_parsers.glm4_moe_tool_parser import (
    Glm4MoeModelToolParser, glm4_deserialize, glm4_is_string_type)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


@ToolParserManager.register_module("glm4_tool_only")
class Glm4ToolOnlyParser(Glm4MoeModelToolParser):
    """
    Parser for tool-only GLM4 output where the template pre-fills <tool_call>.
    The model's first token is the function name, followed only by argument XML.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        # We start assuming we're already inside <tool_call> since the
        # template adds it
        self._current_tool_id: int = 0
        self._current_tool_name: Optional[str] = ""  # Ready for name
        self._current_arg_name: Optional[str] = None
        self._current_arg_value: Optional[str] = None
        self._args_dict: dict[str, Any] = {}
        self._tool_id_counter = 0

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Non-streaming extraction. Since template adds <tool_call>, we prepend it
        to the model output and use the parent parser.
        """
        # The template ends with <tool_call>, so model output starts with
        # function name
        wrapped_output = f"<tool_call>{model_output}"

        # Use parent's extraction logic
        result = super().extract_tool_calls(wrapped_output, request)

        # In tool-only mode, we never return content
        result.content = ""
        return result

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        Streaming extraction for tool-only mode.
        Since the template ends with <tool_call>, we start inside a tool call.
        """

        if delta_text == "<tool_call>":
            if self._current_tool_name is not None:
                logger.error("Misplaced <tool_call> for tool call: %s",
                             current_text)
            self._tool_id_counter += 1
            self._current_tool_id = self._tool_id_counter
            self._current_tool_name = ""
            self._current_arg_name = None
            self._current_arg_value = None
            self._args_dict = {}
            return self._create_result()

        if delta_text == "</tool_call>":
            if self._current_tool_name:
                self._current_tool_name = None
                return self._create_result(
                    delta_args=json.dumps(self._args_dict))
            else:
                logger.error("Misplaced </tool_call> for tool call: %s",
                             current_text)
                return None

        if delta_text == "<arg_key>":
            if self._current_arg_name is not None:
                logger.error("Misplaced <arg_key> for tool call: %s",
                             current_text)
            self._current_arg_name = ""
            return None

        if delta_text == "</arg_key>":
            if not self._current_tool_name:
                logger.error("Misplaced </arg_key> for tool call: %s",
                             current_text)
                return None
            return None

        if delta_text == "<arg_value>":
            if self._current_arg_value is not None:
                logger.error("Misplaced <arg_value> for tool call: %s",
                             current_text)
            self._current_arg_value = ""
            return None

        if delta_text == "</arg_value>":
            if (self._current_tool_name and self._current_arg_value
                    and self._current_arg_name):
                if glm4_is_string_type(self._current_tool_name,
                                       self._current_arg_name, request.tools):
                    arg_val = self._current_arg_value
                else:
                    arg_val = glm4_deserialize(self._current_arg_value)
                self._args_dict[self._current_arg_name] = arg_val
                self._current_arg_name = None
                self._current_arg_value = None
            else:
                logger.error("Misplaced </arg_value> for tool call: %s",
                             current_text)
            return None

        if delta_text == "\n":
            return None

        # Handle content tokens depending on where we are at
        if self._current_arg_value is not None:
            self._current_arg_value += delta_text
            return None
        if self._current_arg_name is not None:
            self._current_arg_name += delta_text
            return None
        if self._current_tool_name is not None:
            self._current_tool_name += delta_text
            return self._create_result(delta_name=delta_text)

        # If we got here, we're likely not in a tool call at all
        return None

    def _create_result(self,
                       delta_name: Optional[str] = None,
                       delta_args: Optional[str] = None) -> DeltaMessage:
        return DeltaMessage(tool_calls=[
            DeltaToolCall(
                id=str(self._current_tool_id),
                index=self._current_tool_id,
                type="function",
                function=DeltaFunctionCall(name=delta_name,
                                           arguments=delta_args),
            )
        ])
