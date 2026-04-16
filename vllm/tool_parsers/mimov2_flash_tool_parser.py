# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
else:
    TokenizerLike = object

logger = init_logger(__name__)


class MimoV2FlashToolParser(ToolParser):
    """
    Tool parser for MiMo-V2-Flash models.

    The model outputs tool calls in XML-like format:

    <tool_call>
    <function=execute_bash>
    <parameter=command>pwd && ls</parameter>
    </function>
    </tool_call>
    """

    def __init__(self, tokenizer: "TokenizerLike", tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        # For <function=NAME> format (in <tool_call>)
        self.function_attr_pattern = re.compile(r"<function=([^>]+)>")
        self.parameter_tag_pattern = re.compile(r"<parameter=([^>]+)>")
        self.parameter_end_tag: str = "</parameter>"

        # Streaming state
        self._sent_content_idx: int = 0
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

    def _parse_single_tool(self, tool_content: str) -> tuple[str, dict]:
        """Parse a single tool block and return (name, arguments dict).

        Format: <function=NAME> followed by <parameter=KEY>VALUE</parameter>
        """
        arguments = {}

        # Attribute format: <function=NAME>
        func_match = self.function_attr_pattern.search(tool_content)
        if func_match:
            func_name = func_match.group(1).strip()
        else:
            return "", {}

        remaining = tool_content
        while True:
            param_match = self.parameter_tag_pattern.search(remaining)
            if not param_match:
                break

            param_name = param_match.group(1).strip()
            param_start = param_match.end()

            param_end_pos = remaining.find(self.parameter_end_tag, param_start)
            if param_end_pos == -1:
                break

            param_value = remaining[param_start:param_end_pos]
            arguments[param_name] = param_value

            remaining = remaining[param_end_pos + len(self.parameter_end_tag) :]

        return func_name, arguments

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from model output for non-streaming mode."""

        has_tool_call = self.tool_call_start_token in model_output

        if not has_tool_call:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls = []
            content = ""

            first_tool_pos = model_output.find(self.tool_call_start_token)
            if first_tool_pos > 0:
                content = model_output[:first_tool_pos].strip()

            start_idx = 0
            while True:
                tool_start = model_output.find(
                    self.tool_call_start_token, start_idx
                )
                if tool_start == -1:
                    break

                tool_end = model_output.find(self.tool_call_end_token, tool_start)
                if tool_end == -1:
                    break

                tool_content = model_output[
                    tool_start + len(self.tool_call_start_token) : tool_end
                ]

                func_name, arguments = self._parse_single_tool(tool_content)
                if func_name:
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=func_name,
                                arguments=json.dumps(arguments, ensure_ascii=False),
                            ),
                        )
                    )

                start_idx = tool_end + len(self.tool_call_end_token)

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent non-tool-call text, or None."""
        tool_pos = (
            current_text.find(self.tool_call_start_token)
            if self.tool_call_start_token in current_text
            else -1
        )

        sendable_idx = tool_pos if tool_pos != -1 else len(current_text)

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def _extract_tool_call_blocks(self, text: str) -> list[tuple[str, bool]]:
        """Extract tool call blocks from text.

        Returns list of (block_content, is_complete) tuples.
        """
        results: list[tuple[str, bool]] = []

        pos = 0
        while True:
            start = text.find(self.tool_call_start_token, pos)
            if start == -1:
                break

            content_start = start + len(self.tool_call_start_token)
            end = text.find(self.tool_call_end_token, content_start)

            if end != -1:
                block = text[content_start:end]
                results.append((block, True))
                pos = end + len(self.tool_call_end_token)
            else:
                # Incomplete block
                block = text[content_start:]
                results.append((block, False))
                break

        return results

    def _parse_tool_block(self, block: str) -> tuple[str, dict]:
        """Parse a tool block and return (name, arguments dict).

        Format: <function=NAME> (attribute format).
        """
        func_match = self.function_attr_pattern.search(block)
        if func_match:
            func_name = func_match.group(1).strip()
        else:
            return "", {}

        arguments = {}

        remaining = block
        while True:
            param_match = self.parameter_tag_pattern.search(remaining)
            if not param_match:
                break

            param_name = param_match.group(1).strip()
            param_start = param_match.end()

            param_end_pos = remaining.find(self.parameter_end_tag, param_start)
            if param_end_pos == -1:
                break

            param_value = remaining[param_start:param_end_pos]
            arguments[param_name] = param_value

            remaining = remaining[param_end_pos + len(self.parameter_end_tag) :]

        return func_name, arguments

    def _get_new_arguments(self, index: int, full_args: str) -> str | None:
        """Return new argument text not yet sent for tool at index."""
        prev_args = (
            self.streamed_args_for_tool[index]
            if index < len(self.streamed_args_for_tool)
            else ""
        )

        if len(full_args) <= len(prev_args):
            return None

        new_args = full_args[len(prev_args) :]
        self.streamed_args_for_tool[index] = full_args
        return new_args

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
        """Stream tool calls incrementally."""
        try:
            # Extract content before tool calls
            content = self._extract_content(current_text)
            tool_call_jsons = self._extract_tool_call_blocks(current_text)
            tool_call_deltas: list[DeltaToolCall] = []

            for i, (block, _) in enumerate(tool_call_jsons):
                if i >= len(self.prev_tool_call_arr):
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")

                # Stream tool name if not already sent
                if "name" not in self.prev_tool_call_arr[i]:
                    func_name, _ = self._parse_tool_block(block)
                    if not func_name:
                        break
                    self.prev_tool_call_arr[i]["name"] = func_name
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(name=func_name).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

                # Stream arguments
                _, arguments = self._parse_tool_block(block)
                args_str = json.dumps(arguments, ensure_ascii=False)
                new_args = self._get_new_arguments(i, args_str)
                if new_args:
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            function=DeltaFunctionCall(arguments=new_args).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

            if content or tool_call_deltas:
                return DeltaMessage(
                    content=content,
                    tool_calls=tool_call_deltas,
                )

            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
