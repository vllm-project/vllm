# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

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
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)


class TrinityToolParser(ToolParser):
    """
    Tool parser for Trinity models using Qwen-style tool call format:

    <tool_call>
    {"name":"func1", "arguments":{...}}
    </tool_call>

    Tool calls can appear inside <think> sections; the tags are stripped before
    parsing while preserving the contents.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        self.current_tool_id = -1

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"

        self.tool_call_regex = re.compile(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
        )

        self._buffer = ""
        self._think_buffer = ""

    def _strip_think_tags(self, text: str) -> str:
        return text.replace(self.think_start_token, "").replace(
            self.think_end_token, ""
        )

    def _ends_with_partial_token(self, text: str, token: str) -> int:
        max_len = min(len(text), len(token) - 1)
        for i in range(max_len, 0, -1):
            if text.endswith(token[:i]):
                return i
        return 0

    def _strip_think_tags_streaming(self, text: str) -> str:
        if not text:
            return ""
        combined = self._think_buffer + text
        combined = combined.replace(self.think_start_token, "").replace(
            self.think_end_token, ""
        )
        pending_len = max(
            self._ends_with_partial_token(combined, self.think_start_token),
            self._ends_with_partial_token(combined, self.think_end_token),
        )
        if pending_len:
            self._think_buffer = combined[-pending_len:]
            return combined[:-pending_len]
        self._think_buffer = ""
        return combined

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        cleaned_output = self._strip_think_tags(model_output)
        if self.tool_call_start_token not in cleaned_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_output
            )

        try:
            tool_call_json_list = self.tool_call_regex.findall(cleaned_output)
            tool_calls = []
            self.prev_tool_call_arr = []
            for tool_call_json in tool_call_json_list:
                tool_call_dict = json.loads(tool_call_json)
                args_str = json.dumps(
                    tool_call_dict.get("arguments", {}), ensure_ascii=False
                )
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tool_call_dict.get("name", ""),
                            arguments=args_str,
                        ),
                    )
                )
                self.prev_tool_call_arr.append(
                    {"name": tool_call_dict.get("name", ""), "arguments": args_str}
                )

            content_idx = cleaned_output.find(self.tool_call_start_token)
            content = cleaned_output[:content_idx].strip()
            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_output
            )

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
        if not previous_text:
            self._buffer = ""
            self._think_buffer = ""
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = []
            self.current_tool_id = -1

        cleaned_delta = self._strip_think_tags_streaming(delta_text)
        if not cleaned_delta and not self._buffer:
            return None

        self._buffer += cleaned_delta

        start_idx = self._buffer.find(self.tool_call_start_token)
        if start_idx == -1:
            partial_len = self._ends_with_partial_token(
                self._buffer, self.tool_call_start_token
            )
            if partial_len:
                content = self._buffer[:-partial_len]
                self._buffer = self._buffer[-partial_len:]
            else:
                content = self._buffer
                self._buffer = ""
            return DeltaMessage(content=content if content else None)

        if start_idx > 0:
            content = self._buffer[:start_idx]
            self._buffer = self._buffer[start_idx:]
            return DeltaMessage(content=content if content else None)

        end_idx = self._buffer.find(self.tool_call_end_token)
        if end_idx == -1:
            return None

        full_call_text = self._buffer[: end_idx + len(self.tool_call_end_token)]
        self._buffer = self._buffer[end_idx + len(self.tool_call_end_token) :]

        match = self.tool_call_regex.search(full_call_text)
        if not match:
            logger.warning("Failed to parse tool call block: %s", full_call_text)
            return None

        try:
            tool_call_dict = json.loads(match.group(1))
            args_str = json.dumps(
                tool_call_dict.get("arguments", {}), ensure_ascii=False
            )
        except json.JSONDecodeError:
            logger.exception("Failed to decode tool call JSON: %s", match.group(1))
            return None

        if self.current_tool_id == -1:
            self.current_tool_id = 0
        tool_index = self.current_tool_id
        self.current_tool_id += 1

        while len(self.prev_tool_call_arr) <= tool_index:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= tool_index:
            self.streamed_args_for_tool.append("")

        self.prev_tool_call_arr[tool_index] = {
            "name": tool_call_dict.get("name", ""),
            "arguments": json.loads(args_str),
        }
        self.streamed_args_for_tool[tool_index] = args_str

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=tool_index,
                    id=make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(
                        name=tool_call_dict.get("name", ""), arguments=args_str
                    ),
                )
            ]
        )
