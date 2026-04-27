# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tool call parser for Apertus models.

Extracts tool calls from the format:
<|tools_prefix|>[{"function_name": {"arg1": "value1", ...}}, ...]<|tools_suffix|>

Used when --enable-auto-tool-choice --tool-call-parser apertus are set.
"""

import json
from collections.abc import Sequence

import regex as re
from partial_json_parser.core.options import Allow

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
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import find_common_prefix, partial_json_loads

logger = init_logger(__name__)

# Apertus special tokens for tool calls
TOOL_CALLS_PREFIX = "<|tools_prefix|>"
TOOL_CALLS_SUFFIX = "<|tools_suffix|>"


class ApertusToolParser(ToolParser):
    """
    Tool call parser for Apertus models.

    Handles the Apertus function call format:
    <|tools_prefix|>[{"function_name": {"arg1": "value1", ...}}, ...]<|tools_suffix|>
    """

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        # Regex for non-streaming: extract complete tool calls.
        self.tool_call_regex = re.compile(
            rf"{re.escape(TOOL_CALLS_PREFIX)}(.*?){re.escape(TOOL_CALLS_SUFFIX)}",
            re.DOTALL,
        )

        # Streaming state
        self._reset_streaming_state()
        self.buffered_delta_text = ""

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new request."""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def _buffer_delta_text(self, delta_text: str) -> str:
        """Buffer incoming delta text to handle multi-token special sequences."""
        combined = self.buffered_delta_text + delta_text

        if combined.endswith(TOOL_CALLS_PREFIX) or combined.endswith(
                TOOL_CALLS_SUFFIX):
            self.buffered_delta_text = ""
            return combined

        for tag in [TOOL_CALLS_PREFIX, TOOL_CALLS_SUFFIX]:
            for i in range(1, len(tag)):
                if combined.endswith(tag[:i]):
                    self.buffered_delta_text = combined[-i:]
                    return combined[:-i]

        self.buffered_delta_text = ""
        return combined

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response."""
        if TOOL_CALLS_PREFIX not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

        match = self.tool_call_regex.search(model_output)
        if not match:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

        try:
            json_str = match.group(1).strip()
            tool_call_objects = json.loads(json_str)

            if not isinstance(tool_call_objects, list):
                tool_call_objects = [tool_call_objects]

            tool_calls: list[ToolCall] = []
            for obj in tool_call_objects:
                if isinstance(obj, dict) and len(obj) == 1:
                    name = next(iter(obj))
                    arguments = obj[name]
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            id=make_tool_call_id(),
                            function=FunctionCall(
                                name=name,
                                arguments=json.dumps(arguments,
                                                     ensure_ascii=False),
                            ),
                        ))

            content_end = model_output.find(TOOL_CALLS_PREFIX)
            content = model_output[:content_end].strip(
            ) if content_end > 0 else None

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error extracting tool calls from Apertus response")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

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
        delta_text = self._buffer_delta_text(delta_text)

        if TOOL_CALLS_PREFIX not in current_text:
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        try:
            return self._extract_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
            )
        except Exception:
            logger.exception("Error in Apertus streaming tool call extraction")
            return None

    def _extract_streaming(self, previous_text: str, current_text: str, delta_text: str) -> DeltaMessage | None:
        start_count = current_text.count(TOOL_CALLS_PREFIX)
        end_count = current_text.count(TOOL_CALLS_SUFFIX)
        prev_start_count = previous_text.count(TOOL_CALLS_PREFIX)
        prev_end_count = previous_text.count(TOOL_CALLS_SUFFIX)

        # Outside tool call
        if start_count == end_count and prev_end_count == end_count and TOOL_CALLS_SUFFIX not in delta_text:
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        # Just finished
        if end_count > prev_end_count:
            delta = self._handle_tool_call_end(current_text)
            content_str = None
            if TOOL_CALLS_SUFFIX in delta_text:
                suffix_idx = delta_text.find(TOOL_CALLS_SUFFIX) + len(TOOL_CALLS_SUFFIX)
                if suffix_idx < len(delta_text):
                    content_str = delta_text[suffix_idx:]

            if delta:
                delta.content = content_str
                return delta
            elif content_str:
                return DeltaMessage(content=content_str)
            return None

        # In the middle
        if start_count > end_count:
            delta = self._handle_tool_call_middle(current_text)
            content_str = None
            if start_count > prev_start_count and TOOL_CALLS_PREFIX in delta_text:
                prefix_idx = delta_text.find(TOOL_CALLS_PREFIX)
                if prefix_idx > 0:
                    content_str = delta_text[:prefix_idx]

            if delta:
                delta.content = content_str
                return delta
            elif content_str:
                return DeltaMessage(content=content_str)
            return None

        if delta_text:
            text = delta_text.replace(TOOL_CALLS_PREFIX, "").replace(TOOL_CALLS_SUFFIX, "")
            if text:
                return DeltaMessage(content=text)
        return None

    def _handle_tool_call_middle(self, current_text: str) -> DeltaMessage | None:
        last_start = current_text.rfind(TOOL_CALLS_PREFIX)
        json_part = current_text[last_start + len(TOOL_CALLS_PREFIX):]

        try:
            # Apertus format is a list: [{"name": {args}}]
            parsed, _ = partial_json_loads(json_part, Allow.ALL)
            if not isinstance(parsed, list):
                parsed = [parsed] if parsed else []
        except Exception:
            return None

        if not parsed:
            return None

        # Check if we moved to a new tool in the list
        new_tool_index = len(parsed) - 1
        if new_tool_index > self.current_tool_id:
            # Finalize previous tool if exists
            delta = None
            if self.current_tool_id >= 0:
                delta = self._finalize_tool(parsed, self.current_tool_id)

            self.current_tool_id = new_tool_index
            self.current_tool_name_sent = False
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            
            if delta:
                return delta

        current_obj = parsed[self.current_tool_id]
        if not isinstance(current_obj, dict) or not current_obj:
            return None

        name = next(iter(current_obj))
        args = current_obj[name]

        # Send name once
        if not self.current_tool_name_sent:
            self.current_tool_name_sent = True
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=name,
                            arguments="",
                        ).model_dump(exclude_none=True),
                    )
                ])

        # Diff arguments
        if args is not None:
            return self._emit_tool_diff(**args)

        return None

    def _emit_tool_diff(self, index: int, args_dict: dict, is_final: bool = False) -> DeltaMessage | None:
        full_args_json = json.dumps(args_dict, ensure_ascii=False)
        safe_json = full_args_json

        # Withhold trailing structural chars unless final
        if not is_final:
            while safe_json and safe_json[-1] in ("}", '"', "]", " ", ","):
                safe_json = safe_json[:-1]

        prev_sent = self.streamed_args_for_tool[index]
        if safe_json == prev_sent:
            return None

        if prev_sent:
            prefix = find_common_prefix(prev_sent, safe_json)
            if len(prefix) < len(prev_sent):
                self.streamed_args_for_tool[index] = prefix
                return None
            diff = safe_json[len(prev_sent):]
        else:
            diff = safe_json

        if diff:
            self.streamed_args_for_tool[index] = safe_json
            return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                                index=index,
                                function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True)
                                )
                        ]
                    )
        return None

    def _finalize_tool(self, parsed: list, index: int) -> DeltaMessage | None:
        if index < 0 or index >= len(parsed):
            return None

        current_obj = parsed[index]
        if not isinstance(current_obj, dict) or not current_obj:
            return None

        name = next(iter(current_obj))
        final_args_json = json.dumps(current_obj[name], ensure_ascii=False)

        prev_sent = self.streamed_args_for_tool[index]
        diff = final_args_json[len(prev_sent):]

        if diff:
            self.streamed_args_for_tool[index] = final_args_json
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=index,
                        function=DeltaFunctionCall(arguments=diff).model_dump(
                            exclude_none=True),
                    )
                ])
        return None

    def _handle_tool_call_end(self, current_text: str) -> DeltaMessage | None:
        match = self.tool_call_regex.search(current_text)
        if not match:
            return None

        try:
            json_str = match.group(1).strip()
            tool_call_objects = json.loads(json_str)
            if not isinstance(tool_call_objects, list):
                tool_call_objects = [tool_call_objects]

            if self.current_tool_id < 0 or self.current_tool_id >= len(
                    tool_call_objects):
                return None

            current_obj = tool_call_objects[self.current_tool_id]
            name = next(iter(current_obj))
            final_args_json = json.dumps(current_obj[name], ensure_ascii=False)

            prev_sent = self.streamed_args_for_tool[self.current_tool_id]
            diff = final_args_json[len(prev_sent):]

            if diff:
                self.streamed_args_for_tool[self.current_tool_id] = final_args_json
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=diff).model_dump(
                                exclude_none=True),
                        )
                    ])
        except Exception:
            pass
        return None
