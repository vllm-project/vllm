# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import json
from collections.abc import Sequence
from typing import Any

import regex as re

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
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.utils import random_uuid

logger = init_logger(__name__)


class Step3ToolParser(ToolParser):
    """
    Tool parser for a model that uses a specific XML-like format for tool calls.
    This version uses a robust, stateful, cursor-based streaming parser and
    consolidates tool arguments into a single message.
    """

    TOOL_CALLS_BEGIN = "<｜tool_calls_begin｜>"
    TOOL_CALLS_END = "<｜tool_calls_end｜>"
    TOOL_CALL_BEGIN = "<｜tool_call_begin｜>"
    TOOL_CALL_END = "<｜tool_call_end｜>"
    TOOL_SEP = "<｜tool_sep｜>"
    SPECIAL_TOKENS = [TOOL_CALLS_BEGIN, TOOL_CALLS_END, TOOL_CALL_BEGIN, TOOL_CALL_END]

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.position = 0
        # Explicit state flags for robust streaming
        self.tool_block_started = False
        self.tool_block_finished = False

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    @staticmethod
    def _parse_steptml_invoke(
        action_text: str,
    ) -> tuple[str | None, dict[str, str] | None]:
        func_name_match = re.search(r'<steptml:invoke name="([^"]+)">', action_text)
        if not func_name_match:
            return None, None
        func_name = func_name_match.group(1)

        params: dict[str, str] = {}
        param_matches = re.findall(
            r'<steptml:parameter name="([^"]+)">([^<]*)</steptml:parameter>',
            action_text,
        )
        for name, value in param_matches:
            params[name] = value.strip()
        return func_name, params

    def _cast_arguments(
        self,
        func_name: str,
        params: dict[str, Any],
        request: ChatCompletionRequest,
    ) -> dict[str, Any]:
        for tool in request.tools or []:
            if tool.function.name == func_name:
                schema = tool.function.parameters or {}
                properties = schema.get("properties", {})
                for key, value in params.items():
                    if not isinstance(value, str):
                        continue
                    prop = properties.get(key, {})
                    typ = prop.get("type")
                    if typ == "string":
                        params[key] = value.strip()
                    elif typ == "integer":
                        with contextlib.suppress(ValueError):
                            params[key] = int(value)
                    elif typ == "number":
                        with contextlib.suppress(ValueError):
                            params[key] = float(value)
                    elif typ == "boolean":
                        lower_val = value.lower()
                        params[key] = (
                            lower_val == "true"
                            if lower_val in ("true", "false")
                            else value
                        )
                    elif typ == "null":
                        params[key] = None if value.lower() == "null" else value
                break
        return params

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
        # The main loop processes the stream from the last known position.
        while True:
            if self.position >= len(current_text):
                return None  # We've processed the entire stream.

            unprocessed_text = current_text[self.position :]

            # STATE: After all tools are done, all subsequent text is content.
            if self.tool_block_finished:
                self.position = len(current_text)
                return DeltaMessage(content=unprocessed_text)

            # STATE: Before the tool block has started.
            if not self.tool_block_started:
                if unprocessed_text.startswith(self.TOOL_CALLS_BEGIN):
                    self.position += len(self.TOOL_CALLS_BEGIN)
                    self.tool_block_started = True
                    continue  # Token consumed, re-loop.

                start_pos = unprocessed_text.find(self.TOOL_CALLS_BEGIN)
                if start_pos == -1:
                    if (
                        self.TOOL_CALLS_BEGIN.startswith(unprocessed_text.strip())
                        and unprocessed_text
                    ):
                        return None  # It's a prefix, wait.
                    self.position = len(current_text)
                    return DeltaMessage(content=unprocessed_text)
                else:
                    content = unprocessed_text[:start_pos]
                    self.position += len(content)
                    return DeltaMessage(content=content)

            # STATE: Inside the main tool block.
            offset = len(unprocessed_text) - len(unprocessed_text.lstrip())
            unprocessed_text = unprocessed_text.lstrip()
            self.position += offset

            if unprocessed_text.startswith(self.TOOL_CALLS_END):
                self.position += len(self.TOOL_CALLS_END)
                self.tool_block_finished = True
                self.current_tool_id = -1
                continue

            # Check if we are between tool calls.
            tool_finished = self.current_tool_id != -1 and self.prev_tool_call_arr[
                self.current_tool_id
            ].get("finished")
            if self.current_tool_id == -1 or tool_finished:
                if unprocessed_text.startswith(self.TOOL_CALL_BEGIN):
                    self.position += len(self.TOOL_CALL_BEGIN)
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                    else:
                        self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    self.prev_tool_call_arr[self.current_tool_id]["finished"] = False
                    continue

                if self.TOOL_CALL_BEGIN.startswith(unprocessed_text):
                    return None

            # STATE: Parsing an active tool call.
            if self.current_tool_id != -1 and not self.prev_tool_call_arr[
                self.current_tool_id
            ].get("finished", False):
                end_tool_pos = unprocessed_text.find(self.TOOL_CALL_END)
                if end_tool_pos == -1:
                    tool_body = unprocessed_text
                else:
                    tool_body = unprocessed_text[:end_tool_pos]

                if end_tool_pos == -1 and self.TOOL_CALL_END.startswith(tool_body):
                    return None

                function_name, arguments = self._parse_steptml_invoke(tool_body)
                if not function_name:
                    return None

                tool_call_arr = {"name": function_name, "parameters": arguments or {}}

                # Send the function name as soon as it's parsed.
                if not self.current_tool_name_sent:
                    self.current_tool_name_sent = True
                    self.prev_tool_call_arr[self.current_tool_id].update(tool_call_arr)
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=f"chatcmpl-tool-{random_uuid()}",
                                function=DeltaFunctionCall(name=function_name),
                            )
                        ]
                    )

                # Update our internal state with the latest parsed arguments.
                self.prev_tool_call_arr[self.current_tool_id].update(  # noqa: E501
                    tool_call_arr
                )

                # Only send arguments when the tool call is complete.
                if end_tool_pos != -1:
                    self.position += end_tool_pos + len(self.TOOL_CALL_END)
                    self.prev_tool_call_arr[self.current_tool_id]["finished"] = True

                    final_args = self._cast_arguments(
                        function_name,
                        tool_call_arr.get("parameters", {}),  # type: ignore
                        request,
                    )
                    if final_args:
                        final_args_json = json.dumps(final_args, ensure_ascii=False)
                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=final_args_json
                                    ),
                                )
                            ]
                        )

                # If tool is not finished, return None to wait for more tokens.
                return None

            return None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.TOOL_CALLS_BEGIN not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        pre_text, rest = model_output.split(self.TOOL_CALLS_BEGIN, 1)
        if self.TOOL_CALLS_END not in rest:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_block, post_text = rest.split(self.TOOL_CALLS_END, 1)
        content = (pre_text + post_text).strip()

        tool_calls: list[ToolCall] = []
        call_parts = tool_block.split(self.TOOL_CALL_BEGIN)

        for part in call_parts:
            if not part or self.TOOL_CALL_END not in part:
                continue

            call_content = part.split(self.TOOL_CALL_END, 1)[0]
            if self.TOOL_SEP not in call_content:
                continue

            type_part, invoke_part = call_content.split(self.TOOL_SEP, 1)
            if type_part.strip() != "function":
                continue

            function_name, params_dict = self._parse_steptml_invoke(invoke_part)

            if function_name and params_dict is not None:
                params_dict = self._cast_arguments(function_name, params_dict, request)
                params_str = json.dumps(params_dict, ensure_ascii=False)
                tool_calls.append(
                    ToolCall(
                        function=FunctionCall(name=function_name, arguments=params_str)
                    )
                )
        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )
