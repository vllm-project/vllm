# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, SIM102

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
from vllm.tool_parsers.utils import consume_space
from vllm.utils import random_uuid

logger = init_logger(__name__)


class HunyuanA13BToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # Initialize state for streaming mode
        self.prev_tool_calls: list[dict] = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args: list[str] = []  # Track arguments sent for each tool

        # For backward compatibility with tests
        self.current_tools_sent: list[bool] = []

        # For backward compatibility with serving code
        self.prev_tool_call_arr = []

        # Regex patterns for preprocessing
        self.answer_tool_calls_pattern = re.compile(
            r"<tool_calls>([\s\S]*?)</tool_calls>", re.DOTALL
        )

        self.tool_name_reg = re.compile(r'"name"\s*:\s*"([^"]+)"')

        self.tool_empty_arg_reg = re.compile(
            r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{\s*\}'
        )

        # TODO: not support nested json object in fc arguments.
        self.tool_non_empty_arg_reg = re.compile(
            r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
        )

        self.bot_string = "<tool_calls>"

        # Define streaming state type to be initialized later
        self.streaming_state: dict[str, Any] = {
            "current_tool_index": -1,
            "tool_ids": [],
            "sent_tools": [],
        }

    def preprocess_model_output(
        self, model_output: str
    ) -> tuple[str | None, str | None]:
        # find the location tool call
        for match in self.answer_tool_calls_pattern.finditer(model_output):
            start, end = match.span()
            # check tool_calls whether in side of <think>
            think_regions = [
                (m.start(), m.end())
                for m in re.finditer(
                    r"<think>(.*?)</think>", model_output, flags=re.DOTALL
                )
            ]
            in_think = any(
                start > t_start and end < t_end for t_start, t_end in think_regions
            )
            if not in_think:
                content = model_output[:start]
                tool_calls_content = match.group(1).strip()
                try:
                    json.loads(tool_calls_content)
                    return content, tool_calls_content
                except Exception:
                    continue
        return model_output, None

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model output.
        """
        try:
            # Preprocess the model output
            content, potential_tool_calls = self.preprocess_model_output(model_output)

            if not potential_tool_calls:
                # some text should be filtered out for no function call
                # this text is in a13b's chat template.
                if content:
                    content = content.replace("助手：", "", 1)
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=content
                )

            # Parse the potential tool calls as JSON
            tool_calls_data = json.loads(potential_tool_calls)

            # Ensure it's an array
            if not isinstance(tool_calls_data, list):
                logger.debug("Tool calls data is not an array")
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=content or model_output,
                )

            tool_calls: list[ToolCall] = []

            for idx, call in enumerate(tool_calls_data):
                if (
                    not isinstance(call, dict)
                    or "name" not in call
                    or "arguments" not in call
                ):
                    continue

                tool_call = ToolCall(
                    id=f"call_{random_uuid()}",
                    type="function",
                    function=FunctionCall(
                        name=call["name"],
                        arguments=(
                            json.dumps(call["arguments"])
                            if isinstance(call["arguments"], dict)
                            else call["arguments"]
                        ),
                    ),
                )
                tool_calls.append(tool_call)

            if not content or len(content.strip()) == 0:
                # clear the whitespace content.
                content = None

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content,
            )

        except Exception:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
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
        """
        Extract tool calls for streaming mode.
        """

        start_idx = consume_space(0, current_text)
        if current_text[start_idx:].startswith(self.bot_string):
            start_idx = consume_space(start_idx + len(self.bot_string), current_text)
        if (
            not current_text
            or start_idx >= len(current_text)
            or current_text[start_idx] != "["
        ):
            return DeltaMessage(content=delta_text)

        self._try_parse_json_tools(current_text[start_idx:])

        test_delta = self._handle_test_compatibility(current_text)
        if test_delta:
            return test_delta

        name_matches = list(self.tool_name_reg.finditer(current_text))
        tool_count = len(name_matches)
        if tool_count == 0:
            return None
        self._ensure_state_arrays(tool_count)
        current_idx = self.streaming_state["current_tool_index"]

        name_delta = self._handle_tool_name_streaming(
            current_idx, tool_count, name_matches
        )
        if name_delta:
            return name_delta

        args_delta = self._handle_tool_args_streaming(
            current_text, current_idx, tool_count
        )
        if args_delta:
            return args_delta

        return None

    def _try_parse_json_tools(self, current_text: str):
        try:
            parsed_tools = json.loads(current_text)
            if isinstance(parsed_tools, list):
                self.prev_tool_call_arr = parsed_tools
        except json.JSONDecodeError:
            pass

    def _handle_test_compatibility(self, current_text: str):
        if len(self.current_tools_sent) > 0:
            if (
                len(self.current_tools_sent) == 1
                and self.current_tools_sent[0] is False
            ):
                name_match = self.tool_name_reg.search(current_text)
                if name_match:
                    function_name = name_match.group(1)
                    tool_id = f"chatcmpl-tool-{random_uuid()}"
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=0,
                                type="function",
                                id=tool_id,
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tools_sent = [True]
                    self.current_tool_id = 0
                    self.streaming_state["current_tool_index"] = 0
                    if len(self.streaming_state["sent_tools"]) == 0:
                        self.streaming_state["sent_tools"].append(
                            {
                                "sent_name": True,
                                "sent_arguments_prefix": False,
                                "sent_arguments": "",
                            }
                        )
                    else:
                        self.streaming_state["sent_tools"][0]["sent_name"] = True
                    self.current_tool_name_sent = True
                    return delta
        return None

    def _ensure_state_arrays(self, tool_count: int):
        while len(self.streaming_state["sent_tools"]) < tool_count:
            self.streaming_state["sent_tools"].append(
                {
                    "sent_name": False,
                    "sent_arguments_prefix": False,
                    "sent_arguments": "",
                }
            )
        while len(self.streaming_state["tool_ids"]) < tool_count:
            self.streaming_state["tool_ids"].append(None)

    def _handle_tool_name_streaming(
        self, current_idx: int, tool_count: int, name_matches
    ):
        if current_idx == -1 or current_idx < tool_count - 1:
            next_idx = current_idx + 1
            if (
                next_idx < tool_count
                and not self.streaming_state["sent_tools"][next_idx]["sent_name"]
            ):
                self.streaming_state["current_tool_index"] = next_idx
                self.current_tool_id = next_idx
                current_idx = next_idx
                tool_name = name_matches[current_idx].group(1)
                tool_id = f"call_{current_idx}_{random_uuid()}"
                self.streaming_state["tool_ids"][current_idx] = tool_id
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=current_idx,
                            type="function",
                            id=tool_id,
                            function=DeltaFunctionCall(name=tool_name).model_dump(
                                exclude_none=True
                            ),
                        )
                    ]
                )
                self.streaming_state["sent_tools"][current_idx]["sent_name"] = True
                self.current_tool_name_sent = True
                while len(self.streamed_args) <= current_idx:
                    self.streamed_args.append("")
                return delta
        return None

    def _handle_tool_args_streaming(
        self, current_text: str, current_idx: int, tool_count: int
    ):
        if current_idx >= 0 and current_idx < tool_count:
            empty_args_match = self.tool_empty_arg_reg.search(current_text)
            if empty_args_match and empty_args_match.start() > 0:
                for i in range(tool_count):
                    if i == current_idx:
                        if not self.streaming_state["sent_tools"][current_idx][
                            "sent_arguments_prefix"
                        ]:
                            self.streaming_state["sent_tools"][current_idx][
                                "sent_arguments_prefix"
                            ] = True
                            self.streaming_state["sent_tools"][current_idx][
                                "sent_arguments"
                            ] = "{}"
                            while len(self.streamed_args) <= current_idx:
                                self.streamed_args.append("")
                            self.streamed_args[current_idx] += "{}"
                            delta = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=current_idx,
                                        function=DeltaFunctionCall(
                                            arguments="{}"
                                        ).model_dump(exclude_none=True),
                                    )
                                ]
                            )
                            if current_idx < tool_count - 1:
                                self.streaming_state["current_tool_index"] += 1
                                self.current_tool_id = self.streaming_state[
                                    "current_tool_index"
                                ]
                            return delta

            args_matches = list(self.tool_non_empty_arg_reg.finditer(current_text))
            if current_idx < len(args_matches):
                args_text = args_matches[current_idx].group(1)
                is_last_tool = current_idx == tool_count - 1
                if not is_last_tool:
                    next_tool_pos = current_text.find(
                        "},{", args_matches[current_idx].start()
                    )
                    if next_tool_pos != -1:
                        args_end_pos = next_tool_pos + 1
                        args_text = (
                            current_text[
                                args_matches[current_idx].start() : args_end_pos
                            ]
                            .split('"arguments":')[1]
                            .strip()
                        )
                sent_args = self.streaming_state["sent_tools"][current_idx][
                    "sent_arguments"
                ]
                if not self.streaming_state["sent_tools"][current_idx][
                    "sent_arguments_prefix"
                ] and args_text.startswith("{"):
                    self.streaming_state["sent_tools"][current_idx][
                        "sent_arguments_prefix"
                    ] = True
                    self.streaming_state["sent_tools"][current_idx][
                        "sent_arguments"
                    ] = "{"
                    while len(self.streamed_args) <= current_idx:
                        self.streamed_args.append("")
                    self.streamed_args[current_idx] += "{"
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=current_idx,
                                function=DeltaFunctionCall(arguments="{").model_dump(
                                    exclude_none=True
                                ),
                            )
                        ]
                    )
                    return delta

                if args_text.startswith(sent_args):
                    args_diff = args_text[len(sent_args) :]
                    if args_diff:
                        self.streaming_state["sent_tools"][current_idx][
                            "sent_arguments"
                        ] = args_text
                        while len(self.streamed_args) <= current_idx:
                            self.streamed_args.append("")
                        self.streamed_args[current_idx] += args_diff
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=current_idx,
                                    function=DeltaFunctionCall(
                                        arguments=args_diff
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        return delta

                if args_text.endswith("}") and args_text == sent_args:
                    if current_idx < tool_count - 1:
                        self.streaming_state["current_tool_index"] += 1
                        self.current_tool_id = self.streaming_state[
                            "current_tool_index"
                        ]
        return None
