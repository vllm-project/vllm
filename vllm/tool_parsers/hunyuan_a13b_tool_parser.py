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
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import consume_space
from vllm.utils import random_uuid

logger = init_logger(__name__)


class HunyuanA13BToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        # Initialize state for streaming mode
        self.prev_tool_calls: list[dict] = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args: list[str] = []  # Track arguments sent for each tool

        # For backward compatibility with tests
        self.current_tools_sent: list[bool] = []

        # For backward compatibility with serving code
        self.prev_tool_call_arr: list[dict[str, Any]] = []

        # Regex patterns for preprocessing
        self.answer_tool_calls_pattern = re.compile(
            r"<tool_calls>([\s\S]*?)</tool_calls>", re.DOTALL
        )

        self.tool_name_reg = re.compile(r'"name"\s*:\s*"([^"]+)"')

        self.tool_empty_arg_reg = re.compile(
            r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{\s*\}'
        )

        # Note: historically we used a regex for nested JSON. Streaming output
        # is incremental, so regex brace matching can fail for deeply nested
        # objects; we now use a brace-depth scanner instead.
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
            current_text, current_idx, tool_count, delta_text
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
                    # Stateful JSON object parsing for streamed `arguments`.
                    # We only scan the new `delta_text` each step to avoid
                    # O(N^2) behavior from repeatedly re-scanning the full
                    # `current_text`.
                    "brace_depth": 0,
                    "in_string": False,
                    "escape": False,
                    # While we haven't seen the opening `{` for the
                    # `arguments` object yet, we keep a small suffix buffer
                    # to detect the `"arguments": {` boundary.
                    "suffix_buffer": "",
                    # Whether we've already emitted the closing `}` for
                    # this tool's `arguments` object.
                    "args_completed": False,
                }
            )
        while len(self.streaming_state["tool_ids"]) < tool_count:
            self.streaming_state["tool_ids"].append(None)

    def _scan_json_object_from_state(
        self,
        text: str,
        tool_state: dict[str, Any],
    ) -> tuple[str, bool, int]:
        """Scan `text` while tracking brace-depth/string state.

        Returns `(emitted, closed, consumed_len)` where `emitted` contains
        all characters up to and including the first matching `}` that
        returns `brace_depth` to 0 (if `closed` is True). If the object isn't
        complete, `closed` is False and `consumed_len == len(text)`.
        """
        emitted_chars: list[str] = []
        for i, ch in enumerate(text):
            emitted_chars.append(ch)
            if tool_state["in_string"]:
                if tool_state["escape"]:
                    tool_state["escape"] = False
                elif ch == "\\":
                    tool_state["escape"] = True
                elif ch == '"':
                    tool_state["in_string"] = False
                continue

            if ch == '"':
                tool_state["in_string"] = True
            elif ch == "{":
                tool_state["brace_depth"] += 1
            elif ch == "}":
                tool_state["brace_depth"] -= 1
                if tool_state["brace_depth"] == 0:
                    emitted = "".join(emitted_chars)
                    return emitted, True, i + 1

        return "".join(emitted_chars), False, len(text)

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

    def _scan_braced_json_object(self, text: str, start_brace: int) -> str | None:
        """Extract a JSON object starting at `start_brace` (which must be '{').

        In streaming mode the object may be incomplete; in that case we return
        the partial text from `start_brace` to the end of `text`.
        """
        assert start_brace < len(text) and text[start_brace] == "{"

        depth = 0
        in_string = False
        escape = False

        for i in range(start_brace, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start_brace : i + 1]

        # Streaming/incomplete JSON: return whatever we have so far.
        return text[start_brace:] if depth > 0 else None

    def _extract_tool_arguments_text(
        self, current_text: str, tool_index: int
    ) -> str | None:
        """Extract the `arguments` JSON object for a given tool index."""
        tool_calls_start = current_text.find(self.bot_string)
        search_start = tool_calls_start if tool_calls_start != -1 else 0

        arguments_keys = list(
            re.finditer(r'"arguments"\s*:', current_text[search_start:])
        )
        if tool_index >= len(arguments_keys):
            return None

        match = arguments_keys[tool_index]
        after_colon = search_start + match.end()
        brace_pos = current_text.find("{", after_colon)
        if brace_pos == -1:
            return None
        return self._scan_braced_json_object(current_text, brace_pos)

    def _handle_tool_args_streaming(
        self,
        current_text: str,
        current_idx: int,
        tool_count: int,
        delta_text: str,
    ) -> DeltaMessage | None:
        if current_idx < 0 or current_idx >= tool_count:
            return None

        CAP = 512
        tool_state: dict[str, Any] = self.streaming_state["sent_tools"][current_idx]

        if tool_state.get("args_completed", False):
            return None

        empty_args_match = self.tool_empty_arg_reg.search(current_text)
        if empty_args_match and empty_args_match.start() > 0:
            if not tool_state["sent_arguments_prefix"]:
                tool_state["sent_arguments_prefix"] = True
                tool_state["args_completed"] = True
                tool_state["sent_arguments"] = "{}"
                while len(self.streamed_args) <= current_idx:
                    self.streamed_args.append("")
                self.streamed_args[current_idx] += "{}"
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=current_idx,
                            function=DeltaFunctionCall(arguments="{}").model_dump(
                                exclude_none=True
                            ),
                        )
                    ]
                )
                if current_idx < tool_count - 1:
                    self.streaming_state["current_tool_index"] += 1
                    self.current_tool_id = self.streaming_state["current_tool_index"]
                return delta

        # Phase 1: find the beginning of the `"arguments": { ... }` object.
        if not tool_state["sent_arguments_prefix"]:
            old_buf: str = tool_state.get("suffix_buffer", "")
            prev_len = len(old_buf)
            new_buf = old_buf + delta_text
            if len(new_buf) > CAP:
                trim = len(new_buf) - CAP
                new_buf = new_buf[trim:]
                prev_len = max(0, prev_len - trim)

            tool_state["suffix_buffer"] = new_buf

            match = re.search(r'"arguments"\s*:\s*{', new_buf)
            if not match:
                return None

            open_brace_pos = match.end() - 1  # index of '{'
            # Ensure the `{` is part of the newly appended buffer; otherwise
            # we might emit already-seen characters.
            if open_brace_pos < prev_len:
                return None

            tool_state["sent_arguments_prefix"] = True
            tool_state["sent_arguments"] = ""
            tool_state["brace_depth"] = 0
            tool_state["in_string"] = False
            tool_state["escape"] = False
            tool_state["args_completed"] = False

            scan_text = new_buf[open_brace_pos:]
            emitted, closed, consumed_len = self._scan_json_object_from_state(
                scan_text, tool_state
            )

            # Clear suffix buffer once we start parsing.
            tool_state["suffix_buffer"] = ""

            if not emitted:
                return None

            while len(self.streamed_args) <= current_idx:
                self.streamed_args.append("")
            tool_state["sent_arguments"] = emitted
            self.streamed_args[current_idx] = emitted

            delta = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=current_idx,
                        function=DeltaFunctionCall(arguments=emitted).model_dump(
                            exclude_none=True
                        ),
                    )
                ]
            )

            if closed and current_idx < tool_count - 1:
                tail = scan_text[consumed_len:]
                self.streaming_state["current_tool_index"] += 1
                self.current_tool_id = self.streaming_state["current_tool_index"]
                next_state = self.streaming_state["sent_tools"][current_idx + 1]
                if not next_state["sent_arguments_prefix"] and tail:
                    next_state["suffix_buffer"] = (
                        next_state.get("suffix_buffer", "") + tail
                    )[-CAP:]
            elif closed:
                tool_state["args_completed"] = True

            return delta

        # Phase 2: we are already inside the `arguments` JSON object; scan only
        # the incremental delta to extend the output until the matching `}`.
        emitted, closed, consumed_len = self._scan_json_object_from_state(
            delta_text, tool_state
        )
        if not emitted:
            return None

        sent_args = tool_state["sent_arguments"]
        tool_state["sent_arguments"] = sent_args + emitted
        while len(self.streamed_args) <= current_idx:
            self.streamed_args.append("")
        self.streamed_args[current_idx] += emitted

        delta = DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=current_idx,
                    function=DeltaFunctionCall(arguments=emitted).model_dump(
                        exclude_none=True
                    ),
                )
            ]
        )

        if closed and current_idx < tool_count - 1:
            tail = delta_text[consumed_len:]
            self.streaming_state["current_tool_index"] += 1
            self.current_tool_id = self.streaming_state["current_tool_index"]
            next_state = self.streaming_state["sent_tools"][current_idx + 1]
            if not next_state["sent_arguments_prefix"] and tail:
                next_state["suffix_buffer"] = (
                    next_state.get("suffix_buffer", "") + tail
                )[-CAP:]
        elif closed:
            tool_state["args_completed"] = True

        return delta
