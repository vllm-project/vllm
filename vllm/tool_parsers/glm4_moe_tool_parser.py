# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GLM-4 Tool Call Parser with incremental string streaming support.

This parser fixes the streaming issue reported in Issue #32829 where long string
parameters (e.g., file content with 4000+ characters of code) are buffered until
complete, causing multi-second delays before the user sees any content.

The fix streams string values incrementally as they arrive, providing a true
streaming experience for long content.
"""

import ast
import json
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
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

logger = init_logger(__name__)


class Glm4MoeModelToolParser(ToolParser):
    """Tool parser for GLM-4 models with incremental string streaming.

    This parser emits tool-call deltas incrementally as arguments arrive.
    For string-type parameters, content is streamed character-by-character
    rather than waiting for the complete </arg_value> tag.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        # Stateful streaming fields
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.arg_key_start: str = "<arg_key>"
        self.arg_key_end: str = "</arg_key>"
        self.arg_val_start: str = "<arg_value>"
        self.arg_val_end: str = "</arg_value>"

        self.tool_calls_start_token = self.tool_call_start_token

        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>([^\n]*)\n(.*)</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self._buffer: str = ""

        # Streaming state for incremental tool-call streaming
        self._in_tool_call: bool = False
        self._current_tool_name: str | None = None
        self._pending_key: str | None = None
        self._streaming_string_value: bool = False
        self._tool_call_ids: list[str] = []
        self._args_started: list[bool] = []
        self._args_closed: list[bool] = []
        self._seen_keys: list[set[str]] = []

    @staticmethod
    def _deserialize(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        return value

    @staticmethod
    def _json_escape_string_content(s: str) -> str:
        """JSON-escape string content for incremental streaming.

        This escapes the content that goes INSIDE a JSON string (between quotes),
        not including the surrounding quotes themselves.
        """
        if not s:
            return ""
        return json.dumps(s, ensure_ascii=False)[1:-1]

    @staticmethod
    def _is_string_type(
        tool_name: str,
        arg_name: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> bool:
        if tools is None:
            return False
        for tool in tools:
            if tool.function.name != tool_name:
                continue
            if tool.function.parameters is None:
                return False
            arg_type = (
                tool.function.parameters.get("properties", {})
                .get(arg_name, {})
                .get("type", None)
            )
            return arg_type == "string"
        logger.debug("No tool named '%s'.", tool_name)
        return False

    @staticmethod
    def _tools_enabled(request: ChatCompletionRequest) -> bool:
        """Return whether tool parsing should be applied for this request."""
        try:
            tools = getattr(request, "tools", None)
            tool_choice = getattr(request, "tool_choice", None)
            return bool(tools) and tool_choice != "none"
        except Exception:
            logger.exception("Failed to determine if tools are enabled.")
            return False

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust request parameters for tool call token handling."""
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Ensure tool call tokens (<tool_call>, </tool_call>) are not skipped
            # during decoding. Even though they are not marked as special tokens,
            # setting skip_special_tokens=False ensures proper handling in
            # transformers 5.x where decoding behavior may have changed.
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        matched_tool_calls = self.func_call_regex.findall(model_output)
        logger.debug("model_output: %s", model_output)
        try:
            tool_calls: list[ToolCall] = []
            for match in matched_tool_calls:
                tc_detail = self.func_detail_regex.search(match)
                if not tc_detail:
                    logger.warning(
                        "Failed to parse tool call details from: %s",
                        match,
                    )
                    continue
                tc_name = tc_detail.group(1).strip()
                tc_args = tc_detail.group(2)
                pairs = self.func_arg_regex.findall(tc_args) if tc_args else []
                arg_dct: dict[str, Any] = {}
                for key, value in pairs:
                    arg_key = key.strip()
                    arg_val = value.strip()
                    if not self._is_string_type(tc_name, arg_key, request.tools):
                        arg_val = self._deserialize(arg_val)
                    logger.debug("arg_key = %s, arg_val = %s", arg_key, arg_val)
                    arg_dct[arg_key] = arg_val
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tc_name,
                            arguments=json.dumps(arg_dct, ensure_ascii=False),
                        ),
                    )
                )
        except Exception:
            logger.exception("Failed to extract tool call spec")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        else:
            if len(tool_calls) > 0:
                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=tool_calls, content=content
                )
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
        if not self._tools_enabled(request):
            return DeltaMessage(content=delta_text) if delta_text else None

        self._buffer += delta_text

        while True:
            if not self._in_tool_call:
                start_idx = self._buffer.find(self.tool_call_start_token)
                if start_idx == -1:
                    # Check for partial start token at end of buffer
                    for i in range(1, len(self.tool_call_start_token)):
                        if self._buffer.endswith(self.tool_call_start_token[:i]):
                            out = self._buffer[:-i]
                            self._buffer = self._buffer[-i:]
                            return DeltaMessage(content=out) if out else None
                    out = self._buffer
                    self._buffer = ""
                    return DeltaMessage(content=out) if out else None

                if start_idx > 0:
                    out = self._buffer[:start_idx]
                    self._buffer = self._buffer[start_idx:]
                    return DeltaMessage(content=out) if out else None

                self._buffer = self._buffer[len(self.tool_call_start_token) :]
                self._begin_tool_call()
                continue

            # Parse tool name first
            if not self.current_tool_name_sent:
                nl = self._buffer.find("\n")
                ak = self._buffer.find(self.arg_key_start)
                end = self._buffer.find(self.tool_call_end_token)
                candidates = [i for i in [nl, ak, end] if i != -1]
                if not candidates:
                    return None
                cut = min(candidates)
                tool_name = self._buffer[:cut].strip()
                if tool_name == "" and cut == end:
                    # Handle empty tool call like `<tool_call></tool_call>`.
                    # Consume the tokens and reset state to avoid infinite loop.
                    self._buffer = self._buffer[end + len(self.tool_call_end_token) :]
                    self._finish_tool_call()
                    self._revert_last_tool_call_state()
                    continue

                if cut == nl:
                    self._buffer = self._buffer[nl + 1 :]
                else:
                    self._buffer = self._buffer[cut:]

                self._current_tool_name = tool_name
                self.current_tool_name_sent = True
                return self._emit_tool_name_delta(tool_name)

            assert self._current_tool_name is not None

            # Handle incremental string value streaming
            if self._streaming_string_value:
                val_end = self._buffer.find(self.arg_val_end)
                if val_end != -1:
                    raw_content = self._buffer[:val_end]
                    self._buffer = self._buffer[val_end + len(self.arg_val_end) :]
                    self._streaming_string_value = False
                    self._pending_key = None

                    escaped = self._json_escape_string_content(raw_content)
                    frag = escaped + '"'
                    self.streamed_args_for_tool[self.current_tool_id] += frag
                    return self._emit_tool_args_delta(frag)
                else:
                    # Check for partial </arg_value> at end
                    safe_len = len(self._buffer)
                    for i in range(1, len(self.arg_val_end)):
                        if self._buffer.endswith(self.arg_val_end[:i]):
                            safe_len = len(self._buffer) - i
                            break

                    if safe_len > 0:
                        to_emit = self._buffer[:safe_len]
                        self._buffer = self._buffer[safe_len:]
                        escaped = self._json_escape_string_content(to_emit)
                        if escaped:
                            self.streamed_args_for_tool[self.current_tool_id] += escaped
                            return self._emit_tool_args_delta(escaped)
                    return None

            # If we have a pending key, parse its value
            if self._pending_key is not None:
                val_pos = self._buffer.find(self.arg_val_start)
                if val_pos == -1:
                    return None
                if val_pos > 0:
                    self._buffer = self._buffer[val_pos:]

                key = (self._pending_key or "").strip()

                is_string = self._is_string_type(
                    self._current_tool_name, key, request.tools
                )

                if is_string:
                    # String type: stream incrementally
                    self._buffer = self._buffer[len(self.arg_val_start) :]

                    if key in self._seen_keys[self.current_tool_id]:
                        self._pending_key = None
                        continue

                    self._seen_keys[self.current_tool_id].add(key)
                    key_json = json.dumps(key, ensure_ascii=False)

                    if not self._args_started[self.current_tool_id]:
                        frag = "{" + key_json + ':"'
                        self._args_started[self.current_tool_id] = True
                    else:
                        frag = "," + key_json + ':"'

                    self.streamed_args_for_tool[self.current_tool_id] += frag
                    self._streaming_string_value = True
                    return self._emit_tool_args_delta(frag)
                else:
                    # Non-string type: wait for complete value
                    val_end = self._buffer.find(self.arg_val_end)
                    if val_end == -1:
                        return None

                    raw_val = self._buffer[len(self.arg_val_start) : val_end].strip()
                    self._buffer = self._buffer[val_end + len(self.arg_val_end) :]
                    self._pending_key = None

                    frag = self._append_arg_fragment(
                        key=key,
                        raw_val=raw_val,
                    )
                    if frag:
                        return self._emit_tool_args_delta(frag)
                    continue

            # Parse next arg or close
            end_pos = self._buffer.find(self.tool_call_end_token)
            key_pos = self._buffer.find(self.arg_key_start)
            if end_pos != -1 and (key_pos == -1 or end_pos < key_pos):
                self._buffer = self._buffer[end_pos + len(self.tool_call_end_token) :]
                frag = self._close_args_if_needed()
                # Finalize prev_tool_call_arr with complete parsed arguments
                if self._current_tool_name:
                    try:
                        full_args_str = self.streamed_args_for_tool[
                            self.current_tool_id
                        ]
                        args_dict = json.loads(full_args_str)
                        self.prev_tool_call_arr[self.current_tool_id] = {
                            "name": self._current_tool_name,
                            "arguments": args_dict,
                        }
                    except (json.JSONDecodeError, IndexError) as e:
                        logger.warning(
                            "Failed to finalize tool call state for tool %d: %s",
                            self.current_tool_id,
                            e,
                        )
                self._finish_tool_call()
                return self._emit_tool_args_delta(frag) if frag else None

            if key_pos == -1:
                return None
            if key_pos > 0:
                self._buffer = self._buffer[key_pos:]
            key_end = self._buffer.find(self.arg_key_end)
            if key_end == -1:
                return None
            key = self._buffer[len(self.arg_key_start) : key_end]
            self._buffer = self._buffer[key_end + len(self.arg_key_end) :]
            self._pending_key = key
            continue

    def _ensure_tool_state(self) -> None:
        while len(self._tool_call_ids) <= self.current_tool_id:
            self._tool_call_ids.append(
                make_tool_call_id(id_type="random", func_name=None, idx=None)
            )
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self._args_started) <= self.current_tool_id:
            self._args_started.append(False)
        while len(self._args_closed) <= self.current_tool_id:
            self._args_closed.append(False)
        while len(self._seen_keys) <= self.current_tool_id:
            self._seen_keys.append(set())

    def _begin_tool_call(self) -> None:
        if self.current_tool_id == -1:
            self.current_tool_id = 0
        else:
            self.current_tool_id += 1
        self._ensure_tool_state()
        self.current_tool_name_sent = False
        self._current_tool_name = None
        self._pending_key = None
        self._streaming_string_value = False
        self._in_tool_call = True

    def _finish_tool_call(self) -> None:
        self._in_tool_call = False
        self._current_tool_name = None
        self._pending_key = None
        self._streaming_string_value = False

    def _revert_last_tool_call_state(self) -> None:
        """Revert the state allocation for the last tool call."""
        if self.current_tool_id < 0:
            return
        self._tool_call_ids.pop()
        self.streamed_args_for_tool.pop()
        self.prev_tool_call_arr.pop()
        self._args_started.pop()
        self._args_closed.pop()
        self._seen_keys.pop()
        self.current_tool_id -= 1

    def _emit_tool_name_delta(self, tool_name: str) -> DeltaMessage:
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    id=self._tool_call_ids[self.current_tool_id],
                    type="function",
                    function=DeltaFunctionCall(
                        name=tool_name,
                        arguments="",
                    ).model_dump(exclude_none=True),
                )
            ]
        )

    def _emit_tool_args_delta(self, fragment: str) -> DeltaMessage:
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    function=DeltaFunctionCall(arguments=fragment).model_dump(
                        exclude_none=True
                    ),
                )
            ]
        )

    def _append_arg_fragment(
        self,
        *,
        key: str,
        raw_val: str,
    ) -> str | None:
        key = key.strip()
        if not key:
            return None
        if key in self._seen_keys[self.current_tool_id]:
            return None

        # This function is only called for non-string types (already checked
        # by _is_string_type in the caller), so we always deserialize.
        val_obj: Any = self._deserialize(raw_val)

        key_json = json.dumps(key, ensure_ascii=False)
        val_json = json.dumps(val_obj, ensure_ascii=False)

        if not self._args_started[self.current_tool_id]:
            fragment = "{" + key_json + ":" + val_json
            self._args_started[self.current_tool_id] = True
        else:
            fragment = "," + key_json + ":" + val_json

        self._seen_keys[self.current_tool_id].add(key)
        self.streamed_args_for_tool[self.current_tool_id] += fragment
        return fragment

    def _close_args_if_needed(self) -> str | None:
        if self._args_closed[self.current_tool_id]:
            return None
        self._args_closed[self.current_tool_id] = True
        if not self._args_started[self.current_tool_id]:
            fragment = "{}"
            self.streamed_args_for_tool[self.current_tool_id] = fragment
        else:
            fragment = "}"
            self.streamed_args_for_tool[self.current_tool_id] += fragment
        return fragment
