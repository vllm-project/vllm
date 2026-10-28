# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)

_ESCAPE = "<escape>"
_ESCAPE_LEN = len(_ESCAPE)
_CALL_END = "<end_function_call>"


def _json_scalar(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _stable_argument_json(args_str: str, *, finished: bool) -> str:
    """Serialize a call body so each snapshot is a prefix of a later one.

    ``json.dumps`` closes every container. An array or object that is still
    open in the source then cannot grow: the next snapshot is not a suffix
    of the text already sent. Leave those closers unwritten until the source
    closes them. A bare value at the end of an unfinished chunk is omitted,
    because another digit may still arrive.
    """
    if finished:
        return _json_scalar(_parse_functiongemma_fields(args_str, partial=False))
    if not args_str or not args_str.strip():
        return ""
    body, _index, closed = _emit_object_body(args_str, 0)
    if not body and not closed:
        return ""
    if closed:
        return "{" + body + "}"
    return "{" + body


def _emit_object_body(src: str, index: int) -> tuple[str, int, bool]:
    fields: list[str] = []
    length = len(src)
    closed = False
    while index < length:
        while index < length and src[index] in " ,\n\t":
            index += 1
        if index >= length or src.startswith(_CALL_END, index):
            break
        if src[index] == "}":
            index += 1
            closed = True
            break
        if src[index] == "]":
            break
        key_start = index
        while index < length and src[index] != ":":
            index += 1
        if index >= length:
            break
        key = src[key_start:index].strip()
        index += 1
        if not key:
            break
        while index < length and src[index] in " \n\t":
            index += 1
        if index >= length:
            break
        value, index, value_closed = _emit_value(src, index)
        if not value_closed and not value:
            break
        fields.append(_json_scalar(key) + ": " + value)
        if not value_closed:
            break
    return ", ".join(fields), index, closed


def _emit_value(src: str, index: int) -> tuple[str, int, bool]:
    if src.startswith(_ESCAPE, index):
        value_start = index + _ESCAPE_LEN
        end_pos = src.find(_ESCAPE, value_start)
        if end_pos == -1:
            return "", index, False
        content = src[value_start:end_pos]
        return _json_scalar(content), end_pos + _ESCAPE_LEN, True
    if src[index] == "{":
        body, index, closed = _emit_object_body(src, index + 1)
        if closed:
            return "{" + body + "}", index, True
        return "{" + body, index, False
    if src[index] == "[":
        return _emit_array(src, index)
    value_start = index
    length = len(src)
    while index < length and src[index] not in ",}]":
        index += 1
    if index >= length:
        return "", value_start, False
    raw = src[value_start:index].strip()
    if raw == "":
        return "", value_start, False
    return _json_scalar(_coerce_bare(raw)), index, True


def _emit_array(src: str, index: int) -> tuple[str, int, bool]:
    index += 1
    items: list[str] = []
    length = len(src)
    closed = False
    while index < length:
        while index < length and src[index] in " ,\n\t":
            index += 1
        if index >= length:
            break
        if src[index] == "]":
            index += 1
            closed = True
            break
        value, index, value_closed = _emit_value(src, index)
        if not value_closed and not value:
            break
        items.append(value)
        if not value_closed:
            break
    body = ", ".join(items)
    if closed:
        return "[" + body + "]", index, True
    return "[" + body, index, False


def _coerce_bare(raw: str):
    """JSON-decode a bare FunctionGemma scalar. Unquoted text stays a string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_functiongemma_fields(args_str: str, *, partial: bool = False) -> dict:
    """Parse ``key:value`` pairs from a FunctionGemma call body.

    ``<escape>...<escape>`` is always a string, including text that looks
    like JSON. Bare ``true`` / ``false`` / ``null`` / numbers keep JSON
    types. Objects and arrays nest. A trailing ``}`` or
    ``<end_function_call>`` ends the object so a streaming slice can
    include the call closer.
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        while i < n and args_str[i] in " ,\n\t":
            i += 1
        if i >= n or args_str[i] in "}]" or args_str.startswith(_CALL_END, i):
            break

        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        if not key:
            break
        i += 1
        while i < n and args_str[i] in " \n\t":
            i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break

        if args_str.startswith(_ESCAPE, i):
            i += _ESCAPE_LEN
            end_pos = args_str.find(_ESCAPE, i)
            if end_pos == -1:
                if not partial:
                    result[key] = args_str[i:]
                break
            result[key] = args_str[i:end_pos]
            i = end_pos + _ESCAPE_LEN
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str.startswith(_ESCAPE, i):
                    i += _ESCAPE_LEN
                    nxt = args_str.find(_ESCAPE, i)
                    i = n if nxt == -1 else nxt + _ESCAPE_LEN
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            inner = args_str[obj_start:] if depth > 0 else args_str[obj_start : i - 1]
            result[key] = _parse_functiongemma_fields(inner, partial=depth > 0)
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str.startswith(_ESCAPE, i):
                    i += _ESCAPE_LEN
                    nxt = args_str.find(_ESCAPE, i)
                    i = n if nxt == -1 else nxt + _ESCAPE_LEN
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            inner = args_str[arr_start:] if depth > 0 else args_str[arr_start : i - 1]
            result[key] = _parse_functiongemma_array(inner, partial=depth > 0)
        else:
            val_start = i
            while i < n and args_str[i] not in ",}]":
                i += 1
            if partial and i >= n:
                break
            raw = args_str[val_start:i].strip()
            if raw == "":
                break
            result[key] = _coerce_bare(raw)

    return result


def _parse_functiongemma_array(arr_str: str, *, partial: bool = False) -> list:
    items: list = []
    i = 0
    n = len(arr_str)
    while i < n:
        while i < n and arr_str[i] in " ,\n\t":
            i += 1
        if i >= n or arr_str[i] == "]":
            break
        if arr_str.startswith(_ESCAPE, i):
            i += _ESCAPE_LEN
            end_pos = arr_str.find(_ESCAPE, i)
            if end_pos == -1:
                if not partial:
                    items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + _ESCAPE_LEN
        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str.startswith(_ESCAPE, i):
                    i += _ESCAPE_LEN
                    nxt = arr_str.find(_ESCAPE, i)
                    i = n if nxt == -1 else nxt + _ESCAPE_LEN
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            inner = arr_str[obj_start:] if depth > 0 else arr_str[obj_start : i - 1]
            items.append(_parse_functiongemma_fields(inner, partial=depth > 0))
        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str.startswith(_ESCAPE, i):
                    i += _ESCAPE_LEN
                    nxt = arr_str.find(_ESCAPE, i)
                    i = n if nxt == -1 else nxt + _ESCAPE_LEN
                    continue
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            inner = arr_str[sub_start:] if depth > 0 else arr_str[sub_start : i - 1]
            items.append(_parse_functiongemma_array(inner, partial=depth > 0))
        else:
            val_start = i
            while i < n and arr_str[i] not in ",]":
                i += 1
            if partial and i >= n:
                break
            raw = arr_str[val_start:i].strip()
            if raw == "":
                break
            items.append(_coerce_bare(raw))
    return items


class FunctionGemmaToolParser(ToolParser):
    """Tool parser for Google's FunctionGemma model (google/functiongemma-270m-it).

    Handles the FunctionGemma function call format:
    <start_function_call>call:func_name{key:<escape>text<escape>,n:1}<end_function_call>

    ``<escape>`` values are strings. Numbers, booleans, null, arrays, and
    objects are written bare, matching the model chat template.
    """

    # FunctionGemma tokens
    tool_call_start_token: str = "<start_function_call>"
    tool_call_end_token: str = "<end_function_call>"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        # Streaming state
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []
        self.buffered_delta_text = ""

    def _parse_arguments(self, args_str: str, *, partial: bool = False) -> dict:
        """Parse a FunctionGemma argument body into a dictionary."""
        return _parse_functiongemma_fields(args_str, partial=partial)

    def _iter_calls(self, text: str):
        """Yield ``(name, args_body, closed)`` for each call in ``text``.

        Brace matching skips ``<escape>`` strings, so a nested object does
        not end the call. An unclosed call yields the text so far and stops.
        """
        token = self.tool_call_start_token
        pos = 0
        length = len(text)
        while pos < length:
            start = text.find(token, pos)
            if start < 0:
                return
            cursor = start + len(token)
            if not text.startswith("call:", cursor):
                pos = cursor
                continue
            cursor += len("call:")
            name_end = cursor
            while name_end < length and (
                text[name_end].isalnum() or text[name_end] == "_"
            ):
                name_end += 1
            if name_end == cursor or name_end >= length or text[name_end] != "{":
                pos = start + len(token)
                continue
            name = text[cursor:name_end]
            args_start = name_end + 1
            depth = 1
            index = args_start
            while index < length and depth:
                if text.startswith(_ESCAPE, index):
                    index += _ESCAPE_LEN
                    end = text.find(_ESCAPE, index)
                    if end < 0:
                        index = length
                        break
                    index = end + _ESCAPE_LEN
                    continue
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                index += 1
            if depth != 0:
                yield name, text[args_start:], False
                return
            yield name, text[args_start : index - 1], True
            if text.startswith(self.tool_call_end_token, index):
                pos = index + len(self.tool_call_end_token)
            else:
                pos = index

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            matches = list(self._iter_calls(model_output))

            if not matches:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls: list[ToolCall] = []

            for func_name, args_str, _closed in matches:
                if not func_name:
                    continue

                arguments = self._parse_arguments(args_str)

                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=func_name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        ),
                    )
                )

            if tool_calls:
                content_end = model_output.find(self.tool_call_start_token)
                content = (
                    model_output[:content_end].strip() if content_end > 0 else None
                )

                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        except Exception:
            logger.exception("Error extracting tool calls from FunctionGemma response")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _buffer_delta_text(self, delta_text: str) -> str:
        """Buffer incoming delta text to handle multi-token special sequences."""
        potential_start = "<start_function_call>"
        potential_end = "<end_function_call>"

        combined = self.buffered_delta_text + delta_text

        if combined.endswith(potential_start) or combined.endswith(potential_end):
            self.buffered_delta_text = ""
            return combined

        for tag in [potential_start, potential_end]:
            for i in range(1, len(tag)):
                if combined.endswith(tag[:i]):
                    self.buffered_delta_text = combined[-(i):]
                    return combined[:-i]

        self.buffered_delta_text = ""
        return combined

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
        current_text = previous_text + delta_text

        if self.tool_call_start_token not in current_text:
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        try:
            start_count = current_text.count(self.tool_call_start_token)
            end_count = current_text.count(self.tool_call_end_token)
            prev_start_count = previous_text.count(self.tool_call_start_token)
            prev_end_count = previous_text.count(self.tool_call_end_token)

            if self.tool_call_start_token not in current_text:
                return DeltaMessage(content=delta_text)

            # Starting a new function call
            if start_count > prev_start_count and start_count > end_count:
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr.append({})
                logger.debug("Starting new tool call %d", self.current_tool_id)
                return None

            # In the middle of a function call
            if start_count > end_count:
                last_start = current_text.rfind(self.tool_call_start_token)
                partial_call = current_text[
                    last_start + len(self.tool_call_start_token) :
                ]

                if partial_call.startswith("call:"):
                    func_part = partial_call[5:]

                    if "{" in func_part:
                        func_name = func_part.split("{")[0]
                        args_part = (
                            func_part.split("{", 1)[1] if "{" in func_part else ""
                        )

                        if not self.current_tool_name_sent and func_name:
                            self.current_tool_name_sent = True
                            self.prev_tool_call_arr[self.current_tool_id] = {
                                "name": func_name,
                                "arguments": {},
                            }
                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        type="function",
                                        id=make_tool_call_id(),
                                        function=DeltaFunctionCall(
                                            name=func_name
                                        ).model_dump(exclude_none=True),
                                    )
                                ]
                            )

                        if self.current_tool_name_sent and args_part:
                            current_args = self._parse_arguments(
                                args_part, partial=True
                            )
                            if current_args or args_part.strip():
                                current_args_json = _stable_argument_json(
                                    args_part, finished=False
                                )
                                prev_streamed = self.streamed_args_for_tool[
                                    self.current_tool_id
                                ]

                                if current_args_json.startswith(prev_streamed) and len(
                                    current_args_json
                                ) > len(prev_streamed):
                                    diff = current_args_json[len(prev_streamed) :]
                                    self.streamed_args_for_tool[
                                        self.current_tool_id
                                    ] = current_args_json
                                    self.prev_tool_call_arr[self.current_tool_id][
                                        "arguments"
                                    ] = current_args

                                    return DeltaMessage(
                                        tool_calls=[
                                            DeltaToolCall(
                                                index=self.current_tool_id,
                                                function=DeltaFunctionCall(
                                                    arguments=diff
                                                ).model_dump(exclude_none=True),
                                            )
                                        ]
                                    )

                return None

            # Function call just ended
            if end_count > prev_end_count:
                if self.current_tool_id >= 0 and self.current_tool_id < len(
                    self.prev_tool_call_arr
                ):
                    all_calls = list(self._iter_calls(current_text))
                    args = {}
                    args_str = ""
                    if self.current_tool_id < len(all_calls):
                        _func_name, args_str, _closed = all_calls[self.current_tool_id]
                        args = self._parse_arguments(args_str)
                        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = (
                            args
                        )

                    if args or args_str:
                        args_json = _stable_argument_json(args_str, finished=True)
                        prev_streamed = self.streamed_args_for_tool[
                            self.current_tool_id
                        ]
                        if args_json.startswith(prev_streamed) and len(args_json) > len(
                            prev_streamed
                        ):
                            diff = args_json[len(prev_streamed) :]
                            self.streamed_args_for_tool[self.current_tool_id] = (
                                args_json
                            )
                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        function=DeltaFunctionCall(
                                            arguments=diff
                                        ).model_dump(exclude_none=True),
                                    )
                                ]
                            )
                return None

            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        except Exception:
            logger.exception("Error in streaming tool call extraction")
            return None
