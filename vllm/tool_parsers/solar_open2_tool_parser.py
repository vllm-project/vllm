# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    find_tool_properties,
)

# Declared types whose values render verbatim as a JSON string, so the
# argument can be streamed incrementally instead of buffered until its end.
_STRING_SCHEMA_TYPES = frozenset({"string", "str", "text", "varchar", "char", "enum"})


class SolarOpen2ToolParser(ToolParser):
    """
    Tool call parser for Solar Open2 model.

    Parses the format:
        <|tool_call:start|>{function_name}
        <|tool_arg:start|>{arg_name}<|tool_arg:value|>{arg_value}<|tool_arg:end|>
        <|tool_call:end|>

    Argument values are surfaced as raw strings in the wire format. When a
    matching JSON-schema entry is found on ``request.tools``, values are
    coerced to the declared type; on lookup miss or conversion failure the
    original string is kept so malformed output still round-trips.
    """

    TOOL_CALL_START = "<|tool_call:start|>"
    TOOL_CALL_END = "<|tool_call:end|>"
    TOOL_ARG_START = "<|tool_arg:start|>"
    TOOL_ARG_VALUE = "<|tool_arg:value|>"
    TOOL_ARG_END = "<|tool_arg:end|>"

    # An argument value ends at ``<|tool_arg:end|>``. The other sentinels
    # bound the scan so a single dropped end sentinel cannot swallow the
    # arguments and calls that follow it.
    _VALUE_STOP_SENTINELS = (
        TOOL_ARG_END,
        TOOL_ARG_START,
        TOOL_CALL_END,
        TOOL_CALL_START,
    )

    # Streaming state machine states.
    _STATE_WAITING_FOR_TOOL = "waiting_for_tool"
    _STATE_READING_FUNCTION_NAME = "reading_function_name"
    _STATE_WAITING_IN_CALL = "waiting_in_call"
    _STATE_READING_ARG_NAME = "reading_arg_name"
    _STATE_READING_ARG_VALUE = "reading_arg_value"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._reset_stream_state()

        tc_start = re.escape(self.TOOL_CALL_START)
        tc_end = re.escape(self.TOOL_CALL_END)
        ta_start = re.escape(self.TOOL_ARG_START)
        ta_value = re.escape(self.TOOL_ARG_VALUE)
        ta_end = re.escape(self.TOOL_ARG_END)
        # Tempered dot: a call body may not cross a call boundary, so a dropped
        # sentinel cannot swallow the calls that follow. Tempering on the full
        # sentinels — not on their shared ``<|tool_call:`` prefix — keeps that
        # literal usable inside an argument value, as the stream already allows.
        not_call = rf"(?:(?!{tc_start}|{tc_end}).)"
        # The function name is the remainder of the start sentinel's line, so it
        # holds no newline. Bounding it keeps the scan linear: a name group that
        # may cross newlines turns every later newline into a candidate
        # terminator, each one costing a backtracking pass over the rest of the
        # output. An empty name is allowed, which is what the stream reads from
        # ``<|tool_call:start|>\n``.
        not_call_nl = rf"(?:(?!{tc_start}|{tc_end})[^\n])"

        # A call ends at ``<|tool_call:end|>`` or, if the model dropped it, at
        # the start of the next call. This mirrors the streaming state machine,
        # which resyncs on a bare start sentinel. A call left unterminated at
        # end of output stays unmatched so a truncated value is never surfaced
        # as a complete argument.
        self.tool_call_pattern = re.compile(
            rf"{tc_start}({not_call_nl}*?)\n({not_call}*)(?:{tc_end}|(?={tc_start}))",
            re.DOTALL,
        )
        self.tool_arg_pattern = re.compile(
            rf"{ta_start}((?:(?!{ta_start}).)*?){ta_value}"
            rf"((?:(?!{ta_end}|{ta_start}).)*)(?:{ta_end}|(?={ta_start})|\Z)",
            re.DOTALL,
        )

    def _get_param_types(
        self,
        func_name: str,
        param_name: str,
        tools: list[Tool] | None,
    ) -> list[str]:
        """Return the JSON-schema types declared for a parameter."""
        spec = find_tool_properties(tools, func_name).get(param_name)
        if not isinstance(spec, dict):
            return ["string"]
        return extract_types_from_schema(spec)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        tools = getattr(request, "tools", None)

        tool_calls: list[ToolCall] = []
        for match in self.tool_call_pattern.finditer(model_output):
            func_name = match.group(1).strip()
            args_block = match.group(2)

            args_dict: dict[str, Any] = {}
            for arg_match in self.tool_arg_pattern.finditer(args_block):
                arg_name = arg_match.group(1)
                if arg_name in args_dict:
                    # A stream has already committed to the first value, so
                    # keep both paths on first-occurrence-wins.
                    continue
                param_types = self._get_param_types(func_name, arg_name, tools)
                args_dict[arg_name] = coerce_to_schema_type(
                    arg_match.group(2), param_types
                )

            tool_calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=func_name,
                        arguments=json.dumps(args_dict, ensure_ascii=False),
                    ),
                    id=make_tool_call_id(),
                )
            )

        if not tool_calls:
            # Nothing parseable (no sentinel at all, or a truncated block):
            # return the raw output rather than dropping generated text.
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        # Content is the text before the first tool call. A whitespace-only
        # prefix is reported as no content so that a streamed and a batched
        # response for the same output carry the same message.
        prefix = model_output[: model_output.index(self.TOOL_CALL_START)]
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=prefix if prefix.strip() else None,
        )

    def _reset_stream_state(self) -> None:
        """Reset streaming-only fields. Safe to call from ``__init__`` and from
        the start of every new stream (detected via empty ``previous_text``).
        """
        self._stream_buffer: str = ""
        # Leading whitespace run, held back because a tool call may follow —
        # in which case the whole prefix is dropped, as in ``extract_tool_calls``.
        self._stream_pending_ws: str = ""
        self._stream_content_emitted: bool = False
        # Content ends at the first ``<|tool_call:start|>``, as in
        # ``extract_tool_calls``, even if that block never becomes a call.
        self._stream_start_seen: bool = False
        self._stream_state: str = self._STATE_WAITING_FOR_TOOL
        self._stream_func_name: str = ""
        self._stream_arg_name: str = ""
        # Argument names already emitted for the current call, so a repeated
        # name cannot produce a duplicate JSON key.
        self._stream_seen_arg_names: set[str] = set()
        self._stream_skip_arg: bool = False
        # First-arg flag per current tool call: on the first completed arg
        # in a call we emit the opening ``{``; after that we emit ``,`` separators.
        self._stream_first_arg_in_call: bool = True
        # Per-argument state for incrementally streaming pure string values.
        # Other schema types stay buffered until TOOL_ARG_END so coercion
        # can preserve the non-streaming conversion/fallback contract.
        self._stream_arg_param_types: list[str] = ["string"]
        self._stream_partial_string: bool = False
        self._stream_value_key_opened: bool = False

    @staticmethod
    def _is_streamable_string_type(param_types: Sequence[str]) -> bool:
        """Return whether ``param_types`` is safe to stream as a JSON string."""
        normalized = {t.strip().lower() for t in param_types}
        return bool(normalized & _STRING_SCHEMA_TYPES) and not (
            normalized - _STRING_SCHEMA_TYPES - {"null"}
        )

    @staticmethod
    def _json_string_body(value: str) -> str:
        """JSON-escape ``value`` without its surrounding quotes."""
        return json.dumps(value, ensure_ascii=False)[1:-1]

    @staticmethod
    def _may_still_be_null_literal(raw: str) -> bool:
        """Return whether appending bytes could still make ``raw`` JSON null.

        A nullable parameter coerces the ``null`` literal to ``None``, so the
        JSON string must not be opened while the partial value can still
        become that literal.
        """
        stripped = raw.strip().lower()
        if not stripped or stripped == "null":
            return True
        return "null".startswith(stripped)

    def _append_arg_fragment(
        self,
        fragment: str,
        tool_calls_out: list[DeltaToolCall],
    ) -> None:
        """Emit one argument delta and keep both serving trackers identical."""
        if not fragment:
            return
        tool_idx = self.current_tool_id
        combined = self.streamed_args_for_tool[tool_idx] + fragment
        self.streamed_args_for_tool[tool_idx] = combined
        self.prev_tool_call_arr[tool_idx]["arguments"] = combined
        tool_calls_out.append(
            DeltaToolCall(
                index=tool_idx,
                function=DeltaFunctionCall(arguments=fragment),
            )
        )

    def _stream_safe_string_prefix(
        self,
        tool_calls_out: list[DeltaToolCall],
    ) -> None:
        """Emit and consume the safe prefix of the current string argument.

        Only a possible suffix of a stop sentinel remains buffered. Consuming
        emitted raw text avoids re-escaping the full value on every token.
        """
        holdback = self._holdback_len(self._stream_buffer, self._VALUE_STOP_SENTINELS)
        flush_end = len(self._stream_buffer) - holdback
        if flush_end <= 0:
            return

        raw_prefix = self._stream_buffer[:flush_end]
        if not self._stream_value_key_opened and self._may_still_be_null_literal(
            raw_prefix
        ):
            return

        if not self._stream_value_key_opened:
            object_prefix = "{" if self._stream_first_arg_in_call else ", "
            self._stream_first_arg_in_call = False
            key_json = json.dumps(self._stream_arg_name, ensure_ascii=False)
            self._append_arg_fragment(f'{object_prefix}{key_json}: "', tool_calls_out)
            self._stream_value_key_opened = True

        self._append_arg_fragment(self._json_string_body(raw_prefix), tool_calls_out)
        self._stream_buffer = self._stream_buffer[flush_end:]

    def _holdback_len(self, buf: str, sentinels: tuple[str, ...]) -> int:
        """Number of trailing bytes of ``buf`` that could be the start of any
        sentinel in ``sentinels`` and therefore must stay in the buffer until
        the next delta arrives.

        Example: if ``buf`` ends with ``"<|tool_ca"`` and one sentinel is
        ``"<|tool_call:start|>"``, return 9 so the caller flushes everything
        before those 9 bytes and carries those 9 bytes forward.
        """
        if not buf or not sentinels:
            return 0
        # Anchor on the last ``<`` — all solar_open2 sentinels start with ``<|``.
        last_lt = buf.rfind("<")
        if last_lt == -1:
            return 0
        tail = buf[last_lt:]
        for s in sentinels:
            # Only holdback if ``tail`` is a proper, non-empty prefix of ``s``.
            # If ``tail == s`` we don't hold back (the full sentinel is already
            # present and will be consumed by the state machine).
            if len(tail) < len(s) and s.startswith(tail):
                return len(tail)
        return 0

    def _sendable_content(self, chunk: str) -> str:
        """Return the part of ``chunk`` that may go out on the content channel."""
        if not self._stream_content_emitted and not chunk.strip():
            self._stream_pending_ws += chunk
            return ""
        out = self._stream_pending_ws + chunk
        self._stream_pending_ws = ""
        self._stream_content_emitted = True
        return out

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
        # Fresh stream: reset both shared (base-class) and streaming-only state.
        if not previous_text:
            self.prev_tool_call_arr = []
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool = []
            self._reset_stream_state()

        if delta_text:
            self._stream_buffer += delta_text

        tools = getattr(request, "tools", None)
        content_out = ""
        tool_calls_out: list[DeltaToolCall] = []

        while True:
            if self._stream_state == self._STATE_WAITING_FOR_TOOL:
                # Content is the text before the FIRST tool call; anything
                # after one is dropped, as in ``extract_tool_calls``.
                is_content = not self._stream_start_seen
                idx = self._stream_buffer.find(self.TOOL_CALL_START)
                if idx == -1:
                    # No sentinel yet — flush everything as content except
                    # any trailing bytes that could be the start of the
                    # sentinel we're waiting on.
                    hb = self._holdback_len(
                        self._stream_buffer, (self.TOOL_CALL_START,)
                    )
                    flush_end = len(self._stream_buffer) - hb
                    if flush_end > 0:
                        chunk = self._stream_buffer[:flush_end]
                        self._stream_buffer = self._stream_buffer[flush_end:]
                        if is_content:
                            content_out += self._sendable_content(chunk)
                    break
                prefix = self._stream_buffer[:idx]
                self._stream_buffer = self._stream_buffer[
                    idx + len(self.TOOL_CALL_START) :
                ]
                if is_content and (prefix.strip() or self._stream_content_emitted):
                    content_out += self._stream_pending_ws + prefix
                    self._stream_content_emitted = True
                self._stream_pending_ws = ""
                self._stream_start_seen = True
                self._stream_state = self._STATE_READING_FUNCTION_NAME
                continue

            if self._stream_state == self._STATE_READING_FUNCTION_NAME:
                nl = self._stream_buffer.find("\n")
                # A call boundary before the newline means this block can never
                # become a call. The non-streaming pattern skips it, so resync
                # on the boundary rather than reading a name across it.
                bounds = [
                    i
                    for i in (
                        self._stream_buffer.find(self.TOOL_CALL_START),
                        self._stream_buffer.find(self.TOOL_CALL_END),
                    )
                    if i != -1
                ]
                if bounds and (nl == -1 or min(bounds) < nl):
                    self._stream_buffer = self._stream_buffer[min(bounds) :]
                    self._stream_state = self._STATE_WAITING_FOR_TOOL
                    continue
                if nl == -1:
                    # Wait for the newline that terminates the function name.
                    break
                self._stream_func_name = self._stream_buffer[:nl].strip()
                self._stream_buffer = self._stream_buffer[nl + 1 :]

                # Register a new tool call and emit the name delta.
                self.current_tool_id += 1
                tool_idx = self.current_tool_id
                while len(self.prev_tool_call_arr) <= tool_idx:
                    self.prev_tool_call_arr.append({"name": "", "arguments": ""})
                while len(self.streamed_args_for_tool) <= tool_idx:
                    self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr[tool_idx]["name"] = self._stream_func_name

                self._stream_first_arg_in_call = True
                self._stream_seen_arg_names = set()
                tool_calls_out.append(
                    DeltaToolCall(
                        index=tool_idx,
                        id=make_tool_call_id(),
                        type="function",
                        function=DeltaFunctionCall(
                            name=self._stream_func_name, arguments=""
                        ),
                    )
                )
                self.current_tool_name_sent = True
                self._stream_state = self._STATE_WAITING_IN_CALL
                continue

            if self._stream_state == self._STATE_WAITING_IN_CALL:
                ta_idx = self._stream_buffer.find(self.TOOL_ARG_START)
                end_idx = self._stream_buffer.find(self.TOOL_CALL_END)
                next_idx = self._stream_buffer.find(self.TOOL_CALL_START)
                found = [i for i in (ta_idx, end_idx, next_idx) if i != -1]
                if not found:
                    # Could be partway through any of them — wait for more.
                    break
                first = min(found)
                if first == ta_idx:
                    self._stream_buffer = self._stream_buffer[
                        ta_idx + len(self.TOOL_ARG_START) :
                    ]
                    self._stream_state = self._STATE_READING_ARG_NAME
                    continue
                # Either this call ended or the next one started without an
                # end sentinel; close the JSON object either way and leave a
                # dangling start sentinel for WAITING_FOR_TOOL to consume.
                close_str = "{}" if self._stream_first_arg_in_call else "}"
                self._append_arg_fragment(close_str, tool_calls_out)
                if first == end_idx:
                    self._stream_buffer = self._stream_buffer[
                        end_idx + len(self.TOOL_CALL_END) :
                    ]
                self._stream_state = self._STATE_WAITING_FOR_TOOL
                continue

            if self._stream_state == self._STATE_READING_ARG_NAME:
                v_idx = self._stream_buffer.find(self.TOOL_ARG_VALUE)
                # An argument name ends at ``<|tool_arg:value|>``; a call or
                # argument boundary before it means this argument never
                # completes. The non-streaming patterns bound the name the same
                # way, so resync on the boundary instead of reading across it.
                # Only a boundary before the nearest argument start can be the
                # first one, so the other two scans stop there rather than
                # rescanning the whole buffer on every resync.
                ta_idx = self._stream_buffer.find(self.TOOL_ARG_START)
                limit = len(self._stream_buffer) if ta_idx == -1 else ta_idx
                bounds = [
                    i
                    for i in (
                        ta_idx,
                        self._stream_buffer.find(self.TOOL_CALL_START, 0, limit),
                        self._stream_buffer.find(self.TOOL_CALL_END, 0, limit),
                    )
                    if i != -1
                ]
                if bounds and (v_idx == -1 or min(bounds) < v_idx):
                    self._stream_buffer = self._stream_buffer[min(bounds) :]
                    self._stream_state = self._STATE_WAITING_IN_CALL
                    continue
                if v_idx == -1:
                    break
                self._stream_arg_name = self._stream_buffer[:v_idx]
                self._stream_buffer = self._stream_buffer[
                    v_idx + len(self.TOOL_ARG_VALUE) :
                ]
                self._stream_skip_arg = (
                    self._stream_arg_name in self._stream_seen_arg_names
                )
                self._stream_seen_arg_names.add(self._stream_arg_name)
                self._stream_arg_param_types = self._get_param_types(
                    self._stream_func_name, self._stream_arg_name, tools
                )
                self._stream_partial_string = not self._stream_skip_arg and (
                    self._is_streamable_string_type(self._stream_arg_param_types)
                )
                self._stream_value_key_opened = False
                self._stream_state = self._STATE_READING_ARG_VALUE
                continue

            if self._stream_state == self._STATE_READING_ARG_VALUE:
                stop_idx, consume = -1, 0
                for sentinel in self._VALUE_STOP_SENTINELS:
                    idx = self._stream_buffer.find(sentinel)
                    if idx == -1 or (stop_idx != -1 and idx >= stop_idx):
                        continue
                    stop_idx = idx
                    # Only the argument terminator is consumed here; call and
                    # argument boundaries belong to the enclosing states.
                    consume = len(sentinel) if sentinel == self.TOOL_ARG_END else 0
                if stop_idx == -1:
                    if self._stream_partial_string:
                        self._stream_safe_string_prefix(tool_calls_out)
                    # Non-string types stay fully buffered so coercion sees
                    # the complete value atomically.
                    break
                raw_value = self._stream_buffer[:stop_idx]
                self._stream_buffer = self._stream_buffer[stop_idx + consume :]
                self._stream_state = self._STATE_WAITING_IN_CALL

                if self._stream_skip_arg:
                    self._stream_skip_arg = False
                    continue

                if self._stream_value_key_opened:
                    # Earlier raw prefixes have already been escaped, emitted,
                    # and consumed. Finish only the remaining suffix.
                    self._append_arg_fragment(
                        self._json_string_body(raw_value) + '"',
                        tool_calls_out,
                    )
                    self._stream_value_key_opened = False
                    self._stream_partial_string = False
                    continue

                # Atomic path: non-string types, null, empty strings, short
                # null-prefix strings, or a complete value in one input delta.
                coerced = coerce_to_schema_type(raw_value, self._stream_arg_param_types)
                prefix = "{" if self._stream_first_arg_in_call else ", "
                self._stream_first_arg_in_call = False
                key_json = json.dumps(self._stream_arg_name, ensure_ascii=False)
                value_json = json.dumps(coerced, ensure_ascii=False)
                self._append_arg_fragment(
                    f"{prefix}{key_json}: {value_json}", tool_calls_out
                )
                self._stream_partial_string = False
                continue

            # Unreachable — every state above either `continue`s or `break`s.
            break

        if not content_out and not tool_calls_out:
            return None
        return DeltaMessage(
            content=content_out or None,
            tool_calls=tool_calls_out,
        )
