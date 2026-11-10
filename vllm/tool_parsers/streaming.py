# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.utils.mistral import is_mistral_tokenizer

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
else:
    TokenizerLike = object


def extract_named_tool_call_streaming(
    *,
    delta_text: str,
    function_name: str,
    function_name_returned: bool,
    tool_call_idx: int | None,
    tool_call_id_type: str,
    tokenizer: "TokenizerLike",
    tool_call_array_index: int = 0,
) -> tuple[DeltaMessage | None, bool]:
    """Build a streaming tool-call delta for forced named tool choice."""
    if function_name_returned:
        delta_tool_call = DeltaToolCall(
            function=DeltaFunctionCall(arguments=delta_text),
            index=tool_call_array_index,
        )
    else:
        if is_mistral_tokenizer(tokenizer):
            # Import mistral_common only if we need it.
            from vllm.parser.mistral import MistralToolCall

            tool_call_id = MistralToolCall.generate_random_id()
        else:
            tool_call_id = make_tool_call_id(
                id_type=tool_call_id_type,
                func_name=function_name,
                idx=tool_call_idx,
            )
        delta_tool_call = DeltaToolCall(
            id=tool_call_id,
            type="function",
            function=DeltaFunctionCall(
                name=function_name,
                arguments=delta_text,
            ),
            index=tool_call_array_index,
        )
        function_name_returned = True
    return (
        DeltaMessage(tool_calls=[delta_tool_call]),
        function_name_returned,
    )


@dataclass
class _RequiredToolCallSpan:
    """Where one element of a ``tool_choice="required"`` array sits in the text.

    The model output for ``tool_choice="required"`` is a JSON array of
    ``{"name": ..., "parameters": ...}`` objects. ``name`` is only set once
    its string value is closed and decodes as JSON; ``name_end`` is the
    offset just past that closing quote. ``args_start``/``args_end`` are the
    character offsets of the ``parameters`` value (``args_end`` is ``None``
    while the value is still being generated).

    All offsets are prefix-stable: scanning any prefix of the text yields the
    same spans truncated to that prefix. That is what lets one scan of the
    current text tell what an earlier, shorter text had already revealed.
    """

    name: str | None = None
    name_end: int | None = None
    args_start: int | None = None
    args_end: int | None = None

    def ready_at(self, length: int) -> bool:
        """Whether the first ``length`` characters of the text already made
        this call streamable: its name was complete and its parameters value
        had started."""
        return (
            self.name is not None
            and self.name_end is not None
            and self.name_end <= length
            and self.args_start is not None
            and self.args_start < length
        )

    @property
    def ready(self) -> bool:
        return self.name is not None and self.args_start is not None


# Parse states for one element object of the required-tool-call array.
_EXPECT_KEY = 0
_EXPECT_COLON = 1
_EXPECT_VALUE = 2
_IN_STRING_VALUE = 3
_IN_CONTAINER_VALUE = 4
_IN_SCALAR_VALUE = 5
_AFTER_VALUE = 6


@dataclass
class RequiredToolCallScanner:
    """Incremental scanner owning the text of one required-tool-call stream."""

    spans: list[_RequiredToolCallSpan] = field(default_factory=list)
    text: str = ""
    depth: int = 0
    in_string: bool = False
    escaped: bool = False
    string_start: int = 0
    cur: _RequiredToolCallSpan | None = None
    expect: int = _EXPECT_KEY
    key: str | None = None
    value_start: int = 0
    done: bool = False

    def scan(self, delta_text: str) -> list[_RequiredToolCallSpan]:
        """Append and scan new text, retaining lexical state across deltas."""
        start = len(self.text)
        self.text += delta_text
        text = self.text
        if self.done:
            return self.spans

        def end_value(pos: int) -> None:
            assert self.cur is not None
            if self.key == "parameters":
                self.cur.args_end = pos
            elif self.key == "name" and self.expect == _IN_STRING_VALUE:
                try:
                    self.cur.name = json.loads(text[self.value_start : pos])
                    self.cur.name_end = pos
                except json.JSONDecodeError:
                    # Not a valid JSON string: the call can never become ready,
                    # which matches the non-streaming path rejecting the array.
                    self.cur.name = None
            self.expect = _AFTER_VALUE

        def start_value(pos: int, value_state: int) -> None:
            assert self.cur is not None
            self.value_start = pos
            if self.key == "parameters":
                self.cur.args_start = pos
            self.expect = value_state

        for i in range(start, len(text)):
            ch = text[i]
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif ch == "\\":
                    self.escaped = True
                elif ch == '"':
                    self.in_string = False
                    if self.depth == 2 and self.cur is not None:
                        if self.expect == _EXPECT_KEY:
                            self.key = text[self.string_start + 1 : i]
                            self.expect = _EXPECT_COLON
                        elif self.expect == _IN_STRING_VALUE:
                            end_value(i + 1)
                continue

            if ch == '"':
                self.in_string = True
                self.string_start = i
                if (
                    self.depth == 2
                    and self.cur is not None
                    and self.expect == _EXPECT_VALUE
                ):
                    start_value(i, _IN_STRING_VALUE)
                continue

            if ch in "{[":
                if self.depth == 0:
                    if ch != "[":
                        self.done = True
                        return self.spans
                elif self.depth == 1:
                    if ch != "{":
                        self.done = True
                        return self.spans
                    self.cur = _RequiredToolCallSpan()
                    self.spans.append(self.cur)
                    self.expect = _EXPECT_KEY
                    self.key = None
                elif (
                    self.depth == 2
                    and self.cur is not None
                    and self.expect == _EXPECT_VALUE
                ):
                    start_value(i, _IN_CONTAINER_VALUE)
                self.depth += 1
                continue

            if ch in "}]":
                if (
                    self.depth == 2
                    and self.cur is not None
                    and self.expect == _IN_SCALAR_VALUE
                ):
                    end_value(i)
                self.depth -= 1
                if (
                    self.depth == 2
                    and self.cur is not None
                    and self.expect == _IN_CONTAINER_VALUE
                ):
                    end_value(i + 1)
                elif self.depth == 1:
                    self.cur = None
                elif self.depth <= 0:
                    self.done = True
                    return self.spans
                continue

            if self.depth != 2 or self.cur is None:
                continue

            if ch == ":":
                if self.expect == _EXPECT_COLON:
                    self.expect = _EXPECT_VALUE
            elif ch == ",":
                if self.expect == _IN_SCALAR_VALUE:
                    end_value(i)
                self.expect = _EXPECT_KEY
                self.key = None
            elif ch.isspace():
                if self.expect == _IN_SCALAR_VALUE:
                    end_value(i)
            elif self.expect == _EXPECT_VALUE:
                start_value(i, _IN_SCALAR_VALUE)

        return self.spans


def extract_required_tool_call_streaming(
    *,
    delta_text: str,
    scanner: RequiredToolCallScanner,
    tool_call_idx: int | None,
    tool_call_id_type: str,
) -> tuple[DeltaMessage | None, bool]:
    """Stream required tool calls from new text using a per-stream scanner.

    The first chunk of each call carries its id, name and available arguments;
    subsequent chunks carry only new argument text. A delta may start several
    calls. The scanner owns accumulated text and lexical state.
    """
    sent_len = len(scanner.text)
    spans = scanner.scan(delta_text)
    current_text = scanner.text

    tool_calls: list[DeltaToolCall] = []
    started = 0
    for index, span in enumerate(spans):
        if not span.ready:
            # Elements are generated in order: nothing later can be ready.
            break
        assert span.args_start is not None
        end = span.args_end if span.args_end is not None else len(current_text)
        if span.ready_at(sent_len):
            # Already announced: send only the new argument text.
            sent_end = min(end, sent_len)
            new_arguments = current_text[sent_end:end]
            if new_arguments:
                tool_calls.append(
                    DeltaToolCall(
                        index=index,
                        function=DeltaFunctionCall(
                            # OpenAI API returns None instead of name every time
                            name=None,
                            arguments=new_arguments,
                        ),
                    )
                )
            continue

        idx = None if tool_call_idx is None else tool_call_idx + started
        started += 1
        tool_calls.append(
            DeltaToolCall(
                id=make_tool_call_id(
                    id_type=tool_call_id_type, func_name=span.name, idx=idx
                ),
                type="function",
                index=index,
                function=DeltaFunctionCall(
                    name=span.name,
                    arguments=current_text[span.args_start : end],
                ),
            )
        )

    function_name_returned = any(span.ready for span in spans)
    if not tool_calls:
        return None, function_name_returned
    return DeltaMessage(tool_calls=tool_calls), function_name_returned
