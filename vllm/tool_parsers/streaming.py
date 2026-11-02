# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass
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


def _scan_required_tool_calls(text: str) -> list[_RequiredToolCallSpan]:
    """Locate every tool call of a (possibly partial) required-tool-call array.

    A single string- and nesting-aware pass over ``text`` that records, for
    each top-level array element, its ``name`` (once complete) and the
    character span of its ``parameters`` value. Text after the closing ``]``
    of the array is ignored. Only the element level of the array is
    interpreted, so braces, brackets and quotes inside nested values and
    strings cannot confuse it.
    """
    spans: list[_RequiredToolCallSpan] = []
    depth = 0  # nesting depth of ``{}`` and ``[]``; element objects sit at 2
    in_string = False
    escaped = False
    string_start = 0
    cur: _RequiredToolCallSpan | None = None
    expect = _EXPECT_KEY
    key: str | None = None
    value_start = 0

    def end_value(pos: int) -> None:
        nonlocal expect
        assert cur is not None
        if key == "parameters":
            cur.args_end = pos
        elif key == "name" and expect == _IN_STRING_VALUE:
            try:
                cur.name = json.loads(text[value_start:pos])
                cur.name_end = pos
            except json.JSONDecodeError:
                # Not a valid JSON string: the call can never become ready,
                # which matches the non-streaming path rejecting the array.
                cur.name = None
        expect = _AFTER_VALUE

    def start_value(pos: int, state: int) -> None:
        nonlocal expect, value_start
        assert cur is not None
        value_start = pos
        if key == "parameters":
            cur.args_start = pos
        expect = state

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                if depth == 2 and cur is not None:
                    if expect == _EXPECT_KEY:
                        key = text[string_start + 1 : i]
                        expect = _EXPECT_COLON
                    elif expect == _IN_STRING_VALUE:
                        end_value(i + 1)
            continue

        if ch == '"':
            in_string = True
            string_start = i
            if depth == 2 and cur is not None and expect == _EXPECT_VALUE:
                start_value(i, _IN_STRING_VALUE)
            continue

        if ch in "{[":
            if depth == 0:
                if ch != "[":
                    return spans
            elif depth == 1:
                if ch != "{":
                    return spans
                cur = _RequiredToolCallSpan()
                spans.append(cur)
                expect = _EXPECT_KEY
                key = None
            elif depth == 2 and cur is not None and expect == _EXPECT_VALUE:
                start_value(i, _IN_CONTAINER_VALUE)
            depth += 1
            continue

        if ch in "}]":
            if depth == 2 and cur is not None and expect == _IN_SCALAR_VALUE:
                end_value(i)
            depth -= 1
            if depth == 2 and cur is not None and expect == _IN_CONTAINER_VALUE:
                end_value(i + 1)
            elif depth == 1:
                cur = None
            elif depth <= 0:
                return spans
            continue

        if depth != 2 or cur is None:
            continue

        if ch == ":":
            if expect == _EXPECT_COLON:
                expect = _EXPECT_VALUE
        elif ch == ",":
            if expect == _IN_SCALAR_VALUE:
                end_value(i)
            expect = _EXPECT_KEY
            key = None
        elif ch.isspace():
            if expect == _IN_SCALAR_VALUE:
                end_value(i)
        elif expect == _EXPECT_VALUE:
            start_value(i, _IN_SCALAR_VALUE)

    return spans


def extract_required_tool_call_streaming(
    *,
    previous_text: str,
    current_text: str | None,
    tool_call_idx: int | None,
    tool_call_id_type: str,
) -> tuple[DeltaMessage | None, bool]:
    """Stream the tool calls of a ``tool_choice="required"`` JSON array.

    ``previous_text`` must be a prefix of ``current_text`` (the accumulated
    text before and after this delta). Every array element that has something
    not yet sent is emitted, in array order, so a delta may carry several
    ``DeltaToolCall`` entries: the first chunk of a call carries its id, name
    and the arguments generated so far; later chunks carry only new argument
    text. What was already sent is read off the same scan, since the spans
    are prefix-stable, so no state is needed beyond the two texts.

    Returns the delta (``None`` when nothing is new) and whether at least one
    call has been announced so far.
    """
    if not current_text:
        return None, False

    spans = _scan_required_tool_calls(current_text)
    sent_len = len(previous_text)

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
