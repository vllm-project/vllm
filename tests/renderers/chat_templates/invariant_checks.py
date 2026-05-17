# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Any

from tests.renderers.chat_templates.conversation_builder import (
    AUTO,
    MISSING,
    SYSTEM,
    USER,
    Assistant,
)

BASIC_CASES = {
    "turn": (SYSTEM, USER, Assistant()),
    "turn_user_follow_up": (USER, Assistant(), USER),
    "multi_turn": (USER, Assistant(), USER, Assistant()),
}

REASONING_CASES = {
    "reasoning": (USER, Assistant(reasoning=AUTO)),
    "reasoning_turn_follow_up": (USER, Assistant(reasoning=AUTO), USER),
    "reasoning_multi_turn": (
        USER,
        Assistant(reasoning=AUTO),
        USER,
        Assistant(reasoning=AUTO),
    ),
    "empty_reasoning": (USER, Assistant(reasoning="")),
    "empty_reasoning_turn_user_follow_up": (USER, Assistant(reasoning=""), USER),
}

TOOL_CALL_CASES = {
    "tool_call": (USER, Assistant(content=MISSING, tool_uses=[(AUTO, MISSING)])),
    "tool_call_w_response": (
        USER,
        Assistant(content=MISSING, tool_uses=[AUTO]),
    ),
    "tool_turn": (
        USER,
        Assistant(content=MISSING, tool_uses=[AUTO]),
        Assistant(),
    ),
    "multi_tool_call": (
        USER,
        Assistant(content=MISSING, tool_uses=[AUTO]),
        Assistant(content=MISSING, tool_uses=[(AUTO, MISSING)]),
    ),
    "multi_tool_call_w_response": (
        USER,
        Assistant(content=MISSING, tool_uses=[AUTO]),
        Assistant(content=MISSING, tool_uses=[AUTO]),
    ),
    "multi_tool_turn": (
        USER,
        Assistant(content=MISSING, tool_uses=[AUTO]),
        Assistant(content=MISSING, tool_uses=[AUTO]),
        Assistant(),
    ),
    "tool_call_w_content": (USER, Assistant(tool_uses=[(AUTO, MISSING)])),
    "tool_call_w_content_w_response": (USER, Assistant(tool_uses=[AUTO])),
    "tool_turn_w_content": (USER, Assistant(tool_uses=[AUTO]), Assistant()),
    "tool_turn_w_content_user_interrupt": (
        USER,
        Assistant(tool_uses=[AUTO]),
        USER,
    ),
    "multi_tool_call_w_content": (
        USER,
        Assistant(tool_uses=[AUTO]),
        Assistant(tool_uses=[(AUTO, MISSING)]),
    ),
    "multi_tool_call_w_content_w_response": (
        USER,
        Assistant(tool_uses=[AUTO]),
        Assistant(tool_uses=[AUTO]),
    ),
    "multi_tool_turn_w_content": (
        USER,
        Assistant(tool_uses=[AUTO]),
        Assistant(tool_uses=[AUTO]),
        Assistant(),
    ),
    "multi_tool_turn_w_content_user_followup": (
        USER,
        Assistant(tool_uses=[AUTO]),
        Assistant(tool_uses=[AUTO]),
        Assistant(),
        USER,
    ),
}

TOOL_CALL_W_REASONING_CASES = {
    "tool_call_w_reasoning": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[(AUTO, MISSING)]),
    ),
    "tool_call_w_reasoning_w_response": (
        USER,
        Assistant(reasoning=AUTO, content=MISSING, tool_uses=[AUTO]),
    ),
    "tool_turn_w_reasoning": (
        USER,
        Assistant(reasoning=AUTO, content=MISSING, tool_uses=[AUTO]),
        Assistant(reasoning=AUTO),
    ),
    "multi_tool_call_w_reasoning": (
        USER,
        Assistant(reasoning=AUTO, content=MISSING, tool_uses=[AUTO]),
        Assistant(reasoning=AUTO, content=MISSING, tool_uses=[(AUTO, MISSING)]),
    ),
    "multi_tool_call_w_reasoning_w_response": (
        USER,
        Assistant(reasoning=AUTO, content=MISSING, tool_uses=[AUTO]),
        Assistant(reasoning=AUTO, content=MISSING, tool_uses=[AUTO]),
    ),
    "multi_tool_turn_w_reasoning": (
        USER,
        Assistant(reasoning=AUTO, content=MISSING, tool_uses=[AUTO]),
        Assistant(reasoning=AUTO, content=MISSING, tool_uses=[AUTO]),
        Assistant(reasoning=AUTO),
    ),
    "tool_call_w_reasoning_w_content": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[(AUTO, MISSING)]),
    ),
    "tool_call_w_reasoning_w_content_w_response": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO]),
    ),
    "tool_turn_w_reasoning_w_content": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO]),
        Assistant(reasoning=AUTO),
    ),
    "tool_turn_w_reasoning_w_content_user_interrupt": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO]),
        USER,
    ),
    "multi_tool_call_w_reasoning_w_content": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO]),
        Assistant(reasoning=AUTO, tool_uses=[(AUTO, MISSING)]),
    ),
    "multi_tool_call_w_reasoning_w_content_w_response": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO]),
        Assistant(reasoning=AUTO, tool_uses=[AUTO]),
    ),
    "multi_tool_turn_w_reasoning_w_content": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO]),
        Assistant(reasoning=AUTO, tool_uses=[AUTO]),
        Assistant(reasoning=AUTO),
    ),
    "multi_tool_turn_w_reasoning_w_content_user_followup": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO]),
        Assistant(reasoning=AUTO, tool_uses=[AUTO]),
        Assistant(reasoning=AUTO),
        USER,
    ),
}

PARALLEL_TOOL_CALL_CASES = {
    "parallel_tool_call": (
        USER,
        Assistant(tool_uses=[(AUTO, MISSING), (AUTO, MISSING)]),
    ),
    "parallel_tool_call_partial_response_1": (
        USER,
        Assistant(tool_uses=[AUTO, (AUTO, MISSING)]),
    ),
    "parallel_tool_call_partial_response_2": (
        USER,
        Assistant(tool_uses=[(AUTO, MISSING), AUTO]),
    ),
    "parallel_tool_call_w_response": (
        USER,
        Assistant(tool_uses=[AUTO, AUTO]),
    ),
    "parallel_tool_call_user_interrupt": (
        USER,
        Assistant(tool_uses=[AUTO, AUTO]),
        USER,
    ),
    "parallel_tool_turn": (
        USER,
        Assistant(tool_uses=[AUTO, AUTO]),
        Assistant(),
    ),
    "parallel_tool_turn_user_follow_up": (
        USER,
        Assistant(tool_uses=[AUTO, AUTO]),
        Assistant(),
        USER,
    ),
    "multi_parallel_tool_call": (
        USER,
        Assistant(tool_uses=[AUTO, AUTO]),
        Assistant(tool_uses=[(AUTO, MISSING), (AUTO, MISSING)]),
    ),
    "multi_parallel_tool_call_partial_response_1": (
        USER,
        Assistant(tool_uses=[AUTO, AUTO]),
        Assistant(tool_uses=[AUTO, (AUTO, MISSING)]),
    ),
    "multi_parallel_tool_call_partial_response_2": (
        USER,
        Assistant(tool_uses=[AUTO, AUTO]),
        Assistant(tool_uses=[(AUTO, MISSING), AUTO]),
    ),
    "multi_parallel_tool_call_w_response": (
        USER,
        Assistant(tool_uses=[AUTO, AUTO]),
        Assistant(tool_uses=[AUTO, AUTO]),
    ),
    "multi_parallel_tool_turn": (
        USER,
        Assistant(tool_uses=[AUTO, AUTO]),
        Assistant(tool_uses=[AUTO, AUTO]),
        Assistant(),
    ),
}

PARALLEL_TOOL_CALL_W_REASONING_CASES = {
    "parallel_tool_call_w_reasoning": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[(AUTO, MISSING), (AUTO, MISSING)]),
    ),
    "parallel_tool_call_w_reasoning_partial_response_1": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO, (AUTO, MISSING)]),
    ),
    "parallel_tool_call_w_reasoning_partial_response_2": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[(AUTO, MISSING), AUTO]),
    ),
    "parallel_tool_call_w_reasoning_w_response": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
    ),
    "parallel_tool_call_w_reasoning_user_interrupt": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
        USER,
    ),
    "parallel_tool_turn_w_reasoning": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
        Assistant(reasoning=AUTO),
    ),
    "parallel_tool_turn_w_reasoning_user_follow_up": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
        Assistant(reasoning=AUTO),
        USER,
    ),
    "multi_parallel_tool_call_w_reasoning": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
        Assistant(reasoning=AUTO, tool_uses=[(AUTO, MISSING), (AUTO, MISSING)]),
    ),
    "multi_parallel_tool_call_w_reasoning_partial_response_1": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
        Assistant(reasoning=AUTO, tool_uses=[AUTO, (AUTO, MISSING)]),
    ),
    "multi_parallel_tool_call_w_reasoning_partial_response_2": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
        Assistant(reasoning=AUTO, tool_uses=[(AUTO, MISSING), AUTO]),
    ),
    "multi_parallel_tool_call_w_reasoning_w_response": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
    ),
    "multi_parallel_tool_turn_w_reasoning": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
        Assistant(reasoning=AUTO),
    ),
    "multi_parallel_tool_turn_w_reasoning_user_follow_up": (
        USER,
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
        Assistant(reasoning=AUTO, tool_uses=[AUTO, AUTO]),
        Assistant(reasoning=AUTO),
        USER,
    ),
}


def delimiter_balance_trace(
    text: str, start_delimiter: str, end_delimiter: str
) -> list[int]:
    """Return the running delimiter balance after each delimiter token."""
    if not start_delimiter or not end_delimiter:
        return []

    balance = 0
    trace: list[int] = []
    search_start = 0
    while True:
        next_start = text.find(start_delimiter, search_start)
        next_end = text.find(end_delimiter, search_start)

        if next_start < 0 and next_end < 0:
            return trace

        # Prefer the start delimiter on ties to preserve the prior ordering.
        if next_end < 0 or (next_start >= 0 and next_start <= next_end):
            balance += 1
            search_start = next_start + len(start_delimiter)
        else:
            balance -= 1
            search_start = next_end + len(end_delimiter)
        trace.append(balance)


def delimiter_state(text: str, delimiter: tuple[str | None, str | None] | None) -> int:
    """Summarize whether a delimiter pair is balanced (0), open (1), or invalid (-1).
    Nested or out-of-order delimiters are treated as invalid test output.
    """
    if delimiter is None:
        return 0

    start, end = delimiter
    if start is None or end is None:
        return 0

    trace = delimiter_balance_trace(text, start, end)
    if any(balance < 0 or balance > 1 for balance in trace):
        return -1
    if not trace:
        return 0
    return trace[-1]


def all_appear_in_order(text: str, substrings: Sequence[str]) -> bool:
    """Return whether all substrings appear in order within ``text``.

    Matches are consumed left-to-right so repeated values must appear as
    distinct, non-overlapping occurrences in the rendered output.
    """
    search_start = 0
    for substring in substrings:
        next_match = text.find(substring, search_start)
        if next_match < 0:
            return False
        search_start = next_match + len(substring)
    return True


class TestChatTemplateInvariants:
    """Mixin-style assertions for concrete chat-template renderer tests."""

    @classmethod
    def _build_markers(cls, messages: list[dict[str, Any]]) -> list[str]:
        markers = []

        last_non_assistant_or_tool_message_idx = 0
        for idx, msg in enumerate(messages):
            if msg.get("role") not in ("assistant", "tool"):
                last_non_assistant_or_tool_message_idx = idx

        for idx, msg in enumerate(messages):
            if idx > last_non_assistant_or_tool_message_idx:
                reasoning = msg.get("reasoning")
                if reasoning is not None:
                    markers.append(reasoning)

            content = msg.get("content")
            if content is not None:
                markers.append(content)

            tool_calls = msg.get("tool_calls", ())
            for tool_call in tool_calls:
                tool_call_name = tool_call.get("function", {}).get("name")
                if tool_call_name is not None:
                    markers.append(tool_call_name)

        return markers

    @classmethod
    def _check_delimiters(cls, messages: list[dict[str, Any]], result: str):
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def _check_markers(cls, messages: list[dict[str, Any]], result: str):
        markers = cls._build_markers(messages)
        assert all_appear_in_order(result, markers), (
            f"Markers are not in order\nMarkers: {markers}\nResult: {result}"
        )

    def _test_case(self, messages: list[dict[str, Any]], result: str):
        self._check_delimiters(messages, result)
        self._check_markers(messages, result)
