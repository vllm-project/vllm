# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.reasoning.kimi_k2_reasoning_parser import KimiK2ReasoningParser
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser

REASONING_MODEL_NAME = "moonshotai/Kimi-K2.5"


@pytest.fixture
def mock_kimi_k2_tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<think>": 100,
        "</think>": 101,
        "<|tool_calls_section_begin|>": 200,
        "<|tool_calls_section_end|>": 201,
        "<|tool_call_begin|>": 202,
        "<|tool_call_end|>": 203,
    }
    return tokenizer


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return get_tokenizer(tokenizer_name=REASONING_MODEL_NAME, trust_remote_code=True)


def test_parser_selection_thinking_enabled(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(
        kimi_k2_tokenizer, chat_template_kwargs={"thinking": True}
    )
    assert parser._identity_parser is None


def test_parser_selection_thinking_disabled(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(
        kimi_k2_tokenizer, chat_template_kwargs={"thinking": False}
    )
    assert isinstance(parser._identity_parser, IdentityReasoningParser)


def test_extract_reasoning_with_think_tags(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think>step by step reasoning</think>final answer", request
    )
    assert reasoning == "step by step reasoning"
    assert content == "final answer"


def test_extract_reasoning_empty_thinking(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think></think>final answer", request
    )
    assert reasoning == ""
    assert content == "final answer"


def test_extract_reasoning_implicit_start(kimi_k2_tokenizer):
    """When there's no <think> tag, everything is treated as reasoning."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "implicit reasoning with no tags", request
    )
    assert reasoning == "implicit reasoning with no tags"
    assert content is None


def test_extract_reasoning_tool_section_ends_reasoning(kimi_k2_tokenizer):
    """<|tool_calls_section_begin|> implicitly ends reasoning."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    text = "some reasoning<|tool_calls_section_begin|>tool call data"
    reasoning, content = parser.extract_reasoning(text, request)
    assert reasoning == "some reasoning"
    assert content == "<|tool_calls_section_begin|>tool call data"


def test_streaming_reasoning_then_content(kimi_k2_tokenizer):
    """Token-by-token streaming: reasoning tokens then content after </think>."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)

    think_id = parser._start_token_id
    end_think_id = parser._end_token_id
    # Use a real token ID from the tokenizer for regular content
    regular_id = kimi_k2_tokenizer.encode("hello", add_special_tokens=False)[0]

    # First token: <think> — single special token should be skipped
    result = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="<think>",
        delta_text="<think>",
        previous_token_ids=[],
        current_token_ids=[think_id],
        delta_token_ids=[think_id],
    )
    assert result is None

    # Reasoning token
    result = parser.extract_reasoning_streaming(
        previous_text="<think>",
        current_text="<think>step one",
        delta_text="step one",
        previous_token_ids=[think_id],
        current_token_ids=[think_id, regular_id],
        delta_token_ids=[regular_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.reasoning == "step one"
    assert result.content is None

    # End token </think> as single token — should be skipped
    result = parser.extract_reasoning_streaming(
        previous_text="<think>step one",
        current_text="<think>step one</think>",
        delta_text="</think>",
        previous_token_ids=[think_id, regular_id],
        current_token_ids=[think_id, regular_id, end_think_id],
        delta_token_ids=[end_think_id],
    )
    assert result is None

    # Content after </think>
    content_id = kimi_k2_tokenizer.encode("world", add_special_tokens=False)[0]
    result = parser.extract_reasoning_streaming(
        previous_text="<think>step one</think>",
        current_text="<think>step one</think>answer",
        delta_text="answer",
        previous_token_ids=[think_id, regular_id, end_think_id],
        current_token_ids=[think_id, regular_id, end_think_id, content_id],
        delta_token_ids=[content_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.content == "answer"


def test_streaming_tool_section_ends_reasoning(kimi_k2_tokenizer):
    """<|tool_calls_section_begin|> in delta ends reasoning during streaming."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)

    think_id = parser._start_token_id
    tool_begin_id = parser._tool_section_start_token_id
    regular_id = kimi_k2_tokenizer.encode("hello", add_special_tokens=False)[0]

    # Tool section token arrives — should transition from reasoning to content
    result = parser.extract_reasoning_streaming(
        previous_text="<think>thinking",
        current_text="<think>thinking<|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[think_id, regular_id],
        current_token_ids=[think_id, regular_id, tool_begin_id],
        delta_token_ids=[tool_begin_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.content == "<|tool_calls_section_begin|>"


def test_streaming_end_token_id_buffered(mock_kimi_k2_tokenizer):
    """When stop sequences buffer text, </think> ID arrives before its text.

    The token ID is present in delta_token_ids but the actual string is not
    yet in delta_text (still buffered). The parser must return None to wait
    for the next delta, instead of calling find() which returns -1 and
    silently corrupting the text split.
    """
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    think_id = parser._start_token_id
    end_think_id = parser._end_token_id

    # Simulate: </think> ID arrived but text not yet flushed.
    # Two token IDs in delta to bypass the single-special-token guard.
    result = parser.extract_reasoning_streaming(
        previous_text="some reasoning",
        current_text="some reasoning extra",
        delta_text="extra",  # </think> text not yet flushed
        previous_token_ids=[think_id],
        current_token_ids=[think_id, end_think_id, 999],
        delta_token_ids=[end_think_id, 999],
    )
    assert result is None


def test_streaming_tool_section_id_buffered(mock_kimi_k2_tokenizer):
    """When stop sequences buffer text, tool section start ID arrives before its text.

    Same buffering scenario as above but for <|tool_calls_section_begin|>.
    Without the guard, find() returns -1 and delta_text[:tool_index] silently
    drops the last character of reasoning.
    """
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    think_id = parser._start_token_id
    tool_begin_id = parser._tool_section_start_token_id

    result = parser.extract_reasoning_streaming(
        previous_text="some reasoning",
        current_text="some reasoning extra",
        delta_text="extra",  # tool section text not yet flushed
        previous_token_ids=[think_id],
        current_token_ids=[think_id, tool_begin_id, 999],
        delta_token_ids=[tool_begin_id, 999],
    )
    assert result is None


# DelegatingParser-level tests for the post-boundary skip that strips the
# buffered </think> text from phase B (see DelegatingParser.parse_delta).


@dataclass
class Chunk:
    """One streamed delta as the engine would produce it."""

    delta_text: str
    token_ids: list[int]


@pytest.fixture
def wrapped_kimi_parser(kimi_k2_tokenizer):
    class _KimiK2WrappedParser(_WrappedParser):
        reasoning_parser_cls = KimiK2ReasoningParser
        tool_parser_cls = KimiK2ToolParser

    return _KimiK2WrappedParser(kimi_k2_tokenizer, tools=[])


def _drive(parser, timeline, request):
    """Feed a chunk timeline through parse_delta and concatenate the deltas."""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in timeline:
        msg = parser.parse_delta(
            delta_text=chunk.delta_text,
            delta_token_ids=chunk.token_ids,
            request=request,
        )
        if msg is None:
            continue
        if msg.reasoning:
            reasoning_parts.append(msg.reasoning)
        if msg.content:
            content_parts.append(msg.content)
    return "".join(reasoning_parts), "".join(content_parts)


def _buffered_flush_timeline(parser, kimi_k2_tokenizer):
    """Boundary chunk carries the </think> id but only an unrelated byte in
    text; the buffered </think> then flushes in two later chunks."""
    start_id = parser._reasoning_parser._start_token_id
    end_id = parser._reasoning_parser._end_token_id

    def encode(s: str) -> int:
        return kimi_k2_tokenizer.encode(s, add_special_tokens=False)[0]

    return [
        Chunk("<think>", [start_id]),
        Chunk("reasoning text", [encode("reasoning text")]),
        Chunk("T", [encode("T"), end_id]),
        Chunk("</", []),
        Chunk("think>", []),
        Chunk(" answer", [encode("answer")]),
    ]


def test_delegating_parser_strips_buffered_end_token(
    wrapped_kimi_parser, kimi_k2_tokenizer
):
    """Phase B must not emit the literal </think> when its text is buffered.

    Exercises the end-to-end flow through DelegatingParser.parse_delta on
    the same scenario as test_streaming_end_token_id_buffered.
    """
    request = ChatCompletionRequest(
        model="test-model", messages=[], temperature=1.0, stream=True
    )
    timeline = _buffered_flush_timeline(wrapped_kimi_parser, kimi_k2_tokenizer)

    reasoning, content = _drive(wrapped_kimi_parser, timeline, request)

    assert "</think>" not in content, f"end-token leaked into content: {content!r}"
    assert "</think>" not in reasoning, (
        f"end-token leaked into reasoning: {reasoning!r}"
    )
    assert "answer" in content, f"post-boundary content lost: {content!r}"


def test_delegating_parser_skip_disarmed_when_parser_splits_in_one_chunk(
    wrapped_kimi_parser, kimi_k2_tokenizer
):
    """No buffered tail follows when </think> id and text arrive together;
    the skip must stay disarmed or it would over-truncate post-boundary
    content."""
    request = ChatCompletionRequest(
        model="test-model", messages=[], temperature=1.0, stream=True
    )
    start_id = wrapped_kimi_parser._reasoning_parser._start_token_id
    end_id = wrapped_kimi_parser._reasoning_parser._end_token_id

    def encode(s: str) -> int:
        return kimi_k2_tokenizer.encode(s, add_special_tokens=False)[0]

    timeline = [
        Chunk("<think>", [start_id]),
        Chunk("thinking", [encode("thinking")]),
        Chunk("</think>answer", [end_id, encode("answer")]),
    ]

    _, content = _drive(wrapped_kimi_parser, timeline, request)

    assert "answer" in content, f"post-boundary content swallowed by skip: {content!r}"
    assert not wrapped_kimi_parser._stream_state.end_token_skip_armed
    assert not wrapped_kimi_parser._stream_state.end_token_skip_done


def test_delegating_parser_boundary_at_end_of_delta_text(
    wrapped_kimi_parser, kimi_k2_tokenizer
):
    """When </think> lands at the very end of delta_text with no content
    after, the parser returns DeltaMessage(reasoning=..., content=None).
    The skip must still stay disarmed: the boundary text was already
    visible, so no buffered tail follows. Otherwise the skip would wait
    for </think> to appear again on subsequent chunks (it won't) and
    suppress all post-boundary content."""
    request = ChatCompletionRequest(
        model="test-model", messages=[], temperature=1.0, stream=True
    )
    start_id = wrapped_kimi_parser._reasoning_parser._start_token_id
    end_id = wrapped_kimi_parser._reasoning_parser._end_token_id

    def encode(s: str) -> int:
        return kimi_k2_tokenizer.encode(s, add_special_tokens=False)[0]

    timeline = [
        Chunk("<think>", [start_id]),
        Chunk("thinking", [encode("thinking")]),
        # Boundary chunk: </think> at end, nothing after it.
        Chunk("</think>", [end_id]),
        # Real content arrives in the next chunk.
        Chunk("answer", [encode("answer")]),
    ]

    _, content = _drive(wrapped_kimi_parser, timeline, request)

    assert "answer" in content, (
        f"post-boundary content suppressed when boundary lands at end of "
        f"delta_text: {content!r}"
    )
    assert "</think>" not in content


def test_delegating_parser_handles_buffered_tool_section_end(
    wrapped_kimi_parser, kimi_k2_tokenizer
):
    """Kimi K2 ends reasoning implicitly via <|tool_calls_section_begin|>.
    Under stop-sequence buffering the section-start id arrives ahead of
    its rendered text, so the skip must wait for the section literal to
    appear (not </think>, which never arrives in this flow). The literal
    must be preserved in the cleaned text passed to the tool parser:
    KimiK2ToolParser uses it to identify the section boundary, and
    without it the post-section payload leaks as raw content (matching
    the immediate-text path in KimiK2ReasoningParser, which keeps the
    literal via `content = delta_text[tool_index:]`)."""
    request = ChatCompletionRequest(
        model="test-model", messages=[], temperature=1.0, stream=True
    )
    start_id = wrapped_kimi_parser._reasoning_parser._start_token_id
    tool_section_id = wrapped_kimi_parser._reasoning_parser._tool_section_start_token_id

    def encode(s: str) -> int:
        return kimi_k2_tokenizer.encode(s, add_special_tokens=False)[0]

    # Boundary chunk: tool-section id arrives but its rendered text
    # (<|tool_calls_section_begin|>) is still buffered. The buffered text
    # then flushes across two later chunks; the in-section payload
    # follows.
    timeline = [
        Chunk("<think>", [start_id]),
        Chunk("reasoning", [encode("reasoning")]),
        Chunk("X", [encode("X"), tool_section_id]),
        Chunk("<|tool_calls_section", []),
        Chunk("_begin|>", []),
        Chunk("payload", [encode("payload")]),
    ]

    _, content = _drive(wrapped_kimi_parser, timeline, request)

    # The in-section payload must NOT leak as raw content: with the
    # tool-section literal preserved, KimiK2ToolParser handles everything
    # past the literal as tool-call territory rather than free content.
    # If the skip stripped the literal, the tool parser wouldn't see the
    # section start and "payload" would leak as raw content.
    assert "payload" not in content, (
        f"in-section payload leaked as content because the skip stripped "
        f"the tool-section literal: {content!r}"
    )
    # Marker must still be visible to the tool parser, i.e. retained in
    # the cumulative state. Without it the tool parser can't identify
    # tool calls.
    assert (
        "<|tool_calls_section_begin|>"
        in wrapped_kimi_parser._stream_state.previous_text
    ), "tool-section marker not preserved for downstream tool parser"


def test_delegating_parser_recovers_partial_end_token_in_boundary_chunk(
    wrapped_kimi_parser, kimi_k2_tokenizer
):
    """The boundary chunk's delta_text can end mid-`</think>` (e.g. the
    output_text_buffer flushed `"reasoning</th"` on the chunk where the
    end-token id arrives). With the buffer initialized to `""` on arm,
    the `</th` prefix is lost and the next chunk's `"ink>..."` never
    matches `</think>` — the skip suppresses every subsequent chunk
    forever and the response truncates at the boundary."""
    request = ChatCompletionRequest(
        model="test-model", messages=[], temperature=1.0, stream=True
    )
    start_id = wrapped_kimi_parser._reasoning_parser._start_token_id
    end_id = wrapped_kimi_parser._reasoning_parser._end_token_id

    def encode(s: str) -> int:
        return kimi_k2_tokenizer.encode(s, add_special_tokens=False)[0]

    # Boundary chunk carries `</th` (partial prefix of `</think>`) plus
    # the end-token id. Subsequent chunks finish flushing `</think>` then
    # emit content.
    timeline = [
        Chunk("<think>", [start_id]),
        Chunk("reasoning", [encode("reasoning")]),
        Chunk("</th", [end_id]),
        Chunk("ink>", []),
        Chunk(" answer", [encode("answer")]),
    ]

    _, content = _drive(wrapped_kimi_parser, timeline, request)

    assert "answer" in content, (
        f"post-boundary content suppressed when </think> spans the "
        f"boundary chunk: {content!r}"
    )
    assert "</think>" not in content, (
        f"partial end-token leaked into content: {content!r}"
    )
