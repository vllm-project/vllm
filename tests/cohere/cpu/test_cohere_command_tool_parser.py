# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from transformers import AutoTokenizer

from vllm.tool_parsers.cohere_command_tool_parser import (
    CohereCommand3ToolParser,
    CohereCommand4ToolParser,
)


@dataclass
class ExpectedToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolCallCase:
    parser_cls: Any
    model_output: str
    expected_tool_calls: list[ExpectedToolCall] = field(default_factory=list)
    expected_reasoning: str | None = None
    expected_content: str | None = None


TOOL_CALL_CASES = [
    pytest.param(
        ToolCallCase(
            parser_cls=CohereCommand3ToolParser,
            model_output="""\
<|START_THINKING|> i will call foo with query1<|END_THINKING|><|START_ACTION|>
[
    {"tool_call_id": "0", "tool_name": "foo", "parameters": {"query": "query1"}}
]
<|END_ACTION|>""",
            expected_tool_calls=[
                ExpectedToolCall(id="0", name="foo", arguments={"query": "query1"}),
            ],
            expected_reasoning="i will call foo with query1",
        ),
        id="cmd3-single_tool_call",
    ),
    pytest.param(
        ToolCallCase(
            parser_cls=CohereCommand4ToolParser,
            model_output="""\
<|START_THINKING|> i will call foo with query1<|END_THINKING|><|START_ACTION|>
[
    {"tool_call_id": "0", "tool_name": "foo", "parameters": {"query": "query1"}}
]
<|END_ACTION|>""",
            expected_tool_calls=[
                ExpectedToolCall(id="0", name="foo", arguments={"query": "query1"}),
            ],
            expected_reasoning="i will call foo with query1",
        ),
        id="cmd4-single_tool_call",
    ),
    pytest.param(
        ToolCallCase(
            parser_cls=CohereCommand3ToolParser,
            model_output="""\
<|START_THINKING|>This is a rainbow <co>emoji: 🌈</co: 0:[1]><|END_THINKING|>
<|START_RESPONSE|>foo <co>bar</co: 0:[1,2],1:[3,4]><|END_RESPONSE|>""",
            expected_reasoning="This is a rainbow emoji: 🌈",
            expected_content="foo bar",
        ),
        id="cmd3-citations_no_tool_calls",
    ),
    pytest.param(
        ToolCallCase(
            parser_cls=CohereCommand4ToolParser,
            model_output="""\
<|START_THINKING|>This is a rainbow <co>emoji: 🌈</co: 0:[1]><|END_THINKING|>
<|START_RESPONSE|>foo <co>bar</co: 0:[1,2],1:[3,4]><|END_RESPONSE|>""",
            expected_reasoning="This is a rainbow emoji: 🌈",
            expected_content="foo bar",
        ),
        id="cmd4-citations_no_tool_calls",
    ),
    pytest.param(
        ToolCallCase(
            parser_cls=CohereCommand3ToolParser,
            model_output="""\
<|START_THINKING|>first I think about foo<|END_THINKING|><|START_ACTION|>
[
    {"tool_call_id": "0", "tool_name": "foo", "parameters": {"query": "query1"}},
    {"tool_call_id": "1", "tool_name": "bar", "parameters": {"x": 42}}
]
<|END_ACTION|>""",
            expected_tool_calls=[
                ExpectedToolCall(id="0", name="foo", arguments={"query": "query1"}),
                ExpectedToolCall(id="1", name="bar", arguments={"x": 42}),
            ],
            expected_reasoning="first I think about foo",
        ),
        id="cmd3-multiple_tool_calls",
    ),
    pytest.param(
        ToolCallCase(
            parser_cls=CohereCommand4ToolParser,
            model_output="""\
<|START_THINKING|>first I think about foo<|END_THINKING|><|START_ACTION|>
[
    {"tool_call_id": "0", "tool_name": "foo", "parameters": {"query": "query1"}},
    {"tool_call_id": "1", "tool_name": "bar", "parameters": {"x": 42}}
]
<|END_ACTION|>""",
            expected_tool_calls=[
                ExpectedToolCall(id="0", name="foo", arguments={"query": "query1"}),
                ExpectedToolCall(id="1", name="bar", arguments={"x": 42}),
            ],
            expected_reasoning="first I think about foo",
        ),
        id="cmd4-multiple_tool_calls",
    ),
    pytest.param(
        ToolCallCase(
            parser_cls=CohereCommand3ToolParser,
            model_output="""\
<|START_THINKING|>just think, no response<|END_THINKING|>""",
            expected_reasoning="just think, no response",
        ),
        id="cmd3-reasoning_only",
    ),
    pytest.param(
        ToolCallCase(
            parser_cls=CohereCommand4ToolParser,
            model_output="""\
<|START_THINKING|>just think, no response<|END_THINKING|>""",
            expected_reasoning="just think, no response",
        ),
        id="cmd4-reasoning_only",
    ),
]


@pytest.fixture(scope="module")
def tokenizer():
    # use command-a-reasoning-08-2025 for both cmd3 and cmd4 for tests
    # update when cmd4 is open sourced
    return AutoTokenizer.from_pretrained("CohereLabs/command-a-reasoning-08-2025")


REPLACEMENT_CHAR = "\ufffd"


def _token_deltas(tokenizer, text: str) -> list[str]:
    """Progressively decode the token sequence and return per-step string
    deltas.  Incomplete multi-byte sequences (trailing U+FFFD) are buffered
    until the next token completes them, matching real streaming behaviour."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    deltas: list[str] = []
    prev = ""
    for i in range(1, len(ids) + 1):
        current = tokenizer.decode(ids[:i], skip_special_tokens=False)
        if current.endswith(REPLACEMENT_CHAR):
            continue
        delta = current[len(prev) :]
        if delta:
            deltas.append(delta)
        prev = current
    return deltas


@dataclass
class StreamingResult:
    tool_calls: dict[int, dict]
    reasoning: str | None
    content: str | None


def _run_streaming(parser, tokenizer, model_output: str) -> StreamingResult:
    """Run streaming extraction and return the accumulated tool calls,
    reasoning text, and content text produced across all deltas."""
    token_strings = _token_deltas(tokenizer, model_output)
    accumulated: dict[int, dict] = {}
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    previous_text = ""

    for token_str in token_strings:
        current_text = previous_text + token_str
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=token_str,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,  # type: ignore[arg-type]
        )
        if delta is not None:
            if delta.reasoning is not None:
                reasoning_parts.append(delta.reasoning)
            if delta.content is not None:
                content_parts.append(delta.content)
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in accumulated:
                    accumulated[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.id:
                    accumulated[idx]["id"] = tc.id
                if tc.function and tc.function.name:
                    accumulated[idx]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    accumulated[idx]["arguments"] += tc.function.arguments
        previous_text = current_text

    return StreamingResult(
        tool_calls=accumulated,
        reasoning="".join(reasoning_parts) if reasoning_parts else None,
        content="".join(content_parts) if content_parts else None,
    )


@pytest.mark.parametrize("case", TOOL_CALL_CASES)
class TestExtractToolCalls:
    def test_streaming(self, tokenizer, case: ToolCallCase):
        """Streaming extraction should yield the expected tool call deltas."""
        parser = case.parser_cls(tokenizer)
        streamed = _run_streaming(parser, tokenizer, case.model_output)

        assert len(streamed.tool_calls) == len(case.expected_tool_calls)

        for i, expected_tc in enumerate(case.expected_tool_calls):
            tc = streamed.tool_calls[i]
            assert tc["id"] == expected_tc.id
            assert tc["name"] == expected_tc.name
            assert json.loads(tc["arguments"]) == expected_tc.arguments

    def test_streaming_reasoning(self, tokenizer, case: ToolCallCase):
        """Streaming extraction should also emit reasoning deltas that match
        the non-streaming reasoning output."""
        parser = case.parser_cls(tokenizer)
        streamed = _run_streaming(parser, tokenizer, case.model_output)

        assert streamed.reasoning == case.expected_reasoning

    def test_streaming_content(self, tokenizer, case: ToolCallCase):
        """Streaming extraction should emit content deltas only when the
        model output contains a response block."""
        parser = case.parser_cls(tokenizer)
        streamed = _run_streaming(parser, tokenizer, case.model_output)

        assert streamed.content == case.expected_content

    def test_nonstreaming(self, tokenizer, case: ToolCallCase):
        """Non-streaming extraction should parse the action block and return
        the correct tool calls."""
        parser = case.parser_cls(tokenizer)
        result = parser.extract_tool_calls(
            case.model_output,
            request=None,  # type: ignore[arg-type]
        )

        assert result.tools_called == (len(case.expected_tool_calls) > 0)
        assert len(result.tool_calls) == len(case.expected_tool_calls)

        for actual_tc, expected_tc in zip(result.tool_calls, case.expected_tool_calls):
            assert actual_tc.type == "function"
            assert actual_tc.function.name == expected_tc.name
            assert json.loads(actual_tc.function.arguments) == expected_tc.arguments

    def test_streaming_nonstreaming_agree(self, tokenizer, case: ToolCallCase):
        """Verify streaming and non-streaming extraction produce the same
        tool calls."""
        parser_streaming = case.parser_cls(tokenizer)
        parser_nonstreaming = case.parser_cls(tokenizer)

        streamed = _run_streaming(parser_streaming, tokenizer, case.model_output)
        result = parser_nonstreaming.extract_tool_calls(
            case.model_output,
            request=None,  # type: ignore[arg-type]
        )

        assert len(streamed.tool_calls) == len(result.tool_calls)

        for i, actual_tc in enumerate(result.tool_calls):
            assert streamed.tool_calls[i]["name"] == actual_tc.function.name
            assert json.loads(streamed.tool_calls[i]["arguments"]) == json.loads(
                actual_tc.function.arguments
            )


class TestStreamingDeltaShape:
    """Fine-grained checks for the shape of individual streaming deltas,
    rather than just the accumulated result."""

    @pytest.mark.parametrize(
        "parser_cls",
        [CohereCommand3ToolParser, CohereCommand4ToolParser],
        ids=["cmd3", "cmd4"],
    )
    def test_reasoning_and_tool_calls_are_separate_deltas(self, tokenizer, parser_cls):
        """Reasoning text and tool call chunks must never be emitted on the
        same delta: a single ``DeltaMessage`` should carry exactly one of
        reasoning / content / tool_calls."""
        parser = parser_cls(tokenizer)
        model_output = (
            "<|START_THINKING|> i will call foo with query1<|END_THINKING|>"
            "<|START_ACTION|>\n"
            '[\n    {"tool_call_id": "0", "tool_name": "foo", '
            '"parameters": {"query": "query1"}}\n]\n'
            "<|END_ACTION|>"
        )

        token_strings = _token_deltas(tokenizer, model_output)
        previous_text = ""
        saw_reasoning = False
        saw_tool_call = False

        for token_str in token_strings:
            current_text = previous_text + token_str
            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=token_str,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=None,  # type: ignore[arg-type]
            )
            if delta is not None:
                populated = [
                    delta.content is not None,
                    delta.reasoning is not None,
                    bool(delta.tool_calls),
                ]
                assert sum(populated) == 1, (
                    "A single streaming delta must carry exactly one of "
                    f"content/reasoning/tool_calls, got {delta!r}"
                )
                if delta.reasoning is not None:
                    saw_reasoning = True
                if delta.tool_calls:
                    saw_tool_call = True
            previous_text = current_text

        assert saw_reasoning, "expected at least one reasoning delta"
        assert saw_tool_call, "expected at least one tool-call delta"
