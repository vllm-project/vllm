# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool-call parsing for the unified Cohere Command parser.

The ``cohere_command3`` / ``cohere_command4`` tool parsers are registry shims;
``CohereCommandParser`` (resolved through ``ParserManager``) does the parsing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import Parser

from .utils import MockCohereTokenizer, drive_parser, make_parser, token_deltas

SPECIAL_TOKEN_MARKERS = (
    "<|START_THINKING|>",
    "<|END_THINKING|>",
    "<|START_RESPONSE|>",
    "<|END_RESPONSE|>",
    "<|START_ACTION|>",
    "<|END_ACTION|>",
    "<|START_TEXT|>",
    "<|END_TEXT|>",
)

DUMMY_TOOLS = [
    {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
    for name in ("foo", "bar")
]

# (registry name, answer framing tag)
VARIANTS = (("cohere_command3", "RESPONSE"), ("cohere_command4", "TEXT"))


@dataclass
class ExpectedToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolCallCase:
    parser_name: str
    model_output: str
    expected_tool_calls: list[ExpectedToolCall] = field(default_factory=list)
    expected_reasoning: str | None = None
    expected_content: str | None = None


SINGLE_TOOL_CALL = """\
<|START_THINKING|> i will call foo with query1<|END_THINKING|><|START_ACTION|>
[
    {"tool_call_id": "0", "tool_name": "foo", "parameters": {"query": "query1"}}
]
<|END_ACTION|>"""
MULTI_TOOL_CALL = """\
<|START_THINKING|>first I think about foo<|END_THINKING|><|START_ACTION|>
[
    {"tool_call_id": "0", "tool_name": "foo", "parameters": {"query": "query1"}},
    {"tool_call_id": "1", "tool_name": "bar", "parameters": {"x": 42}}
]
<|END_ACTION|>"""
CITATIONS_NO_TOOL_CALL = (
    "<|START_THINKING|>This is a rainbow <co>emoji: 🌈</co: 0:[1]><|END_THINKING|>\n"
    "<|START_{tag}|>foo <co>bar</co: 0:[1,2],1:[3,4]><|END_{tag}|>"
)

# (case id, model output template, expectations)
_CASE_TABLE = (
    (
        "single_tool_call",
        SINGLE_TOOL_CALL,
        dict(
            expected_tool_calls=[ExpectedToolCall("0", "foo", {"query": "query1"})],
            expected_reasoning="i will call foo with query1",
        ),
    ),
    (
        "citations_no_tool_calls",
        CITATIONS_NO_TOOL_CALL,
        dict(
            expected_reasoning="This is a rainbow emoji: 🌈", expected_content="foo bar"
        ),
    ),
    (
        "multiple_tool_calls",
        MULTI_TOOL_CALL,
        dict(
            expected_tool_calls=[
                ExpectedToolCall("0", "foo", {"query": "query1"}),
                ExpectedToolCall("1", "bar", {"x": 42}),
            ],
            expected_reasoning="first I think about foo",
        ),
    ),
    (
        "reasoning_only",
        "<|START_THINKING|>just think, no response<|END_THINKING|>",
        dict(expected_reasoning="just think, no response"),
    ),
)

TOOL_CALL_CASES = [
    pytest.param(
        ToolCallCase(name, output.replace("{tag}", tag), **expected),
        id=f"{name}-{case_id}",
    )
    for name, tag in VARIANTS
    for case_id, output, expected in _CASE_TABLE
]

PARSER_NAMES = pytest.mark.parametrize("parser_name", [name for name, _ in VARIANTS])


def _tool_parser(tokenizer: MockCohereTokenizer, parser_name: str) -> Parser:
    return make_parser(tokenizer, parser_name, DUMMY_TOOLS)


def _tool_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[], model="test-model", tools=DUMMY_TOOLS, tool_choice="auto"
    )


@dataclass
class StreamingResult:
    tool_calls: dict[int, dict]
    reasoning: str | None
    content: str | None


def run_streaming(
    tokenizer: MockCohereTokenizer,
    parser_name: str,
    model_output: str,
    chunk_size: int = 1,
) -> StreamingResult:
    deltas = drive_parser(
        _tool_parser(tokenizer, parser_name),
        _tool_request(),
        token_deltas(tokenizer, model_output, chunk_size),
    )
    accumulated: dict[int, dict] = {}
    for delta in deltas:
        for tc in delta.tool_calls or []:
            slot = accumulated.setdefault(
                tc.index, {"id": "", "name": "", "arguments": ""}
            )
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
    return StreamingResult(
        tool_calls=accumulated,
        reasoning="".join(d.reasoning for d in deltas if d.reasoning) or None,
        content="".join(d.content for d in deltas if d.content) or None,
    )


def run_nonstreaming(
    tokenizer: MockCohereTokenizer, parser_name: str, model_output: str
) -> tuple[str | None, str | None, list]:
    reasoning, content, tool_calls = _tool_parser(tokenizer, parser_name).parse(
        model_output, _tool_request(), enable_auto_tools=True
    )
    return reasoning, content, tool_calls or []


@pytest.mark.parametrize("case", TOOL_CALL_CASES)
class TestExtractToolCalls:
    def test_streaming(self, tokenizer: MockCohereTokenizer, case: ToolCallCase):
        streamed = run_streaming(tokenizer, case.parser_name, case.model_output)
        assert len(streamed.tool_calls) == len(case.expected_tool_calls)
        for i, expected in enumerate(case.expected_tool_calls):
            tc = streamed.tool_calls[i]
            assert tc["id"] == expected.id
            assert tc["name"] == expected.name
            assert json.loads(tc["arguments"]) == expected.arguments

    def test_streaming_reasoning(
        self, tokenizer: MockCohereTokenizer, case: ToolCallCase
    ):
        streamed = run_streaming(tokenizer, case.parser_name, case.model_output)
        assert streamed.reasoning == case.expected_reasoning

    def test_streaming_content(
        self, tokenizer: MockCohereTokenizer, case: ToolCallCase
    ):
        streamed = run_streaming(tokenizer, case.parser_name, case.model_output)
        assert streamed.content == case.expected_content

    def test_nonstreaming(self, tokenizer: MockCohereTokenizer, case: ToolCallCase):
        reasoning, content, tool_calls = run_nonstreaming(
            tokenizer, case.parser_name, case.model_output
        )
        assert reasoning == case.expected_reasoning
        assert content == case.expected_content
        assert len(tool_calls) == len(case.expected_tool_calls)
        for actual, expected in zip(tool_calls, case.expected_tool_calls):
            assert actual.name == expected.name
            assert json.loads(actual.arguments) == expected.arguments

    def test_streaming_nonstreaming_agree(
        self, tokenizer: MockCohereTokenizer, case: ToolCallCase
    ):
        streamed = run_streaming(tokenizer, case.parser_name, case.model_output)
        _, _, tool_calls = run_nonstreaming(
            tokenizer, case.parser_name, case.model_output
        )
        assert len(streamed.tool_calls) == len(tool_calls)
        for i, actual in enumerate(tool_calls):
            assert streamed.tool_calls[i]["name"] == actual.name
            assert json.loads(streamed.tool_calls[i]["arguments"]) == json.loads(
                actual.arguments
            )


class TestSpeculativeDecodingMultiTokenDelta:
    @PARSER_NAMES
    @pytest.mark.parametrize("chunk_size", [2, 3, 4, 6])
    def test_no_special_token_leak_in_streaming_deltas(
        self, tokenizer: MockCohereTokenizer, parser_name: str, chunk_size: int
    ):
        deltas = drive_parser(
            _tool_parser(tokenizer, parser_name),
            _tool_request(),
            token_deltas(tokenizer, SINGLE_TOOL_CALL, chunk_size),
        )
        for delta in deltas:
            fields: list[tuple[str, str | None]] = [
                ("reasoning", delta.reasoning),
                ("content", delta.content),
            ]
            for tc in delta.tool_calls or []:
                if tc.function:
                    fields.append(("tool_call.name", tc.function.name))
                    fields.append(("tool_call.arguments", tc.function.arguments))
            for marker in SPECIAL_TOKEN_MARKERS:
                for field_name, value in fields:
                    assert value is None or marker not in value, (
                        f"special token {marker!r} leaked into {field_name} "
                        f"with chunk_size={chunk_size} delta={delta!r}"
                    )

    @PARSER_NAMES
    @pytest.mark.parametrize("chunk_size", [2, 3, 4, 6])
    def test_multi_token_chunks_still_produce_correct_tool_call(
        self, tokenizer: MockCohereTokenizer, parser_name: str, chunk_size: int
    ):
        streamed = run_streaming(tokenizer, parser_name, SINGLE_TOOL_CALL, chunk_size)
        assert len(streamed.tool_calls) == 1
        tc = streamed.tool_calls[0]
        assert tc["id"] == "0"
        assert tc["name"] == "foo"
        assert json.loads(tc["arguments"]) == {"query": "query1"}


class TestStreamingDeltaShape:
    @PARSER_NAMES
    def test_reasoning_and_tool_calls_are_separate_deltas(
        self, tokenizer: MockCohereTokenizer, parser_name: str
    ):
        deltas = drive_parser(
            _tool_parser(tokenizer, parser_name),
            _tool_request(),
            token_deltas(tokenizer, SINGLE_TOOL_CALL),
        )
        for delta in deltas:
            populated = [
                delta.content is not None,
                delta.reasoning is not None,
                bool(delta.tool_calls),
            ]
            assert sum(populated) == 1, (
                "A single streaming delta must carry exactly one of "
                f"content/reasoning/tool_calls, got {delta!r}"
            )
        assert any(d.reasoning is not None for d in deltas)
        assert any(d.tool_calls for d in deltas)
