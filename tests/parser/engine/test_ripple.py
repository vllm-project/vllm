# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ripple parser: optional `<think>` reasoning and JSON tool calls.

The generation prompt does not open `<think>`, so output without it is the
answer. The chat template frames markers with newlines, which must not leak
into reasoning or content.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.replay_harness import (
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.parser.engine.registered_adapters import (
    RippleParserReasoningAdapter,
    RippleParserToolAdapter,
)
from vllm.parser.parser_manager import ParserManager
from vllm.parser.ripple import RippleParser
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    },
]


def _call(city: str) -> str:
    body = {"name": "get_weather", "arguments": {"city": city}}
    return f"\n<tool_call>\n{json.dumps(body)}\n</tool_call>"


# (model output, reasoning, content, tool-call cities)
CASES = {
    "direct_answer": ("17*23 = 391.", "", "17*23 = 391.", []),
    "reasoning": (
        "<think>\nCompute.\n17*23=391.\n</think>\nThe answer is 391.",
        "Compute.\n17*23=391.",
        "The answer is 391.",
        [],
    ),
    "reasoning_then_tool": (
        "<think>\nNeed the tool.\n</think>" + _call("Paris"),
        "Need the tool.",
        "",
        ["Paris"],
    ),
    "content_then_tools": (
        "Checking both." + _call("Paris") + _call("Rome"),
        "",
        "Checking both.",
        ["Paris", "Rome"],
    ),
    "truncated_reasoning": ("<think>\nStill thinking", "Still thinking", "", []),
    # Text before a tool call keeps its whitespace in both parse paths.
    "extra_newline_before_tool_content": (
        "<think>\nNeed the tool.\n</think>\n\nChecking now." + _call("Paris"),
        "Need the tool.",
        "\nChecking now.",
        ["Paris"],
    ),
}


@pytest.fixture
def tokenizer():
    return make_mock_tokenizer({"<|im_start|>": 1, "<|im_end|>": 2})


@pytest.fixture
def parser_cls():
    return ParserManager.get_parser(
        tool_parser_name="ripple",
        reasoning_parser_name="ripple",
        enable_auto_tools=True,
    )


def test_parser_names_resolve_to_engine_adapters():
    tool_parser_cls = ToolParserManager.get_tool_parser("ripple")
    assert issubclass(tool_parser_cls, RippleParserToolAdapter)
    assert tool_parser_cls.structural_tag_model == "hermes"
    assert (
        ReasoningParserManager.get_reasoning_parser("ripple")
        is RippleParserReasoningAdapter
    )


@pytest.mark.parametrize("case", CASES)
def test_parse(parser_cls, tokenizer, case):
    text, reasoning, content, cities = CASES[case]
    parsed_reasoning, parsed_content, tool_calls = parser_cls(tokenizer).parse(
        text, _test_request(TOOLS), enable_auto_tools=True
    )
    assert (parsed_reasoning or "") == reasoning
    assert (parsed_content or "") == content
    assert [json.loads(tc.arguments)["city"] for tc in tool_calls or []] == cities
    assert all(tc.name == "get_weather" for tc in tool_calls or [])


@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("case", CASES)
def test_streaming_matches_parse(parser_cls, tokenizer, case, chunk_size):
    text, reasoning, content, cities = CASES[case]
    tokens = [(ord(ch), ch) for ch in text]
    output = collect_output(
        replay_streaming(
            parser_cls(tokenizer),
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=TOOLS,
        )
    )
    assert output.reasoning == reasoning
    assert output.content == content
    assert [json.loads(tc["arguments"])["city"] for tc in output.tool_calls] == cities


@pytest.mark.parametrize("case", ["direct_answer", "reasoning"])
def test_streaming_without_tool_parser(tokenizer, case):
    # Without --enable-auto-tool-choice no tool parser is composed, so the
    # reasoning parser alone must still split out <think> while streaming.
    parser_cls = ParserManager.get_parser(reasoning_parser_name="ripple")
    assert parser_cls is not None
    text, reasoning, content, _ = CASES[case]
    output = collect_output(
        replay_streaming(
            parser_cls(tokenizer),
            [(ord(ch), ch) for ch in text],
            chunk_size=1,
            finished_on_last=True,
            prompt_token_ids=[1, ord("u"), 2, 1],
        )
    )
    assert (output.reasoning, output.content) == (reasoning, content)


def test_structured_output_applies_from_first_token(tokenizer):
    # Thinking is optional, so a constrained response must not wait for a
    # "</think>" the model may never emit.
    parser = RippleParser(tokenizer)
    assert parser.is_reasoning_end([1, ord("u"), 2, 1])
