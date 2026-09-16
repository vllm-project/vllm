# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning extraction, reasoning-end gating and parser selection for the
unified Cohere Command parser."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.reasoning.cohere_command_reasoning_parser import (
    CohereCommand3ReasoningParser,
    CohereCommand4ReasoningParser,
)

from .utils import make_parser, stream_parser


@dataclass
class ReasoningCase:
    parser_name: str
    model_output: str
    expected_reasoning: str | None
    expected_content: str | None


REASONING_CASES = [
    pytest.param(
        ReasoningCase(
            parser_name="cohere_command3",
            model_output="""\
<|START_THINKING|> i will call foo with query1<|END_THINKING|><|START_ACTION|>
[
    {"tool_call_id": "0", "tool_name": "foo", "parameters": {"query": "query1"}}
]
<|END_ACTION|>""",
            expected_reasoning="i will call foo with query1",
            expected_content=None,
        ),
        id="cmd3-single_tool_call",
    ),
    pytest.param(
        ReasoningCase(
            parser_name="cohere_command4",
            model_output="""\
<|START_THINKING|> i will call foo with query1<|END_THINKING|><|START_ACTION|>
[
    {"tool_call_id": "0", "tool_name": "foo", "parameters": {"query": "query1"}}
]
<|END_ACTION|>""",
            expected_reasoning="i will call foo with query1",
            expected_content=None,
        ),
        id="cmd4-single_tool_call",
    ),
    pytest.param(
        ReasoningCase(
            parser_name="cohere_command3",
            model_output="""\
<|START_THINKING|>This is a rainbow <co>emoji: 🌈</co: 0:[1]><|END_THINKING|>
<|START_RESPONSE|>foo <co>bar</co: 0:[1,2],1:[3,4]><|END_RESPONSE|>""",
            expected_reasoning="This is a rainbow emoji: 🌈",
            expected_content="foo bar",
        ),
        id="cmd3-citations_with_emoji",
    ),
    pytest.param(
        ReasoningCase(
            parser_name="cohere_command4",
            model_output="""\
<|START_THINKING|>This is a rainbow <co>emoji: 🌈</co: 0:[1]><|END_THINKING|>
<|START_RESPONSE|>foo <co>bar</co: 0:[1,2],1:[3,4]><|END_RESPONSE|>""",
            expected_reasoning="This is a rainbow emoji: 🌈",
            expected_content="foo bar",
        ),
        id="cmd4-citations_with_emoji",
    ),
]


@pytest.mark.parametrize("case", REASONING_CASES)
class TestExtractReasoning:
    def test_nonstreaming(self, tokenizer, request_obj, case: ReasoningCase):
        parser = make_parser(tokenizer, case.parser_name)
        reasoning, content, tool_calls = parser.parse(
            case.model_output, request_obj, enable_auto_tools=True
        )
        assert reasoning == case.expected_reasoning
        assert content == case.expected_content
        # No tools on the request, so any tool call is dropped, not surfaced.
        assert not tool_calls

    def test_streaming(self, tokenizer, request_obj, case: ReasoningCase):
        parser = make_parser(tokenizer, case.parser_name)
        deltas = stream_parser(parser, request_obj, tokenizer, case.model_output)

        reasoning = "".join(d.reasoning for d in deltas if d.reasoning) or None
        content = "".join(d.content for d in deltas if d.content) or None
        assert reasoning == case.expected_reasoning
        assert content == case.expected_content
        assert all(not d.tool_calls for d in deltas)


class TestFramingTokensStripped:
    @pytest.mark.parametrize(
        ("parser_name", "content_tags"),
        [
            pytest.param(
                "cohere_command4", ("<|START_TEXT|>", "<|END_TEXT|>"), id="cmd4"
            ),
            pytest.param(
                "cohere_command3", ("<|START_RESPONSE|>", "<|END_RESPONSE|>"), id="cmd3"
            ),
        ],
    )
    @pytest.mark.parametrize("chunk_size", [1, 4], ids=["per_token", "batched"])
    @pytest.mark.parametrize("with_tools", [False, True], ids=["no_tools", "tools"])
    def test_content_framing_tokens_stripped(
        self, tokenizer, parser_name, content_tags, chunk_size, with_tools
    ):
        parser = make_parser(tokenizer, parser_name)
        tools = [{"type": "function", "function": {"name": "foo"}}]
        request = ChatCompletionRequest(
            messages=[], model="test-model", tools=tools if with_tools else None
        )
        parser.adjust_request(request)

        start_tag, end_tag = content_tags
        generation = (
            f"<|START_THINKING|>Think deeply. The user greets us.<|END_THINKING|>"
            f"{start_tag}I'm doing well, thank you{end_tag}"
        )
        deltas = stream_parser(parser, request, tokenizer, generation, chunk_size)

        assert "".join(d.reasoning or "" for d in deltas) == (
            "Think deeply. The user greets us."
        )
        assert not any(d.tool_calls for d in deltas)
        assert "".join(d.content or "" for d in deltas) == "I'm doing well, thank you"


class TestIsReasoningEnd:
    @pytest.mark.parametrize(
        "parser_cls",
        [CohereCommand3ReasoningParser, CohereCommand4ReasoningParser],
        ids=["cmd3", "cmd4"],
    )
    def test_is_reasoning_end(self, tokenizer, parser_cls):
        parser = parser_cls(tokenizer)
        start_id = tokenizer.convert_tokens_to_ids("<|START_THINKING|>")
        end_id = tokenizer.convert_tokens_to_ids("<|END_THINKING|>")
        chatbot_id = tokenizer.convert_tokens_to_ids("<|CHATBOT_TOKEN|>")
        content_ids = [99, 100]

        # Generation-only tokens have no chatbot marker, so the whole sequence
        # is considered.
        assert parser.is_reasoning_end([end_id])
        assert parser.is_reasoning_end([start_id, *content_ids, end_id])
        assert not parser.is_reasoning_end([start_id, *content_ids])

        # Full prompt/history tokens are scoped to the latest chatbot marker,
        # so stray thinking tokens from the preamble or previous turns are ignored.
        assert not parser.is_reasoning_end([start_id, end_id, chatbot_id, *content_ids])
        assert parser.is_reasoning_end(
            [start_id, end_id, chatbot_id, start_id, *content_ids, end_id]
        )


@pytest.mark.parametrize("parser_name", ["cohere_command3", "cohere_command4"])
def test_count_reasoning_tokens(tokenizer, parser_name):
    parser = make_parser(tokenizer, parser_name)
    start = tokenizer.convert_tokens_to_ids("<|START_THINKING|>")
    end = tokenizer.convert_tokens_to_ids("<|END_THINKING|>")

    assert parser.count_reasoning_tokens([99, start, 11, 12, end, 100]) == 2
    assert parser.count_reasoning_tokens([end, start, 1, start, 2, end, 3, end]) == 3
    assert parser.count_reasoning_tokens([start, 1, 2]) == 2
    assert parser.count_reasoning_tokens([1, 2, end]) == 0


class TestParserSelection:
    @pytest.mark.parametrize("with_tool_parser", [True, False])
    def test_adjust_request_keeps_framing_tokens(self, tokenizer, with_tool_parser):
        """Melody needs the special tokens, with or without a tool parser."""
        cls = ParserManager.get_parser(
            "cohere_command4" if with_tool_parser else None,
            "cohere_command4",
            enable_auto_tools=True,
        )
        request = ChatCompletionRequest(messages=[], model="m")
        assert request.skip_special_tokens
        cls(tokenizer).adjust_request(request)
        assert request.skip_special_tokens is False

    def test_non_cohere_selection_does_not_import_melody_parser(self, monkeypatch):
        """``cohere_melody`` is optional; other parsers must not depend on it."""
        import sys

        monkeypatch.setitem(sys.modules, "vllm.parser.cohere_command", None)
        assert ParserManager.get_parser("hermes", "deepseek_r1", True) is not None
