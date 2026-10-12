# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning-boundary tests for the GLM-4.7 parser.

GLM can transition straight from reasoning into a tool call without ever
emitting ``</think>``, so ``<tool_call>`` is an implicit reasoning
terminator. These cover that path plus the multi-turn prompts where an
earlier turn's markers must not be read as the current turn's state.
"""

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.glm47_moe import (
    THINK_END,
    THINK_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Glm47MoeParser,
)

THINK_S, THINK_E = 1000, 1001
TOOL_S, TOOL_E = 1002, 1003
ASSISTANT, OBSERVATION, USER = 1004, 1005, 1006
OBSERVATION_TEXT = "<|observation|>"
TEXT = 42  # stand-in for an ordinary reasoning/content token

# GLM-5 ordinary-token pieces captured in issue #58315.
SPLIT_TOOL_VOCAB = {
    "<": 27,
    "tool": 14163,
    "_call": 13420,
    ">": 29,
    "get": 455,
    "_current": 11075,
    "_time": 3009,
    "</": 522,
}
SPLIT_TOOL_IDS = [
    27,
    14163,
    13420,
    29,
    455,
    11075,
    3009,
    522,
    14163,
    13420,
    29,
    154829,  # <|observation|>
]
ATOMIC_TOOL_IDS = [154843, 455, 11075, 3009, 154844, 154829]
SPLIT_TOOL_TEXT = "<tool_call>get_current_time</tool_call>"

VOCAB = {
    THINK_START: THINK_S,
    THINK_END: THINK_E,
    TOOL_CALL_START: TOOL_S,
    TOOL_CALL_END: TOOL_E,
    "<|assistant|>": ASSISTANT,
    "<|observation|>": OBSERVATION,
    "<|user|>": USER,
}


def _make_glm5_tokenizer():
    return make_mock_tokenizer(
        {
            **VOCAB,
            **SPLIT_TOOL_VOCAB,
            TOOL_CALL_START: 154843,
            TOOL_CALL_END: 154844,
            OBSERVATION_TEXT: 154829,
        },
        special_tokens=list(VOCAB),
    )


def _make_glm5_parser(tokenizer, request, tool_choice, *, skip_reasoning_parsing=False):
    request.tool_choice = tool_choice
    request.tools = [
        ChatCompletionToolsParam(
            function={
                "name": "get_current_time",
                "parameters": {"type": "object", "properties": {}},
            }
        )
    ]
    parser = Glm47MoeParser(tokenizer)
    parser.skip_reasoning_parsing = skip_reasoning_parsing
    parser.initialize_streaming(
        initial_state=(ParserState.CONTENT if skip_reasoning_parsing else None)
    )
    return parser


def _parse_one_token_at_a_time(parser, tokenizer, request, token_ids):
    deltas = []
    for index, token_id in enumerate(token_ids):
        delta = parser.parse_delta(
            tokenizer.decode([token_id]),
            [token_id],
            request,
            finished=index == len(token_ids) - 1,
        )
        if delta is not None:
            deltas.append(delta)
    return deltas


@pytest.fixture
def parser():
    return Glm47MoeParser(make_mock_tokenizer(VOCAB))


@pytest.fixture
def no_thinking_parser():
    return Glm47MoeParser(
        make_mock_tokenizer(VOCAB),
        chat_template_kwargs={"thinking": False},
    )


class TestIsReasoningEnd:
    def test_open_reasoning(self, parser):
        assert not parser.is_reasoning_end([THINK_S, TEXT])

    def test_think_end(self, parser):
        assert parser.is_reasoning_end([THINK_S, TEXT, THINK_E, TEXT])

    def test_tool_call_without_think_end(self, parser):
        """Reasoning -> tool call with no ``</think>`` still ends reasoning."""
        assert parser.is_reasoning_end([THINK_S, TEXT, TOOL_S, TEXT])

    def test_previous_turn_tool_call_ignored(self, parser):
        """A finished tool call from an earlier turn says nothing about the
        turn currently being generated."""
        prompt = [THINK_S, TEXT, TOOL_S, TEXT, TOOL_E, OBSERVATION, TEXT, ASSISTANT]
        assert not parser.is_reasoning_end(prompt)

    def test_previous_turn_think_end_ignored(self, parser):
        prompt = [THINK_S, TEXT, THINK_E, TEXT, USER, TEXT, ASSISTANT]
        assert not parser.is_reasoning_end(prompt)

    def test_empty_input(self, parser):
        assert not parser.is_reasoning_end([])

    def test_thinking_disabled(self, no_thinking_parser):
        assert no_thinking_parser.is_reasoning_end([THINK_S, TEXT])


class TestExtractContentIds:
    def test_think_end_wins_over_later_tool_call(self, parser):
        """Content between ``</think>`` and a tool call must survive."""
        ids = [THINK_S, TEXT, THINK_E, 20, 21, TOOL_S, 30, TOOL_E]
        assert parser.extract_content_ids(ids) == [20, 21, TOOL_S, 30, TOOL_E]

    def test_every_tool_call_after_think_end_kept(self, parser):
        ids = [THINK_E, 20, TOOL_S, 30, TOOL_E, 21, TOOL_S, 31, TOOL_E]
        assert parser.extract_content_ids(ids) == ids[1:]

    def test_falls_back_to_tool_call(self, parser):
        """Without ``</think>``, content starts at the opener itself so the
        tool parser still receives a well-formed call."""
        ids = [THINK_S, TEXT, TOOL_S, 30, TOOL_E]
        assert parser.extract_content_ids(ids) == [TOOL_S, 30, TOOL_E]

    def test_falls_back_to_first_tool_call_of_turn(self, parser):
        ids = [THINK_S, TEXT, TOOL_S, 30, TOOL_E, TOOL_S, 31, TOOL_E]
        assert parser.extract_content_ids(ids) == ids[2:]

    def test_previous_turn_tool_call_ignored(self, parser):
        ids = [TOOL_S, 30, TOOL_E, OBSERVATION, TEXT, ASSISTANT, THINK_S, TEXT]
        assert parser.extract_content_ids(ids) == ids

    def test_no_markers_returns_input_ids(self, parser):
        assert parser.extract_content_ids([20, 21]) == [20, 21]

    def test_thinking_disabled(self, no_thinking_parser):
        ids = [THINK_S, TEXT, TOOL_S]
        assert no_thinking_parser.extract_content_ids(ids) == ids


class TestRequiredToolChoice:
    @pytest.mark.parametrize(
        ("tool_choice", "token_ids", "expected_tool_call"),
        [
            pytest.param("required", SPLIT_TOOL_IDS, True, id="required-split-markers"),
            pytest.param("auto", SPLIT_TOOL_IDS, False, id="auto-split-markers"),
            pytest.param(
                "auto", ATOMIC_TOOL_IDS, True, id="auto-special-token-markers"
            ),
        ],
    )
    def test_tool_marker_encoding_follows_tool_choice(
        self, mock_request, tool_choice, token_ids, expected_tool_call
    ):
        tokenizer = _make_glm5_tokenizer()
        assert tokenizer.decode(token_ids) == f"{SPLIT_TOOL_TEXT}{OBSERVATION_TEXT}"
        parser = _make_glm5_parser(tokenizer, mock_request, tool_choice)

        output_ids = [THINK_S, TEXT, THINK_E, *token_ids]
        deltas = _parse_one_token_at_a_time(parser, tokenizer, mock_request, output_ids)

        tool_calls = [
            tool_call for delta in deltas for tool_call in (delta.tool_calls or [])
        ]
        content = "".join(delta.content or "" for delta in deltas)
        assert bool(tool_calls) == expected_tool_call
        if expected_tool_call:
            assert tool_calls[0].function.name == "get_current_time"
            assert content == ""
        else:
            assert content == SPLIT_TOOL_TEXT

    def test_required_choice_keeps_split_markers_in_reasoning(self, mock_request):
        tokenizer = _make_glm5_tokenizer()
        parser = _make_glm5_parser(tokenizer, mock_request, "required")

        output_ids = [THINK_S, TEXT, *SPLIT_TOOL_IDS[:-1], THINK_E, 154829]
        deltas = _parse_one_token_at_a_time(parser, tokenizer, mock_request, output_ids)

        tool_calls = [
            tool_call for delta in deltas for tool_call in (delta.tool_calls or [])
        ]
        reasoning = "".join(delta.reasoning or "" for delta in deltas)
        assert tool_calls == []
        assert reasoning == f"*{SPLIT_TOOL_TEXT}"

    def test_required_choice_waits_for_reasoning_boundary_when_skipped(
        self, mock_request
    ):
        tokenizer = _make_glm5_tokenizer()
        parser = _make_glm5_parser(
            tokenizer, mock_request, "required", skip_reasoning_parsing=True
        )
        output_ids = [THINK_S, *SPLIT_TOOL_IDS[:-1], THINK_E, *SPLIT_TOOL_IDS]
        deltas = _parse_one_token_at_a_time(parser, tokenizer, mock_request, output_ids)

        tool_calls = [
            tool_call for delta in deltas for tool_call in (delta.tool_calls or [])
        ]
        content = "".join(delta.content or "" for delta in deltas)
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_current_time"
        assert content == f"{THINK_START}{SPLIT_TOOL_TEXT}{THINK_END}"
