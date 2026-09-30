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
TEXT = 42  # stand-in for an ordinary reasoning/content token

VOCAB = {
    THINK_START: THINK_S,
    THINK_END: THINK_E,
    TOOL_CALL_START: TOOL_S,
    TOOL_CALL_END: TOOL_E,
    "<|assistant|>": ASSISTANT,
    "<|observation|>": OBSERVATION,
    "<|user|>": USER,
}


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
