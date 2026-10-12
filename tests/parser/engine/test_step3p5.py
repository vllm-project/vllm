# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Step-3.5 parser.

Step-3.5 reuses the Qwen3 grammar, which ``test_qwen3.py`` covers. These tests
cover what differs: thinking is always on, the newline the model emits before
``</think>`` is dropped, prior-turn reasoning does not end the current turn,
and the ``step3p5`` names resolve to the engine adapters.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import simulate_reasoning_streaming
from vllm.parser.engine.registered_adapters import (
    Step3p5ParserReasoningAdapter,
    Step3p5ParserToolAdapter,
)
from vllm.parser.step3p5 import Step3p5Parser
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

_THINK_START_ID = 50
_THINK_END_ID = 51
_IM_START_ID = 70
_IM_END_ID = 71
_TEXT_ID = 100

_STEP3P5_VOCAB = {
    "<think>": _THINK_START_ID,
    "</think>": _THINK_END_ID,
    "<tool_call>": 60,
    "</tool_call>": 61,
    "<|im_start|>": _IM_START_ID,
    "<|im_end|>": _IM_END_ID,
}


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(_STEP3P5_VOCAB)


@pytest.fixture
def parser(mock_tokenizer):
    return Step3p5Parser(mock_tokenizer)


def test_parser_names_resolve_to_engine_adapters():
    tool_parser_cls = ToolParserManager.get_tool_parser("step3p5")
    assert issubclass(tool_parser_cls, Step3p5ParserToolAdapter)
    assert tool_parser_cls.structural_tag_model == "qwen_3_coder"
    reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser("step3p5")
    assert reasoning_parser_cls is Step3p5ParserReasoningAdapter


def test_enable_thinking_false_is_ignored(mock_tokenizer):
    # The chat template always prefills "<think>\n" and has no thinking switch.
    parser = Step3p5Parser(
        mock_tokenizer, chat_template_kwargs={"enable_thinking": False}
    )
    assert parser.extract_reasoning("plan</think>answer", None) == ("plan", "answer")
    assert not parser.is_reasoning_end([_IM_START_ID, _TEXT_ID, _THINK_START_ID])


def test_newline_before_think_end_dropped(parser):
    text = "Line one.\nLine two.\n</think>\nThe answer."
    assert parser.extract_reasoning(text, None) == (
        "Line one.\nLine two.",
        "\nThe answer.",
    )


def test_newline_before_think_end_dropped_streaming(parser):
    reasoning, content = simulate_reasoning_streaming(
        parser,
        ["Line one.\n", "Line two.\n", "</think>", "\nThe answer."],
        [(_TEXT_ID,), (_TEXT_ID,), (_THINK_END_ID,), (_TEXT_ID,)],
    )
    assert reasoning == "Line one.\nLine two."
    assert content == "\nThe answer."


class TestPriorTurnReasoning:
    """The template replays earlier assistant turns as
    ``<think>\\n...\\n</think>\\n``, so a ``</think>`` from history must not end
    reasoning for the new turn (#34211)."""

    _HISTORY = [
        _IM_START_ID,
        _TEXT_ID,
        _THINK_START_ID,
        _TEXT_ID,
        _THINK_END_ID,
        _TEXT_ID,
        _IM_END_ID,
    ]

    def test_prior_turn_think_end_not_end(self, parser):
        assert not parser.is_reasoning_end([*self._HISTORY, _IM_START_ID, _TEXT_ID])

    def test_think_end_in_current_turn_is_end(self, parser):
        assert parser.is_reasoning_end(
            [*self._HISTORY, _IM_START_ID, _THINK_START_ID, _TEXT_ID, _THINK_END_ID]
        )


def test_reasoning_then_tool_call(parser, mock_request):
    text = (
        "Need the weather.\n</think>\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>"
    )
    reasoning, content, tool_calls = parser.parse(
        text, mock_request, enable_auto_tools=True
    )
    assert reasoning == "Need the weather."
    assert content is None
    assert tool_calls is not None
    assert tool_calls[0].name == "get_weather"
    assert json.loads(tool_calls[0].arguments) == {"city": "Tokyo"}
