# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based BerryLM parser
(``--reasoning-parser berrylm`` / ``--tool-call-parser berrylm``).

BerryLM emits ChatML turns with ``<think>``/``</think>`` reasoning and XML
tool calls::

    <tool_call>
    <function=get_weather>
    <parameter=city>Tokyo</parameter>
    </function>
    </tool_call>

Covered here: the reasoning grammar (``<tool_call>`` as implicit reasoning
end, a duplicate ``</think>`` absorbed, thinking disabled through
``chat_template_kwargs``, a replayed ``</think>`` of an earlier turn not
taken as the end of the current one), the tool grammar (arguments coerced to
the request's tool schema, parallel calls, the malformed-header regression,
streaming), and the registered adapters.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_function_name,
    collect_tool_arguments,
    simulate_reasoning_streaming,
    simulate_tool_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.parser.berrylm import BerryLMParser
from vllm.parser.engine.registered_adapters import (
    BerryLMParserReasoningAdapter,
    BerryLMParserToolAdapter,
)

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

_THINK_START_ID = 50
_THINK_END_ID = 51
_TOOL_CALL_START_ID = 60
_TOOL_CALL_END_ID = 61
_IM_START_ID = 70
_IM_END_ID = 71
_TEXT_ID = 100

_BERRYLM_VOCAB = {
    THINK_START: _THINK_START_ID,
    THINK_END: _THINK_END_ID,
    TOOL_CALL_START: _TOOL_CALL_START_ID,
    TOOL_CALL_END: _TOOL_CALL_END_ID,
    IM_START: _IM_START_ID,
    IM_END: _IM_END_ID,
}

WEATHER_CALL = (
    f"{TOOL_CALL_START}\n<function=get_weather>\n"
    "<parameter=city>Tokyo</parameter>\n"
    "<parameter=days>3</parameter>\n"
    f"</function>\n{TOOL_CALL_END}"
)


def _weather_tool() -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_weather",
            "description": "Weather forecast for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    )


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(_BERRYLM_VOCAB)


@pytest.fixture
def parser(mock_tokenizer):
    return BerryLMParser(mock_tokenizer)


@pytest.fixture
def tool_parser(mock_tokenizer):
    return BerryLMParser(
        mock_tokenizer, chat_template_kwargs={"enable_thinking": False}
    )


@pytest.fixture
def typed_request(mock_request):
    mock_request.tools = [_weather_tool()]
    return mock_request


def test_config_wired(parser):
    assert parser.parser_engine_config.name == "berrylm"
    assert parser.reasoning_start_str == THINK_START
    assert parser.reasoning_end_str == THINK_END
    assert parser.parser_engine_config.turn_boundary_tokens == frozenset(
        (IM_START, IM_END)
    )


class TestReasoningNonStreaming:
    def test_reasoning_then_content(self, parser):
        reasoning, content = parser.extract_reasoning(
            f"{THINK_START}Plan the answer.{THINK_END}The answer is 42.", None
        )
        assert reasoning == "Plan the answer."
        assert content == "The answer is 42."

    def test_template_opened_think(self, parser):
        """The chat template ends the prompt with ``<think>``: the generated
        text starts inside the reasoning block."""
        reasoning, content = parser.extract_reasoning(
            f"Plan the answer.{THINK_END}The answer is 42.", None
        )
        assert reasoning == "Plan the answer."
        assert content == "The answer is 42."

    def test_duplicate_think_end_absorbed(self, parser):
        _, content = parser.extract_reasoning(
            f"Plan.{THINK_END}{THINK_END}The answer is 42.", None
        )
        assert THINK_END not in content
        assert content.endswith("The answer is 42.")

    def test_tool_call_is_implicit_reasoning_end(self, parser):
        """``<tool_call>`` without ``</think>`` ends the reasoning; the tool
        call itself is the tool parser's business (see the adapter test)."""
        reasoning, _ = parser.extract_reasoning(
            f"I need the forecast.{WEATHER_CALL}", None
        )
        assert reasoning == "I need the forecast."
        assert TOOL_CALL_START not in reasoning
        assert THINK_END not in reasoning

    def test_thinking_disabled_returns_everything_as_content(self, mock_tokenizer):
        no_think = BerryLMParser(
            mock_tokenizer, chat_template_kwargs={"enable_thinking": False}
        )
        reasoning, content = no_think.extract_reasoning("The answer is 42.", None)
        assert reasoning is None
        assert content == "The answer is 42."


class TestReasoningStreaming:
    def test_reasoning_then_content(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["Plan the ", "answer.", THINK_END, "The answer", " is 42."],
            [(1,), (2,), (_THINK_END_ID,), (3,), (4,)],
        )
        assert reasoning == "Plan the answer."
        assert content == "The answer is 42."

    def test_think_end_and_tool_call_same_delta(self, parser):
        """``</think>`` and ``<tool_call>`` arriving in one delta must not leak
        either terminal into the reasoning text."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            [
                "Let me check the forecast.",
                f"{THINK_END}{TOOL_CALL_START}",
                "<function=get_weather>",
            ],
            [(1,), (_THINK_END_ID, _TOOL_CALL_START_ID), (2,)],
        )
        assert reasoning == "Let me check the forecast."
        assert THINK_END not in reasoning
        assert TOOL_CALL_START not in reasoning
        assert content is not None


class TestTurnBoundaries:
    """The chat template replays earlier assistant turns with their reasoning,
    so a ``</think>`` from a previous turn sits in the prompt ids. The backward
    walk for ``is_reasoning_end`` must stop at the ChatML turn boundary."""

    def test_replayed_reasoning_in_history_is_not_end(self, parser):
        assert not parser.is_reasoning_end(
            [
                _IM_START_ID,
                _THINK_START_ID,
                _TEXT_ID,
                _THINK_END_ID,
                _TEXT_ID,
                _IM_END_ID,
                _IM_START_ID,
            ]
        )

    def test_think_end_in_current_turn_is_end(self, parser):
        assert parser.is_reasoning_end(
            [_IM_START_ID, _TEXT_ID, _IM_END_ID, _IM_START_ID, _THINK_END_ID]
        )

    def test_thinking_disabled_is_always_end(self, tool_parser):
        assert tool_parser.is_reasoning_end([_IM_START_ID, _TEXT_ID])


class TestToolCalls:
    def test_single_call_typed_arguments(self, tool_parser, typed_request):
        result = tool_parser.extract_tool_calls(WEATHER_CALL, typed_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0].function
        assert call.name == "get_weather"
        # ``days`` is an integer in the tool schema: coerced from the XML text
        assert json.loads(call.arguments) == {"city": "Tokyo", "days": 3}

    def test_content_before_call_is_kept(self, tool_parser, typed_request):
        result = tool_parser.extract_tool_calls(
            f"Checking the forecast.\n{WEATHER_CALL}", typed_request
        )
        assert result.tools_called is True
        assert result.content is not None
        assert result.content.strip() == "Checking the forecast."

    def test_no_call(self, tool_parser, mock_request):
        result = tool_parser.extract_tool_calls("Just an answer.", mock_request)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "Just an answer."

    def test_parallel_calls(self, tool_parser, typed_request):
        text = WEATHER_CALL + (
            f"\n{TOOL_CALL_START}\n<function=get_weather>\n"
            "<parameter=city>Osaka</parameter>\n"
            f"</function>\n{TOOL_CALL_END}"
        )
        result = tool_parser.extract_tool_calls(text, typed_request)
        names = [tc.function.name for tc in result.tool_calls]
        assert names == ["get_weather", "get_weather"]
        args = [json.loads(tc.function.arguments) for tc in result.tool_calls]
        assert args[0] == {"city": "Tokyo", "days": 3}
        assert args[1] == {"city": "Osaka"}

    def test_multiline_value(self, tool_parser, mock_request):
        text = (
            f"{TOOL_CALL_START}\n<function=write_file>\n"
            "<parameter=path>/tmp/x.py</parameter>\n"
            "<parameter=content>\nline 1\nline 2\n</parameter>\n"
            f"</function>\n{TOOL_CALL_END}"
        )
        result = tool_parser.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"path": "/tmp/x.py", "content": "line 1\nline 2"}

    def test_malformed_function_end_does_not_drop_siblings(
        self, tool_parser, typed_request
    ):
        text = (
            f"{TOOL_CALL_START}\n<function=broken</function>\n{TOOL_CALL_END}"
            + WEATHER_CALL
        )
        result = tool_parser.extract_tool_calls(text, typed_request)
        weather = next(
            tc for tc in result.tool_calls if tc.function.name == "get_weather"
        )
        assert json.loads(weather.function.arguments) == {"city": "Tokyo", "days": 3}

    def test_basic_streaming(self, tool_parser, typed_request):
        chunks = [
            f"{TOOL_CALL_START}\n",
            "<function=get_weather>\n",
            "<parameter=city>Tok",
            "yo</parameter>\n",
            "<parameter=days>3</parameter>\n",
            "</function>\n",
            f"{TOOL_CALL_END}",
        ]
        results = simulate_tool_streaming(tool_parser, typed_request, chunks)
        assert collect_function_name(results) == "get_weather"
        assert json.loads(collect_tool_arguments(results)) == {
            "city": "Tokyo",
            "days": 3,
        }


def test_end_to_end_through_registered_adapters(mock_tokenizer, typed_request):
    reasoning_parser = BerryLMParserReasoningAdapter(mock_tokenizer)
    tool_parser = BerryLMParserToolAdapter(mock_tokenizer)
    text = f"{THINK_START}Plan the call.{THINK_END}{WEATHER_CALL}"

    reasoning, remaining = reasoning_parser.extract_reasoning(text, typed_request)
    assert reasoning == "Plan the call."

    tool_result = tool_parser.extract_tool_calls(remaining, typed_request)
    assert tool_result.tools_called is True
    assert tool_result.tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_result.tool_calls[0].function.arguments) == {
        "city": "Tokyo",
        "days": 3,
    }
