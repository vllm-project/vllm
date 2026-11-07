# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Nemotron V3 parser.

Validates that ``NemotronV3Parser`` correctly handles:
- ``<think>``/``</think>`` reasoning with ``<tool_call>`` XML tool calls
  (same format as Qwen3)
- Nemotron-specific reasoning/content swap when ``enable_thinking=False``
  or ``force_nonempty_content=True``
"""

import json
from unittest.mock import MagicMock

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.nemotron_v3 import NemotronV3Parser
from vllm.parser.parser_manager import ParserManager

_THINK_START_ID = 50
_THINK_END_ID = 51
_TOOL_CALL_ID = 60
_TOOL_CALL_END_ID = 61
_TEXT_ID = 100

_VOCAB = {
    "<think>": _THINK_START_ID,
    "</think>": _THINK_END_ID,
    "<tool_call>": _TOOL_CALL_ID,
    "</tool_call>": _TOOL_CALL_END_ID,
}


def _make_request(**chat_template_kwargs):
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    request.include_reasoning = True
    request.chat_template_kwargs = chat_template_kwargs or None
    return request


def _weather_tool() -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function=FunctionDefinition(
            name="get_current_weather",
            description="Get current weather",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["city"],
            },
        ),
    )


@pytest.fixture
def parser():
    return NemotronV3Parser(make_mock_tokenizer(_VOCAB))


class TestNemotronSwap:
    def test_enable_thinking_false_swaps(self, parser):
        """When enable_thinking=False, model output without think tags
        should have reasoning swapped to content."""
        text = "The answer is 42."
        request = _make_request(enable_thinking=False)
        reasoning, content = parser.extract_reasoning(text, request)
        assert content == "The answer is 42."
        assert reasoning is None

    def test_force_nonempty_content_swaps(self, parser):
        """force_nonempty_content=True triggers swap when content empty."""
        text = "The answer is 42."
        request = _make_request(force_nonempty_content=True)
        reasoning, content = parser.extract_reasoning(text, request)
        assert content == "The answer is 42."
        assert reasoning is None

    def test_no_swap_when_content_exists(self, parser):
        """With enable_thinking=False but real </think> giving content,
        no swap occurs."""
        text = "Some reasoning.</think>Actual content here."
        request = _make_request(enable_thinking=False)
        reasoning, content = parser.extract_reasoning(text, request)
        assert reasoning == "Some reasoning."
        assert content == "Actual content here."

    def test_no_swap_when_enable_thinking_true(self, parser):
        """Normal thinking mode: no swap, even when content is empty."""
        text = "Still thinking..."
        request = _make_request(enable_thinking=True)
        reasoning, content = parser.extract_reasoning(text, request)
        assert reasoning == "Still thinking..."
        assert content is None

    def test_no_swap_with_none_request(self, parser):
        """Graceful handling when request is None."""
        text = "Some text."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Some text."
        assert content is None

    def test_no_swap_with_no_kwargs(self, parser):
        """No swap when chat_template_kwargs is absent."""
        text = "Some text."
        request = _make_request()
        reasoning, content = parser.extract_reasoning(text, request)
        assert reasoning == "Some text."
        assert content is None

    def test_swap_with_whitespace_only_content(self, parser):
        """Swap occurs when content is whitespace-only."""
        text = "The answer.</think>   "
        request = _make_request(enable_thinking=False)
        reasoning, content = parser.extract_reasoning(text, request)
        assert content == "The answer."
        assert reasoning == "   "


class TestNonStreamingToolCalls:
    def test_single_tool_call(self, parser):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        request = _make_request()
        result = parser.extract_tool_calls(text, request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "Tokyo"}

    @pytest.mark.parametrize(
        "tool_choice",
        [
            "required",
            {
                "type": "function",
                "function": {"name": "get_current_weather"},
            },
        ],
    )
    def test_required_and_named_tool_choice_parse_xml_after_reasoning(
        self, tool_choice
    ):
        """Regression: Nemotron V3 must not send Qwen XML to JSON parsing."""
        tool_payload = _weather_tool().model_dump(mode="json", exclude_none=True)
        request = ChatCompletionRequest(
            model="model",
            messages=[],
            tools=[tool_payload],
            tool_choice=tool_choice,
        )
        tool = request.tools[0]
        parser_cls = ParserManager.get_parser(
            tool_parser_name="qwen3_coder",
            reasoning_parser_name="nemotron_v3",
            enable_auto_tools=True,
            model_name="model",
        )
        parser = parser_cls(make_mock_tokenizer(_VOCAB), [tool])
        model_output = (
            "The user asked for weather.</think>\n"
            "<tool_call>\n"
            "<function=get_current_weather>\n"
            "<parameter=city>Dallas</parameter>\n"
            "<parameter=state>TX</parameter>\n"
            "<parameter=unit>fahrenheit</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        reasoning, content, tool_calls = parser.parse(
            model_output, request, enable_auto_tools=True
        )

        assert reasoning == "The user asked for weather."
        assert content is None
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_current_weather"
        assert json.loads(tool_calls[0].arguments) == {
            "city": "Dallas",
            "state": "TX",
            "unit": "fahrenheit",
        }

    def test_parallel_tool_calls(self, parser):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
            "<tool_call>\n"
            "<function=get_time>\n"
            "<parameter=timezone>Asia/Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        request = _make_request()
        result = parser.extract_tool_calls(text, request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_no_tool_calls(self, parser):
        request = _make_request()
        result = parser.extract_tool_calls("Hello, how can I help?", request)
        assert result.tools_called is False
        # Parser starts in REASONING state, so plain text is classified
        # as reasoning (not content) when there are no tool calls.
        assert result.content is None


class TestStreaming:
    def test_streaming_tool_calls(self, parser):
        request = _make_request()
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser, request, chunks)
        name = collect_function_name(results)
        assert name == "get_weather"
        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"city": "Tokyo"}


class TestParseDeltaTokenIdFiltering:
    """parse_delta must not trigger tool call parsing when <tool_call>
    appears as regular text rather than as a special token ID."""

    def test_tool_call_text_in_reasoning_is_not_parsed(self, parser):
        """Literal <tool_call> in model reasoning should be content,
        not a tool call."""
        request = _make_request()

        text = (
            "The test uses <tool_call> syntax:\n"
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>ls</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.parse_delta(
            delta_text=text,
            delta_token_ids=[_TEXT_ID] * 6,
            request=request,
            prompt_token_ids=[],
            finished=True,
        )

        assert result is not None
        assert result.reasoning is not None
        assert "<tool_call>" in result.reasoning
        assert not result.tool_calls

    def test_special_token_id_still_triggers_tool_call(self, parser):
        """When the scanner matches a special token ID, the tool call
        must still be parsed correctly."""
        request = _make_request()

        parser.parse_delta(
            delta_text="Let me check.",
            delta_token_ids=[_TEXT_ID, _TEXT_ID, _TEXT_ID],
            request=request,
            prompt_token_ids=[],
            finished=False,
        )

        parser.parse_delta(
            delta_text="<tool_call>",
            delta_token_ids=[_TOOL_CALL_ID],
            request=request,
            finished=False,
        )

        parser.parse_delta(
            delta_text=(
                "\n<function=get_weather>\n"
                "<parameter=city>Tokyo</parameter>\n"
                "</function>\n"
            ),
            delta_token_ids=[_TEXT_ID] * 5,
            request=request,
            finished=False,
        )

        parser.parse_delta(
            delta_text="</tool_call>",
            delta_token_ids=[_TOOL_CALL_END_ID],
            request=request,
            finished=True,
        )

        assert any(s.name == "get_weather" for s in parser._tool_slots)

    def test_text_discussion_then_real_tool_call(self, parser):
        """Model discusses tool syntax in reasoning, then makes a real
        tool call via special tokens."""
        request = _make_request()

        r1 = parser.parse_delta(
            delta_text="Use <tool_call> to invoke tools.",
            delta_token_ids=[_TEXT_ID] * 6,
            request=request,
            prompt_token_ids=[],
            finished=False,
        )

        r2 = parser.parse_delta(
            delta_text="</think>",
            delta_token_ids=[_THINK_END_ID],
            request=request,
            finished=False,
        )

        r3 = parser.parse_delta(
            delta_text="<tool_call>",
            delta_token_ids=[_TOOL_CALL_ID],
            request=request,
            finished=False,
        )

        r4 = parser.parse_delta(
            delta_text=("\n<function=test>\n<parameter=x>1</parameter>\n</function>\n"),
            delta_token_ids=[_TEXT_ID] * 4,
            request=request,
            finished=False,
        )

        r5 = parser.parse_delta(
            delta_text="</tool_call>",
            delta_token_ids=[_TOOL_CALL_END_ID],
            request=request,
            finished=True,
        )

        results = [r1, r2, r3, r4, r5]
        reasoning = "".join(r.reasoning for r in results if r and r.reasoning)
        assert "<tool_call>" in reasoning

        names = [
            tc.function.name
            for r in results
            if r and r.tool_calls
            for tc in r.tool_calls
            if tc.function and tc.function.name
        ]
        assert "test" in names
