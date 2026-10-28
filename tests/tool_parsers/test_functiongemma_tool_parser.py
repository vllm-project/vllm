# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.functiongemma_tool_parser import FunctionGemmaToolParser


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = {}
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return FunctionGemmaToolParser(mock_tokenizer)


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    request.parallel_tool_calls = True
    return request


class TestExtractToolCalls:
    def test_no_tool_calls(self, parser, mock_request):
        model_output = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == model_output

    def test_single_tool_call(self, parser, mock_request):
        model_output = (
            "<start_function_call>call:get_weather{location:<escape>London<escape>}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert '"location": "London"' in result.tool_calls[0].function.arguments

    def test_multiple_arguments(self, parser, mock_request):
        model_output = (
            "<start_function_call>call:get_weather{"
            "location:<escape>San Francisco<escape>,"
            "unit:<escape>celsius<escape>}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = result.tool_calls[0].function.arguments
        assert "San Francisco" in args
        assert "celsius" in args

    def test_text_before_tool_call(self, parser, mock_request):
        model_output = (
            "Let me check the weather for you. "
            "<start_function_call>call:get_weather{location:<escape>Paris<escape>}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.content == "Let me check the weather for you."

    def test_multiple_tool_calls(self, parser, mock_request):
        model_output = (
            "<start_function_call>call:get_weather{location:<escape>London<escape>}"
            "<end_function_call>"
            "<start_function_call>call:get_time{timezone:<escape>UTC<escape>}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_template_argument_grammar(self, parser, mock_request):
        model_output = (
            "<start_function_call>call:set_alarm{"
            "days:[<escape>mon<escape>,<escape>tue<escape>],"
            "enabled:true,hour:7,label:<escape>Wake up<escape>,"
            "snooze:{minutes:5}}<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "set_alarm"
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "days": ["mon", "tue"],
            "enabled": True,
            "hour": 7,
            "label": "Wake up",
            "snooze": {"minutes": 5},
        }


class TestParseArguments:
    def test_empty_arguments(self, parser):
        result = parser._parse_arguments("")
        assert result == {}

    def test_single_string_argument(self, parser):
        result = parser._parse_arguments("city:<escape>Tokyo<escape>")
        assert result == {"city": "Tokyo"}

    def test_multiple_arguments(self, parser):
        args_str = "city:<escape>Tokyo<escape>,country:<escape>Japan<escape>"
        result = parser._parse_arguments(args_str)
        assert result == {"city": "Tokyo", "country": "Japan"}

    def test_numeric_argument(self, parser):
        # The chat template escapes only strings. A bare number stays a number.
        assert parser._parse_arguments("count:42") == {"count": 42}
        assert parser._parse_arguments("count:<escape>42<escape>") == {"count": "42"}

    def test_boolean_argument(self, parser):
        assert parser._parse_arguments("enabled:true") == {"enabled": True}
        assert parser._parse_arguments("enabled:<escape>true<escape>") == {
            "enabled": "true"
        }

    def test_argument_with_spaces(self, parser):
        result = parser._parse_arguments("message:<escape>Hello World<escape>")
        assert result == {"message": "Hello World"}

    @pytest.mark.parametrize(
        "literal",
        ["123", "true", "null", "[1,2]", '{"x":1}', '"quoted"'],
    )
    def test_escape_delimited_literal_stays_string(self, parser, literal):
        result = parser._parse_arguments(f"value:<escape>{literal}<escape>")
        assert result == {"value": literal}

    def test_bare_scalars_keep_json_types(self, parser):
        result = parser._parse_arguments("hour:7,enabled:true,note:null")
        assert result == {"hour": 7, "enabled": True, "note": None}


class TestAdjustRequest:
    def test_skip_special_tokens_disabled(self, parser, mock_request):
        mock_request.tools = [{"type": "function", "function": {"name": "test"}}]
        mock_request.tool_choice = "auto"
        mock_request.skip_special_tokens = True

        result = parser.adjust_request(mock_request)
        assert result.skip_special_tokens is False

    def test_skip_special_tokens_when_tool_choice_none(self, parser, mock_request):
        mock_request.tools = [{"type": "function", "function": {"name": "test"}}]
        mock_request.tool_choice = "none"
        mock_request.skip_special_tokens = True

        result = parser.adjust_request(mock_request)
        assert result.skip_special_tokens is True


class TestBufferDeltaText:
    def test_regular_text_not_buffered(self, parser):
        result = parser._buffer_delta_text("hello")
        assert result == "hello"
        assert parser.buffered_delta_text == ""

    def test_complete_tag_flushed(self, parser):
        parser.buffered_delta_text = "<start_function_"
        result = parser._buffer_delta_text("call>")
        assert "<start_function_call>" in result


class TestTemplateGrammarStreaming:
    def test_template_argument_grammar_streaming(self, parser, mock_request):
        text = (
            "<start_function_call>call:set_alarm{"
            "days:[<escape>mon<escape>,<escape>tue<escape>],"
            "enabled:true,hour:7,label:<escape>Wake up<escape>,"
            "snooze:{minutes:5}}<end_function_call>"
        )
        # Name first, then a partial argument body, then the rest of the call.
        # Argument deltas must concatenate to one JSON object.
        cuts = (
            len("<start_function_call>"),
            text.index("{") + 1,
            text.index("hour:"),
            len(text),
        )
        previous = ""
        argument_deltas: list[str] = []
        for end in cuts:
            chunk = text[len(previous) : end]
            delta = parser.extract_tool_calls_streaming(
                previous,
                previous + chunk,
                chunk,
                [],
                [],
                [],
                mock_request,
            )
            if delta and delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.function and tool_call.function.arguments:
                        argument_deltas.append(tool_call.function.arguments)
            previous = text[:end]
        assert json.loads("".join(argument_deltas)) == {
            "days": ["mon", "tue"],
            "enabled": True,
            "hour": 7,
            "label": "Wake up",
            "snooze": {"minutes": 5},
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
