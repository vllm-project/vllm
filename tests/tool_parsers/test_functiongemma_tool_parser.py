# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
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
        result = parser._parse_arguments("count:<escape>42<escape>")
        assert result == {"count": 42}

    def test_boolean_argument(self, parser):
        result = parser._parse_arguments("enabled:<escape>true<escape>")
        assert result == {"enabled": True}

    def test_argument_with_spaces(self, parser):
        result = parser._parse_arguments("message:<escape>Hello World<escape>")
        assert result == {"message": "Hello World"}


# Rendered by the model's chat template (``format_argument``): only strings are
# ``<escape>``-delimited; booleans, numbers, arrays and objects are bare.
TEMPLATE_CALL = (
    "<start_function_call>call:set_alarm{"
    "days:[<escape>mon<escape>,<escape>tue<escape>],enabled:true,hour:7,"
    "label:<escape>Wake up<escape>,snooze:{minutes:5}}"
    "<end_function_call>"
)
TEMPLATE_ARGS = {
    "days": ["mon", "tue"],
    "enabled": True,
    "hour": 7,
    "label": "Wake up",
    "snooze": {"minutes": 5},
}

_TAG_RE = re.compile(r"(<start_function_call>|<end_function_call>)")


def _chunks(text: str, size: int) -> list[str]:
    """Split ``text`` into ``size``-character chunks, keeping the tags whole.

    The tags are single special tokens, so a delta never splits them.
    """
    chunks = []
    for part in _TAG_RE.split(text):
        if _TAG_RE.fullmatch(part):
            chunks.append(part)
        else:
            chunks.extend(part[i : i + size] for i in range(0, len(part), size))
    return [chunk for chunk in chunks if chunk]


def _stream(parser, request, text: str, size: int) -> list[tuple[str, dict]]:
    """Reassemble streamed tool calls as a client would: by index."""
    previous = ""
    calls: dict[int, dict] = {}
    for i, chunk in enumerate(_chunks(text, size), start=1):
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous, current, chunk, [], [i], [i], request
        )
        previous = current
        for call in (delta.tool_calls if delta else None) or []:
            entry = calls.setdefault(call.index, {"name": None, "arguments": ""})
            if call.function.name:
                entry["name"] = call.function.name
            entry["arguments"] += call.function.arguments or ""
    return [
        (entry["name"], json.loads(entry["arguments"]))
        for _, entry in sorted(calls.items())
    ]


class TestTemplateArgumentTypes:
    """Regression tests for #63983: bare (non-string) arguments were dropped."""

    def test_non_streaming_keeps_every_argument(self, parser, mock_request):
        result = parser.extract_tool_calls(TEMPLATE_CALL, mock_request)

        assert result.tools_called is True
        assert json.loads(result.tool_calls[0].function.arguments) == TEMPLATE_ARGS

    @pytest.mark.parametrize("size", [1, 3, 1000])
    def test_streaming_keeps_every_argument(self, parser, mock_request, size):
        assert _stream(parser, mock_request, TEMPLATE_CALL, size) == [
            ("set_alarm", TEMPLATE_ARGS)
        ]

    @pytest.mark.parametrize("size", [1, 1000])
    def test_streaming_multiple_arguments_reassemble_to_json(
        self, parser, mock_request, size
    ):
        """The closing brace is held back until the call ends, so the
        concatenated deltas of a multi-argument call form one JSON object."""
        text = (
            "<start_function_call>call:get_weather{"
            "location:<escape>San Francisco<escape>,unit:<escape>celsius<escape>}"
            "<end_function_call>"
            "<start_function_call>call:set_alarm{hour:7,snooze:{minutes:5}}"
            "<end_function_call>"
        )
        assert _stream(parser, mock_request, text, size) == [
            ("get_weather", {"location": "San Francisco", "unit": "celsius"}),
            ("set_alarm", {"hour": 7, "snooze": {"minutes": 5}}),
        ]

    def test_escaped_strings_keep_delimiters(self, parser):
        args = "note:<escape>a, b: {c} [d]<escape>,count:2"
        assert parser._parse_arguments(args) == {"note": "a, b: {c} [d]", "count": 2}

    def test_nested_containers_and_literals(self, parser):
        args = "cfg:{on:false,ids:[1,[2,3],{k:<escape>v<escape>}],none:None},f:1.5"
        assert parser._parse_arguments(args) == {
            "cfg": {"on": False, "ids": [1, [2, 3], {"k": "v"}], "none": None},
            "f": 1.5,
        }

    @pytest.mark.parametrize(
        ("prefix", "expected"),
        [
            ("hour:7", {}),
            ("hour:7,", {"hour": 7}),
            ("hour:7,label:<escape>Wake", {"hour": 7}),
            ("hour:7,days:[<escape>mon<escape>", {"hour": 7}),
            ("hour:7,snooze:{minutes:5", {"hour": 7}),
            ("hour:7,snooze:{minutes:5}", {"hour": 7, "snooze": {"minutes": 5}}),
        ],
    )
    def test_partial_parsing_holds_back_incomplete_values(
        self, parser, prefix, expected
    ):
        assert parser._parse_arguments(prefix, partial=True) == expected

    @pytest.mark.parametrize(
        "args",
        ["a:1,items:[1}", "a:1,cfg:{x:1]", "a:1,b:,c:2", "a:1,b:}"],
    )
    @pytest.mark.parametrize("partial", [False, True])
    def test_mismatched_delimiters_stop_parsing(self, parser, args, partial):
        """A stray closing brace must not spin the parser; the arguments
        parsed before it are kept and the rest is dropped."""
        assert parser._parse_arguments(args, partial=partial) == {"a": 1}


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
