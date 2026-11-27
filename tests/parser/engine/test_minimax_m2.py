# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.minimax_m2 import (
    THINK_END,
    TOOL_CALL_END,
    TOOL_CALL_START,
    MinimaxM2Parser,
    _minimax_m2_arg_value_extender,
    minimax_m2_config,
)


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            THINK_END: 99,
            TOOL_CALL_START: 100,
            TOOL_CALL_END: 101,
        }
    )


@pytest.fixture
def parser(mock_tokenizer):
    return MinimaxM2Parser(mock_tokenizer)


def make_tools(*names: str):
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name=name,
                parameters={
                    "type": "object",
                    "properties": {},
                },
            ),
        )
        for name in names
    ]


class TestNonStreaming:
    def test_no_tool_calls(self, parser, mock_request):
        result = parser.extract_tool_calls(
            "</think>This is a regular response without tool calls.",
            mock_request,
        )
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "This is a regular response without tool calls."

    def test_single_tool_call(self, parser, mock_request):
        result = parser.extract_tool_calls(
            '<minimax:tool_call><invoke name="get_weather">'
            '<parameter name="city">Seattle</parameter>'
            "</invoke></minimax:tool_call>",
            mock_request,
        )

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "city": "Seattle",
        }

    def test_multiple_invokes(self, parser, mock_request):
        result = parser.extract_tool_calls(
            "<minimax:tool_call>"
            '<invoke name="search"><parameter name="q">OpenAI</parameter></invoke>'
            '<invoke name="search"><parameter name="q">vLLM</parameter></invoke>'
            "</minimax:tool_call>",
            mock_request,
        )

        assert result.tools_called is True
        assert [tc.function.name for tc in result.tool_calls] == ["search", "search"]
        assert json.loads(result.tool_calls[0].function.arguments) == {"q": "OpenAI"}
        assert json.loads(result.tool_calls[1].function.arguments) == {"q": "vLLM"}

    def test_schema_type_coercion(self, mock_tokenizer, mock_request):
        tools = [
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="forecast",
                    parameters={
                        "type": "object",
                        "properties": {
                            "days": {"type": "integer"},
                            "include_hourly": {"type": "boolean"},
                        },
                    },
                ),
            )
        ]
        parser = MinimaxM2Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        result = parser.extract_tool_calls(
            '<minimax:tool_call><invoke name="forecast">'
            '<parameter name="days">5</parameter>'
            '<parameter name="include_hourly">true</parameter>'
            "</invoke></minimax:tool_call>",
            mock_request,
        )

        assert json.loads(result.tool_calls[0].function.arguments) == {
            "days": 5,
            "include_hourly": True,
        }

    def test_invalid_tool_name_is_rejected(self, mock_tokenizer, mock_request):
        tools = make_tools("search")
        parser = MinimaxM2Parser(mock_tokenizer)
        mock_request.tools = tools

        result = parser.extract_tool_calls(
            '<minimax:tool_call><invoke name="img_gen">'
            '<parameter name="prompt">a cat</parameter>'
            "</invoke></minimax:tool_call>",
            mock_request,
        )

        assert result.tools_called is False
        assert result.tool_calls == []

    def test_mixed_tool_names_only_return_valid(self, mock_tokenizer, mock_request):
        tools = make_tools("search")
        parser = MinimaxM2Parser(mock_tokenizer)
        mock_request.tools = tools

        result = parser.extract_tool_calls(
            "<minimax:tool_call>"
            '<invoke name="img_gen"><parameter name="prompt">cat</parameter></invoke>'
            '<invoke name="search"><parameter name="query">news</parameter></invoke>'
            "</minimax:tool_call>",
            mock_request,
        )

        assert result.tools_called is True
        assert [tc.function.name for tc in result.tool_calls] == ["search"]
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "query": "news",
        }


class TestStreaming:
    def test_streaming_single_tool_call(self, parser, mock_request):
        results = simulate_tool_streaming(
            parser,
            mock_request,
            [
                "<minimax:tool_call>",
                '<invoke name="get_weather">',
                '<parameter name="city">Seattle</parameter>',
                "</invoke></minimax:tool_call>",
            ],
        )

        assert collect_function_name(results) == "get_weather"
        assert json.loads(collect_tool_arguments(results)) == {
            "city": "Seattle",
        }

    def test_streaming_multiple_invokes(self, parser, mock_request):
        results = simulate_tool_streaming(
            parser,
            mock_request,
            [
                "<minimax:tool_call>",
                '<invoke name="a"><parameter name="x">1</parameter></invoke>',
                '<invoke name="b"><parameter name="y">2</parameter></invoke>',
                "</minimax:tool_call>",
            ],
        )

        tool_names = [
            tc.function.name
            for delta, _ in results
            if delta and delta.tool_calls
            for tc in delta.tool_calls
            if tc.function and tc.function.name
        ]
        assert tool_names == ["a", "b"]

    def test_streaming_invoke_prefix_split_before_quote(self, parser, mock_request):
        results = simulate_tool_streaming(
            parser,
            mock_request,
            [
                "<minimax:tool_call>",
                "<invoke name=",
                '"get_weather">',
                '<parameter name="city">Seattle</parameter>',
                "</invoke></minimax:tool_call>",
            ],
        )

        assert collect_function_name(results) == "get_weather"
        assert json.loads(collect_tool_arguments(results)) == {
            "city": "Seattle",
        }

    def test_streaming_invalid_tool_name_is_rejected(
        self, mock_tokenizer, mock_request
    ):
        tools = make_tools("search")
        parser = MinimaxM2Parser(mock_tokenizer)
        mock_request.tools = tools

        results = simulate_tool_streaming(
            parser,
            mock_request,
            [
                "<minimax:tool_call>",
                '<invoke name="img_gen">',
                '<parameter name="prompt">cat</parameter>',
                "</invoke></minimax:tool_call>",
            ],
        )

        assert collect_function_name(results) is None
        assert collect_tool_arguments(results) == ""


WRITE_CALL_HEAD = (
    '<minimax:tool_call>\n<invoke name="write_file">\n'
    '<parameter name="path">/repo/app.py</parameter>\n'
    '<parameter name="mode">644</parameter>\n'
    '<parameter name="content">'
)
WRITE_CALL_TAIL = "</parameter>\n</invoke>\n</minimax:tool_call>"
# Escapes, non-ASCII, blank lines and tag-like text that must stay inside
# the value, including prefixes of the closing and next-parameter tags.
CONTENT_LINE = 'if a < b > c: print("x\\y")  # </p> </param <parameter\té\n\n'
# A model stuck in a loop repeats closed parameters inside one invoke block.
LOOP_HEAD = (
    '<minimax:tool_call>\n<invoke name="write_file">\n'
    '<parameter name="path">/repo/app.py</parameter>\n'
)
LOOP_PARAMS = (
    '<parameter name="content">a > b</parameter>\n'
    '<parameter name="mode">644</parameter>\n'
)
LOOP_SECOND_INVOKE = (
    '<invoke name="write_file">\n<parameter name="path">/repo/app.py</parameter>\n'
)
LOOP_INVOKE_TAIL = "</invoke>\n"
LOOP_TAIL = "</invoke>\n</minimax:tool_call>"


class TestArgValueExtender:
    """The open parameter value streams without re-running the converter."""

    @pytest.fixture
    def write_tool(self):
        return ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="write_file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "mode": {"type": "integer"},
                        "content": {"type": "string"},
                    },
                },
            ),
        )

    @pytest.mark.parametrize("use_schema", [False, True], ids=["no_schema", "schema"])
    def test_long_call_streams_non_streaming_arguments(
        self, mock_tokenizer, mock_request, write_tool, use_schema
    ):
        mock_request.tools = [write_tool] if use_schema else []
        text = WRITE_CALL_HEAD + CONTENT_LINE * 200 + WRITE_CALL_TAIL

        results = simulate_tool_streaming(
            MinimaxM2Parser(mock_tokenizer),
            mock_request,
            [text[i : i + 7] for i in range(0, len(text), 7)],
        )
        result = MinimaxM2Parser(mock_tokenizer).extract_tool_calls(text, mock_request)

        arguments = collect_tool_arguments(results)
        assert arguments == result.tool_calls[0].function.arguments
        assert json.loads(arguments)["content"] == CONTENT_LINE * 200
        assert ('"mode": 644,' if use_schema else '"mode": "644",') in arguments

    def test_conversions_do_not_grow_with_argument_length(
        self, mock_tokenizer, mock_request, write_tool
    ):
        mock_request.tools = [write_tool]
        base = minimax_m2_config()
        converter = base.arg_converter
        assert converter is not None

        def streaming_conversions(lines: int) -> int:
            calls = 0

            def counting_converter(raw_args: str, partial: bool) -> str:
                nonlocal calls
                calls += partial
                return converter(raw_args, partial)

            parser = ParserEngine(
                mock_tokenizer,
                parser_engine_config=dataclasses.replace(
                    base, arg_converter=counting_converter
                ),
            )
            text = WRITE_CALL_HEAD + CONTENT_LINE * lines + WRITE_CALL_TAIL
            simulate_tool_streaming(
                parser,
                mock_request,
                [text[i : i + 4] for i in range(0, len(text), 4)],
            )
            return calls

        assert streaming_conversions(lines=20) == streaming_conversions(lines=200)

    @pytest.mark.parametrize("size", [1, 7])
    def test_repeated_closed_params_stream_non_streaming_arguments(
        self, mock_tokenizer, mock_request, write_tool, size
    ):
        """Covers a duplicate open key and a second call sharing the first pair."""
        mock_request.tools = [write_tool]
        text = (
            LOOP_HEAD
            + LOOP_PARAMS * 20
            + '<parameter name="content">a > b'
            + "</parameter>\n"
            + LOOP_INVOKE_TAIL
            + LOOP_SECOND_INVOKE
            + LOOP_TAIL
        )

        results = simulate_tool_streaming(
            MinimaxM2Parser(mock_tokenizer),
            mock_request,
            [text[i : i + size] for i in range(0, len(text), size)],
        )
        result = MinimaxM2Parser(mock_tokenizer).extract_tool_calls(text, mock_request)

        assert len(result.tool_calls) == 2
        assert collect_tool_arguments(results) == "".join(
            c.function.arguments for c in result.tool_calls
        )

    def test_closed_params_are_parsed_once(
        self, mock_tokenizer, mock_request, write_tool, monkeypatch
    ):
        """Streaming cost must not grow with the closed parameters already seen."""
        from vllm.parser import minimax_m2

        mock_request.tools = [write_tool]
        param_re = minimax_m2._PARAM_RE
        matches = 0

        class CountingPattern:
            def __getattr__(self, name):
                return getattr(param_re, name)

            def finditer(self, *args):
                nonlocal matches
                for match in param_re.finditer(*args):
                    matches += 1
                    yield match

            def sub(self, repl, *args):
                def count(match):
                    nonlocal matches
                    matches += 1
                    return repl

                return param_re.sub(count, *args)

        monkeypatch.setattr(minimax_m2, "_PARAM_RE", CountingPattern())
        text = LOOP_HEAD + LOOP_PARAMS * 200 + LOOP_TAIL
        simulate_tool_streaming(
            minimax_m2.MinimaxM2Parser(mock_tokenizer),
            mock_request,
            [text[i : i + 4] for i in range(0, len(text), 4)],
        )

        assert matches == len(param_re.findall(text))

    def test_value_closed_by_next_parameter_needs_that_parameter(self):
        from vllm.parser.minimax_m2 import _ClosedParams, _minimax_m2_arg_converter

        closed = _ClosedParams()
        _minimax_m2_arg_converter(
            '<parameter name="path">/a.py<parameter name="mode">6', True, closed
        )
        text = '<parameter name="path">/a.py.bak</parameter>'
        assert _minimax_m2_arg_converter(
            text, False, closed
        ) == _minimax_m2_arg_converter(text, False)

    @pytest.mark.parametrize(
        ("prefix", "suffix"),
        [
            (
                '<parameter name="a<parameter name=b>1</parameter>',
                '">2</parameter>',
            ),
            (
                '<parameter name="a>1<parameter name=b>2</parameter>',
                '">3</parameter>',
            ),
        ],
        ids=["unmatched_quoted_opener", "bare_name_starts_with_quote"],
    )
    def test_unstable_quote_prefix_matches_uncached_parse(
        self, mock_tokenizer, mock_request, prefix, suffix
    ):
        from vllm.parser.minimax_m2 import _ClosedParams, _minimax_m2_arg_converter

        closed = _ClosedParams()
        _minimax_m2_arg_converter(prefix, True, closed)
        raw = prefix + suffix
        assert _minimax_m2_arg_converter(
            raw, False, closed
        ) == _minimax_m2_arg_converter(raw, False)

        full = f'<minimax:tool_call><invoke name="f">{raw}</invoke></minimax:tool_call>'
        parser = MinimaxM2Parser(mock_tokenizer)
        _, _, parsed = parser.parse(full, mock_request, enable_auto_tools=True)
        extracted = parser.extract_tool_calls(full, mock_request)

        assert extracted.tool_calls[0].function.arguments == parsed[0].arguments

    def test_extender_stops_at_split_end_tag(self):
        extend = _minimax_m2_arg_value_extender('<parameter name="k">')
        assert extend is not None
        deltas = ['a"\n', "</param", "eter>"]
        assert [extend(d) for d in deltas] == ['a\\"\\n', "</param", None]

    def test_extender_stops_at_split_next_parameter(self):
        extend = _minimax_m2_arg_value_extender("<parameter name='k'>x")
        assert extend is not None
        assert [extend(d) for d in ["< parameter", " name ="]] == ["< parameter", None]

    @pytest.mark.parametrize(
        "raw_args",
        [
            '<parameter name="k">x</parameter>',
            '<parameter name="k',
            (
                '<parameter name="k">1</parameter><parameter name="j">2</parameter>'
                '<parameter name="k">3'
            ),
            '<parameter name="a<parameter name=k>x',
            '<parameter name="k>x',
        ],
        ids=[
            "closed_value",
            "open_name",
            "repeated_key_not_last",
            "earlier_unmatched_opener",
            "bare_name_opens_quote",
        ],
    )
    def test_no_extender_unless_open_value_is_stable(self, raw_args):
        assert _minimax_m2_arg_value_extender(raw_args) is None


class TestReasoning:
    def test_extract_reasoning_without_start_token(self, parser, mock_request):
        reasoning, content = parser.extract_reasoning(
            "This is reasoning</think>This is content",
            mock_request,
        )

        assert reasoning == "This is reasoning"
        assert content == "This is content"

    def test_extract_reasoning_without_end_token(self, parser, mock_request):
        reasoning, content = parser.extract_reasoning(
            "This is still reasoning",
            mock_request,
        )

        assert reasoning == "This is still reasoning"
        assert content is None

    def test_extract_content_ids_without_end_token(self, parser):
        assert parser.extract_content_ids([1, 2, 3]) == []

    def test_extract_content_ids_after_end_token(self, parser):
        assert parser.extract_content_ids([1, 99, 2, 3]) == [2, 3]
