# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Qwen3 tool call parser.

These validate that the engine-driven parser correctly handles
Qwen3 XML-style tool calls.
"""

import json
from unittest.mock import MagicMock

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.qwen3 import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    qwen3_config,
)


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            TOOL_CALL_START: 100,
            TOOL_CALL_END: 101,
        }
    )


@pytest.fixture
def parser(mock_tokenizer):
    return ParserEngine(
        mock_tokenizer,
        parser_engine_config=qwen3_config(thinking=False),
    )


class TestNonStreaming:
    def test_no_tool_calls(self, parser, mock_request):
        result = parser.extract_tool_calls(
            "This is a regular response without any tool calls.",
            mock_request,
        )
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == ("This is a regular response without any tool calls.")

    def test_single_tool_call(self, parser, mock_request):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "Tokyo"}

    def test_parallel_tool_calls(self, parser, mock_request):
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
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert args0 == {"city": "Tokyo"}
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert args1 == {"timezone": "Asia/Tokyo"}

    def test_various_data_types(self, parser, mock_request):
        text = (
            "<tool_call>\n<function=test_function>\n"
            "<parameter=string_field>hello</parameter>\n"
            "<parameter=int_field>42</parameter>\n"
            "<parameter=float_field>3.14</parameter>\n"
            "<parameter=bool_field>true</parameter>\n"
            "<parameter=null_field>null</parameter>\n"
            '<parameter=array_field>["a", "b", "c"]</parameter>\n'
            '<parameter=object_field>{"nested": "value"}</parameter>\n'
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["string_field"] == "hello"
        assert args["int_field"] == "42"
        assert args["float_field"] == "3.14"
        assert args["bool_field"] == "true"
        assert args["null_field"] == "null"
        assert args["array_field"] == '["a", "b", "c"]'
        assert args["object_field"] == '{"nested": "value"}'

    def test_empty_arguments(self, parser, mock_request):
        text = "<tool_call>\n<function=refresh>\n</function>\n</tool_call>"
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "refresh"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}

    def test_surrounding_text(self, parser, mock_request):
        text = (
            "Let me check the weather for you.\n\n"
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n</tool_call>\n\n"
            "I will get that information."
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.content is not None
        assert "Let me check the weather" in result.content
        assert result.tool_calls[0].function.name == "get_weather"

    def test_escaped_strings(self, parser, mock_request):
        text = (
            "<tool_call>\n<function=test_function>\n"
            '<parameter=quoted>He said "hello"</parameter>\n'
            "<parameter=path>C:\\Users\\file.txt</parameter>\n"
            "<parameter=newline>line1\nline2</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["quoted"] == 'He said "hello"'
        assert args["path"] == "C:\\Users\\file.txt"
        assert args["newline"] == "line1\nline2"

    def test_multiple_parameters(self, parser, mock_request):
        text = (
            "<tool_call>\n<function=search>\n"
            "<parameter=query>vllm parsing</parameter>\n"
            "<parameter=limit>10</parameter>\n"
            "<parameter=exact_match>false</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {
            "query": "vllm parsing",
            "limit": "10",
            "exact_match": "false",
        }

    def test_multiline_param_values(self, parser, mock_request):
        """Parameter values spanning multiple lines."""
        text = (
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>\n"
            "ls -la /tmp\n"
            "</parameter>\n"
            "<parameter=description>\n"
            "List files in /tmp directory\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "Bash"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["command"] == "ls -la /tmp"
        assert args["description"] == "List files in /tmp directory"

    def test_multiline_two_tool_calls(self, parser, mock_request):
        """Two tool calls with multi-line parameter values (bug report)."""
        text = (
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>\n"
            "find /workspace -name '*.py' | head -20\n"
            "</parameter>\n"
            "<parameter=description>\n"
            "Find Python files\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
            "<tool_call>\n"
            "<function=Read>\n"
            "<parameter=file_path>\n"
            "/workspace/main.py\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "Bash"
        assert result.tool_calls[1].function.name == "Read"
        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert "find /workspace" in args0["command"]
        assert "Find Python files" in args0["description"]
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert "/workspace/main.py" in args1["file_path"]

    def test_consecutive_tool_calls_without_tool_end(self, parser, mock_request):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Paris</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert args0 == {"city": "Tokyo"}
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert args1 == {"city": "Paris"}

    def test_nested_json_array_parameter(self, parser, mock_request):
        text = (
            "<tool_call>\n"
            "<function=AskUserQuestion>\n"
            "<parameter=questions>"
            '[{"question": "Pick a color",'
            ' "multiSelect": false, "answer": null}]'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {
            "questions": '[{"question": "Pick a color",'
            ' "multiSelect": false, "answer": null}]',
        }


class TestStreaming:
    def test_basic_streaming(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "get_weather"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"city": "Tokyo"}

    def test_streaming_multi_param(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n",
            "<parameter=unit>celsius</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "get_weather"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"city": "Tokyo", "unit": "celsius"}

    def test_streaming_args_arrive_incrementally(self, parser, mock_request):
        """Arguments must stream as intermediate deltas, not batch at
        tool-end."""
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n",
            "<parameter=unit>celsius</parameter>\n",
            "<parameter=days>5</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        arg_deltas: list[str] = []
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.arguments:
                        arg_deltas.append(tc.function.arguments)

        assert len(arg_deltas) > 1, (
            f"Expected arguments across multiple deltas, got {len(arg_deltas)}: "
            f"{arg_deltas}"
        )
        concatenated = "".join(arg_deltas)
        parsed = json.loads(concatenated)
        assert parsed == {"city": "Tokyo", "unit": "celsius", "days": "5"}

    def test_streaming_long_string_arg_before_parameter_end(self, parser, mock_request):
        """Long string arguments should stream before the closing parameter tag."""
        chunks = [
            "<tool_call>\n",
            "<function=write_report>\n",
            "<parameter=content>",
            "Artificial intelligence has rapidly transformed the way ",
            "developers build dynamic applications with external tools.",
            "</param",
            "eter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        pre_close_arg_deltas: list[str] = []
        all_arg_deltas: list[str] = []
        for idx, (delta, _) in enumerate(results):
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.arguments:
                        all_arg_deltas.append(tc.function.arguments)
                        if idx < 5:
                            pre_close_arg_deltas.append(tc.function.arguments)

        assert len(pre_close_arg_deltas) > 1, (
            "Expected long string arguments to stream incrementally before "
            f"</parameter>, got {pre_close_arg_deltas}"
        )
        partial_args = "".join(pre_close_arg_deltas)
        assert partial_args.startswith('{"content": "Artificial intelligence')
        assert partial_args.endswith("external tools.")
        assert not partial_args.endswith('"}')

        all_args = "".join(all_arg_deltas)
        assert json.loads(all_args) == {
            "content": (
                "Artificial intelligence has rapidly transformed the way "
                "developers build dynamic applications with external tools."
            )
        }

    def test_streaming_text_before_tool(self, parser, mock_request):
        chunks = [
            "Let me check ",
            "the weather. ",
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)
        assert collect_content(results).strip().startswith("Let me check")

    def test_streaming_empty_args(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=refresh>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "refresh"

    def test_streaming_split_parameter_tag(self, parser, mock_request):
        """Parameter tag split across chunks."""
        chunks = [
            "<tool_call>\n",
            "<function=test>\n",
            "<parameter=",
            "name>Alice",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "test"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed["name"] == "Alice"

    def test_streaming_split_next_parameter_tag_is_buffered(self, parser, mock_request):
        """A split opening parameter tag must not leak into previous value."""
        chunks = [
            "<tool_call>\n",
            "<function=search>\n",
            "<parameter=query>hello ",
            "<param",
            "eter=limit>10</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        args_after_partial_tag = collect_tool_arguments(results[:4])
        assert "<param" not in args_after_partial_tag
        assert args_after_partial_tag == '{"query": "hello'

        args_text = collect_tool_arguments(results)
        assert json.loads(args_text) == {"query": "hello", "limit": "10"}

    def test_truncated_parameter_value_matches_non_streaming(
        self,
        parser,
        mock_tokenizer,
        mock_request,
    ):
        chunks = [
            "Let me check.",
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>\n",
            "San Fr",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)
        results.append((parser.finish_streaming(), "".join(chunks)))
        streamed_args = collect_tool_arguments(results)

        assert json.loads(streamed_args) == {"city": "San Fr"}

        non_streaming_parser = ParserEngine(
            mock_tokenizer,
            parser_engine_config=qwen3_config(thinking=False),
        )
        result = non_streaming_parser.extract_tool_calls(
            "".join(chunks),
            mock_request,
        )

        assert result.tools_called is True
        assert result.content == "Let me check."
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].function.arguments == streamed_args

    def test_streaming_numeric_values(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=set_config>\n",
            "<parameter=count>42</parameter>\n",
            "<parameter=active>true</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        if args_text:
            parsed = json.loads(args_text)
            assert parsed["count"] == "42"
            assert parsed["active"] == "true"

    def test_streaming_parallel_calls(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n",
            "</function>\n",
            "</tool_call>",
            "<tool_call>\n",
            "<function=get_time>\n",
            "<parameter=tz>JST</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        names = []
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)

        assert "get_weather" in names
        assert "get_time" in names

    def test_streaming_value_split_across_chunks(self, parser, mock_request):
        """Parameter value split across multiple chunks."""
        chunks = [
            "<tool_call>\n",
            "<function=search>\n",
            "<parameter=query>hello ",
            "world",
            " test</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed["query"] == "hello world test"

    def test_streaming_split_tool_call_tag(self, parser, mock_request):
        """<tool_call> arrives as a single special token; the rest of
        the content is split into fine-grained chunks."""
        chunks = [
            "<tool_call>\n",
            "<function=test>\n",
            "<parameter=x>1",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "test"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed["x"] == "1"

    def test_char_by_char_streaming(self, mock_request):
        """Feed text character-by-character to test lexer robustness.

        Uses a tokenizer without special token IDs because char-by-char
        delivery only occurs when the tokenizer splits the tag across
        multiple sub-word tokens (i.e., no dedicated special token).
        """
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.get_vocab.return_value = {}
        tokenizer.decode.side_effect = lambda ids: "".join(
            chr(i) if i < 128 else f"<{i}>" for i in ids
        )
        no_tid_parser = ParserEngine(
            tokenizer, parser_engine_config=qwen3_config(thinking=False)
        )

        full_text = (
            "<tool_call>\n"
            "<function=echo>\n"
            "<parameter=msg>hi</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        chunks = list(full_text)
        results = simulate_tool_streaming(no_tid_parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "echo"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"msg": "hi"}

    def test_streaming_multiline_param_values(self, parser, mock_request):
        """Multi-line parameter values in streaming mode."""
        chunks = [
            "<tool_call>\n",
            "<function=Bash>\n",
            "<parameter=command>\n",
            "ls -la /tmp\n",
            "</parameter>\n",
            "<parameter=description>\n",
            "List files\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "Bash"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert "ls -la /tmp" in parsed["command"]
        assert "List files" in parsed["description"]

    def test_streaming_multiline_two_tool_calls(self, parser, mock_request):
        """Two tool calls with multi-line values — matches bug report."""
        chunks = [
            "<tool_call>\n",
            "<function=Bash>\n",
            "<parameter=command>\n",
            "find /workspace -name '*.py' | head -20\n",
            "</parameter>\n",
            "<parameter=description>\n",
            "Find Python files\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
            "<tool_call>\n",
            "<function=Read>\n",
            "<parameter=file_path>\n",
            "/workspace/main.py\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        names = []
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)

        assert "Bash" in names
        assert "Read" in names


class TestArgConverter:
    """Direct tests for the Qwen3 arg_converter with multi-line values."""

    def test_multiline_param_values(self):
        from vllm.parser.qwen3 import (
            _qwen3_arg_converter,
        )

        raw = (
            "<parameter=command>\n"
            "ls -la /tmp\n"
            "</parameter>\n"
            "<parameter=description>\n"
            "List files\n"
            "</parameter>\n"
        )
        result = json.loads(_qwen3_arg_converter(raw, partial=False))
        assert result["command"] == "ls -la /tmp"
        assert result["description"] == "List files"

    def test_two_multiline_params(self):
        from vllm.parser.qwen3 import (
            _qwen3_arg_converter,
        )

        raw = (
            "<parameter=a>\nfoo\nbar\n</parameter>\n"
            "<parameter=b>\nbaz\nqux\n</parameter>\n"
        )
        result = json.loads(_qwen3_arg_converter(raw, partial=False))
        assert result["a"] == "foo\nbar"
        assert result["b"] == "baz\nqux"

    def test_partial_multiline(self):
        from vllm.parser.qwen3 import (
            _qwen3_arg_converter,
        )

        raw = "<parameter=command>\nls -la</parameter>\n<parameter=desc>\npartial value"
        result = json.loads(_qwen3_arg_converter(raw, partial=True))
        assert result["command"] == "ls -la"
        assert result["desc"] == "partial value"

    def test_partial_value_with_angle_bracket(self):
        from vllm.parser.qwen3 import (
            _qwen3_arg_converter,
        )

        raw = "<parameter=expr>x<5"
        result = json.loads(_qwen3_arg_converter(raw, partial=True))
        assert result == {"expr": "x<5"}

    def test_partial_value_with_angle_bracket_and_complete_param(self):
        from vllm.parser.qwen3 import (
            _qwen3_arg_converter,
        )

        raw = "<parameter=city>Tokyo</parameter>\n<parameter=expr>x<5"
        result = json.loads(_qwen3_arg_converter(raw, partial=True))
        assert result == {"city": "Tokyo", "expr": "x<5"}


class TestSchemaAwareTypeCoercion:
    """Verify that _fix_arg_types corrects miscoerced values using the
    tool schema."""

    @pytest.fixture
    def tools(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "TaskUpdate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "taskId": {"type": "string"},
                            "count": {"type": "integer"},
                            "ratio": {"type": "number"},
                            "flag": {"type": "string"},
                        },
                    },
                },
            )
        ]

    @pytest.fixture
    def parser_with_tools(self, mock_tokenizer, tools):
        return ParserEngine(
            mock_tokenizer,
            tools=tools,
            parser_engine_config=qwen3_config(thinking=False),
        )

    def test_string_param_not_coerced_to_int(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=TaskUpdate>\n"
            "<parameter=taskId>1</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["taskId"] == "1"
        assert isinstance(args["taskId"], str)

    def test_string_param_not_coerced_to_bool(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=TaskUpdate>\n"
            "<parameter=flag>true</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["flag"] == "true"
        assert isinstance(args["flag"], str)

    def test_int_param_still_coerced(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=TaskUpdate>\n"
            "<parameter=count>42</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["count"] == 42
        assert isinstance(args["count"], int)

    def test_no_tools_keeps_strings(self, parser, mock_request):
        text = (
            "<tool_call>\n"
            "<function=TaskUpdate>\n"
            "<parameter=taskId>1</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["taskId"] == "1"
        assert isinstance(args["taskId"], str)

    def test_streaming_string_param_not_coerced(self, parser_with_tools, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=TaskUpdate>\n",
            "<parameter=taskId>1</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        args_str = collect_tool_arguments(results)
        args = json.loads(args_str)
        assert args["taskId"] == "1"
        assert isinstance(args["taskId"], str)


class TestAnyOfTypeCoercion:
    """Verify that _fix_arg_types handles union types (anyOf/oneOf)."""

    @pytest.fixture
    def tools_with_anyof(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "set_config",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "port": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "null"},
                                ],
                            },
                            "count": {"type": "integer"},
                        },
                    },
                },
            )
        ]

    @pytest.fixture
    def parser_with_anyof(self, mock_tokenizer, tools_with_anyof):
        return ParserEngine(
            mock_tokenizer,
            tools=tools_with_anyof,
            parser_engine_config=qwen3_config(thinking=False),
        )

    def test_anyof_string_param_not_coerced(self, parser_with_anyof, mock_request):
        """A param with anyOf including 'string' must not be coerced
        to integer."""
        text = (
            "<tool_call>\n"
            "<function=set_config>\n"
            "<parameter=port>8080</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_anyof.extract_tool_calls(text, mock_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["port"] == "8080"


class TestSchemaCoercionBoolNumberNull:
    """Verify that _fix_arg_types coerces string values to non-string
    schema types using coerce_to_schema_type."""

    @pytest.fixture
    def tools(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "configure",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "ratio": {"type": "number"},
                            "count": {"type": "integer"},
                            "value": {"type": ["integer", "null"]},
                            "label": {"type": "string"},
                        },
                    },
                },
            )
        ]

    @pytest.fixture
    def parser_with_tools(self, mock_tokenizer, tools):
        return ParserEngine(
            mock_tokenizer,
            tools=tools,
            parser_engine_config=qwen3_config(thinking=False),
        )

    def test_bool_param_coerced(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=enabled>true</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["enabled"] is True
        assert isinstance(args["enabled"], bool)

    def test_number_param_whole_normalized(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=ratio>5.0</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["ratio"] == 5
        assert isinstance(args["ratio"], int)

    def test_number_param_fractional(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=ratio>3.14</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["ratio"] == pytest.approx(3.14)
        assert isinstance(args["ratio"], float)

    def test_null_coerced_when_in_schema(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=value>null</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["value"] is None

    def test_null_stays_string_without_null_schema(
        self, parser_with_tools, mock_request
    ):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=label>null</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["label"] == "null"
        assert isinstance(args["label"], str)

    def test_streaming_bool_param_coerced(self, parser_with_tools, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=configure>\n",
            "<parameter=enabled>true</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        args_str = collect_tool_arguments(results)
        args = json.loads(args_str)
        assert args["enabled"] is True
        assert isinstance(args["enabled"], bool)

    def test_streaming_number_param_coerced(self, parser_with_tools, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=configure>\n",
            "<parameter=ratio>3.14</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        args_str = collect_tool_arguments(results)
        args = json.loads(args_str)
        assert args["ratio"] == pytest.approx(3.14)
        assert isinstance(args["ratio"], float)

    def test_streaming_matches_non_streaming_comprehensive(
        self, parser_with_tools, mock_request
    ):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=enabled>true</parameter>\n"
            "<parameter=ratio>5.0</parameter>\n"
            "<parameter=count>42</parameter>\n"
            "<parameter=value>null</parameter>\n"
            "<parameter=label>hello</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        non_stream = parser_with_tools.extract_tool_calls(text, mock_request)
        ns_args = json.loads(non_stream.tool_calls[0].function.arguments)

        chunks = [line + "\n" for line in text.split("\n") if line]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        s_args = json.loads(collect_tool_arguments(results))

        assert s_args == ns_args
        assert ns_args == {
            "enabled": True,
            "ratio": 5,
            "count": 42,
            "value": None,
            "label": "hello",
        }


class TestNestedSchemaCoercion:
    """Verify that _fix_arg_types recurses into nested objects and arrays."""

    @pytest.fixture
    def tools(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "filters": {
                                "type": "object",
                                "properties": {
                                    "language": {"type": "string"},
                                    "min_stars": {"type": "integer"},
                                },
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "limits": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "verbose": {"type": "boolean"},
                        },
                    },
                },
            ),
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "AskUserQuestion",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                        "multiSelect": {
                                            "type": "boolean",
                                        },
                                        "answer": {
                                            "type": ["string", "null"],
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            ),
        ]

    @pytest.fixture
    def parser_with_tools(self, mock_tokenizer, tools):
        return ParserEngine(
            mock_tokenizer,
            tools=tools,
            parser_engine_config=qwen3_config(thinking=False),
        )

    def test_nested_object_coerced(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=search>\n"
            '<parameter=filters>{"language": "python",'
            ' "min_stars": 100}</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["filters"] == {"language": "python", "min_stars": 100}
        assert isinstance(args["filters"]["min_stars"], int)

    def test_nested_array_items_coerced(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=limits>[10, 20, 30]</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["limits"] == [10, 20, 30]
        assert all(isinstance(v, int) for v in args["limits"])

    def test_nested_string_array_not_coerced(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=search>\n"
            '<parameter=tags>["ml", "42"]</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["tags"] == ["ml", "42"]
        assert all(isinstance(v, str) for v in args["tags"])

    def test_array_of_objects_with_bool_and_null_coerced(
        self, parser_with_tools, mock_request
    ):
        text = (
            "<tool_call>\n"
            "<function=AskUserQuestion>\n"
            "<parameter=questions>"
            '[{"question": "Pick a color",'
            ' "multiSelect": false, "answer": null}]'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        questions = args["questions"]
        assert isinstance(questions, list)
        assert len(questions) == 1
        assert questions[0]["question"] == "Pick a color"
        assert questions[0]["multiSelect"] is False
        assert questions[0]["answer"] is None

    def test_streaming_array_of_objects_with_bool_and_null_coerced(
        self, parser_with_tools, mock_request
    ):
        chunks = [
            "<tool_call>\n",
            "<function=AskUserQuestion>\n",
            '<parameter=questions>[{"question": "Pick a color",',
            ' "multiSelect": false, "answer": null}]',
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        args_str = collect_tool_arguments(results)
        args = json.loads(args_str)
        questions = args["questions"]
        assert isinstance(questions, list)
        assert len(questions) == 1
        assert questions[0]["question"] == "Pick a color"
        assert questions[0]["multiSelect"] is False
        assert questions[0]["answer"] is None
