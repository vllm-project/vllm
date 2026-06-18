# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek V4-specific parser engine semantics."""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_reasoning_streaming,
    simulate_tool_streaming,
)
from vllm.parser.deepseek_v4 import (
    DSML_INVOKE_END,
    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
    DSML_THINK_END,
    DSML_THINK_START,
    DSML_TOOL_END,
    DSML_TOOL_START,
    DeepSeekV4Parser,
    _dsml_arg_converter,
    deepseek_v4_config,
)

_THINK_START_ID = 50
_THINK_END_ID = 51

_PARAM_OPEN = '｜DSML｜parameter name="{name}" string="{is_str}">'
_PARAM_CLOSE = "</｜DSML｜parameter>"


def _param(name: str, is_str: str, value: str) -> str:
    return f"<{_PARAM_OPEN.format(name=name, is_str=is_str)}{value}{_PARAM_CLOSE}"


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            DSML_THINK_START: _THINK_START_ID,
            DSML_THINK_END: _THINK_END_ID,
        }
    )


# ── Arg converter unit tests ─────────────────────────────────────────


class TestArgConverter:
    def _raw(self, *params: tuple[str, str, str]) -> str:
        lines = [_param(n, s, v) for n, s, v in params]
        return "\n" + "\n".join(lines) + "\n"

    def test_string_param(self):
        raw = self._raw(("city", "true", "杭州"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result == {"city": "杭州"}

    def test_string_with_spaces_and_quotes(self):
        raw = self._raw(("msg", "true", 'He said "hello world"'))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["msg"] == 'He said "hello world"'

    def test_integer_param(self):
        raw = self._raw(("count", "false", "42"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_float_param(self):
        raw = self._raw(("ratio", "false", "3.14"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert abs(result["ratio"] - 3.14) < 1e-9

    def test_bool_param(self):
        raw = self._raw(("flag", "false", "true"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["flag"] is True

    def test_array_param(self):
        raw = self._raw(("items", "false", '["a", "b", "c"]'))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["items"] == ["a", "b", "c"]

    def test_object_param(self):
        raw = self._raw(("opts", "false", '{"key": "val"}'))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["opts"] == {"key": "val"}

    def test_mixed_types(self):
        raw = self._raw(
            ("location", "true", "Tokyo"),
            ("limit", "false", "10"),
            ("active", "false", "false"),
        )
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result == {"location": "Tokyo", "limit": 10, "active": False}

    def test_empty_args(self):
        result = json.loads(_dsml_arg_converter("", partial=False))
        assert result == {}

    def test_invalid_json_fallback(self):
        raw = self._raw(("data", "false", "[broken"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["data"] == "[broken"

    def test_chinese_chars_preserved_in_json(self):
        raw = self._raw(("query", "true", "你好世界"))
        raw_json = _dsml_arg_converter(raw, partial=False)
        assert "你好世界" in raw_json
        result = json.loads(raw_json)
        assert result["query"] == "你好世界"

    def test_partial_complete_plus_in_progress(self):
        raw = self._raw(("city", "true", "Tokyo"))
        raw += f"<{_PARAM_OPEN.format(name='unit', is_str='true')}celsi"
        result = json.loads(_dsml_arg_converter(raw, partial=True))
        assert result["city"] == "Tokyo"
        assert result["unit"] == "celsi"

    def test_partial_no_in_progress(self):
        raw = self._raw(("city", "true", "Tokyo"))
        result = json.loads(_dsml_arg_converter(raw, partial=True))
        assert result == {"city": "Tokyo"}

    def test_partial_value_with_angle_bracket(self):
        raw = f"<{_PARAM_OPEN.format(name='code', is_str='true')}a<b"
        result = json.loads(_dsml_arg_converter(raw, partial=True))
        assert result == {"code": "a<b"}

    def test_partial_value_with_angle_bracket_and_complete_param(self):
        raw = self._raw(("city", "true", "Tokyo"))
        raw += f"<{_PARAM_OPEN.format(name='expr', is_str='true')}x<5"
        result = json.loads(_dsml_arg_converter(raw, partial=True))
        assert result["city"] == "Tokyo"
        assert result["expr"] == "x<5"

    def test_null_string_false(self):
        raw = self._raw(("val", "false", "null"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["val"] is None

    def test_string_true_not_json_parsed(self):
        raw = self._raw(("n", "true", "42"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["n"] == "42"
        assert isinstance(result["n"], str)


# ── Bare </think> absorption and duplicate <think> absorption ─────────


class TestThinkTagAbsorption:
    def test_bare_think_end_not_leaked(self, mock_tokenizer):
        parser = DeepSeekV4Parser(mock_tokenizer)
        chunks = ["</think>", "Here is the direct answer."]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert reasoning == ""
        assert "</think>" not in content
        assert "Here is the direct answer" in content

    def test_duplicate_think_start_absorbed(self, mock_tokenizer):
        parser = DeepSeekV4Parser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        chunks = [
            "<think>\n",
            "Some reasoning.\n",
            "</think>\n",
            "Answer.",
        ]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert "Some reasoning" in reasoning
        assert "Answer" in content


# ── Missing </｜DSML｜invoke> before </｜DSML｜tool_calls> ────────────


class TestMissingInvokeEnd:
    def test_non_streaming(self, mock_tokenizer, mock_request):
        parser = DeepSeekV4Parser(mock_tokenizer)
        text = (
            f"{DSML_TOOL_START}"
            f"{DSML_INVOKE_PREFIX}get_weather{DSML_INVOKE_NAME_END}\n"
            f"{_param('location', 'true', 'NYC')}\n"
            f"{DSML_TOOL_END}"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "NYC"}

    def test_streaming_with_trailing_content(self, mock_tokenizer, mock_request):
        parser = DeepSeekV4Parser(mock_tokenizer)
        chunks = [
            DSML_TOOL_START,
            f"{DSML_INVOKE_PREFIX}get_weather{DSML_INVOKE_NAME_END}\n"
            f"{_param('location', 'true', 'NYC')}\n",
            DSML_TOOL_END,
            "Done.",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        assert collect_function_name(results) == "get_weather"
        args = json.loads(collect_tool_arguments(results))
        assert args == {"location": "NYC"}
        assert "Done." in collect_content(results)


# ── Thinking mode initial state ──────────────────────────────────────


class TestThinkingModeConfig:
    def test_thinking_true_starts_in_reasoning(self):
        cfg = deepseek_v4_config(thinking=True)
        assert cfg.initial_state.name == "REASONING"

    def test_thinking_false_starts_in_content(self):
        cfg = deepseek_v4_config(thinking=False)
        assert cfg.initial_state.name == "CONTENT"

    def test_enable_thinking_kwarg(self, mock_tokenizer):
        p = DeepSeekV4Parser(
            mock_tokenizer, chat_template_kwargs={"enable_thinking": True}
        )
        assert p.parser_engine_config.initial_state.name == "REASONING"

    def test_no_thinking_kwarg_defaults_to_content(self, mock_tokenizer):
        p = DeepSeekV4Parser(mock_tokenizer)
        assert p.parser_engine_config.initial_state.name == "CONTENT"

    def test_thinking_mode_reasoning_without_tags(self, mock_tokenizer):
        parser = DeepSeekV4Parser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        chunks = [
            "\n\nLet me consider ",
            "this carefully.\n",
            "</think>\n",
            "Here is the result.",
        ]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert "Let me consider" in reasoning
        assert "Here is the result" in content

    def test_thinking_mode_all_reasoning_no_end_tag(self, mock_tokenizer):
        parser = DeepSeekV4Parser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        chunks = ["I'll review ", "the PR."]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert "review" in reasoning
        assert "the PR" in reasoning
        assert content == ""


# ── Wrapper argument unwrapping ──────────────────────────────────────


class TestWrapperUnwrapping:
    def test_unwrap_arguments_wrapper(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        )

        result = DeepSeekV4Parser._unwrap_wrapper_args(
            '{"arguments": {"location": "Beijing"}}',
            [tool],
            "get_weather",
        )
        assert json.loads(result) == {"location": "Beijing"}

    def test_unwrap_input_wrapper(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        )

        result = DeepSeekV4Parser._unwrap_wrapper_args(
            '{"input": {"location": "Beijing"}}',
            [tool],
            "get_weather",
        )
        assert json.loads(result) == {"location": "Beijing"}

    def test_no_unwrap_when_key_in_schema(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "func",
                "parameters": {
                    "type": "object",
                    "properties": {"arguments": {"type": "string"}},
                },
            },
        )

        result = DeepSeekV4Parser._unwrap_wrapper_args(
            '{"arguments": "some value"}',
            [tool],
            "func",
        )
        assert json.loads(result) == {"arguments": "some value"}

    def test_no_unwrap_when_no_tools(self):
        result = DeepSeekV4Parser._unwrap_wrapper_args(
            '{"arguments": {"location": "Beijing"}}',
            None,
            "get_weather",
        )
        assert json.loads(result) == {"arguments": {"location": "Beijing"}}

    def test_unwrap_json_string_inner(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        )

        result = DeepSeekV4Parser._unwrap_wrapper_args(
            '{"arguments": "{\\"location\\": \\"Beijing\\"}"}',
            [tool],
            "get_weather",
        )
        assert json.loads(result) == {"location": "Beijing"}


# ── Parallel tool call wrapper unwrapping ───────────────────────────


def _make_tool(name, properties):
    from vllm.entrypoints.openai.chat_completion.protocol import (  # noqa: E501
        ChatCompletionToolsParam,
    )

    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        },
    )


def _invoke(name, *params):
    body = "\n".join(_param(n, s, v) for n, s, v in params)
    return (
        f"{DSML_INVOKE_PREFIX}{name}{DSML_INVOKE_NAME_END}\n{body}\n{DSML_INVOKE_END}"
    )


def _tool_calls(*invokes):
    return DSML_TOOL_START + "\n".join(invokes) + DSML_TOOL_END


class TestParallelUnwrapping:
    @pytest.fixture
    def weather_tool(self):
        return _make_tool(
            "get_weather",
            {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        )

    @pytest.fixture
    def time_tool(self):
        return _make_tool(
            "get_time",
            {"timezone": {"type": "string"}},
        )

    @pytest.mark.parametrize(
        "weather_args, expected",
        [
            (
                '{"location": "NYC", "unit": "celsius"}',
                {"location": "NYC", "unit": "celsius"},
            ),
            ('{"location": "NYC"}', {"location": "NYC"}),
        ],
        ids=["all_props", "subset_props"],
    )
    def test_unwrap_parallel_uses_correct_schema(
        self,
        mock_tokenizer,
        mock_request,
        weather_tool,
        time_tool,
        weather_args,
        expected,
    ):
        tools = [weather_tool, time_tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        text = _tool_calls(
            _invoke("get_weather", ("arguments", "false", weather_args)),
            _invoke("get_time", ("timezone", "true", "EST")),
        )

        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert args0 == expected
        assert result.tool_calls[1].function.name == "get_time"
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert args1 == {"timezone": "EST"}

    def test_unwrap_parallel_streaming(
        self, mock_tokenizer, mock_request, weather_tool, time_tool
    ):
        tools = [weather_tool, time_tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        chunks = [
            DSML_TOOL_START,
            _invoke(
                "get_weather",
                ("arguments", "false", '{"location": "NYC"}'),
            ),
            _invoke("get_time", ("timezone", "true", "EST")),
            DSML_TOOL_END,
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)
        final_delta, _ = results[-1]
        finish_delta = parser.finish_streaming()
        extracted = parser._build_extracted_result(final_delta, finish_delta)

        assert extracted.tools_called is True
        assert len(extracted.tool_calls) == 2
        args0 = json.loads(extracted.tool_calls[0].function.arguments)
        assert args0 == {"location": "NYC"}
        args1 = json.loads(extracted.tool_calls[1].function.arguments)
        assert args1 == {"timezone": "EST"}

    def test_no_unwrap_parallel_when_no_match(
        self, mock_tokenizer, mock_request, weather_tool, time_tool
    ):
        tools = [weather_tool, time_tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        text = _tool_calls(
            _invoke(
                "get_weather",
                ("arguments", "false", '{"unknown_key": "val"}'),
            ),
            _invoke("get_time", ("timezone", "true", "EST")),
        )

        result = parser.extract_tool_calls(text, mock_request)

        assert len(result.tool_calls) == 2
        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert args0 == {"arguments": {"unknown_key": "val"}}
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert args1 == {"timezone": "EST"}

    def test_unwrap_single_tool_still_works(self, mock_tokenizer, mock_request):
        tool = _make_tool("get_weather", {"location": {"type": "string"}})
        tools = [tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        text = _tool_calls(
            _invoke(
                "get_weather",
                ("arguments", "false", '{"location": "Beijing"}'),
            ),
        )

        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "Beijing"}
