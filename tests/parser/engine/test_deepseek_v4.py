# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek V4-specific parser engine semantics."""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.replay_harness import (
    DUMMY_TOOLS,
    MockTokenizer,
    _test_request,
    collect_output,
    replay_streaming,
)
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_reasoning_streaming,
    simulate_tool_streaming,
)
from vllm.parser.abstract_parser import DelegatingParser
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
    _unwrap_wrapper_args,
    deepseek_v4_config,
)
from vllm.parser.engine.registered_adapters import (
    DeepSeekV4ParserReasoningAdapter,
    DeepSeekV4ParserToolAdapter,
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

    def test_reasoning_effort_none_overrides_enable_thinking(self, mock_tokenizer):
        p = DeepSeekV4Parser(
            mock_tokenizer,
            chat_template_kwargs={
                "enable_thinking": True,
                "reasoning_effort": "none",
            },
        )
        assert p.parser_engine_config.initial_state.name == "CONTENT"


# ── Implicit reasoning end (missing </think> before tool calls) ─────


class TestImplicitReasoningEnd:
    """Tool call markers end reasoning implicitly when </think> is missing.

    DeepSeek V4 models occasionally omit </think> before emitting tool calls.
    The (REASONING, TOOL_START) transition handles this gracefully.
    """

    @pytest.fixture
    def thinking_parser(self, mock_tokenizer):
        return DeepSeekV4Parser(mock_tokenizer, chat_template_kwargs={"thinking": True})

    def _reasoning_then_tool(self, reasoning_text: str) -> str:
        return reasoning_text + _tool_calls(
            _invoke("get_weather", ("location", "true", "NYC")),
        )

    def test_non_streaming_extract_reasoning_implicit_end(self, thinking_parser):
        text = self._reasoning_then_tool("Let me look up the weather.\n\n")
        reasoning, content = thinking_parser.extract_reasoning(text, None)
        assert reasoning == "Let me look up the weather."
        assert DSML_TOOL_START not in reasoning
        assert DSML_INVOKE_PREFIX not in reasoning
        assert content is None

    def test_non_streaming_extract_tool_calls_implicit_end(
        self, thinking_parser, mock_request
    ):
        text = self._reasoning_then_tool("Let me look up the weather.\n\n")
        result = thinking_parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "NYC"}

    def test_non_streaming_parse_implicit_end(self, thinking_parser, mock_request):
        text = self._reasoning_then_tool("Let me look up the weather.\n\n")
        reasoning, content, tool_calls = thinking_parser.parse(text, mock_request)
        assert reasoning == "Let me look up the weather."
        assert content is None
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        args = json.loads(tool_calls[0].arguments)
        assert args == {"location": "NYC"}

    def test_streaming_reasoning_implicit_end(self, thinking_parser):
        chunks = [
            "Let me look up the weather.\n\n",
            DSML_TOOL_START,
            DSML_INVOKE_PREFIX + "get_weather" + DSML_INVOKE_NAME_END,
        ]
        reasoning, content = simulate_reasoning_streaming(thinking_parser, chunks)
        assert reasoning == "Let me look up the weather."
        assert DSML_TOOL_START not in reasoning
        assert DSML_INVOKE_PREFIX not in reasoning

    def test_streaming_tool_extraction_implicit_end(
        self, thinking_parser, mock_request
    ):
        chunks = [
            "Let me check.\n\n",
            DSML_TOOL_START,
            DSML_INVOKE_PREFIX
            + "get_weather"
            + DSML_INVOKE_NAME_END
            + "\n"
            + _param("location", "true", "NYC")
            + "\n"
            + DSML_INVOKE_END,
            DSML_TOOL_END,
        ]
        results = simulate_tool_streaming(thinking_parser, mock_request, chunks)
        assert collect_function_name(results) == "get_weather"
        args = json.loads(collect_tool_arguments(results))
        assert args == {"location": "NYC"}

    def test_thinking_false_explicit_think_then_tool_call(self, mock_tokenizer):
        parser = DeepSeekV4Parser(mock_tokenizer)
        chunks = [
            DSML_THINK_START,
            "Let me check the weather.",
            DSML_TOOL_START,
            DSML_INVOKE_PREFIX + "get_weather" + DSML_INVOKE_NAME_END,
        ]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert "Let me check the weather" in reasoning
        assert DSML_TOOL_START not in reasoning
        assert DSML_THINK_START not in reasoning

    def test_non_streaming_parallel_tools_after_implicit_end(
        self, thinking_parser, mock_request
    ):
        text = "I need both.\n\n" + _tool_calls(
            _invoke("get_weather", ("location", "true", "NYC")),
            _invoke("get_time", ("timezone", "true", "EST")),
        )
        result = thinking_parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_streaming_implicit_end_trailing_whitespace_stripped(self, thinking_parser):
        chunks = [
            "Reasoning.\n\n\n",
            DSML_TOOL_START,
            DSML_INVOKE_PREFIX + "func" + DSML_INVOKE_NAME_END,
        ]
        reasoning, content = simulate_reasoning_streaming(thinking_parser, chunks)
        assert reasoning == "Reasoning."


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

        result = _unwrap_wrapper_args(
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

        result = _unwrap_wrapper_args(
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

        result = _unwrap_wrapper_args(
            '{"arguments": "some value"}',
            [tool],
            "func",
        )
        assert json.loads(result) == {"arguments": "some value"}

    def test_no_unwrap_when_no_tools(self):
        result = _unwrap_wrapper_args(
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

        result = _unwrap_wrapper_args(
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


# ── Streaming wrapper consistency ─────────────────────────────────────


class TestStreamingWrapperConsistency:
    """Streamed arg deltas must stay consistent with final extraction
    when wrapper params like 'arguments' are unwrapped."""

    def test_streaming_wrapper_unwrap_consistency(self, mock_tokenizer, mock_request):
        tool = _make_tool("get_weather", {"location": {"type": "string"}})
        tools = [tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        chunks = [
            DSML_TOOL_START,
            _invoke(
                "get_weather",
                ("arguments", "false", '{"location": "NYC"}'),
            ),
            DSML_TOOL_END,
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)
        streamed_args = collect_tool_arguments(results)

        final_delta, _ = results[-1]
        finish_delta = parser.finish_streaming()
        extracted = parser._build_extracted_result(final_delta, finish_delta)

        assert extracted.tools_called is True
        assert len(extracted.tool_calls) == 1

        final_args = extracted.tool_calls[0].function.arguments
        assert json.loads(final_args) == {"location": "NYC"}

        assert '"arguments"' not in streamed_args, (
            f"Streamed args should not contain wrapper key, got: {streamed_args!r}"
        )

        assert final_args.startswith(streamed_args), (
            f"Extracted args {final_args!r} "
            f"should start with streamed args {streamed_args!r}"
        )


# ── DelegatingParser: large delta with </think> + tool calls ─────────

_DSV4_FULL_VOCAB = {
    DSML_THINK_START: 128821,
    DSML_THINK_END: 128822,
    DSML_TOOL_START: 128823,
    DSML_TOOL_END: 128824,
}


class _DeepSeekV4Delegating(DelegatingParser):
    reasoning_parser_cls = DeepSeekV4ParserReasoningAdapter
    tool_parser_cls = DeepSeekV4ParserToolAdapter


def _dsv4_tokens(
    reasoning: str,
    tool_name: str,
    params: list[tuple[str, str, str]],
) -> list[tuple[int, str]]:
    """Build a token sequence: reasoning + </think> + DSML tool block."""
    tokens: list[tuple[int, str]] = []
    tid = 100

    for word in reasoning.split(" "):
        prefix = " " if tokens else ""
        tokens.append((tid, prefix + word))
        tid += 1

    tokens.append((_DSV4_FULL_VOCAB[DSML_THINK_END], DSML_THINK_END))

    tokens.append((tid, "\n\n"))
    tid += 1

    tokens.append((_DSV4_FULL_VOCAB[DSML_TOOL_START], DSML_TOOL_START))

    tokens.append((tid, "\n"))
    tid += 1

    invoke_prefix_text = f"{DSML_INVOKE_PREFIX}{tool_name}{DSML_INVOKE_NAME_END}"
    tokens.append((tid, invoke_prefix_text))
    tid += 1

    tokens.append((tid, "\n"))
    tid += 1

    for name, is_str, value in params:
        param_text = _param(name, is_str, value)
        tokens.append((tid, param_text))
        tid += 1
        tokens.append((tid, "\n"))
        tid += 1

    tokens.append((tid, DSML_INVOKE_END))
    tid += 1

    tokens.append((tid, "\n"))
    tid += 1

    tokens.append((_DSV4_FULL_VOCAB[DSML_TOOL_END], DSML_TOOL_END))

    return tokens


class TestDelegatingParserLargeDelta:
    """Regression: tool calls lost when </think> + DSML arrive in same delta.

    The DelegatingParser used by the serving layer splits reasoning and
    tool parsing across two separate engine instances.  When </think> and
    the entire DSML tool block arrive in a single large streaming delta,
    the content transfer from reasoning adapter to tool adapter must
    preserve the tool call text.
    """

    @pytest.fixture
    def dsv4_tokens(self):
        return _dsv4_tokens(
            reasoning="The user wants the current weather in Berlin.",
            tool_name="get_weather",
            params=[
                ("location", "true", "Berlin"),
                ("units", "true", "celsius"),
            ],
        )

    @pytest.fixture
    def dsv4_tokenizer(self, dsv4_tokens):
        return MockTokenizer(
            vocab=dict(_DSV4_FULL_VOCAB),
            tokens=dsv4_tokens,
        )

    @pytest.mark.parametrize(
        "chunk_size",
        [1, 2, 3, 5, None],
        ids=lambda c: f"chunk={c}",
    )
    def test_tool_calls_extracted_at_all_chunk_sizes(
        self, dsv4_tokenizer, dsv4_tokens, chunk_size
    ):
        parser = _DeepSeekV4Delegating(
            dsv4_tokenizer,
            chat_template_kwargs={"thinking": True},
        )
        deltas = replay_streaming(
            parser,
            dsv4_tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
        )
        output = collect_output(deltas)

        assert "The user wants" in output.reasoning
        assert len(output.tool_calls) == 1, (
            f"Expected 1 tool call but got {len(output.tool_calls)}; "
            f"reasoning={output.reasoning!r}, content={output.content!r}"
        )
        assert output.tool_calls[0]["name"] == "get_weather"
        args = json.loads(output.tool_calls[0]["arguments"])
        assert args == {"location": "Berlin", "units": "celsius"}

    def test_eos_drop_token_does_not_swallow_tool_calls(self):
        """Tool calls must survive when an EOS DROP token's ID is in
        delta_token_ids but its text is absent from delta_text.

        At large stream_interval the EOS token ID arrives in the same
        delta as </think> + tool calls but the detokenizer strips the
        EOS text.  The scanner's _rebuild_from_anchors defers all text
        after </think> when it can't find the EOS anchor text.  The
        reasoning adapter's finish_streaming must flush deferred text
        as content (with skip_tool_parsing), not as tool calls.
        """
        eos_text = "<｜end▁of▁sentence｜>"
        eos_id = 128801
        vocab = {
            DSML_THINK_START: 128821,
            DSML_THINK_END: 128822,
            eos_text: eos_id,
        }

        reasoning = "The user wants weather."
        tool_block = (
            "\n\n"
            + DSML_TOOL_START
            + "\n"
            + DSML_INVOKE_PREFIX
            + "get_weather"
            + DSML_INVOKE_NAME_END
            + "\n"
            + _param("location", "true", "Berlin")
            + "\n"
            + DSML_INVOKE_END
            + "\n"
            + DSML_TOOL_END
        )
        # delta_text does NOT include EOS text (detokenizer strips it)
        full_text = reasoning + DSML_THINK_END + tool_block
        # Build token list: word-split reasoning, then special tokens,
        # then word-split tool block content, then EOS.
        # EOS ID is present but its text is NOT in delta_text.
        tokens: list[tuple[int, str]] = []
        tid = 100
        for word in reasoning.split(" "):
            pfx = " " if tokens else ""
            tokens.append((tid, pfx + word))
            tid += 1
        tokens.append((128822, DSML_THINK_END))
        for ch in tool_block:
            tokens.append((tid, ch))
            tid += 1
        tokens.append((eos_id, eos_text))

        all_ids = [t[0] for t in tokens]
        tokenizer = MockTokenizer(vocab=vocab, tokens=tokens)
        request = _test_request(tools=DUMMY_TOOLS)

        # All-in-one delta: EOS ID in token_ids but text NOT in
        # delta_text (detokenizer strips EOS).  This is the scenario
        # at large stream_interval.
        parser = _DeepSeekV4Delegating(
            tokenizer,
            chat_template_kwargs={"thinking": True},
        )
        deltas = [
            parser.parse_delta(
                full_text,
                all_ids,
                request,
                prompt_token_ids=[],
                finished=True,
            )
        ]

        output = collect_output(deltas)

        assert "The user wants" in output.reasoning
        assert len(output.tool_calls) == 1, (
            f"Expected 1 tool call but got {len(output.tool_calls)}; "
            f"reasoning={output.reasoning!r}, content={output.content!r}"
        )
        assert output.tool_calls[0]["name"] == "get_weather"
        args = json.loads(output.tool_calls[0]["arguments"])
        assert args == {"location": "Berlin"}

    @pytest.mark.parametrize(
        "chunk_size",
        [1, 2, 3, 5, None],
        ids=lambda c: f"chunk={c}",
    )
    def test_eos_not_leaked_when_reasoning_never_ends(self, chunk_size):
        """EOS must not leak into reasoning_content when the model never
        emits </think> (generation ends while still in REASONING state)."""
        eos_text = "<｜end▁of▁sentence｜>"
        eos_id = 128801
        vocab = {
            **_DSV4_FULL_VOCAB,
            eos_text: eos_id,
        }

        reasoning_text = "Good morning! How can I help you today?"
        tokens: list[tuple[int, str]] = []
        tid = 100
        for word in reasoning_text.split(" "):
            prefix = " " if tokens else ""
            tokens.append((tid, prefix + word))
            tid += 1
        tokens.append((eos_id, eos_text))

        tokenizer = MockTokenizer(vocab=vocab, tokens=tokens)
        parser = _DeepSeekV4Delegating(
            tokenizer,
            chat_template_kwargs={"thinking": True},
        )
        deltas = replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
        )
        output = collect_output(deltas)

        assert reasoning_text in output.reasoning
        assert eos_text not in output.reasoning
        assert output.content == ""
        assert output.tool_calls == []


# ── #48645: force_nonempty_content -- surface unterminated reasoning as
# content ─────────────────────────────────────────────────────────────
#
# Ports the Nemotron V3 pattern (PR #39091, vllm/parser/nemotron_v3.py) to
# deepseek_v4, per bbrowning's guidance on #48645 (2026-07-15): opt-in via
# a chat_template_kwargs flag, never touching a properly closed </think>.
# See DeepSeekV4Parser.get_streaming_fallback_content / .extract_reasoning
# for the implementation these tests exercise.


class TestForceNonemptyContentNonStreaming:
    """Non-streaming: DeepSeekV4Parser.extract_reasoning() swap."""

    def test_misroute_flushed(self, mock_tokenizer, mock_request):
        parser = DeepSeekV4Parser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        mock_request.chat_template_kwargs = {"force_nonempty_content": True}
        text = "Good morning! How can I help you today?"

        reasoning, content = parser.extract_reasoning(text, mock_request)

        assert reasoning is None
        assert content == text

    def test_no_flush_without_opt_in(self, mock_tokenizer, mock_request):
        parser = DeepSeekV4Parser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        mock_request.chat_template_kwargs = None
        text = "Good morning! How can I help you today?"

        reasoning, content = parser.extract_reasoning(text, mock_request)

        assert reasoning == text
        assert content is None

    def test_end_token_seen_not_flushed(self, mock_tokenizer, mock_request):
        """``</think>`` WAS seen: real, deliberate CoT -- never promoted,
        even though nothing followed it.

        Deliberately narrower than the Nemotron V3 precedent, whose gate
        is only "content ended up empty" and promotes in this case too;
        see PR description for the rationale.
        """
        parser = DeepSeekV4Parser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        mock_request.chat_template_kwargs = {"force_nonempty_content": True}
        text = f"Deliberate reasoning.{DSML_THINK_END}"

        reasoning, content = parser.extract_reasoning(text, mock_request)

        assert reasoning == "Deliberate reasoning."
        assert content is None

    def test_tool_call_not_flushed(self, mock_tokenizer, mock_request):
        """A tool call started directly from reasoning (no ``</think>``)
        structurally ends reasoning via the ``(REASONING, TOOL_START)``
        transition, so it is never flushed here -- it's real deliberation
        that led to a real tool call, not a misroute.
        """
        parser = DeepSeekV4Parser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        mock_request.chat_template_kwargs = {"force_nonempty_content": True}
        text = "Let me check the weather.\n\n" + _tool_calls(
            _invoke("get_weather", ("location", "true", "NYC"))
        )

        reasoning, content = parser.extract_reasoning(text, mock_request)

        assert reasoning == "Let me check the weather."
        assert content is None

    def test_non_streaming_has_no_finish_reason_gate(
        self, mock_tokenizer, mock_request
    ):
        """Documents an accepted gap (see PR description): ``extract_reasoning``
        is a shared abstract method with no ``finish_reason`` parameter, so
        unlike the streaming path (below) the non-streaming swap cannot
        distinguish a natural stop from a length-truncated generation. An
        opted-in request with a truncated, never-closed trace *is*
        promoted here.
        """
        parser = DeepSeekV4Parser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        mock_request.chat_template_kwargs = {"force_nonempty_content": True}
        text = "This trace got cut off mid-sent"

        reasoning, content = parser.extract_reasoning(text, mock_request)

        assert reasoning is None
        assert content == text


def _word_tokens(text: str, start_id: int = 100) -> list[tuple[int, str]]:
    tokens: list[tuple[int, str]] = []
    tid = start_id
    for word in text.split(" "):
        prefix = " " if tokens else ""
        tokens.append((tid, prefix + word))
        tid += 1
    return tokens


class TestForceNonemptyContentStreaming:
    """Streaming: ``DelegatingParser.finalize_generation`` ->
    ``DeepSeekV4Parser.get_streaming_fallback_content``, through the same
    reasoning+tool adapter split serving wires up for
    ``--reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4``
    (see vllm/parser/engine/registered_adapters.py).
    """

    @pytest.mark.parametrize(
        "chunk_size", [1, 2, 3, 5, None], ids=lambda c: f"chunk={c}"
    )
    def test_misroute_flushed(self, chunk_size):
        text = "Good morning! How can I help you today?"
        tokens = _word_tokens(text)
        tokenizer = MockTokenizer(vocab=dict(_DSV4_FULL_VOCAB), tokens=tokens)
        parser = _DeepSeekV4Delegating(
            tokenizer, chat_template_kwargs={"thinking": True}
        )

        deltas = replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            finish_reason="stop",
            chat_template_kwargs={"force_nonempty_content": True},
        )
        output = collect_output(deltas)

        assert output.reasoning == text
        assert output.content == text

    @pytest.mark.parametrize(
        "chunk_size", [1, 2, 3, 5, None], ids=lambda c: f"chunk={c}"
    )
    def test_no_flush_without_opt_in(self, chunk_size):
        text = "Good morning! How can I help you today?"
        tokens = _word_tokens(text)
        tokenizer = MockTokenizer(vocab=dict(_DSV4_FULL_VOCAB), tokens=tokens)
        parser = _DeepSeekV4Delegating(
            tokenizer, chat_template_kwargs={"thinking": True}
        )

        deltas = replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            finish_reason="stop",
        )
        output = collect_output(deltas)

        assert output.reasoning == text
        assert output.content == ""

    @pytest.mark.parametrize(
        "chunk_size", [1, 2, 3, 5, None], ids=lambda c: f"chunk={c}"
    )
    def test_length_truncation_not_flushed(self, chunk_size):
        """``finish_reason="length"``: generation was cut off by the token
        budget, not a natural stop. Truncated, mid-sentence CoT must never
        become a user-facing "answer".
        """
        text = "This is a very long chain of thought that got cut off"
        tokens = _word_tokens(text)
        tokenizer = MockTokenizer(vocab=dict(_DSV4_FULL_VOCAB), tokens=tokens)
        parser = _DeepSeekV4Delegating(
            tokenizer, chat_template_kwargs={"thinking": True}
        )

        deltas = replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            finish_reason="length",
            chat_template_kwargs={"force_nonempty_content": True},
        )
        output = collect_output(deltas)

        assert output.reasoning == text
        assert output.content == ""

    @pytest.mark.parametrize(
        "chunk_size", [1, 2, 3, 5, None], ids=lambda c: f"chunk={c}"
    )
    def test_no_finish_reason_not_flushed(self, chunk_size):
        """``finish_reason`` unset (``None``) is treated the same as "not
        confirmed stop": deny, not allow. A caller that doesn't thread
        ``finish_reason`` through gets the pre-existing (no-flush)
        behavior, never a silently-unsafe default.
        """
        text = "Good morning! How can I help you today?"
        tokens = _word_tokens(text)
        tokenizer = MockTokenizer(vocab=dict(_DSV4_FULL_VOCAB), tokens=tokens)
        parser = _DeepSeekV4Delegating(
            tokenizer, chat_template_kwargs={"thinking": True}
        )

        deltas = replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            chat_template_kwargs={"force_nonempty_content": True},
        )
        output = collect_output(deltas)

        assert output.reasoning == text
        assert output.content == ""

    @pytest.mark.parametrize(
        "chunk_size", [1, 2, 3, 5, None], ids=lambda c: f"chunk={c}"
    )
    def test_end_token_seen_not_flushed(self, chunk_size):
        """``</think>`` WAS closed (even though nothing followed): real
        CoT, never promoted."""
        tokens = _word_tokens("Deliberate reasoning.") + [
            (_DSV4_FULL_VOCAB[DSML_THINK_END], DSML_THINK_END)
        ]
        tokenizer = MockTokenizer(vocab=dict(_DSV4_FULL_VOCAB), tokens=tokens)
        parser = _DeepSeekV4Delegating(
            tokenizer, chat_template_kwargs={"thinking": True}
        )

        deltas = replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            finish_reason="stop",
            chat_template_kwargs={"force_nonempty_content": True},
        )
        output = collect_output(deltas)

        assert output.reasoning == "Deliberate reasoning."
        assert output.content == ""

    def test_tool_call_not_flushed(self):
        """Tool call started directly from reasoning (no ``</think>``):
        ``(REASONING, TOOL_START)`` flips ``reasoning_ended`` True before
        any tool event, so this is never flushed even though ``</think>``
        was never seen.
        """
        # DSML_TOOL_START/END must carry their real vocab token IDs (as
        # _dsv4_tokens does) rather than _word_tokens' arbitrary sequential
        # IDs: once any real special-token ID has been seen, the engine
        # only accepts a *token-ID-confirmed* TOOL_START as a state
        # transition, treating literal "looks like a marker" text as
        # ordinary content instead (the same safety net that keeps a
        # user's prose mentioning "<tool_call>" from being misparsed).
        tokens = _word_tokens("Let me check the weather")
        tid = 900
        tokens.append((tid, "\n\n"))
        tid += 1
        tokens.append((_DSV4_FULL_VOCAB[DSML_TOOL_START], DSML_TOOL_START))
        tokens.append((tid, f"{DSML_INVOKE_PREFIX}get_weather{DSML_INVOKE_NAME_END}"))
        tid += 1
        tokens.append((tid, "\n"))
        tid += 1
        tokens.append((tid, _param("location", "true", "NYC")))
        tid += 1
        tokens.append((tid, "\n"))
        tid += 1
        tokens.append((tid, DSML_INVOKE_END))
        tid += 1
        tokens.append((_DSV4_FULL_VOCAB[DSML_TOOL_END], DSML_TOOL_END))

        tokenizer = MockTokenizer(vocab=dict(_DSV4_FULL_VOCAB), tokens=tokens)
        parser = _DeepSeekV4Delegating(
            tokenizer, chat_template_kwargs={"thinking": True}
        )

        deltas = replay_streaming(
            parser,
            tokens,
            chunk_size=None,
            finished_on_last=True,
            finish_reason="stop",
            chat_template_kwargs={"force_nonempty_content": True},
            tools=DUMMY_TOOLS,
        )
        output = collect_output(deltas)

        assert "Let me check the weather" in output.reasoning
        assert output.content == ""
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0]["name"] == "get_weather"

    def test_flushed_content_has_no_trailing_eos(self):
        """The flushed content is built from the same already-EOS-stripped
        reasoning chunks validated by
        ``TestDelegatingParserLargeDelta.test_eos_not_leaked_when_reasoning_never_ends``
        (fixed upstream by #48748) -- so promoting them to content must
        not reintroduce the EOS text there either.
        """
        eos_text = "<｜end▁of▁sentence｜>"
        eos_id = 128801
        vocab = {**_DSV4_FULL_VOCAB, eos_text: eos_id}

        reasoning_text = "Good morning! How can I help you today?"
        tokens = _word_tokens(reasoning_text) + [(eos_id, eos_text)]
        tokenizer = MockTokenizer(vocab=vocab, tokens=tokens)
        parser = _DeepSeekV4Delegating(
            tokenizer, chat_template_kwargs={"thinking": True}
        )

        deltas = replay_streaming(
            parser,
            tokens,
            chunk_size=None,
            finished_on_last=True,
            finish_reason="stop",
            chat_template_kwargs={"force_nonempty_content": True},
        )
        output = collect_output(deltas)

        assert output.reasoning == reasoning_text
        assert eos_text not in output.reasoning
        assert output.content == reasoning_text
        assert eos_text not in output.content
