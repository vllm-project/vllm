# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek V3.2 parser engine semantics.

V3.2 uses the same DSML parameter format as V4 but wraps tool calls in
``<｜DSML｜function_calls>`` instead of ``<｜DSML｜tool_calls>`` and has
no reasoning (``<think>``/``</think>``) support.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.parser.deepseek_v4 import (
    DSML_INVOKE_END,
    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
)
from vllm.parser.deepseek_v32 import (
    DSML_FUNC_END,
    DSML_FUNC_START,
    DeepSeekV32Parser,
)
from vllm.parser.engine.parser_engine_config import ParserState

_PARAM_OPEN = '｜DSML｜parameter name="{name}" string="{is_str}">'
_PARAM_CLOSE = "</｜DSML｜parameter>"


def _param(name: str, is_str: str, value: str) -> str:
    return f"<{_PARAM_OPEN.format(name=name, is_str=is_str)}{value}{_PARAM_CLOSE}"


def _invoke(name: str, *params: str) -> str:
    body = "\n".join(params)
    return (
        f"{DSML_INVOKE_PREFIX}{name}{DSML_INVOKE_NAME_END}\n{body}\n{DSML_INVOKE_END}"
    )


def _func_calls(*invocations: str) -> str:
    body = "\n".join(invocations)
    return f"{DSML_FUNC_START}\n{body}\n{DSML_FUNC_END}"


def _make_tool(name, properties):
    from vllm.entrypoints.openai.chat_completion.protocol import (
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


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer({})


@pytest.fixture
def mock_request():
    from unittest.mock import MagicMock

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = []
    req.tool_choice = "auto"
    return req


# ── Non-streaming extraction ────────────────────────────────────────


class TestNonStreaming:
    def test_no_tool_call(self, mock_tokenizer, mock_request):
        parser = DeepSeekV32Parser(mock_tokenizer)
        result = parser.extract_tool_calls("Hello world", mock_request)
        assert not result.tools_called
        assert result.content == "Hello world"

    def test_single_tool(self, mock_tokenizer, mock_request):
        text = _func_calls(
            _invoke("get_weather", _param("city", "true", "SF")),
        )
        parser = DeepSeekV32Parser(mock_tokenizer)
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "SF"}

    def test_parallel_tools(self, mock_tokenizer, mock_request):
        text = _func_calls(
            _invoke("get_weather", _param("city", "true", "SF")),
            _invoke("get_weather", _param("city", "true", "NYC")),
        )
        parser = DeepSeekV32Parser(mock_tokenizer)
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "SF"}
        assert json.loads(result.tool_calls[1].function.arguments) == {"city": "NYC"}

    def test_content_before_tool_call(self, mock_tokenizer, mock_request):
        text = "Let me check. " + _func_calls(
            _invoke("search", _param("q", "true", "vllm")),
        )
        parser = DeepSeekV32Parser(mock_tokenizer)
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called
        assert result.content is not None
        assert "Let me check" in result.content

    def test_non_string_params_json_parsed(self, mock_tokenizer, mock_request):
        text = _func_calls(
            _invoke(
                "toggle",
                _param("enabled", "false", "true"),
                _param("count", "false", "42"),
            ),
        )
        parser = DeepSeekV32Parser(mock_tokenizer)
        result = parser.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["enabled"] is True
        assert args["count"] == 42

    def test_wrapper_unwrapping(self, mock_tokenizer, mock_request):
        tool = _make_tool("get_weather", {"location": {"type": "string"}})
        mock_request.tools = [tool]
        text = _func_calls(
            _invoke(
                "get_weather",
                _param("arguments", "false", '{"location":"Beijing"}'),
            ),
        )
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[tool])
        result = parser.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "Beijing"}


# ── Initial state ────────────────────────────────────────────────────


class TestInitialState:
    def test_always_content(self, mock_tokenizer):
        parser = DeepSeekV32Parser(mock_tokenizer)
        cfg = parser.parser_engine_config
        assert cfg.initial_state == ParserState.CONTENT

    def test_ignores_thinking_kwargs(self, mock_tokenizer):
        parser = DeepSeekV32Parser(
            mock_tokenizer,
            chat_template_kwargs={"thinking": True, "enable_thinking": True},
        )
        cfg = parser.parser_engine_config
        assert cfg.initial_state == ParserState.CONTENT


# ── Streaming ────────────────────────────────────────────────────────


class TestStreaming:
    def test_single_tool_streaming(self, mock_tokenizer, mock_request):
        text = _func_calls(
            _invoke("get_weather", _param("city", "true", "SF")),
        )
        parser = DeepSeekV32Parser(mock_tokenizer)
        results = simulate_tool_streaming(parser, mock_request, list(text))
        assert collect_function_name(results) == "get_weather"
        args_json = collect_tool_arguments(results)
        assert json.loads(args_json) == {"city": "SF"}

    def test_content_before_tool_streaming(self, mock_tokenizer, mock_request):
        text = "Checking... " + _func_calls(
            _invoke("fn", _param("k", "true", "v")),
        )
        parser = DeepSeekV32Parser(mock_tokenizer)
        results = simulate_tool_streaming(parser, mock_request, list(text))
        content = collect_content(results)
        assert "Checking" in content

    def test_parallel_tools_streaming(self, mock_tokenizer, mock_request):
        text = _func_calls(
            _invoke("fn_a", _param("x", "true", "1")),
            _invoke("fn_b", _param("y", "true", "2")),
        )
        parser = DeepSeekV32Parser(mock_tokenizer)
        results = simulate_tool_streaming(parser, mock_request, list(text))

        names = []
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)
        assert "fn_a" in names
        assert "fn_b" in names

    def test_no_tool_content_only(self, mock_tokenizer, mock_request):
        text = "Just some text, no tools."
        parser = DeepSeekV32Parser(mock_tokenizer)
        results = simulate_tool_streaming(parser, mock_request, list(text))
        content = collect_content(results)
        assert "Just some text" in content
        args = collect_tool_arguments(results)
        assert args == ""

    def test_streaming_wrapper_unwrap_consistency(self, mock_tokenizer, mock_request):
        tool = _make_tool("get_weather", {"location": {"type": "string"}})
        mock_request.tools = [tool]
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[tool])

        chunks = [
            DSML_FUNC_START,
            _invoke(
                "get_weather",
                _param("arguments", "false", '{"location": "NYC"}'),
            ),
            DSML_FUNC_END,
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
        assert '"arguments"' not in streamed_args
        assert final_args.startswith(streamed_args)

    def test_missing_invoke_end(self, mock_tokenizer, mock_request):
        text = (
            f"{DSML_FUNC_START}\n"
            f"{DSML_INVOKE_PREFIX}fn{DSML_INVOKE_NAME_END}\n"
            f"{_param('k', 'true', 'v')}\n"
            f"{DSML_FUNC_END}"
        )
        parser = DeepSeekV32Parser(mock_tokenizer)
        results = simulate_tool_streaming(parser, mock_request, list(text))
        assert collect_function_name(results) == "fn"
        args = json.loads(collect_tool_arguments(results))
        assert args == {"k": "v"}
