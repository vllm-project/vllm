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
    DSML_TOOL_START,
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


def _request_without_tools():
    from unittest.mock import MagicMock

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = []
    req.tool_choice = "auto"
    return req


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer({})


@pytest.fixture
def mock_request():
    return _request_without_tools()


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

    def test_missing_func_start_orphan_invoke(self, mock_tokenizer, mock_request):
        """Orphan invoke without the <｜DSML｜function_calls> wrapper is
        still parsed as a tool call when the request declared the tool
        (see gh-48931)."""
        tool = _make_tool("get_weather", {"city": {"type": "string"}})
        mock_request.tools = [tool]
        text = _invoke("get_weather", _param("city", "true", "SF")) + DSML_FUNC_END
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[tool])
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "SF"}
        assert result.content is None

    def test_orphan_invoke_without_declared_tools_stays_content(
        self, mock_tokenizer, mock_request
    ):
        """A request that declared no tools can never accept a recovered
        name, so the orphan invoke stays plain content."""
        text = _invoke("get_weather", _param("city", "true", "SF")) + DSML_FUNC_END
        parser = DeepSeekV32Parser(mock_tokenizer)
        result = parser.extract_tool_calls(text, mock_request)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == text

    def test_unclosed_foreign_wrapper_then_native_call(
        self, mock_tokenizer, mock_request
    ):
        """A foreign wrapper that never closes must not disable native
        tool parsing: the token backed function_calls wrapper still
        wins."""
        text = (
            DSML_TOOL_START
            + "\nStray foreign text.\n"
            + _func_calls(_invoke("get_weather", _param("city", "true", "SF")))
        )
        parser = DeepSeekV32Parser(mock_tokenizer)
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "SF"}
        assert "Stray foreign text." in result.content

    def test_foreign_tool_calls_wrapper_rejected(self, mock_tokenizer, mock_request):
        """An invoke inside the V4-style tool_calls wrapper stays plain
        content: the orphan fallback must not fire inside a foreign
        wrapper."""
        text = _func_calls(
            _invoke("get_weather", _param("city", "true", "SF")),
        ).replace("function_calls", "tool_calls")
        parser = DeepSeekV32Parser(mock_tokenizer)
        result = parser.extract_tool_calls(text, mock_request)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == text

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


# ── Orphan invoke name validation ────────────────────────────────────


class TestOrphanInvokeNameValidation:
    """Recovered (orphan) invokes must carry a plausible tool name.

    Mirrors the V4 coverage: when the request declares tools, the
    (CONTENT, INVOKE_PREFIX) recovery path only commits to a tool call
    if the parsed name is one of the declared functions; otherwise the
    consumed text is re-emitted as plain content.
    """

    @pytest.fixture
    def weather_tool(self):
        return _make_tool("get_weather", {"city": {"type": "string"}})

    def test_declared_name_recovered(self, mock_tokenizer, mock_request, weather_tool):
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool])
        mock_request.tools = [weather_tool]
        text = _invoke("get_weather", _param("city", "true", "SF")) + DSML_FUNC_END
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "SF"}
        assert result.content is None

    def test_undeclared_name_stays_content(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool])
        mock_request.tools = [weather_tool]
        text = (
            "Quoting "
            + DSML_INVOKE_PREFIX
            + "made_up_tool"
            + DSML_INVOKE_NAME_END
            + " literally."
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == text

    def test_char_by_char_undeclared_name_stays_content(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool])
        mock_request.tools = [weather_tool]
        text = DSML_INVOKE_PREFIX + "made_up_tool" + DSML_INVOKE_NAME_END + " after."
        results = simulate_tool_streaming(parser, mock_request, list(text))
        finish_delta = parser.finish_streaming()

        assert collect_function_name(results) is None
        content = collect_content(results) + (
            finish_delta.content if finish_delta and finish_delta.content else ""
        )
        assert content == text

    def test_quoted_marker_then_wrapped_call_non_streaming(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        """Prose that quotes the invoke marker and never closes it must
        not swallow a real wrapped tool call that follows."""
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool])
        mock_request.tools = [weather_tool]
        text = (
            "Docs quote "
            + DSML_INVOKE_PREFIX
            + " as the marker. "
            + _func_calls(_invoke("get_weather", _param("city", "true", "SF")))
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "SF"}
        assert DSML_INVOKE_PREFIX in result.content

    def test_quoted_marker_directly_before_wrapped_call(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        """A quoted marker followed immediately by the real wrapper must
        release the hold and parse the wrapped call."""
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool])
        mock_request.tools = [weather_tool]
        text = (
            "See "
            + DSML_INVOKE_PREFIX
            + _func_calls(_invoke("get_weather", _param("city", "true", "SF")))
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "SF"}
        assert DSML_INVOKE_PREFIX in result.content
        # The wrapper token opens the real tool call, so it must be
        # consumed by the parser rather than left in the content.
        assert DSML_FUNC_START not in result.content

    def test_streaming_quoted_marker_then_wrapped_call(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool])
        mock_request.tools = [weather_tool]
        chunks = [
            "Docs quote ",
            DSML_INVOKE_PREFIX,
            " as the marker. ",
            DSML_FUNC_START,
            _invoke("get_weather", _param("city", "true", "SF")),
            DSML_FUNC_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        assert collect_function_name(results) == "get_weather"
        args = json.loads(collect_tool_arguments(results))
        assert args == {"city": "SF"}
        assert DSML_INVOKE_PREFIX in collect_content(results)

    def test_streaming_quoted_marker_prose_released_before_finish(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        """Prose after a quoted marker must stream out promptly instead
        of being buffered until the end of the response."""
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool])
        mock_request.tools = [weather_tool]
        prose = "this marker starts a tool call block in the raw output."
        chunks = ["Quote: ", DSML_INVOKE_PREFIX, prose, " More prose."]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        assert collect_function_name(results) is None
        content = collect_content(results)
        assert prose in content
        assert " More prose." in content

    def test_trailing_prose_after_orphan_invoke_is_kept(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        """A model that drops the opening wrapper often drops the
        closing one too, which leaves the response ending between
        invokes.  The text after the invoke is real output and must
        survive as content."""
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool])
        mock_request.tools = [weather_tool]
        text = _invoke("get_weather", _param("city", "true", "SF")) + "\nThanks!"
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.content == "\nThanks!"

    def test_streaming_trailing_prose_after_orphan_invoke_is_kept(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool])
        mock_request.tools = [weather_tool]
        chunks = [_invoke("get_weather", _param("city", "true", "SF")), "\nThanks!"]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        assert collect_function_name(results) == "get_weather"
        # Streamed out as it arrives, not buffered until finish.
        assert collect_content(results) == "\nThanks!"

    def test_whitespace_between_parallel_orphan_invokes_is_ignored(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        """A response that is only two invokes and the padding between
        them comes back with no content at all.

        This case is already covered by the parser dropping content that
        is nothing but whitespace when the response called tools, so it
        passes whether or not the engine holds the padding back.  The
        test that actually pins the holding back is
        ``test_padding_between_orphan_invokes_is_dropped_after_prose``.
        """
        time_tool = _make_tool("get_time", {"timezone": {"type": "string"}})
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool, time_tool])
        mock_request.tools = [weather_tool, time_tool]
        text = (
            _invoke("get_weather", _param("city", "true", "SF"))
            + "\n  \n"
            + _invoke("get_time", _param("timezone", "true", "EST"))
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.content is None

    def test_padding_between_orphan_invokes_is_dropped_after_prose(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        """Padding between two recovered invokes is dropped even when
        the response already produced real text.

        The prose in front means the content is no longer whitespace
        only, so the parser's own whitespace dropping does not apply and
        the engine holding the padding back is the only thing keeping it
        out.  A wrapped call written the same way returns just the
        prose, and the recovered call has to match it.
        """
        time_tool = _make_tool("get_time", {"timezone": {"type": "string"}})
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool, time_tool])
        mock_request.tools = [weather_tool, time_tool]
        text = (
            "Some prose "
            + _invoke("get_weather", _param("city", "true", "SF"))
            + "\n  \n"
            + _invoke("get_time", _param("timezone", "true", "EST"))
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.content == "Some prose "

    def test_streaming_padding_between_orphan_invokes_is_dropped_after_prose(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        time_tool = _make_tool("get_time", {"timezone": {"type": "string"}})
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool, time_tool])
        mock_request.tools = [weather_tool, time_tool]
        chunks = [
            "Some prose ",
            _invoke("get_weather", _param("city", "true", "SF")),
            "\n  \n",
            _invoke("get_time", _param("timezone", "true", "EST")),
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        assert collect_content(results) == "Some prose "

    def test_recovery_does_not_carry_into_a_later_wrapped_call(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        """Once a recovered sequence ends, a later wrapped call in the
        same response is treated as an ordinary wrapped call.

        Text between the invokes of a wrapped call is dropped, so if the
        engine still thought it was inside a recovered sequence the
        stray text below would come back as content.
        """
        time_tool = _make_tool("get_time", {"timezone": {"type": "string"}})
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool, time_tool])
        mock_request.tools = [weather_tool, time_tool]
        text = (
            _invoke("get_weather", _param("city", "true", "SF"))
            + DSML_FUNC_END
            + DSML_FUNC_START
            + _invoke("get_weather", _param("city", "true", "SF"))
            + "stray between wrapped invokes"
            + _invoke("get_time", _param("timezone", "true", "EST"))
            + DSML_FUNC_END
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 3
        assert result.content is None

    def test_streaming_recovery_does_not_carry_into_a_later_wrapped_call(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        time_tool = _make_tool("get_time", {"timezone": {"type": "string"}})
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool, time_tool])
        mock_request.tools = [weather_tool, time_tool]
        chunks = [
            _invoke("get_weather", _param("city", "true", "SF")),
            DSML_FUNC_END,
            DSML_FUNC_START,
            _invoke("get_weather", _param("city", "true", "SF")),
            "stray between wrapped invokes",
            _invoke("get_time", _param("timezone", "true", "EST")),
            DSML_FUNC_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        assert collect_content(results) == ""

    def test_padding_held_before_one_invoke_does_not_reach_a_later_gap(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        """Padding held before one invoke is dropped when that invoke
        starts, so it cannot reappear in front of later text.

        The first gap is padding and belongs to nothing.  Only the
        second gap runs into real text, so only that one is content.
        """
        time_tool = _make_tool("get_time", {"timezone": {"type": "string"}})
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool, time_tool])
        mock_request.tools = [weather_tool, time_tool]
        text = (
            _invoke("get_weather", _param("city", "true", "SF"))
            + "\n\n"
            + _invoke("get_time", _param("timezone", "true", "EST"))
            + "  "
            + "Real text"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.content == "  Real text"

    def test_streaming_padding_held_before_one_invoke_does_not_reach_a_later_gap(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        time_tool = _make_tool("get_time", {"timezone": {"type": "string"}})
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool, time_tool])
        mock_request.tools = [weather_tool, time_tool]
        chunks = [
            _invoke("get_weather", _param("city", "true", "SF")),
            "\n\n",
            _invoke("get_time", _param("timezone", "true", "EST")),
            "  ",
            "Real text",
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        assert collect_content(results) == "  Real text"

    def test_abandoned_recovery_does_not_affect_a_later_wrapped_call(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        """A recovery attempt that turns out not to name a declared tool
        must leave nothing behind.

        The invoke below is held while its name is read, then given up
        on because ``get_nothing`` was never declared.  The wrapped call
        after it is ordinary, so the stray text between its invokes is
        dropped.
        """
        time_tool = _make_tool("get_time", {"timezone": {"type": "string"}})
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool, time_tool])
        mock_request.tools = [weather_tool, time_tool]
        abandoned = _invoke("get_nothing", _param("city", "true", "SF"))
        text = (
            abandoned
            + DSML_FUNC_START
            + _invoke("get_weather", _param("city", "true", "SF"))
            + "stray between wrapped invokes"
            + _invoke("get_time", _param("timezone", "true", "EST"))
            + DSML_FUNC_END
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.content == abandoned

    def test_recovery_does_not_leak_into_the_next_request(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        """A response that ends part way through a recovered sequence
        must not leave the engine set up for recovery.

        The engine is reused, so without a clean start the next
        response would treat an ordinary wrapped call as a recovered
        one and hand back the text between its invokes as content.
        """
        time_tool = _make_tool("get_time", {"timezone": {"type": "string"}})
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool, time_tool])
        mock_request.tools = [weather_tool, time_tool]

        first = parser.extract_tool_calls(
            _invoke("get_weather", _param("city", "true", "SF")), mock_request
        )
        assert first.tools_called

        second = parser.extract_tool_calls(
            DSML_FUNC_START
            + _invoke("get_weather", _param("city", "true", "SF"))
            + "stray between wrapped invokes"
            + _invoke("get_time", _param("timezone", "true", "EST"))
            + DSML_FUNC_END,
            mock_request,
        )

        assert second.tools_called
        assert len(second.tool_calls) == 2
        assert second.content is None

    def test_declared_names_do_not_leak_into_the_next_request(
        self, mock_tokenizer, mock_request, weather_tool
    ):
        """The engine is reused across requests, so a request that
        declares no tools must not recover a tool that an earlier
        request declared."""
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[weather_tool])
        mock_request.tools = [weather_tool]
        text = _invoke("get_weather", _param("city", "true", "SF")) + DSML_FUNC_END

        first = parser.extract_tool_calls(text, mock_request)
        assert first.tools_called

        second = parser.extract_tool_calls(text, _request_without_tools())

        assert not second.tools_called
        assert second.tool_calls == []
        assert second.content == text


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

    def test_missing_func_start_orphan_invoke(self, mock_tokenizer, mock_request):
        """Orphan invoke without the <｜DSML｜function_calls> wrapper is
        still parsed as a tool call when the request declared the tool
        (see gh-48931)."""
        tool = _make_tool("get_weather", {"city": {"type": "string"}})
        mock_request.tools = [tool]
        text = _invoke("get_weather", _param("city", "true", "SF")) + DSML_FUNC_END
        parser = DeepSeekV32Parser(mock_tokenizer, tools=[tool])
        results = simulate_tool_streaming(parser, mock_request, list(text))
        assert collect_function_name(results) == "get_weather"
        args = json.loads(collect_tool_arguments(results))
        assert args == {"city": "SF"}
        assert "DSML" not in collect_content(results)

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
