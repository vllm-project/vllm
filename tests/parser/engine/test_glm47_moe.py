# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning-boundary tests for the GLM-4.7 parser.

GLM can transition straight from reasoning into a tool call without ever
emitting ``</think>``, so ``<tool_call>`` is an implicit reasoning
terminator. These cover that path plus the multi-turn prompts where an
earlier turn's markers must not be read as the current turn's state. The
streaming tests check that long tool arguments stream the same JSON as a
non-streaming parse, without converting the whole argument on every delta.
"""

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
)
from vllm.parser.glm47_moe import (
    THINK_END,
    THINK_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Glm47MoeParser,
    _glm47_arg_value_extender,
    glm47_moe_config,
)

THINK_S, THINK_E = 1000, 1001
TOOL_S, TOOL_E = 1002, 1003
ASSISTANT, OBSERVATION, USER = 1004, 1005, 1006
TEXT = 42  # stand-in for an ordinary reasoning/content token

VOCAB = {
    THINK_START: THINK_S,
    THINK_END: THINK_E,
    TOOL_CALL_START: TOOL_S,
    TOOL_CALL_END: TOOL_E,
    "<|assistant|>": ASSISTANT,
    "<|observation|>": OBSERVATION,
    "<|user|>": USER,
}


@pytest.fixture
def parser():
    return Glm47MoeParser(make_mock_tokenizer(VOCAB))


@pytest.fixture
def no_thinking_parser():
    return Glm47MoeParser(
        make_mock_tokenizer(VOCAB),
        chat_template_kwargs={"thinking": False},
    )


class TestIsReasoningEnd:
    def test_open_reasoning(self, parser):
        assert not parser.is_reasoning_end([THINK_S, TEXT])

    def test_think_end(self, parser):
        assert parser.is_reasoning_end([THINK_S, TEXT, THINK_E, TEXT])

    def test_tool_call_without_think_end(self, parser):
        """Reasoning -> tool call with no ``</think>`` still ends reasoning."""
        assert parser.is_reasoning_end([THINK_S, TEXT, TOOL_S, TEXT])

    def test_previous_turn_tool_call_ignored(self, parser):
        """A finished tool call from an earlier turn says nothing about the
        turn currently being generated."""
        prompt = [THINK_S, TEXT, TOOL_S, TEXT, TOOL_E, OBSERVATION, TEXT, ASSISTANT]
        assert not parser.is_reasoning_end(prompt)

    def test_previous_turn_think_end_ignored(self, parser):
        prompt = [THINK_S, TEXT, THINK_E, TEXT, USER, TEXT, ASSISTANT]
        assert not parser.is_reasoning_end(prompt)

    def test_empty_input(self, parser):
        assert not parser.is_reasoning_end([])

    def test_thinking_disabled(self, no_thinking_parser):
        assert no_thinking_parser.is_reasoning_end([THINK_S, TEXT])


class TestExtractContentIds:
    def test_think_end_wins_over_later_tool_call(self, parser):
        """Content between ``</think>`` and a tool call must survive."""
        ids = [THINK_S, TEXT, THINK_E, 20, 21, TOOL_S, 30, TOOL_E]
        assert parser.extract_content_ids(ids) == [20, 21, TOOL_S, 30, TOOL_E]

    def test_every_tool_call_after_think_end_kept(self, parser):
        ids = [THINK_E, 20, TOOL_S, 30, TOOL_E, 21, TOOL_S, 31, TOOL_E]
        assert parser.extract_content_ids(ids) == ids[1:]

    def test_falls_back_to_tool_call(self, parser):
        """Without ``</think>``, content starts at the opener itself so the
        tool parser still receives a well-formed call."""
        ids = [THINK_S, TEXT, TOOL_S, 30, TOOL_E]
        assert parser.extract_content_ids(ids) == [TOOL_S, 30, TOOL_E]

    def test_falls_back_to_first_tool_call_of_turn(self, parser):
        ids = [THINK_S, TEXT, TOOL_S, 30, TOOL_E, TOOL_S, 31, TOOL_E]
        assert parser.extract_content_ids(ids) == ids[2:]

    def test_previous_turn_tool_call_ignored(self, parser):
        ids = [TOOL_S, 30, TOOL_E, OBSERVATION, TEXT, ASSISTANT, THINK_S, TEXT]
        assert parser.extract_content_ids(ids) == ids

    def test_no_markers_returns_input_ids(self, parser):
        assert parser.extract_content_ids([20, 21]) == [20, 21]

    def test_thinking_disabled(self, no_thinking_parser):
        ids = [THINK_S, TEXT, TOOL_S]
        assert no_thinking_parser.extract_content_ids(ids) == ids


WRITE_TOOL = ChatCompletionToolsParam(
    type="function",
    function={
        "name": "Write",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "line_count": {"type": "integer"},
                "content": {"type": "string"},
            },
        },
    },
)
CALL_HEAD = (
    f"{THINK_END}{TOOL_CALL_START}Write\n"
    "<arg_key>file_path</arg_key><arg_value>/repo/app.py</arg_value>\n"
    "<arg_key>line_count</arg_key><arg_value>120</arg_value>\n"
    "<arg_key>content</arg_key><arg_value>"
)
CALL_TAIL = f"</arg_value>\n{TOOL_CALL_END}"
# Escapes, non-ASCII, and tag-like text that must stay inside the value.
CONTENT_LINE = 'if a < b > c: print("x\\y")  # </div> </arg_ <arg_key>\t\u00e9\n'


def _pieces(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _stream_arguments(parser, request, chunks: list[str]) -> str:
    args = []
    for i, chunk in enumerate(chunks):
        delta = parser.parse_delta(chunk, [], request, finished=i == len(chunks) - 1)
        for tool_call in (delta and delta.tool_calls) or []:
            if tool_call.function and tool_call.function.arguments:
                args.append(tool_call.function.arguments)
    return "".join(args)


class TestStreamedToolArguments:
    @pytest.mark.parametrize(
        "tools, line_count",
        [([], '"120"'), ([WRITE_TOOL], "120")],
        ids=["no_schema", "schema"],
    )
    def test_long_call_streams_non_streaming_arguments(
        self, mock_request, tools, line_count
    ):
        mock_request.tools = tools
        text = CALL_HEAD + CONTENT_LINE * 400 + CALL_TAIL

        streamed = _stream_arguments(
            Glm47MoeParser(make_mock_tokenizer(VOCAB)), mock_request, _pieces(text, 7)
        )
        result = Glm47MoeParser(make_mock_tokenizer(VOCAB)).extract_tool_calls(
            text, mock_request
        )

        assert streamed == result.tool_calls[0].function.arguments
        assert f'"line_count": {line_count},' in streamed

    def test_conversions_do_not_grow_with_argument_length(self, mock_request):
        """The open value streams without re-running the converter."""
        mock_request.tools = [WRITE_TOOL]

        def streaming_conversions(lines: int) -> int:
            calls = 0
            base = glm47_moe_config()

            def counting_converter(raw_args: str, partial: bool) -> str:
                nonlocal calls
                if partial:
                    calls += 1
                return base.arg_converter(raw_args, partial)

            parser = Glm47MoeParser(
                make_mock_tokenizer(VOCAB),
                parser_engine_config=dataclasses.replace(
                    base, arg_converter=counting_converter
                ),
            )
            chunks = [
                *_pieces(CALL_HEAD, 4),
                *_pieces(CONTENT_LINE * lines, 4),
                *_pieces(CALL_TAIL, 4),
            ]
            _stream_arguments(parser, mock_request, chunks)
            return calls

        assert streaming_conversions(lines=40) == streaming_conversions(lines=400)

    def test_value_extender_stops_at_split_value_end(self):
        extend = _glm47_arg_value_extender("<arg_key>k</arg_key><arg_value>x")
        assert [extend(d) for d in ['a"\n', "</arg_", "value>"]] == [
            'a\\"\\n',
            "</arg_",
            None,
        ]

    @pytest.mark.parametrize(
        "raw_args",
        [
            "<arg_key>k</arg_key><arg_value>x</arg_value>",
            "<arg_key>k</arg_key><arg_value>1</arg_value>"
            "<arg_key>j</arg_key><arg_value>2</arg_value>"
            "<arg_key>k</arg_key><arg_value>3",
        ],
        ids=["closed_value", "repeated_key_not_last"],
    )
    def test_no_value_extender_unless_open_value_is_last(self, raw_args):
        assert _glm47_arg_value_extender(raw_args) is None


class TestStrayToolCallTag:
    """A stray ``<tool_call>`` the model emits mid-prose must not swallow
    the remainder of the stream (vllm-project/vllm#60703)."""

    STRAY_THEN_REAL = (
        "Fetching the page. I will issue a <tool_call> for it.\n"
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Tokyo</arg_value></tool_call>"
    )

    @pytest.fixture
    def content_parser(self):
        return Glm47MoeParser(
            make_mock_tokenizer(VOCAB),
            chat_template_kwargs={"thinking": False},
        )

    def test_stray_tag_in_prose_does_not_swallow_later_tool_call(
        self, content_parser, mock_request
    ):
        result = content_parser.extract_tool_calls(self.STRAY_THEN_REAL, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[-1].function.name == "get_weather"
        assert json.loads(result.tool_calls[-1].function.arguments) == {
            "city": "Tokyo",
        }

    def test_streaming_stray_tag_in_prose_does_not_swallow_later_tool_call(
        self, content_parser, mock_request
    ):
        chunks = [
            "Fetching the page. I will issue a ",
            "<tool_call>",
            " for it.\n",
            "<tool_call>",
            "get_weather",
            "<arg_key>city</arg_key>",
            "<arg_value>Tokyo</arg_value>",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(content_parser, mock_request, chunks)

        calls_by_index: dict[int, dict[str, str]] = {}
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    entry = calls_by_index.setdefault(
                        tc.index, {"name": "", "arguments": ""}
                    )
                    if tc.function:
                        if tc.function.name:
                            entry["name"] += tc.function.name
                        if tc.function.arguments:
                            entry["arguments"] += tc.function.arguments

        last_call = calls_by_index[max(calls_by_index)]
        assert last_call["name"] == "get_weather"
        assert json.loads(last_call["arguments"]) == {"city": "Tokyo"}

    def test_stray_tag_in_reasoning_does_not_swallow_tool_call(
        self, parser, mock_request
    ):
        """REASONING entry: the stray tag implicitly ends reasoning and the
        genuine call that follows must still parse."""
        text = (
            "<think>The docs show a <tool_call> example; fetch for real."
            "</think>" + self.STRAY_THEN_REAL
        )
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[-1].function.name == "get_weather"
        assert json.loads(result.tool_calls[-1].function.arguments) == {
            "city": "Tokyo",
        }

    def test_phantom_dropped_when_request_has_tools(self, mock_request):
        """With a populated tools list, validate_tool_names drops the
        phantom: exactly one call must come back, not a prose-named extra."""
        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        )
        mock_request.tools = [tool]
        parser = Glm47MoeParser(
            make_mock_tokenizer(VOCAB),
            tools=[tool],
            chat_template_kwargs={"thinking": False},
        )
        result = parser.extract_tool_calls(self.STRAY_THEN_REAL, mock_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "city": "Tokyo",
        }

    def test_streaming_stray_tag_inside_arg_value_preserved(
        self, content_parser, mock_request
    ):
        """TOOL_ARGS has no recovery transition on purpose: a stray tag in
        an <arg_value> is quoted data and must round-trip."""
        chunks = [
            "<tool_call>",
            "write_file",
            "<arg_key>content</arg_key>",
            "<arg_value>Document ",
            "<tool_call>",
            " usage</arg_value>",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(content_parser, mock_request, chunks)

        assert collect_function_name(results) == "write_file"
        assert json.loads(collect_tool_arguments(results)) == {
            "content": "Document <tool_call> usage",
        }
