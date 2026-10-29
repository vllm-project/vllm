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

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
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
            base_converter = base.arg_converter
            assert base_converter is not None

            def counting_converter(raw_args: str, partial: bool) -> str:
                nonlocal calls
                if partial:
                    calls += 1
                return base_converter(raw_args, partial)

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
        assert extend is not None
        assert [extend(d) for d in ['a"\n', "</arg_", "value>"]] == [
            'a\\"\\n',
            "</arg_",
            None,
        ]

    @pytest.mark.parametrize(
        "raw_args",
        [
            "<arg_key>k</arg_key><arg_value>x</arg_value>",
            (
                "<arg_key>k</arg_key><arg_value>1</arg_value>"
                "<arg_key>j</arg_key><arg_value>2</arg_value>"
                "<arg_key>k</arg_key><arg_value>3"
            ),
        ],
        ids=["closed_value", "repeated_key_not_last"],
    )
    def test_no_value_extender_unless_open_value_is_last(self, raw_args):
        assert _glm47_arg_value_extender(raw_args) is None
