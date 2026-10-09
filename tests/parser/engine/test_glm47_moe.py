# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning-boundary tests for the GLM-4.7 parser.

GLM can transition straight from reasoning into a tool call without ever
emitting ``</think>``, so ``<tool_call>`` is an implicit reasoning
terminator. These cover that path plus the multi-turn prompts where an
earlier turn's markers must not be read as the current turn's state.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.parser.glm47_moe import (
    THINK_END,
    THINK_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Glm47MoeParser,
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
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

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
