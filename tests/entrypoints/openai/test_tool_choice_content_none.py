# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser

pytestmark = pytest.mark.skip_global_cleanup


class _DummyDelegatingParser(DelegatingParser):
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids

    def extract_reasoning(self, model_output: str, request):
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int],
        current_token_ids: list[int],
        delta_token_ids: list[int],
    ):
        return None


def test_chat_completion_named_tool_choice_with_none_content():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }
    )
    parser = _DummyDelegatingParser(tokenizer=None)

    tool_calls, content = parser._extract_tool_calls(
        content=None,
        request=request,
        enable_auto_tools=True,
    )

    assert content is None
    assert tool_calls == []


class _AlwaysFireToolParser(ToolParser):
    """Stub tool parser that always emits a tool call.

    Used to assert that ``DelegatingParser.parse_delta`` bypasses the tool
    parser entirely when ``tool_choice="none"``; without the stub mutating
    ``prev_tool_call_arr``, the bypass check would silently pass on broken
    code.
    """

    def __init__(self, tokenizer=None, tools=None):  # type: ignore[no-untyped-def]
        super().__init__(tokenizer=tokenizer, tools=tools)

    def extract_tool_calls(self, model_output, request):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def extract_tool_calls_streaming(  # type: ignore[no-untyped-def]
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        # Pretend the model is mid-way through emitting a tool call.
        self.prev_tool_call_arr = [{"name": "get_weather", "arguments": {}}]
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    id="call_abc",
                    type="function",
                    function=DeltaFunctionCall(
                        name="get_weather", arguments=delta_text
                    ),
                )
            ]
        )


def _make_streaming_parser_with_tool_only() -> _DummyDelegatingParser:
    """Build a DelegatingParser with only a tool parser (no reasoning)."""
    parser = _DummyDelegatingParser(tokenizer=None)
    parser._tool_parser = _AlwaysFireToolParser()
    # No reasoning parser configured -> tool phase starts immediately.
    parser._stream_state.reasoning_ended = True
    return parser


def _make_request(tool_choice: str | None) -> ChatCompletionRequest:
    """Build a streaming request carrying a tool.

    ``tool_choice=None`` is sent explicitly as JSON ``null`` so that
    ``request.tool_choice`` resolves to ``None`` (rather than the ``"none"``
    field default that an omitted key would pick up), mirroring a client that
    explicitly disables tool calls.
    """
    payload: dict = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": tool_choice,
    }
    return ChatCompletionRequest.model_validate(payload)


def test_parse_delta_with_tool_choice_none_skips_tool_parser():
    """Streaming must honor ``tool_choice="none"``: no tool parser invocation."""
    parser = _make_streaming_parser_with_tool_only()
    request = _make_request(tool_choice="none")

    delta = parser.parse_delta(
        delta_text="hello ",
        delta_token_ids=[1, 2],
        request=request,
        prompt_token_ids=[10],
    )

    assert delta is not None
    assert delta.tool_calls == []
    assert delta.content == "hello "
    # The stub tool parser must not have been called.
    assert parser._tool_parser.prev_tool_call_arr == []


def test_parse_delta_with_tool_choice_null_skips_tool_parser():
    """Explicit ``tool_choice: null`` (request.tool_choice is None) must also
    bypass the tool parser, matching the non-streaming ``not request.tool_choice``
    guard."""
    parser = _make_streaming_parser_with_tool_only()
    request = _make_request(tool_choice=None)
    assert request.tool_choice is None

    delta = parser.parse_delta(
        delta_text="hello ",
        delta_token_ids=[1, 2],
        request=request,
        prompt_token_ids=[10],
    )

    assert delta is not None
    assert delta.tool_calls == []
    assert delta.content == "hello "
    # The stub tool parser must not have been called.
    assert parser._tool_parser.prev_tool_call_arr == []


def test_parse_delta_with_tool_choice_auto_still_runs_tool_parser():
    """Sanity check: the bypass only kicks in when tool_choice is falsy or
    "none"; tool_choice="auto" must still hit the tool parser (no regression)."""
    parser = _make_streaming_parser_with_tool_only()
    request = _make_request(tool_choice="auto")

    delta = parser.parse_delta(
        delta_text="hello ",
        delta_token_ids=[1, 2],
        request=request,
        prompt_token_ids=[10],
    )

    assert delta is not None
    assert delta.tool_calls and delta.tool_calls[0].function is not None
    assert delta.tool_calls[0].function.name == "get_weather"


def test_parse_delta_tool_choice_none_multiple_chunks_remain_content():
    """Multi-chunk streaming under tool_choice="none" stays as content."""
    parser = _make_streaming_parser_with_tool_only()
    request = _make_request(tool_choice="none")

    first = parser.parse_delta(
        delta_text="<tool_call>",
        delta_token_ids=[1],
        request=request,
        prompt_token_ids=[10],
    )
    second = parser.parse_delta(
        delta_text='{"name": "x"}',
        delta_token_ids=[2],
        request=request,
    )

    for chunk in (first, second):
        assert chunk is not None
        assert chunk.tool_calls == []
    # Only the first chunk swallows the accumulated prefix via
    # ``tool_call_text_started``; both deltas must surface as content.
    assert first.content == "<tool_call>"
    assert second.content == '{"name": "x"}'
    assert parser._tool_parser.prev_tool_call_arr == []


def test_parse_delta_responses_tool_choice_none_skips_tool_parser():
    """Responses streaming also routes through ``parse_delta``; the shared
    guard must bypass the tool parser there too when ``tool_choice="none"``."""
    parser = _make_streaming_parser_with_tool_only()
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": "none",
        }
    )

    delta = parser.parse_delta(
        delta_text="hello ",
        delta_token_ids=[1, 2],
        request=request,
        prompt_token_ids=[10],
    )

    assert delta is not None
    assert delta.tool_calls == []
    assert delta.content == "hello "
    # The stub tool parser must not have been called.
    assert parser._tool_parser.prev_tool_call_arr == []


def test_responses_parser_allows_named_tool_choice_with_none_content():
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": "test",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "function", "name": "get_weather"},
        }
    )
    parser = _DummyDelegatingParser(tokenizer=None)

    tool_calls, content = parser._parse_tool_calls(
        request=request,
        content=None,
        enable_auto_tools=False,
    )

    assert content is None
    assert tool_calls == []
