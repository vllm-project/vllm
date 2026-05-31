# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.abstract_parser import DelegatingParser

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

    def extract_tool_calls(self, model_output: str, request):
        return None


def test_parse_tool_calls_from_content_allows_named_tool_choice_with_none_content():
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

    tool_calls, content = OpenAIServing._parse_tool_calls_from_content(
        request=request,
        tokenizer=None,
        enable_auto_tools=True,
        tool_parser_cls=None,
        content=None,
    )

    assert content is None
    assert tool_calls is not None
    assert tool_calls == []


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


class _StubParser(DelegatingParser):
    """DelegatingParser stub that records whether the tool parser ran.

    Skips the real reasoning / tool parsers so ``parse_delta`` jumps directly
    into the tool-call phase on the first delta. ``_extract_tool_calls_streaming``
    flips ``tool_parser_called`` to ``True`` and emits a synthetic tool call,
    so a passing test (``tool_parser_called is False``) proves the new
    ``tool_choice=None``/``"none"`` guard short-circuited before the parser.
    """

    def __init__(self):
        super().__init__(tokenizer=None)
        # Start past the reasoning phase so parse_delta enters the tool-call
        # block on the very first delta.
        self._stream_state.reasoning_ended = True
        self.tool_parser_called = False

    def _in_tool_call_phase(self, state) -> bool:  # type: ignore[override]
        # Decouple from self._tool_parser (which is None on the stub).
        return state.reasoning_ended

    def is_reasoning_end(self, input_ids):
        return False

    def extract_content_ids(self, input_ids):
        return input_ids

    def extract_reasoning(self, model_output, request):
        return None, model_output

    def extract_reasoning_streaming(self, **kwargs):
        return None

    def extract_tool_calls(self, model_output, request):
        return None

    def _extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
        tool_call_idx,
        tool_call_id_type,
        function_name_returned,
    ):
        from vllm.entrypoints.openai.engine.protocol import (
            DeltaFunctionCall,
            DeltaToolCall,
        )

        self.tool_parser_called = True
        delta_message = DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    id="call_test",
                    type="function",
                    function=DeltaFunctionCall(name="t", arguments=""),
                )
            ]
        )
        return delta_message, True


def _make_streaming_request(tool_choice):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
        "stream": True,
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    return ChatCompletionRequest.model_validate(payload)


def test_parse_delta_with_tool_choice_none_skips_tool_parser():
    """Explicit ``tool_choice="none"`` must not invoke the streaming tool parser
    and must surface the raw delta text as ``DeltaMessage.content``."""
    parser = _StubParser()
    request = _make_streaming_request(tool_choice="none")

    delta = parser.parse_delta(
        delta_text="<tool_call>{}</tool_call>",
        delta_token_ids=[1, 2, 3],
        request=request,
    )

    assert parser.tool_parser_called is False
    assert delta is not None
    assert delta.content == "<tool_call>{}</tool_call>"
    assert delta.tool_calls == []


def test_parse_delta_with_omitted_tool_choice_skips_tool_parser():
    """No-tools request with omitted ``tool_choice`` (``request.tool_choice
    is None``) must not invoke the streaming tool parser either, mirroring the
    non-streaming guard ``not request.tool_choice or request.tool_choice ==
    "none"`` in chat_completion/serving.py."""
    parser = _StubParser()
    request = _make_streaming_request(tool_choice=None)
    assert request.tool_choice in (None, "none")

    delta = parser.parse_delta(
        delta_text="<tool_call>{}</tool_call>",
        delta_token_ids=[1, 2, 3],
        request=request,
    )

    assert parser.tool_parser_called is False
    assert delta is not None
    assert delta.content == "<tool_call>{}</tool_call>"
    assert delta.tool_calls == []


def test_parse_delta_without_tool_choice_none_still_runs_tool_parser():
    """Sanity: the bypass is gated on ``tool_choice in (None, "none")``;
    other values (e.g. ``"auto"``) must still hit the tool parser."""
    parser = _StubParser()
    request = _make_streaming_request(tool_choice="auto")

    delta = parser.parse_delta(
        delta_text="<tool_call>{}</tool_call>",
        delta_token_ids=[1, 2, 3],
        request=request,
    )

    assert parser.tool_parser_called is True
    assert delta is not None
    assert delta.tool_calls
    assert delta.tool_calls[0].id == "call_test"


def test_parse_delta_tool_choice_none_multiple_chunks_remain_content():
    """Multiple deltas under ``tool_choice="none"`` must all stay in content."""
    parser = _StubParser()
    request = _make_streaming_request(tool_choice="none")

    chunks = ["<tool_", "call>{", "}</tool_", "call>"]
    contents: list[str | None] = []
    for i, chunk in enumerate(chunks):
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=[i],
            request=request,
        )
        assert delta is not None
        contents.append(delta.content)

    assert parser.tool_parser_called is False
    assert contents[0] == "<tool_"
    assert "".join(c or "" for c in contents) == "".join(chunks)
