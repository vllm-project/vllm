# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
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

    tool_calls, content = parser._extract_tool_calls(
        content=None,
        request=request,
        enable_auto_tools=False,
    )

    assert content is None
    assert tool_calls == []


def _chat_response(message: ChatMessage) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=message,
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_chat_completion_response_omits_empty_tool_calls_payload():
    response = _chat_response(ChatMessage(role="assistant", content="done"))

    payload = response.model_dump()
    payload_exclude_unset = response.model_dump(exclude_unset=True)

    assert "tool_calls" not in payload["choices"][0]["message"]
    assert "tool_calls" not in payload_exclude_unset["choices"][0]["message"]
    parsed = OpenAIChatCompletion.model_validate(payload)
    assert parsed.choices[0].message.tool_calls is None


def test_chat_completion_response_keeps_non_empty_tool_calls_payload():
    response = _chat_response(
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments='{"city": "Beijing"}',
                    )
                )
            ],
        )
    )

    message = response.model_dump()["choices"][0]["message"]

    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"


def _stream_response(delta: DeltaMessage) -> ChatCompletionStreamResponse:
    return ChatCompletionStreamResponse(
        id="chatcmpl-test",
        object="chat.completion.chunk",
        created=1,
        model="test-model",
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=delta,
                finish_reason=None,
            )
        ],
    )


def test_chat_completion_stream_response_omits_empty_tool_calls_payload():
    response = _stream_response(DeltaMessage(content="done"))

    payload = response.model_dump(exclude_unset=True)
    payload_json = response.model_dump_json(exclude_unset=True)

    assert "tool_calls" not in payload["choices"][0]["delta"]
    parsed = ChatCompletionChunk.model_validate_json(payload_json)
    assert parsed.choices[0].delta.tool_calls is None


def test_chat_completion_stream_response_keeps_non_empty_tool_calls_payload():
    response = _stream_response(
        DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    id="call-test",
                    type="function",
                    function=DeltaFunctionCall(
                        name="get_weather",
                        arguments='{"city": "Beijing"}',
                    ),
                )
            ]
        )
    )

    delta = response.model_dump(exclude_unset=True)["choices"][0]["delta"]

    assert len(delta["tool_calls"]) == 1
    assert delta["tool_calls"][0]["function"]["name"] == "get_weather"
