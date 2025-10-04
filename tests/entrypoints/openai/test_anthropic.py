# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Anthropic Messages API endpoint."""

import json

from vllm.entrypoints.openai.api_server import (_anthropic_to_openai_request,
                                                _openai_to_anthropic_response)
from vllm.entrypoints.openai.protocol import (AnthropicMessage,
                                              AnthropicMessageRequest,
                                              ChatCompletionResponse,
                                              ChatCompletionResponseChoice,
                                              ChatMessage, FunctionCall,
                                              ToolCall, UsageInfo)


def test_anthropic_to_openai_simple():
    """Test basic Anthropic to OpenAI request conversion."""
    anthropic_req = AnthropicMessageRequest(
        model="claude-3-sonnet",
        messages=[AnthropicMessage(role="user", content="Hello")],
        max_tokens=100)
    openai_req = _anthropic_to_openai_request(anthropic_req)
    assert (openai_req.model == "claude-3-sonnet"
            and openai_req.max_tokens == 100 and len(openai_req.messages) == 1
            and openai_req.messages[0]["role"] == "user"
            and openai_req.messages[0]["content"] == "Hello")


def test_anthropic_to_openai_with_system():
    """Test conversion with system prompt."""
    anthropic_req = AnthropicMessageRequest(
        model="claude-3-sonnet",
        system="You are helpful",
        messages=[AnthropicMessage(role="user", content="Hi")],
        max_tokens=50)
    openai_req = _anthropic_to_openai_request(anthropic_req)
    assert (len(openai_req.messages) == 2
            and openai_req.messages[0]["role"] == "system"
            and openai_req.messages[0]["content"] == "You are helpful")


def test_openai_to_anthropic_response():
    """Test OpenAI to Anthropic response conversion."""
    openai_resp = ChatCompletionResponse(
        id="chatcmpl-123",
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(index=0,
                                         message=ChatMessage(role="assistant",
                                                             content="Hi!"),
                                         finish_reason="stop")
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5,
                        total_tokens=15))
    anthropic_resp = _openai_to_anthropic_response(openai_resp,
                                                   "claude-3-sonnet")
    assert (anthropic_resp.id.startswith("msg_")
            and anthropic_resp.role == "assistant"
            and len(anthropic_resp.content) == 1
            and anthropic_resp.content[0].text == "Hi!"
            and anthropic_resp.stop_reason == "end_turn"
            and anthropic_resp.stop_sequence is None
            and anthropic_resp.usage.input_tokens == 10
            and anthropic_resp.usage.output_tokens == 5)


def test_stop_reason_mapping():
    """Test finish_reason to stop_reason mapping."""
    test_cases = [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
    ]

    for finish_reason, expected_stop_reason in test_cases:
        openai_resp = ChatCompletionResponse(
            id="chatcmpl-123",
            model="test",
            choices=[
                ChatCompletionResponseChoice(index=0,
                                             message=ChatMessage(
                                                 role="assistant",
                                                 content="ok"),
                                             finish_reason=finish_reason)
            ],
            usage=UsageInfo(prompt_tokens=1,
                            completion_tokens=1,
                            total_tokens=2))

        anthropic_resp = _openai_to_anthropic_response(openai_resp, "claude")
        assert anthropic_resp.stop_reason == expected_stop_reason


def test_tool_calls_conversion():
    """Test conversion of tool calls from OpenAI to Anthropic format."""
    openai_resp = ChatCompletionResponse(
        id="chatcmpl-123",
        model="test",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant",
                                    content="Using tool",
                                    tool_calls=[
                                        ToolCall(id="call_123",
                                                 type="function",
                                                 function=FunctionCall(
                                                     name="get_weather",
                                                     arguments=json.dumps(
                                                         {"location": "NYC"})))
                                    ]),
                finish_reason="tool_calls")
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5,
                        total_tokens=15))
    anthropic_resp = _openai_to_anthropic_response(openai_resp, "claude")
    assert (len(anthropic_resp.content) == 2
            and anthropic_resp.content[0].type == "text"
            and anthropic_resp.content[0].text == "Using tool"
            and anthropic_resp.content[1].type == "tool_use"
            and anthropic_resp.content[1].id == "call_123"
            and anthropic_resp.content[1].name == "get_weather"
            and anthropic_resp.content[1].input == {
                "location": "NYC"
            } and anthropic_resp.stop_reason == "tool_use")


def test_temperature_and_sampling_params():
    """Test that sampling parameters are correctly converted."""
    anthropic_req = AnthropicMessageRequest(
        model="claude-3-sonnet",
        messages=[AnthropicMessage(role="user", content="Test")],
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        stop_sequences=["STOP", "END"])
    openai_req = _anthropic_to_openai_request(anthropic_req)
    assert (openai_req.temperature == 0.7 and openai_req.top_p == 0.9
            and openai_req.top_k == 50 and openai_req.stop == ["STOP", "END"])
