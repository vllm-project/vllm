# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Anthropic Messages API reasoning/thinking support."""

import json

import pytest

from vllm.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicDelta,
    AnthropicMessagesRequest,
    AnthropicStreamEvent,
    AnthropicThinkingConfig,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatMessage,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage

# -- Protocol model tests --------------------------------------------------


class TestAnthropicThinkingConfig:
    def test_enabled_requires_budget_tokens(self):
        with pytest.raises(ValueError, match="budget_tokens is required"):
            AnthropicThinkingConfig(type="enabled")

    def test_enabled_with_budget_tokens(self):
        cfg = AnthropicThinkingConfig(type="enabled", budget_tokens=1024)
        assert cfg.type == "enabled"
        assert cfg.budget_tokens == 1024

    def test_disabled(self):
        cfg = AnthropicThinkingConfig(type="disabled")
        assert cfg.type == "disabled"
        assert cfg.budget_tokens is None


class TestAnthropicContentBlockThinking:
    def test_thinking_type(self):
        block = AnthropicContentBlock(
            type="thinking", thinking="Let me reason about this..."
        )
        assert block.type == "thinking"
        assert block.thinking == "Let me reason about this..."
        assert block.text is None


class TestAnthropicDeltaThinking:
    def test_thinking_delta_type(self):
        delta = AnthropicDelta(type="thinking_delta", thinking="partial reasoning")
        assert delta.type == "thinking_delta"
        assert delta.thinking == "partial reasoning"
        assert delta.text is None


class TestAnthropicMessagesRequestThinking:
    def test_request_with_thinking(self):
        req = AnthropicMessagesRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            thinking={"type": "enabled", "budget_tokens": 2048},
        )
        assert req.thinking is not None
        assert req.thinking.type == "enabled"
        assert req.thinking.budget_tokens == 2048

    def test_request_without_thinking(self):
        req = AnthropicMessagesRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
        )
        assert req.thinking is None


# -- Request conversion tests ----------------------------------------------


class TestConvertAnthropicToOpenAIRequest:
    """Test _convert_anthropic_to_openai_request reasoning handling."""

    def _make_handler(self):
        """Create a minimal AnthropicServingMessages for testing conversion."""
        # We only need the conversion method, not the full server.
        # Use __new__ to bypass __init__ which requires engine_client etc.
        handler = object.__new__(AnthropicServingMessages)
        handler.stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }
        return handler

    def test_thinking_enabled_sets_include_reasoning(self):
        handler = self._make_handler()
        req = AnthropicMessagesRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            thinking={"type": "enabled", "budget_tokens": 2048},
        )
        openai_req = handler._convert_anthropic_to_openai_request(req)
        assert openai_req.include_reasoning is True

    def test_thinking_disabled_sets_include_reasoning_false(self):
        handler = self._make_handler()
        req = AnthropicMessagesRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            thinking={"type": "disabled"},
        )
        openai_req = handler._convert_anthropic_to_openai_request(req)
        assert openai_req.include_reasoning is False

    def test_no_thinking_sets_include_reasoning_false(self):
        handler = self._make_handler()
        req = AnthropicMessagesRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
        )
        openai_req = handler._convert_anthropic_to_openai_request(req)
        assert openai_req.include_reasoning is False

    def test_thinking_blocks_in_messages_converted(self):
        handler = self._make_handler()
        req = AnthropicMessagesRequest(
            model="test-model",
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I need to think about this...",
                        },
                        {"type": "text", "text": "Here is my answer."},
                    ],
                },
                {"role": "user", "content": "follow up question"},
            ],
            max_tokens=1024,
        )
        openai_req = handler._convert_anthropic_to_openai_request(req)
        # The assistant message should contain both thinking (as text) and
        # the actual text content
        assistant_msg = openai_req.messages[0]
        assert assistant_msg["role"] == "assistant"
        content = assistant_msg["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["text"] == "I need to think about this..."
        assert content[1]["text"] == "Here is my answer."


# -- Non-streaming response converter tests ---------------------------------


class TestMessagesFullConverterReasoning:
    def _make_handler(self):
        handler = object.__new__(AnthropicServingMessages)
        handler.stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }
        return handler

    def test_response_with_reasoning(self):
        handler = self._make_handler()
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="The answer is 42.",
                        reasoning="Let me think step by step...",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        result = handler.messages_full_converter(response)

        # Should have thinking block before text block
        assert len(result.content) == 2
        assert result.content[0].type == "thinking"
        assert result.content[0].thinking == "Let me think step by step..."
        assert result.content[1].type == "text"
        assert result.content[1].text == "The answer is 42."

    def test_response_without_reasoning(self):
        handler = self._make_handler()
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="The answer is 42.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        result = handler.messages_full_converter(response)

        # Should have only text block
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "The answer is 42."

    def test_response_with_reasoning_and_tool_calls(self):
        handler = self._make_handler()
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="",
                        reasoning="I should call the weather tool.",
                        tool_calls=[
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": json.dumps({"city": "NYC"}),
                                },
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        result = handler.messages_full_converter(response)

        assert result.stop_reason == "tool_use"
        assert len(result.content) == 3
        assert result.content[0].type == "thinking"
        assert result.content[0].thinking == "I should call the weather tool."
        assert result.content[1].type == "text"
        assert result.content[2].type == "tool_use"
        assert result.content[2].name == "get_weather"


# -- Streaming response converter tests -------------------------------------


class TestMessageStreamConverterReasoning:
    def _make_handler(self):
        handler = object.__new__(AnthropicServingMessages)
        handler.stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }
        return handler

    async def _collect_events(self, handler, chunks):
        """Helper to run stream converter and collect parsed events."""

        async def mock_generator():
            for chunk_obj in chunks:
                yield f"data: {chunk_obj.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        events = []
        async for event_str in handler.message_stream_converter(mock_generator()):
            if event_str.startswith("event:"):
                lines = event_str.strip().split("\n")
                event_type = lines[0].split(": ", 1)[1]
                data_json = lines[1].split(": ", 1)[1]
                events.append((event_type, json.loads(data_json)))
            elif event_str == "data: [DONE]\n\n":
                events.append(("done", None))
        return events

    @pytest.mark.asyncio
    async def test_stream_with_reasoning_then_text(self):
        handler = self._make_handler()

        chunks = [
            # First chunk: message_start
            ChatCompletionStreamResponse(
                id="chatcmpl-1",
                model="test-model",
                choices=[],
                usage=UsageInfo(prompt_tokens=10, completion_tokens=0, total_tokens=10),
            ),
            # Reasoning delta
            ChatCompletionStreamResponse(
                id="chatcmpl-1",
                model="test-model",
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(reasoning="Let me think..."),
                    )
                ],
            ),
            # More reasoning
            ChatCompletionStreamResponse(
                id="chatcmpl-1",
                model="test-model",
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(reasoning=" step by step."),
                    )
                ],
            ),
            # Text content starts
            ChatCompletionStreamResponse(
                id="chatcmpl-1",
                model="test-model",
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(content="The answer is "),
                    )
                ],
            ),
            # More text
            ChatCompletionStreamResponse(
                id="chatcmpl-1",
                model="test-model",
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(content="42."),
                    )
                ],
            ),
            # Finish reason
            ChatCompletionStreamResponse(
                id="chatcmpl-1",
                model="test-model",
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason="stop",
                    )
                ],
            ),
            # Usage chunk (empty choices)
            ChatCompletionStreamResponse(
                id="chatcmpl-1",
                model="test-model",
                choices=[],
                usage=UsageInfo(
                    prompt_tokens=10, completion_tokens=20, total_tokens=30
                ),
            ),
        ]

        events = await self._collect_events(handler, chunks)
        event_types = [e[0] for e in events]

        # Verify event sequence
        assert "message_start" in event_types

        # Find thinking block events
        thinking_start = None
        thinking_deltas = []
        text_start = None
        text_deltas = []
        block_stops = 0

        for event_type, data in events:
            if data is None:
                continue
            if event_type == "content_block_start":
                cb = data.get("content_block", {})
                if cb.get("type") == "thinking":
                    thinking_start = data
                elif cb.get("type") == "text":
                    text_start = data
            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "thinking_delta":
                    thinking_deltas.append(delta)
                elif delta.get("type") == "text_delta":
                    text_deltas.append(delta)
            elif event_type == "content_block_stop":
                block_stops += 1

        # Thinking block at index 0
        assert thinking_start is not None
        assert thinking_start["index"] == 0
        assert len(thinking_deltas) == 2
        assert thinking_deltas[0]["thinking"] == "Let me think..."
        assert thinking_deltas[1]["thinking"] == " step by step."

        # Text block at index 1
        assert text_start is not None
        assert text_start["index"] == 1
        assert len(text_deltas) == 2
        assert text_deltas[0]["text"] == "The answer is "
        assert text_deltas[1]["text"] == "42."

        # Two block stops (thinking + text)
        assert block_stops == 2

    @pytest.mark.asyncio
    async def test_stream_without_reasoning(self):
        handler = self._make_handler()

        chunks = [
            ChatCompletionStreamResponse(
                id="chatcmpl-1",
                model="test-model",
                choices=[],
                usage=UsageInfo(prompt_tokens=10, completion_tokens=0, total_tokens=10),
            ),
            ChatCompletionStreamResponse(
                id="chatcmpl-1",
                model="test-model",
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(content="Hello!"),
                    )
                ],
            ),
            ChatCompletionStreamResponse(
                id="chatcmpl-1",
                model="test-model",
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason="stop",
                    )
                ],
            ),
            ChatCompletionStreamResponse(
                id="chatcmpl-1",
                model="test-model",
                choices=[],
                usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        events = await self._collect_events(handler, chunks)

        # No thinking events should be present
        for event_type, data in events:
            if data is None:
                continue
            if event_type == "content_block_start":
                cb = data.get("content_block", {})
                assert cb.get("type") != "thinking"
            if event_type == "content_block_delta":
                delta = data.get("delta", {})
                assert delta.get("type") != "thinking_delta"


# -- Serialization round-trip tests -----------------------------------------


class TestThinkingContentBlockSerialization:
    def test_thinking_block_serialized_correctly(self):
        block = AnthropicContentBlock(type="thinking", thinking="reasoning text")
        data = block.model_dump(exclude_none=True)
        assert data == {"type": "thinking", "thinking": "reasoning text"}

    def test_thinking_delta_serialized_correctly(self):
        delta = AnthropicDelta(type="thinking_delta", thinking="partial reasoning")
        data = delta.model_dump(exclude_none=True)
        assert data == {
            "type": "thinking_delta",
            "thinking": "partial reasoning",
        }

    def test_stream_event_with_thinking_delta(self):
        event = AnthropicStreamEvent(
            type="content_block_delta",
            index=0,
            delta=AnthropicDelta(type="thinking_delta", thinking="some reasoning"),
        )
        data = json.loads(event.model_dump_json(exclude_unset=True))
        assert data["type"] == "content_block_delta"
        assert data["delta"]["type"] == "thinking_delta"
        assert data["delta"]["thinking"] == "some reasoning"

    def test_thinking_config_enabled_serialized(self):
        cfg = AnthropicThinkingConfig(type="enabled", budget_tokens=4096)
        data = cfg.model_dump()
        assert data == {"type": "enabled", "budget_tokens": 4096}

    def test_request_with_thinking_serialized(self):
        req = AnthropicMessagesRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            thinking={"type": "enabled", "budget_tokens": 2048},
        )
        data = req.model_dump()
        assert data["thinking"]["type"] == "enabled"
        assert data["thinking"]["budget_tokens"] == 2048
