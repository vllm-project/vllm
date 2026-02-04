# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that reasoning parser works correctly with multi-turn conversations
where previous assistant messages may contain thinking content.

This test validates the fix for the bug where `is_reasoning_end(res.prompt_token_ids)`
incorrectly set `reasoning_end_arr = True` when conversation history contained
`</think>` from previous assistant turns, causing new `<think>` blocks in the
current response to be treated as content instead of reasoning.
"""

import openai
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

# Use a small reasoning model for testing
MODEL_NAME = "Qwen/QwQ-32B"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--reasoning-parser",
        "deepseek_r1",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

# Conversation history where the previous assistant message
# contains thinking content with </think> tags.
# This is the scenario that triggered the bug.
MESSAGES_WITH_THINKING_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {
        "role": "assistant",
        # Simulating a previous response that had thinking content
        # The </think> token in this message was incorrectly causing
        # the reasoning parser to skip new <think> blocks
        "content": "The answer is 4.",
        # Note: In a real scenario, the thinking content would have been
        # in a separate field, but the tokenized prompt would still
        # contain the </think> token from the chat template
    },
    {"role": "user", "content": "What's the weather in Seattle?"},
]

# Simple multi-turn without thinking in history (for comparison)
MESSAGES_WITHOUT_THINKING_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Hello! How can I help you today?"},
    {"role": "user", "content": "What's the weather in Seattle?"},
]


def extract_reasoning_from_chunks(chunks: list) -> str:
    """Extract reasoning content from streaming chunks."""
    reasoning = ""
    for chunk in chunks:
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning") and delta.reasoning:
            reasoning += delta.reasoning
    return reasoning


def extract_content_from_chunks(chunks: list) -> str:
    """Extract content from streaming chunks."""
    content = ""
    for chunk in chunks:
        if chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    return content


@pytest.mark.asyncio
async def test_reasoning_with_multi_turn_conversation(client: openai.AsyncOpenAI):
    """
    Test that reasoning is correctly extracted in a multi-turn conversation.

    This validates that even when conversation history is present,
    new <think> blocks are correctly parsed as reasoning, not content.
    """
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_WITHOUT_THINKING_HISTORY,
        tools=TOOLS,
        temperature=0.0,
        stream=True,
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    reasoning = extract_reasoning_from_chunks(chunks)
    content = extract_content_from_chunks(chunks)

    # The model should produce reasoning when asked about weather with tools
    # The key assertion: reasoning should be non-empty (not all in content)
    assert len(reasoning) > 0, (
        "Reasoning should be extracted, not embedded in content. "
        f"Got reasoning='{reasoning[:100]}...' content='{content[:100]}...'"
    )

    # Content should NOT contain <think> tags - they should be in reasoning
    assert "<think>" not in content, (
        f"<think> tags should be in reasoning, not content: {content[:100]}"
    )
    assert "</think>" not in content, (
        f"</think> tags should be in reasoning, not content: {content[:100]}"
    )


@pytest.mark.asyncio
async def test_reasoning_after_previous_thinking_response(client: openai.AsyncOpenAI):
    """
    Test that reasoning works correctly even when previous assistant
    messages in the conversation history contained thinking content.

    This is the specific regression test for the bug where
    is_reasoning_end(res.prompt_token_ids) incorrectly detected
    </think> from previous turns and disabled reasoning parsing.
    """
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_WITH_THINKING_HISTORY,
        tools=TOOLS,
        temperature=0.0,
        stream=True,
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    reasoning = extract_reasoning_from_chunks(chunks)
    content = extract_content_from_chunks(chunks)

    # Even with conversation history, reasoning should be extracted correctly
    # This was the bug: reasoning was empty and everything went to content
    assert len(reasoning) > 0, (
        "Reasoning should be extracted even with conversation history. "
        "Bug: is_reasoning_end(prompt_token_ids) was incorrectly detecting "
        "</think> from previous turns. "
        f"Got reasoning='{reasoning[:100] if reasoning else ''}' "
        f"content='{content[:100] if content else ''}'"
    )

    # Content should NOT contain <think> tags
    assert "<think>" not in content, (
        f"<think> tags leaked into content (bug not fixed). Content: {content[:200]}"
    )
    assert "</think>" not in content, (
        f"</think> tags leaked into content (bug not fixed). Content: {content[:200]}"
    )


@pytest.mark.asyncio
async def test_non_streaming_reasoning_with_history(client: openai.AsyncOpenAI):
    """
    Test non-streaming reasoning extraction with conversation history.
    """
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES_WITH_THINKING_HISTORY,
        tools=TOOLS,
        temperature=0.0,
        stream=False,
    )

    reasoning = response.choices[0].message.reasoning
    content = response.choices[0].message.content

    # Reasoning should be present
    assert reasoning is not None and len(reasoning) > 0, (
        f"Reasoning should be extracted. Got reasoning={reasoning}, content={content}"
    )

    # Content should not contain think tags
    if content:
        assert "<think>" not in content, f"<think> in content: {content[:200]}"
        assert "</think>" not in content, f"</think> in content: {content[:200]}"
