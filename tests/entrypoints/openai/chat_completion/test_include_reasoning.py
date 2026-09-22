# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for ``include_reasoning`` with non-Harmony reasoning models.

Verifies that reasoning content is included by default and suppressed
when ``include_reasoning=False``, for both streaming and non-streaming
Chat Completions.
"""

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
MESSAGES = [{"role": "user", "content": "What is 1+1? Be concise."}]


@pytest.fixture(scope="module")
def server():
    args = [
        "--reasoning-parser",
        "qwen3",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_include_reasoning_true_non_streaming(client: openai.AsyncOpenAI):
    """Default: reasoning content appears in non-streaming response."""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=200,
        extra_body={"include_reasoning": True},
    )

    msg = response.choices[0].message
    reasoning = getattr(msg, "reasoning", None) or getattr(
        msg, "reasoning_content", None
    )
    assert reasoning, "Expected reasoning content when include_reasoning=True"
    assert msg.content, "Expected content in response"


@pytest.mark.asyncio
async def test_include_reasoning_false_non_streaming(client: openai.AsyncOpenAI):
    """Reasoning content is suppressed when include_reasoning=False."""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=200,
        extra_body={"include_reasoning": False},
    )

    msg = response.choices[0].message
    reasoning = getattr(msg, "reasoning", None) or getattr(
        msg, "reasoning_content", None
    )
    assert not reasoning, (
        f"Expected no reasoning when include_reasoning=False, got: {reasoning}"
    )
    assert msg.content, "Expected content in response even without reasoning"


@pytest.mark.asyncio
async def test_include_reasoning_true_streaming(client: openai.AsyncOpenAI):
    """Default: reasoning deltas appear in streaming response."""
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=200,
        stream=True,
        extra_body={"include_reasoning": True},
    )

    reasoning_parts = []
    content_parts = []
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta:
            r = getattr(delta, "reasoning", None) or getattr(
                delta, "reasoning_content", None
            )
            if r:
                reasoning_parts.append(r)
            if delta.content:
                content_parts.append(delta.content)

    reasoning_text = "".join(reasoning_parts)
    content_text = "".join(content_parts)

    assert reasoning_text, "Expected reasoning deltas when include_reasoning=True"
    assert content_text, "Expected content deltas in streaming response"


@pytest.mark.asyncio
async def test_include_reasoning_false_streaming(client: openai.AsyncOpenAI):
    """Reasoning deltas are suppressed in streaming when include_reasoning=False."""
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=200,
        stream=True,
        extra_body={"include_reasoning": False},
    )

    reasoning_parts = []
    content_parts = []
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta:
            r = getattr(delta, "reasoning", None) or getattr(
                delta, "reasoning_content", None
            )
            if r:
                reasoning_parts.append(r)
            if delta.content:
                content_parts.append(delta.content)

    reasoning_text = "".join(reasoning_parts)
    content_text = "".join(content_parts)

    assert not reasoning_text, (
        f"Expected no reasoning deltas when include_reasoning=False, "
        f"got: {reasoning_text[:100]}"
    )
    assert content_text, "Expected content deltas even without reasoning"


@pytest.mark.asyncio
async def test_default_includes_reasoning(client: openai.AsyncOpenAI):
    """Without specifying include_reasoning, reasoning appears (default=True)."""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=200,
    )

    msg = response.choices[0].message
    reasoning = getattr(msg, "reasoning", None) or getattr(
        msg, "reasoning_content", None
    )
    assert reasoning, "Expected reasoning content by default"
