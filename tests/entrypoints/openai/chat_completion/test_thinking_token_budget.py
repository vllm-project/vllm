# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for thinking_token_budget with reasoning models."""

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
MESSAGES = [{"role": "user", "content": "What is 1+1? Be concise."}]
THINK_BUDGET = 5


@pytest.fixture(scope="module")
def server():
    args = [
        "--reasoning-parser",
        "qwen3",
        "--reasoning-config",
        '{"think_start_str": "<think>", "think_end_str": "</think>"}',
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
        "--no-async-scheduling",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def server_with_auto_reasoning_config():
    args = [
        "--reasoning-parser",
        "qwen3",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
        "--no-async-scheduling",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(request, server, server_with_auto_reasoning_config):
    server_map = {
        "default": server,
        "auto_config": server_with_auto_reasoning_config,
    }
    target_server = server_map[request.param]
    async with target_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["default", "auto_config"], indirect=True)
async def test_thinking_token_budget_mixed_requests(client: openai.AsyncOpenAI):
    """Test that mixed requests (some with thinking_token_budget, some without)
    complete successfully without errors."""

    response_with_budget = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=100,
        extra_body={"thinking_token_budget": THINK_BUDGET},
    )
    response_without_budget = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=100,
    )

    msg_with = response_with_budget.choices[0].message
    msg_without = response_without_budget.choices[0].message

    assert msg_with.content or getattr(msg_with, "reasoning", None)
    assert msg_without.content or getattr(msg_without, "reasoning", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["default", "auto_config"], indirect=True)
async def test_thinking_token_budget_limits_reasoning(client: openai.AsyncOpenAI):
    """Test that thinking_token_budget limits the number of reasoning tokens.

    In streaming mode each reasoning delta corresponds to one token, so
    counting non-empty reasoning_content chunks gives the exact token count.
    """

    reasoning_token_count = 0
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=100,
        stream=True,
        extra_body={"thinking_token_budget": THINK_BUDGET},
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if getattr(delta, "reasoning", None):
            reasoning_token_count += 1

    assert reasoning_token_count == THINK_BUDGET, (
        f"reasoning tokens ({reasoning_token_count}) exceeded "
        f"thinking_token_budget ({THINK_BUDGET})"
    )
