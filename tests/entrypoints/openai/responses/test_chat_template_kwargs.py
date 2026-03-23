# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import pytest_asyncio
from openai import OpenAI

from tests.utils import RemoteOpenAIServer

from .conftest import BASE_TEST_ENV

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server():
    args = [
        "--reasoning-parser",
        "qwen3",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--max-model-len",
        "4096",
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
    ]
    env_dict = {
        **BASE_TEST_ENV,
        "VLLM_ENABLE_RESPONSES_API_STORE": "1",
    }
    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env_dict) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_responses_honors_default_chat_template_kwargs(
    client: OpenAI, model_name: str
):
    response = await client.responses.create(
        model=model_name,
        input="Compute 17 * 19 and explain briefly.",
        reasoning={"effort": "low"},
        temperature=0.0,
    )

    reasoning_items = [item for item in response.output if item.type == "reasoning"]

    assert response.status == "completed"
    assert response.output_text
    assert not reasoning_items


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_responses_request_chat_template_kwargs_override_server_default(
    client: OpenAI, model_name: str
):
    response = await client.responses.create(
        model=model_name,
        input="Compute 23 * 17 and explain briefly.",
        reasoning={"effort": "low"},
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    reasoning_items = [item for item in response.output if item.type == "reasoning"]

    assert response.status == "completed"
    assert response.usage is not None
    assert response.usage.output_tokens_details.reasoning_tokens > 0
    assert reasoning_items
    assert reasoning_items[0].content
