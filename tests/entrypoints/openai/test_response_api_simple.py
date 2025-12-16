# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import pytest_asyncio
from openai import OpenAI

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-8B"


@pytest.fixture(scope="module")
def server():
    args = ["--reasoning-parser", "qwen3", "--max_model_len", "5000"]
    env_dict = dict(
        VLLM_ENABLE_RESPONSES_API_STORE="1",
        # uncomment for tool calling
        # PYTHON_EXECUTION_BACKEND="dangerously_use_uv",
    )

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env_dict) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_basic(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is 13 * 24?",
    )
    assert response is not None
    print("response: ", response)
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_enable_response_messages(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="Hello?",
        extra_body={"enable_response_messages": True},
    )
    assert response.status == "completed"
    assert response.input_messages[0]["type"] == "raw_message_tokens"
    assert type(response.input_messages[0]["message"]) is str
    assert len(response.input_messages[0]["message"]) > 10
    assert type(response.input_messages[0]["tokens"][0]) is int
    assert type(response.output_messages[0]["message"]) is str
    assert len(response.output_messages[0]["message"]) > 10
    assert type(response.output_messages[0]["tokens"][0]) is int


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_reasoning_item(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input=[
            {"type": "message", "content": "Hello.", "role": "user"},
            {
                "type": "reasoning",
                "id": "lol",
                "content": [
                    {
                        "type": "reasoning_text",
                        "text": "We need to respond: greeting.",
                    }
                ],
                "summary": [],
            },
        ],
        temperature=0.0,
    )
    assert response is not None
    assert response.status == "completed"
    # make sure we get a reasoning and text output
    assert response.output[0].type == "reasoning"
    assert response.output[1].type == "message"
    assert type(response.output[1].content[0].text) is str
