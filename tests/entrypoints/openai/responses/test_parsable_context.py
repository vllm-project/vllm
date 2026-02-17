# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json

import pytest
import pytest_asyncio
from openai import OpenAI

from ....utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-8B"


@pytest.fixture(scope="module")
def server():
    assert importlib.util.find_spec("gpt_oss") is not None, (
        "Harmony tests require gpt_oss package to be installed"
    )

    args = [
        "--reasoning-parser",
        "qwen3",
        "--max_model_len",
        "5000",
        "--structured-outputs-config.backend",
        "xgrammar",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--tool-server",
        "demo",
    ]
    env_dict = dict(
        VLLM_ENABLE_RESPONSES_API_STORE="1",
        VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT="1",
        PYTHON_EXECUTION_BACKEND="dangerously_use_uv",
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
        input="What is 123 * 456?",
    )
    assert response is not None
    print("response: ", response)
    assert response.status == "completed"
    assert response.incomplete_details is None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_reasoning_and_function_items(client: OpenAI, model_name: str):
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
            {
                "arguments": '{"location": "Paris", "unit": "celsius"}',
                "call_id": "call_5f7b38f3b81e4b8380fd0ba74f3ca3ab",
                "name": "get_weather",
                "type": "function_call",
                "id": "fc_4fe5d6fc5b6c4d6fa5f24cc80aa27f78",
                "status": "completed",
            },
            {
                "call_id": "call_5f7b38f3b81e4b8380fd0ba74f3ca3ab",
                "id": "fc_4fe5d6fc5b6c4d6fa5f24cc80aa27f78",
                "output": "The weather in Paris is 20 Celsius",
                "status": "completed",
                "type": "function_call_output",
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


def get_horoscope(sign):
    return f"{sign}: Next Tuesday you will befriend a baby otter."


def call_function(name, args):
    if name == "get_horoscope":
        return get_horoscope(**args)
    else:
        raise ValueError(f"Unknown function: {name}")


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_function_call_first_turn(client: OpenAI, model_name: str):
    tools = [
        {
            "type": "function",
            "name": "get_horoscope",
            "description": "Get today's horoscope for an astrological sign.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sign": {"type": "string"},
                },
                "required": ["sign"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    response = await client.responses.create(
        model=model_name,
        input="What is the horoscope for Aquarius today?",
        tools=tools,
        temperature=0.0,
    )
    assert response is not None
    assert response.status == "completed"
    assert len(response.output) == 2
    assert response.output[0].type == "reasoning"
    assert response.output[1].type == "function_call"

    function_call = response.output[1]
    assert function_call.name == "get_horoscope"
    assert function_call.call_id is not None

    args = json.loads(function_call.arguments)
    assert "sign" in args

    # the multi turn function call is tested above in
    # test_reasoning_and_function_items


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_mcp_tool_call(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is 123 * 456? Use python to calculate the result.",
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        extra_body={"enable_response_messages": True},
        temperature=0.0,
    )

    assert response is not None
    assert response.status == "completed"

    # The model may produce multiple reasoning/mcp_call rounds before the
    # final message, so validate structurally rather than by exact index.
    output_types = [o.type for o in response.output]
    assert "reasoning" in output_types
    mcp_calls = [o for o in response.output if o.type == "mcp_call"]
    assert len(mcp_calls) >= 1
    assert type(mcp_calls[0].arguments) is str
    assert type(mcp_calls[0].output) is str

    # The final output should be a message containing the correct answer
    assert response.output[-1].type == "message"
    assert any(s in response.output[-1].content[0].text for s in ("56088", "56,088"))

    # Test raw input_messages / output_messages
    assert len(response.input_messages) == 1
    assert len(response.output_messages) >= 3
    assert any(
        s in response.output_messages[-1]["message"] for s in ("56088", "56,088")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_max_tokens(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is the first paragraph of Moby Dick?",
        reasoning={"effort": "low"},
        max_output_tokens=30,
    )
    assert response is not None
    assert response.status == "incomplete"
    assert response.incomplete_details.reason == "max_output_tokens"
