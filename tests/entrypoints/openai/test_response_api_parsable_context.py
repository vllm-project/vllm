# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json

import pytest
import pytest_asyncio
from openai import OpenAI

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-8B"


def assert_token_usage_consistency(response):
    """Validate that token usage statistics are internally consistent.

    This function verifies:
    1. input_tokens == sum(input_tokens_per_turn)
    2. output_tokens == sum(output_tokens_per_turn)
    3. tool_output_tokens == sum(tool_output_tokens_per_turn)
    4. cached_tokens == sum(cached_tokens_per_turn)
    """
    usage = response.usage
    assert usage is not None, "Response should have usage information"

    input_details = usage.input_tokens_details
    output_details = usage.output_tokens_details

    # Verify input_tokens == sum(input_tokens_per_turn)
    if input_details and input_details.input_tokens_per_turn:
        expected_input_tokens = sum(input_details.input_tokens_per_turn)
        assert usage.input_tokens == expected_input_tokens, (
            f"input_tokens ({usage.input_tokens}) != "
            f"sum(input_tokens_per_turn) ({expected_input_tokens})"
        )

    # Verify cached_tokens == sum(cached_tokens_per_turn)
    if input_details and input_details.cached_tokens_per_turn:
        expected_cached_tokens = sum(input_details.cached_tokens_per_turn)
        assert input_details.cached_tokens == expected_cached_tokens, (
            f"cached_tokens ({input_details.cached_tokens}) != "
            f"sum(cached_tokens_per_turn) ({expected_cached_tokens})"
        )

    # Verify output_tokens == sum(output_tokens_per_turn)
    if output_details and output_details.output_tokens_per_turn:
        expected_output_tokens = sum(output_details.output_tokens_per_turn)
        assert usage.output_tokens == expected_output_tokens, (
            f"output_tokens ({usage.output_tokens}) != "
            f"sum(output_tokens_per_turn) ({expected_output_tokens})"
        )

    # Verify tool_output_tokens == sum(tool_output_tokens_per_turn)
    if output_details and output_details.tool_output_tokens_per_turn:
        expected_tool_tokens = sum(output_details.tool_output_tokens_per_turn)
        assert output_details.tool_output_tokens == expected_tool_tokens, (
            f"tool_output_tokens ({output_details.tool_output_tokens}) != "
            f"sum(tool_output_tokens_per_turn) ({expected_tool_tokens})"
        )

    # Verify total_tokens == input_tokens + output_tokens
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens, (
        f"total_tokens ({usage.total_tokens}) != "
        f"input_tokens + output_tokens "
        f"({usage.input_tokens} + {usage.output_tokens})"
    )


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
        input="What is 13 * 24?",
    )
    assert response is not None
    print("response: ", response)
    assert response.status == "completed"

    # Verify token usage consistency
    assert_token_usage_consistency(response)

    # Verify input_tokens > 0
    assert response.usage.input_tokens > 0, "input_tokens should be > 0"


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

    # Verify token usage consistency
    assert_token_usage_consistency(response)

    # Verify input_tokens > 0
    assert response.usage.input_tokens > 0, "input_tokens should be > 0"

    # Verify reasoning tokens > 0 for reasoning output
    output_details = response.usage.output_tokens_details
    assert output_details.reasoning_tokens > 0, (
        "reasoning_tokens should be > 0 for reasoning output"
    )


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

    # Verify token usage consistency
    assert_token_usage_consistency(response)

    # Verify input_tokens > 0
    assert response.usage.input_tokens > 0, "input_tokens should be > 0"

    # Verify reasoning tokens > 0 for reasoning output
    output_details = response.usage.output_tokens_details
    assert output_details.reasoning_tokens > 0, (
        "reasoning_tokens should be > 0 for reasoning output"
    )

    # the multi turn function call is tested above in
    # test_reasoning_and_function_items


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_mcp_tool_call(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is 13 * 24? Use python to calculate the result.",
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        extra_body={"enable_response_messages": True},
        temperature=0.0,
    )

    assert response is not None
    assert response.status == "completed"
    assert response.output[0].type == "reasoning"
    assert response.output[1].type == "mcp_call"
    assert type(response.output[1].arguments) is str
    assert type(response.output[1].output) is str
    assert response.output[2].type == "reasoning"
    # make sure the correct math is in the final output
    assert response.output[3].type == "message"
    assert "312" in response.output[3].content[0].text

    # test raw input_messages / output_messages
    assert len(response.input_messages) == 1
    assert len(response.output_messages) == 3
    assert "312" in response.output_messages[2]["message"]

    # Verify token usage consistency (multi-turn with tool call)
    assert_token_usage_consistency(response)

    # Verify input_tokens > 0
    assert response.usage.input_tokens > 0, "input_tokens should be > 0"

    # Verify reasoning tokens > 0 for reasoning output
    output_details = response.usage.output_tokens_details
    assert output_details.reasoning_tokens > 0, (
        "reasoning_tokens should be > 0 for reasoning output"
    )

    # Verify cached tokens > 0 for multi-turn conversation
    input_details = response.usage.input_tokens_details
    assert input_details.cached_tokens > 0, (
        "cached_tokens should be > 0 for multi-turn conversation"
    )

    # Verify tool output tokens > 0 for tool call
    assert output_details.tool_output_tokens > 0, (
        "tool_output_tokens should be > 0 for tool call"
    )
