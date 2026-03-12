# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import logging

import pytest
import pytest_asyncio
from openai import OpenAI

from ....utils import RemoteOpenAIServer
from .conftest import (
    BASE_TEST_ENV,
    has_output_type,
    log_response_diagnostics,
    retry_for_tool_call,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-8B"

_PYTHON_TOOL_INSTRUCTION = (
    "You must use the Python tool to execute code. "
    "Never simulate execution. You must print the final answer."
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
    env_dict = {
        **BASE_TEST_ENV,
        "VLLM_ENABLE_RESPONSES_API_STORE": "1",
        "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT": "1",
        "PYTHON_EXECUTION_BACKEND": "dangerously_use_uv",
    }
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
        temperature=0.0,
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

    output_types = [getattr(o, "type", None) for o in response.output]
    assert "reasoning" in output_types, (
        f"Expected reasoning in output, got: {output_types}"
    )
    assert "message" in output_types, f"Expected message in output, got: {output_types}"

    msg = next(o for o in response.output if o.type == "message")
    assert type(msg.content[0].text) is str


def get_horoscope(sign):
    return f"{sign}: Next Tuesday you will befriend a baby otter."


def call_function(name, args):
    logger.info("Calling function %s with args %s", name, args)
    if name == "get_horoscope":
        return get_horoscope(**args)
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

    response = await retry_for_tool_call(
        client,
        model=model_name,
        expected_tool_type="function_call",
        input="What is the horoscope for Aquarius today?",
        tools=tools,
        temperature=0.0,
    )
    assert response is not None
    assert response.status == "completed"

    output_types = [getattr(o, "type", None) for o in response.output]
    assert "reasoning" in output_types, (
        f"Expected reasoning in output, got: {output_types}"
    )
    assert has_output_type(response, "function_call"), (
        f"Expected function_call in output, got: {output_types}"
    )

    function_call = next(o for o in response.output if o.type == "function_call")
    assert function_call.name == "get_horoscope"
    assert function_call.call_id is not None

    args = json.loads(function_call.arguments)
    assert "sign" in args


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_mcp_tool_call(client: OpenAI, model_name: str):
    """MCP tool calling with code_interpreter.

    The model may make one or more tool calls before producing a final
    message.  We validate server invariants (mcp_call items have correct
    fields) with hard assertions.  Output indices are never hardcoded
    since the model can produce multiple tool-call rounds.
    """
    # MCP + container init + code execution can be slow
    client_with_timeout = client.with_options(timeout=client.timeout * 3)

    response = await retry_for_tool_call(
        client_with_timeout,
        model=model_name,
        expected_tool_type="mcp_call",
        input=(
            "What is 123 * 456? Use python to calculate the result. "
            "Print the result with print()."
        ),
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        instructions=_PYTHON_TOOL_INSTRUCTION,
        temperature=0.0,
        extra_body={"enable_response_messages": True},
    )

    assert response is not None

    output_types = [getattr(o, "type", None) for o in response.output]
    log_response_diagnostics(response, label="test_mcp_tool_call")

    assert response.status == "completed", (
        f"Response status={response.status} "
        f"(details={getattr(response, 'incomplete_details', None)}). "
        f"Output types: {output_types}."
    )

    assert "reasoning" in output_types, (
        f"Expected reasoning in output, got: {output_types}"
    )
    assert "mcp_call" in output_types, (
        f"Expected mcp_call in output, got: {output_types}"
    )

    # Every mcp_call item must have well-typed fields
    for item in response.output:
        if getattr(item, "type", None) == "mcp_call":
            assert type(item.arguments) is str, (
                f"mcp_call.arguments should be str, got {type(item.arguments)}"
            )
            assert type(item.output) is str, (
                f"mcp_call.output should be str, got {type(item.output)}"
            )

    # The model may make 1+ tool-call rounds but must still produce
    # a final message for a trivial calculation like 123 * 456.
    message_outputs = [
        o for o in response.output if getattr(o, "type", None) == "message"
    ]
    assert message_outputs, (
        f"Model did not produce a final message. Output types: {output_types}"
    )

    final_message = message_outputs[-1]
    assert any(s in final_message.content[0].text for s in ("56088", "56,088")), (
        f"Expected 56088 in final message, got: {final_message.content[0].text!r}"
    )

    # Validate raw input_messages / output_messages
    assert len(response.input_messages) >= 1, "Expected at least 1 input message"
    assert len(response.output_messages) >= 1, "Expected at least 1 output message"
    assert any(
        any(s in str(msg) for s in ("56088", "56,088"))
        for msg in response.output_messages
    ), (
        f"Expected 56088 in at least one output_message, "
        f"got {len(response.output_messages)} messages"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_max_tokens(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is the first paragraph of Moby Dick?",
        reasoning={"effort": "low"},
        max_output_tokens=30,
        temperature=0.0,
    )
    assert response is not None
    assert response.status == "incomplete"
    assert response.incomplete_details.reason == "max_output_tokens"
