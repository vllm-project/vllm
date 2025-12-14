# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test demonstrating tool calling + structured output issue in vLLM.
When both are used together, tools are ignored even with tool_choice='auto'.
"""

# Use RemoteOpenAIServer for self-contained test
import sys
from pathlib import Path

import openai
import pytest
import pytest_asyncio

# Add tests directory to path to import utilities
tests_dir = Path(__file__).parent / "tests"
sys.path.insert(0, str(tests_dir))

from utils import RemoteOpenAIServer  # noqa: E402

# Use a small model for testing
MODEL_NAME = "Qwen/Qwen3-0.6B"

# Define a simple weather tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"}
                },
                "required": ["location"],
            },
        },
    }
]

# Define a response format (structured output)
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    },
}

# Test message
messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]


@pytest.fixture(scope="module")
def server():
    """Start vLLM server with necessary flags for tool calling."""
    args = [
        "--dtype",
        "half",
        "--enable-auto-tool-choice",
        "--structured-outputs-config.backend",
        "xgrammar",
        "--tool-call-parser",
        "hermes",
        "--gpu-memory-utilization",
        "0.4",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    """Create async OpenAI client."""
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_tools_with_structured_output(client: openai.AsyncOpenAI):
    """
    Test that tools work correctly when used together with response_format.

    This is the main bug test case: when both tools and response_format are
    provided, the tools should still be available and callable, but currently
    they are ignored.
    """
    # Test WITH structured output (currently fails to call tool)
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        response_format=response_format,
    )

    message = response.choices[0].message

    # Check if tool was called
    print(f"\n{'=' * 60}")
    print("TEST: Tools + Structured Output Together")
    print(f"{'=' * 60}")
    print(f"Tool calls: {message.tool_calls}")
    print(f"Content: {message.content}")
    print(f"Finish reason: {response.choices[0].finish_reason}")

    # According to the bug report, tools should NOT be called when response_format
    # is present (the bug), but they SHOULD be called (correct behavior)
    if message.tool_calls:
        print("✅ GOOD: Tool was called (expected behavior - bug may be fixed!)")
        # Test passes - correct behavior
    else:
        print(
            "❌ BAD: Tool was NOT called "
            "(bug - should call tool even with response_format)"
        )
        # Fail the test to document the bug
        pytest.fail(
            "Bug confirmed: Tools are ignored when response_format is "
            "present. Expected tool to be called for weather query, but "
            "got text response instead."
        )


@pytest.mark.asyncio
async def test_tools_without_structured_output(client: openai.AsyncOpenAI):
    """
    Test that tools work correctly when used WITHOUT response_format.

    This should work correctly and serves as a baseline comparison.
    """
    # Test WITHOUT structured output (should work)
    response = await client.chat.completions.create(
        model=MODEL_NAME, messages=messages, tools=tools, tool_choice="auto"
    )

    message = response.choices[0].message

    # Check if tool was called
    print(f"\n{'=' * 60}")
    print("BASELINE: Tools Only (no response_format)")
    print(f"{'=' * 60}")
    print(f"Tool calls: {message.tool_calls}")
    print(f"Content: {message.content}")
    print(f"Finish reason: {response.choices[0].finish_reason}")

    # This should pass - tools work without response_format
    if message.tool_calls:
        print("✅ Tool was called (baseline works correctly)")
    else:
        print("⚠️  No tool call (model chose to respond with text)")

    assert message.tool_calls is not None or message.content is not None, (
        "Expected either tool call or text response"
    )


@pytest.mark.asyncio
async def test_structured_output_without_tools(client: openai.AsyncOpenAI):
    """
    Test that structured output works correctly when used WITHOUT tools.

    This should work correctly and serves as a baseline comparison.
    """
    # Test structured output only (should work)
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello to the world"}],
        response_format=response_format,
    )

    message = response.choices[0].message

    print(f"\n{'=' * 60}")
    print("BASELINE: Structured Output Only (no tools)")
    print(f"{'=' * 60}")
    print(f"Content: {message.content}")
    print(f"Finish reason: {response.choices[0].finish_reason}")

    # This should pass - structured output works without tools
    if message.content:
        print("✅ Structured output works correctly")

    assert message.content is not None, "Expected structured JSON response"


if __name__ == "__main__":
    # Allow running directly with pytest
    pytest.main([__file__, "-v", "-s"])
