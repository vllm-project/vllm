# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess

import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

MODEL_NAME = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="module")
def memory_mcp_server():
    """Start Memory MCP server as subprocess."""
    from tests.utils import find_free_port

    from .memory_mcp_server import start_test_server

    # Find a free port
    port = find_free_port()

    # Start memory MCP server using helper
    process = start_test_server(port)

    yield f"http://localhost:{port}/sse", port

    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@pytest.fixture(scope="module")
def memory_custom_server(monkeypatch_module, memory_mcp_server):
    """vLLM server with Memory MCP tool as custom (not elevated)."""
    server_url, port = memory_mcp_server
    args = ["--enforce-eager", "--tool-server", f"localhost:{port}"]

    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
        # NO GPT_OSS_SYSTEM_TOOL_MCP_LABELS - memory is custom tool
        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def memory_custom_client(memory_custom_server):
    async with memory_custom_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_memory_mcp_custom(memory_custom_client, model_name: str):
    """Test Memory MCP tool as custom (not elevated).

    When memory is NOT in GPT_OSS_SYSTEM_TOOL_MCP_LABELS:
    - Tool should be in developer message (not system message)
    - Tool calls should be on commentary channel
    - Tool responses should be on commentary channel
    """
    response = await memory_custom_client.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.store and memory.retrieve tools. "
            "Never simulate tool execution."
        ),
        input=("Store the key 'test_key' with value 'test_value' and then retrieve it"),
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                # URL unused, connection via --tool-server
                "server_url": "http://unused",
                "headers": {"x-memory-id": "test-session-custom"},
            }
        ],
        extra_body={"enable_response_messages": True},
    )

    assert response is not None
    assert response.status == "completed"
    assert response.usage.output_tokens_details.tool_output_tokens > 0

    # Verify input messages: Should have developer message with tool
    developer_messages = [
        msg for msg in response.input_messages if msg["author"]["role"] == "developer"
    ]
    assert len(developer_messages) > 0, "Developer message expected for custom tools"

    # Verify output messages: Tool calls and responses on commentary channel
    tool_call_found = False
    tool_response_found = False
    for message in response.output_messages:
        recipient = message.get("recipient")
        if recipient and recipient.startswith("memory."):
            tool_call_found = True
            assert message.get("channel") == "commentary", (
                "Tool call should be on commentary channel"
            )
        author = message.get("author", {})
        if (
            author.get("role") == "tool"
            and author.get("name")
            and author.get("name").startswith("memory.")
        ):
            tool_response_found = True
            assert message.get("channel") == "commentary", (
                "Tool response should be on commentary channel"
            )

    assert tool_call_found, "Should have found at least one memory tool call"
    assert tool_response_found, "Should have found at least one memory tool response"

    # Verify McpCall items (tool invocations)
    from openai.types.responses.response_output_item import McpCall

    mcp_calls = [
        item for item in reversed(response.output) if isinstance(item, McpCall)
    ]

    assert len(mcp_calls) > 0, "Should have at least one McpCall"

    for mcp_call in mcp_calls:
        # Verify it's a memory tool call
        assert mcp_call.server_label == "memory"
        assert mcp_call.name in ["store", "retrieve"]

        # Verify arguments make sense
        assert mcp_call.arguments is not None
        args = json.loads(mcp_call.arguments)
        assert "key" in args

        # Verify output was populated
        assert mcp_call.output is not None
        assert len(mcp_call.output) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_memory_mcp_with_headers(memory_custom_client, model_name: str):
    """Test Memory MCP tool with custom headers for memory isolation.

    Different x-memory-id headers should provide isolated memory spaces.
    This tests that headers are properly passed through the MCP protocol.
    """
    # First request with session-1
    response1 = await memory_custom_client.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.store tool. Never simulate tool execution."
        ),
        input="Store the key 'isolated_key' with value 'session_1_value'",
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                # URL unused, connection via --tool-server
                "server_url": "http://unused",
                "headers": {"x-memory-id": "session-1"},
            }
        ],
    )

    assert response1.status == "completed"
    assert response1.usage.output_tokens_details.tool_output_tokens > 0

    # Second request with session-2 (different memory space)
    response2 = await memory_custom_client.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.retrieve tool. Never simulate tool execution."
        ),
        input="Retrieve the value for key 'isolated_key'",
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                # URL unused, connection via --tool-server
                "server_url": "http://unused",
                "headers": {"x-memory-id": "session-2"},
            }
        ],
    )

    assert response2.status == "completed"
    # The key should NOT be found in session-2 (memory isolation working)
    # Check McpCall output field for exact error message
    from openai.types.responses.response_output_item import McpCall

    mcp_call_output = None
    for item in response2.output:
        if isinstance(item, McpCall) and item.output:
            mcp_call_output = item.output
            break

    # Memory isolation: key from session-1 should not be in session-2
    assert mcp_call_output is not None, "Should have McpCall with output"
    assert "Key 'isolated_key' not found in memory space 'session-2'" in mcp_call_output
