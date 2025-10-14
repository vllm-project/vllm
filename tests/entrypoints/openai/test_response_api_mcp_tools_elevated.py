# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess

import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer, find_free_port
from .memory_mcp_server import start_test_server

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
def memory_elevated_server(monkeypatch_module, memory_mcp_server):
    """vLLM server with Memory MCP tool elevated."""
    server_url, port = memory_mcp_server
    args = ["--enforce-eager", "--tool-server", f"localhost:{port}"]

    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
        # Use system instructions to ensure model follows directions
        m.setenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "1")
        m.setenv("GPT_OSS_SYSTEM_TOOL_MCP_LABELS", "memory")  # Elevate memory tool
        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def memory_elevated_client(memory_elevated_server):
    async with memory_elevated_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_memory_mcp_elevated(memory_elevated_client, model_name: str):
    """Test Memory MCP tool as elevated.

    When memory IS in GPT_OSS_SYSTEM_TOOL_MCP_LABELS:
    - Tool should be in system message (not developer message)
    - Tool calls should be on analysis channel
    - Tool responses should be on analysis channel
    """
    response = await memory_elevated_client.responses.create(
        model=model_name,
        instructions=(
            "You must use the memory.store and memory.retrieve tools. "
            "Never simulate tool execution. Call the tool using json "
            "on the analysis channel like a normal system tool."
        ),
        input=(
            "Store the key 'elevated_key' with value 'elevated_value' and retrieve it"
        ),
        tools=[
            {
                "type": "mcp",
                "server_label": "memory",
                # URL unused, connection via --tool-server
                "server_url": "http://unused",
                "headers": {"x-memory-id": "test-session-elevated"},
            }
        ],
        extra_body={"enable_response_messages": True},
    )

    assert response is not None
    assert response.status == "completed"
    assert response.usage.output_tokens_details.tool_output_tokens > 0

    # Verify input messages: Should have system message with tool, NO developer message
    # (since all tools are elevated)
    developer_messages = [
        msg for msg in response.input_messages if msg["author"]["role"] == "developer"
    ]
    assert len(developer_messages) == 0, (
        "No developer message expected for elevated tools"
    )

    # Verify output messages: Tool calls and responses on analysis channel
    tool_call_found = False
    tool_response_found = False
    for message in response.output_messages:
        recipient = message.get("recipient")
        if recipient and recipient.startswith("memory."):
            tool_call_found = True
            assert message.get("channel") == "analysis", (
                "Tool call should be on analysis channel"
            )
        author = message.get("author", {})
        if (
            author.get("role") == "tool"
            and author.get("name")
            and author.get("name").startswith("memory.")
        ):
            tool_response_found = True
            assert message.get("channel") == "analysis", (
                "Tool response should be on analysis channel"
            )

    assert tool_call_found, "Should have found at least one memory tool call"
    assert tool_response_found, "Should have found at least one memory tool response"

    # Verify functional correctness
    output_text = ""
    for item in response.output:
        if hasattr(item, "content"):
            for content_item in item.content:
                if hasattr(content_item, "text"):
                    output_text += content_item.text
    assert (
        "elevated_value" in output_text.lower() or "successfully" in output_text.lower()
    )
