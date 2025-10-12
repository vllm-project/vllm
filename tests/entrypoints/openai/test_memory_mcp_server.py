# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the standalone Memory MCP Server.

These tests verify that the memory MCP server works correctly:
- Basic store/retrieve operations
- Memory isolation with x-memory-id headers
- Default memory pool behavior
- List keys functionality
- Delete functionality
"""

import asyncio
import os
import socket
import subprocess
import sys
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

from tests.utils import find_free_port


@asynccontextmanager
async def mcp_client_session(server_url: str, headers: dict[str, str] | None = None):
    """Create an MCP client session with optional custom headers."""
    async with (
        sse_client(url=server_url, headers=headers or {}) as streams,
        ClientSession(*streams) as session,
    ):
        await session.initialize()
        yield session


@pytest_asyncio.fixture(scope="function")
async def memory_server():
    """Start memory MCP server as subprocess on random port."""
    port = find_free_port()
    server_url = f"http://127.0.0.1:{port}/sse"

    # Get the path to the memory_mcp_server.py script
    script_path = os.path.join(
        os.path.dirname(__file__),
        "memory_mcp_server.py",
    )

    # Start server process
    process = subprocess.Popen(
        [sys.executable, script_path, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for server to be ready (max 10 seconds)
    server_ready = False
    for attempt in range(50):  # 50 attempts * 0.2s = 10s max
        try:
            # Simple TCP connection check
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            if result == 0:
                server_ready = True
                # Give it a tiny bit more time to fully initialize
                await asyncio.sleep(0.1)
                break
        except Exception:
            pass
        await asyncio.sleep(0.2)

    if not server_ready:
        process.kill()
        stdout, stderr = process.communicate()
        pytest.fail(
            f"Memory MCP server failed to start within 10 seconds.\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )

    # Yield server URL for tests
    yield server_url

    # Cleanup: kill process
    process.kill()
    process.wait()


@pytest.mark.asyncio
async def test_basic_store_retrieve(memory_server):
    """Test basic store and retrieve operations."""
    async with mcp_client_session(memory_server) as session:
        # Store a value
        store_result = await session.call_tool(
            "store", arguments={"key": "test_key", "value": "test_value"}
        )
        assert store_result is not None
        assert len(store_result.content) > 0
        assert "Successfully stored" in store_result.content[0].text

        # Retrieve the value
        retrieve_result = await session.call_tool(
            "retrieve", arguments={"key": "test_key"}
        )
        assert retrieve_result is not None
        assert len(retrieve_result.content) > 0
        assert "test_value" in retrieve_result.content[0].text


@pytest.mark.asyncio
async def test_retrieve_nonexistent_key(memory_server):
    """Test retrieving a key that doesn't exist."""
    async with mcp_client_session(memory_server) as session:
        retrieve_result = await session.call_tool(
            "retrieve", arguments={"key": "nonexistent_key"}
        )
        assert retrieve_result is not None
        assert len(retrieve_result.content) > 0
        assert "not found" in retrieve_result.content[0].text


@pytest.mark.asyncio
async def test_memory_isolation_with_headers(memory_server):
    """Test that different x-memory-id headers isolate data."""
    # Session 1 with x-memory-id: "user1"
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "user1"}
    ) as session1:
        # Store in user1's space
        await session1.call_tool("store", arguments={"key": "name", "value": "Alice"})

    # Session 2 with x-memory-id: "user2"
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "user2"}
    ) as session2:
        # Store in user2's space
        await session2.call_tool("store", arguments={"key": "name", "value": "Bob"})

    # Verify session1 gets Alice
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "user1"}
    ) as session1:
        result = await session1.call_tool("retrieve", arguments={"key": "name"})
        assert "Alice" in result.content[0].text
        assert "Bob" not in result.content[0].text

    # Verify session2 gets Bob
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "user2"}
    ) as session2:
        result = await session2.call_tool("retrieve", arguments={"key": "name"})
        assert "Bob" in result.content[0].text
        assert "Alice" not in result.content[0].text


@pytest.mark.asyncio
async def test_default_memory_pool(memory_server):
    """Test that no header uses default pool."""
    # Session without header
    async with mcp_client_session(memory_server) as session1:
        await session1.call_tool("store", arguments={"key": "shared", "value": "data"})

    # Another session without header should see the same data
    async with mcp_client_session(memory_server) as session2:
        result = await session2.call_tool("retrieve", arguments={"key": "shared"})
        assert "data" in result.content[0].text


@pytest.mark.asyncio
async def test_default_pool_isolated_from_custom_headers(memory_server):
    """Test that default pool is isolated from custom memory IDs."""
    # Store in default pool
    async with mcp_client_session(memory_server) as session_default:
        await session_default.call_tool(
            "store", arguments={"key": "isolation_test", "value": "default_value"}
        )

    # Try to retrieve from custom memory ID - should not find it
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "custom"}
    ) as session_custom:
        result = await session_custom.call_tool(
            "retrieve", arguments={"key": "isolation_test"}
        )
        assert "not found" in result.content[0].text


@pytest.mark.asyncio
async def test_list_keys(memory_server):
    """Test listing keys in memory space."""
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "list_test"}
    ) as session:
        # Initially should be empty
        result = await session.call_tool("list_keys", arguments={})
        assert "No keys found" in result.content[0].text

        # Store multiple keys
        await session.call_tool("store", arguments={"key": "key1", "value": "value1"})
        await session.call_tool("store", arguments={"key": "key2", "value": "value2"})
        await session.call_tool("store", arguments={"key": "key3", "value": "value3"})

        # List keys - should see all three
        result = await session.call_tool("list_keys", arguments={})
        result_text = result.content[0].text
        assert "key1" in result_text
        assert "key2" in result_text
        assert "key3" in result_text


@pytest.mark.asyncio
async def test_list_keys_isolation(memory_server):
    """Test that list_keys respects memory isolation."""
    # Store keys in different memory spaces
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "space1"}
    ) as session1:
        await session1.call_tool(
            "store", arguments={"key": "space1_key", "value": "val1"}
        )

    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "space2"}
    ) as session2:
        await session2.call_tool(
            "store", arguments={"key": "space2_key", "value": "val2"}
        )

    # List keys in space1 - should only see space1_key
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "space1"}
    ) as session1:
        result = await session1.call_tool("list_keys", arguments={})
        result_text = result.content[0].text
        assert "space1_key" in result_text
        assert "space2_key" not in result_text

    # List keys in space2 - should only see space2_key
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "space2"}
    ) as session2:
        result = await session2.call_tool("list_keys", arguments={})
        result_text = result.content[0].text
        assert "space2_key" in result_text
        assert "space1_key" not in result_text


@pytest.mark.asyncio
async def test_delete(memory_server):
    """Test deleting keys."""
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "delete_test"}
    ) as session:
        # Store a key
        await session.call_tool(
            "store", arguments={"key": "to_delete", "value": "temp"}
        )

        # Verify it exists
        result = await session.call_tool("retrieve", arguments={"key": "to_delete"})
        assert "temp" in result.content[0].text

        # Delete it
        delete_result = await session.call_tool(
            "delete", arguments={"key": "to_delete"}
        )
        assert "Successfully deleted" in delete_result.content[0].text

        # Verify it's gone
        result = await session.call_tool("retrieve", arguments={"key": "to_delete"})
        assert "not found" in result.content[0].text


@pytest.mark.asyncio
async def test_delete_nonexistent_key(memory_server):
    """Test deleting a key that doesn't exist."""
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "delete_test2"}
    ) as session:
        result = await session.call_tool("delete", arguments={"key": "nonexistent"})
        assert "not found" in result.content[0].text


@pytest.mark.asyncio
async def test_delete_isolation(memory_server):
    """Test that delete respects memory isolation."""
    # Store same key in different memory spaces
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "del_space1"}
    ) as session1:
        await session1.call_tool(
            "store", arguments={"key": "shared_key", "value": "value1"}
        )

    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "del_space2"}
    ) as session2:
        await session2.call_tool(
            "store", arguments={"key": "shared_key", "value": "value2"}
        )

    # Delete from space1
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "del_space1"}
    ) as session1:
        await session1.call_tool("delete", arguments={"key": "shared_key"})

        # Verify it's gone from space1
        result = await session1.call_tool("retrieve", arguments={"key": "shared_key"})
        assert "not found" in result.content[0].text

    # Verify it still exists in space2
    async with mcp_client_session(
        memory_server, headers={"x-memory-id": "del_space2"}
    ) as session2:
        result = await session2.call_tool("retrieve", arguments={"key": "shared_key"})
        assert "value2" in result.content[0].text


@pytest.mark.asyncio
async def test_server_info(memory_server):
    """Test that server identifies itself correctly."""
    async with mcp_client_session(memory_server) as session:
        # The session.initialize() call should have set the server info
        # We can verify the server name through list_tools
        tools_result = await session.list_tools()
        assert tools_result is not None
        assert len(tools_result.tools) == 4
        tool_names = [tool.name for tool in tools_result.tools]
        assert "store" in tool_names
        assert "retrieve" in tool_names
        assert "list_keys" in tool_names
        assert "delete" in tool_names
