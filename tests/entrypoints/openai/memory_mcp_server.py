#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone Memory MCP Server

This is a standalone MCP server that provides memory storage capabilities
with isolation based on x-memory-id headers.

Tools provided:
- store: Store a key-value pair in memory
- retrieve: Retrieve a value by key
- list_keys: List all keys in the current memory space
- delete: Delete a key from memory

Run standalone:
    python memory_mcp_server.py --port 8765
"""

import argparse
import os
import socket
import subprocess
import sys
import time

from fastmcp import Context, FastMCP

# In-memory storage: {memory_id: {key: value}}
memories: dict[str, dict[str, str]] = {}

# Default memory space for requests without x-memory-id header
DEFAULT_MEMORY_ID = "default"

# Create FastMCP app
mcp = FastMCP("memory")


def extract_memory_id(ctx: Context) -> str:
    """Extract memory_id from request context headers."""
    try:
        # Try to get HTTP request from context
        http_request = ctx.get_http_request()
        if http_request and hasattr(http_request, "headers"):
            headers = http_request.headers
            # Headers may be case-insensitive, check variations
            memory_id = headers.get("x-memory-id") or headers.get("X-Memory-Id")
            if memory_id:
                return memory_id
    except Exception:
        pass

    return DEFAULT_MEMORY_ID


@mcp.tool()
def store(key: str, value: str, ctx: Context) -> str:
    """Store a key-value pair in memory.

    Args:
        key: The key to store
        value: The value to store
    """
    memory_id = extract_memory_id(ctx)

    # Ensure memory space exists
    if memory_id not in memories:
        memories[memory_id] = {}

    memories[memory_id][key] = value
    return (
        f"Successfully stored key '{key}' with value '{value}' "
        f"in memory space '{memory_id}'"
    )


@mcp.tool()
def retrieve(key: str, ctx: Context) -> str:
    """Retrieve a value by key from memory.

    Args:
        key: The key to retrieve
    """
    memory_id = extract_memory_id(ctx)

    # Ensure memory space exists
    if memory_id not in memories:
        memories[memory_id] = {}

    value = memories[memory_id].get(key)
    if value is None:
        return f"Key '{key}' not found in memory space '{memory_id}'"
    return f"Retrieved value for key '{key}': {value}"


@mcp.tool()
def list_keys(ctx: Context) -> str:
    """List all keys in the current memory space."""
    memory_id = extract_memory_id(ctx)

    # Ensure memory space exists
    if memory_id not in memories:
        memories[memory_id] = {}

    keys = list(memories[memory_id].keys())
    if not keys:
        return f"No keys found in memory space '{memory_id}'"
    return f"Keys in memory space '{memory_id}': {', '.join(keys)}"


@mcp.tool()
def delete(key: str, ctx: Context) -> str:
    """Delete a key from memory.

    Args:
        key: The key to delete
    """
    memory_id = extract_memory_id(ctx)

    # Ensure memory space exists
    if memory_id not in memories:
        memories[memory_id] = {}

    if key in memories[memory_id]:
        del memories[memory_id][key]
        return f"Successfully deleted key '{key}' from memory space '{memory_id}'"
    return f"Key '{key}' not found in memory space '{memory_id}'"


def start_test_server(port: int) -> subprocess.Popen:
    """Start memory MCP server for testing.

    Args:
        port: Port to run server on

    Returns:
        subprocess.Popen object for the running server
    """
    script_path = os.path.abspath(__file__)
    process = subprocess.Popen(
        [sys.executable, script_path, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready (TCP check)
    for _ in range(30):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            if result == 0:
                return process
        except Exception:
            pass
        time.sleep(0.1)

    # Failed to start
    process.kill()
    stdout, stderr = process.communicate()
    raise RuntimeError(
        f"Memory MCP server failed to start.\n"
        f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
    )


def main():
    parser = argparse.ArgumentParser(description="Memory MCP Server")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to run the server on (default: 8765)",
    )
    args = parser.parse_args()

    print(f"Starting Memory MCP Server on port {args.port}...")
    mcp.run(port=args.port, transport="sse")


if __name__ == "__main__":
    main()
