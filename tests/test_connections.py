# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import socket
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
from aiohttp import web

from vllm.connections import global_http_connection


@pytest_asyncio.fixture
async def httpbin_echo_server() -> AsyncGenerator[str, Any]:
    """
    A pytest fixture that creates a local aiohttp server to mimic httpbin.org/anything.
    It captures request details and returns them as JSON.
    This makes the test self-contained and not dependent on external services.

    Yields:
        str: The base URL of the local server.
    """

    async def echo_handler(request: web.Request) -> web.Response:
        # CRITICAL CHANGE: Use request.raw_path to get the undecoded path.
        # This is the only way to verify that the client sent the encoded path.
        return web.json_response({"raw_path": request.raw_path})

    app = web.Application()
    # This route captures everything after /anything/
    app.router.add_get("/anything/{tail:.*}", echo_handler)

    # Find a random available port to avoid conflicts
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    base_url = f"http://127.0.0.1:{port}"

    yield base_url

    await runner.cleanup()

@pytest.mark.asyncio
async def test_async_client_encodes_unencoded_path(httpbin_echo_server: str):
    """
    Test that the AsyncHttpClient correctly encodes special characters in the URL path.
    """
    unencoded_segment = "path with spaces"
    url = f"{httpbin_echo_server}/anything/{unencoded_segment}"

    async with await global_http_connection.get_async_response(url) as resp:
        resp.raise_for_status()
        data = await resp.json()

        expected_raw_path = "/anything/path%20with%20spaces"
        assert data["raw_path"] == expected_raw_path

@pytest.mark.asyncio
async def test_async_client_preserves_encoded_path(httpbin_echo_server: str):
    """
    Test that the AsyncHttpClient preserves encoded characters in the URL path
    by sending a request to a local mock server.
    """
    # The segment we want to test, containing an encoded slash (%2F)
    encoded_segment = "path%2Fwith%2Fencoded%2Fslash"

    # Construct the URL with the encoded segment IN THE PATH
    url = f"{httpbin_echo_server}/anything/{encoded_segment}"

    async with await global_http_connection.get_async_response(url) as resp:
        resp.raise_for_status()
        data = await resp.json()

        # The expected raw path the server should receive
        expected_raw_path = f"/anything/{encoded_segment}"

        # CRITICAL CHANGE: Assert against the 'raw_path' key from the server response.
        assert data["raw_path"] == expected_raw_path, (
            f"URL path was not preserved. Expected '{expected_raw_path}', "
            f"but got '{data.get('raw_path')}'."
        )
