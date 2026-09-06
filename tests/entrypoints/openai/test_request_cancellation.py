# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cancel a live request by the id the client was actually given.

The serving layer fans a single client request out to one engine request per
prompt, under derived ids such as ``cmpl-abc-0``. These tests run against a
real server so that they catch a regression in that resolution, which a mocked
engine cannot.
"""

import openai
import pytest
import pytest_asyncio
import requests

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "facebook/opt-125m"


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "float32",
        "--max-model-len",
        "2048",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def _cancel(server: RemoteOpenAIServer, request_id: str) -> requests.Response:
    return requests.post(server.url_for("v1", "requests", request_id, "cancel"))


@pytest.mark.asyncio
async def test_cancel_completion_by_response_id(
    server: RemoteOpenAIServer, client: openai.AsyncOpenAI
):
    """Completions register ``{id}-{i}``, one per prompt."""
    stream = await client.completions.create(
        model=MODEL_NAME,
        prompt=["Hello, my name is", "The capital of France is"],
        max_tokens=2048,
        temperature=0.0,
        stream=True,
    )

    first_chunk = await stream.__anext__()
    request_id = first_chunk.id

    assert _cancel(server, request_id).status_code == 200

    # The stream ends rather than running to max_tokens.
    async for _ in stream:
        pass

    # A second cancel finds nothing left to abort.
    assert _cancel(server, request_id).status_code == 404


@pytest.mark.asyncio
async def test_cancel_chat_completion_by_response_id(
    server: RemoteOpenAIServer, client: openai.AsyncOpenAI
):
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Hello, my name is"}],
        max_tokens=2048,
        temperature=0.0,
        stream=True,
    )

    first_chunk = await stream.__anext__()
    request_id = first_chunk.id

    assert _cancel(server, request_id).status_code == 200

    async for _ in stream:
        pass

    assert _cancel(server, request_id).status_code == 404


@pytest.mark.asyncio
async def test_cancel_unknown_request_id(server: RemoteOpenAIServer):
    assert _cancel(server, "cmpl-does-not-exist").status_code == 404
