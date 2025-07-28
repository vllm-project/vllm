# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

# Use a small reasoning model to test the responses API.
MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def default_server_args():
    return [
        "--max-model-len",
        "8192",
        "--enforce-eager",  # For faster startup.
        "--reasoning-parser",
        "deepseek_r1",
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client
