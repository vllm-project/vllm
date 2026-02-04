# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

# Use a small reasoning model to test the responses API.
MODEL_NAME = "/mnt/data4/models/Qwen/Qwen3-8B"


@pytest.fixture(scope="module")
def default_server_args():
    return [
        "--max-model-len",
        "8192",
        "--enforce-eager",  # For faster startup.
        "--enable-auto-tool-choice",
        "--structured-outputs-config.backend",
        "xgrammar",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "qwen3",
    ]


@pytest.fixture(scope="module")
def server_with_store(default_server_args):
    with RemoteOpenAIServer(
        MODEL_NAME,
        default_server_args,
        env_dict={
            "VLLM_ENABLE_RESPONSES_API_STORE": "1",
            "VLLM_SERVER_DEV_MODE": "1",
        },
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server_with_store):
    async with server_with_store.get_async_client() as async_client:
        yield async_client
