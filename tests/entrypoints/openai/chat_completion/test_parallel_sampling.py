# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai
import pytest
import pytest_asyncio
from openai import BadRequestError

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-num-seqs",
        "8",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.3",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(
        MODEL_NAME, args, env_dict={"VLLM_MAX_N_SEQUENCES": "16384"}
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_chat_completions_reject_n_above_server_capacity(
    client: openai.AsyncOpenAI,
):
    with pytest.raises(BadRequestError, match="max_num_seqs"):
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello!"}],
            max_completion_tokens=1,
            n=9,
        )


@pytest.mark.asyncio
async def test_completions_reject_n_above_server_capacity(
    client: openai.AsyncOpenAI,
):
    with pytest.raises(BadRequestError, match="max_num_seqs"):
        await client.completions.create(
            model=MODEL_NAME,
            prompt="Hello!",
            max_tokens=1,
            n=9,
        )
