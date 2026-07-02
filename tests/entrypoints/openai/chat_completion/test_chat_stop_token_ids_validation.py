# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from vllm.config import ModelConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def get_vocab_size(model_name):
    config = ModelConfig(
        model=model_name,
        seed=0,
        dtype="bfloat16",
    )
    return config.get_vocab_size()


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--enforce-eager",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_chat_stop_token_ids_valid(client):
    """Test that valid stop_token_ids are accepted in chat completions."""
    vocab_size = get_vocab_size(MODEL_NAME)
    valid_token_id = vocab_size - 1

    completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Testing valid stop token ids"}],
        max_tokens=5,
        extra_body={"stop_token_ids": [valid_token_id]},
    )

    assert completion.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_chat_stop_token_ids_invalid(client):
    """Test that out-of-vocab stop_token_ids are rejected."""
    vocab_size = get_vocab_size(MODEL_NAME)
    invalid_token_id = vocab_size + 1

    with pytest.raises(openai.BadRequestError) as excinfo:
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Testing invalid stop token ids"}],
            max_tokens=5,
            extra_body={"stop_token_ids": [invalid_token_id]},
        )

    error = excinfo.value
    error_message = str(error)

    assert error.status_code == 400
    assert str(invalid_token_id) in error_message
    assert str(vocab_size) in error_message
