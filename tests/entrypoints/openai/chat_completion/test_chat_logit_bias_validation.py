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
async def test_chat_logit_bias_valid(client):
    """Test that valid logit_bias values are accepted in chat completions."""
    vocab_size = get_vocab_size(MODEL_NAME)
    valid_token_id = vocab_size - 1

    completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Testing valid logit bias"}],
        max_tokens=5,
        logit_bias={str(valid_token_id): 1.0},
    )

    assert completion.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_chat_logit_bias_invalid(client):
    """Test that invalid logit_bias values are rejected in chat completions."""
    vocab_size = get_vocab_size(MODEL_NAME)
    invalid_token_id = vocab_size + 1

    with pytest.raises(openai.BadRequestError) as excinfo:
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Testing invalid logit bias"}],
            max_tokens=5,
            logit_bias={str(invalid_token_id): 1.0},
        )

    error = excinfo.value
    error_message = str(error)

    assert error.status_code == 400
    assert str(invalid_token_id) in error_message
    assert str(vocab_size) in error_message


@pytest.mark.asyncio
async def test_chat_logit_bias_non_integer_key(client):
    """Test that a non-integer logit_bias key is rejected with a clean,
    informative error instead of a raw 'invalid literal for int()' message."""
    with pytest.raises(openai.BadRequestError) as excinfo:
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Testing invalid logit bias key"}],
            max_tokens=5,
            logit_bias={"not_a_token_id": 50},
        )

    error = excinfo.value
    error_message = str(error)

    assert error.status_code == 400
    assert "not_a_token_id" in error_message
    assert "logit_bias" in error_message


@pytest.mark.asyncio
async def test_chat_logit_bias_non_numeric_value(client):
    """Test that a non-numeric logit_bias value is rejected with a message
    that names the specific offending token, not just a generic TypeError."""
    with pytest.raises(openai.BadRequestError) as excinfo:
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Testing invalid logit bias value"}],
            max_tokens=5,
            logit_bias={"1": "not_a_number"},
        )

    error = excinfo.value
    error_message = str(error)

    assert error.status_code == 400
    assert "logit_bias" in error_message


@pytest.mark.asyncio
async def test_chat_logit_bias_multiple_non_integer_keys(client):
    """Test that ALL invalid logit_bias keys are reported together,
    not just the first one encountered."""
    with pytest.raises(openai.BadRequestError) as excinfo:
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Testing multiple bad keys"}],
            max_tokens=5,
            logit_bias={"bad1": 50.0, "bad2": 20.0},
        )

    error_message = str(excinfo.value)
    assert excinfo.value.status_code == 400
    assert "bad1" in error_message
    assert "bad2" in error_message
