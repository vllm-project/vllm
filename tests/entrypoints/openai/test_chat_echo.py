# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from vllm.config import ModelConfig

from ...utils import RemoteOpenAIServer

# # any model with a chat template should work here
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"


def get_vocab_size(model_name):
    config = ModelConfig(
        model=model_name,
        seed=0,
        dtype="float16",
    )
    return config.get_vocab_size()


@pytest.fixture(scope="module")
def server():
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--enforce-eager",
        "--max-model-len",
        "4080",
        "--max-logprobs",  # test prompt_logprobs equal to -1
        "151936",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


class TestCase(NamedTuple):
    model_name: str
    echo: bool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(model_name=MODEL_NAME, echo=True),
        TestCase(model_name=MODEL_NAME, echo=False),
    ],
)
async def test_chat_session_with_echo_and_continue_final_message(
    client: openai.AsyncOpenAI, test_case: TestCase
):
    saying: str = "Here is a common saying about apple. An apple a day, keeps"
    # test echo with continue_final_message parameter
    chat_completion = await client.chat.completions.create(
        model=test_case.model_name,
        messages=[
            {"role": "user", "content": "tell me a common saying"},
            {"role": "assistant", "content": saying},
        ],
        extra_body={
            "echo": test_case.echo,
            "continue_final_message": True,
            "add_generation_prompt": False,
        },
    )
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "stop"

    message = choice.message
    if test_case.echo:
        assert message.content is not None and saying in message.content
    else:
        assert message.content is not None and saying not in message.content
    assert message.role == "assistant"


@pytest.mark.asyncio
async def test_prompt_logprobs(client: openai.AsyncOpenAI):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Beijing is the capital of which country?"},
    ]

    completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_body={"prompt_logprobs": -1},
    )

    assert completion.prompt_logprobs is not None
    assert len(completion.prompt_logprobs) > 0


@pytest.mark.asyncio
async def test_top_logprobs(client: openai.AsyncOpenAI):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Beijing is the capital of which country?"},
    ]

    completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1,
        extra_body={
            "top_logprobs": -1,
            "logprobs": "true",
        },
    )
    assert completion.choices[0].logprobs is not None
    assert completion.choices[0].logprobs.content is not None
    assert len(completion.choices[0].logprobs.content) > 0
    assert len(
        completion.choices[0].logprobs.content[0].top_logprobs
    ) == get_vocab_size(MODEL_NAME)
