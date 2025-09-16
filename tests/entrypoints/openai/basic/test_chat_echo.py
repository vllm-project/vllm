# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", [True, False])
async def test_chat_session_with_echo_and_continue_final_message(
        client: openai.AsyncOpenAI, echo: bool, model_name):
    saying: str = "Here is a common saying about apple. An apple a day, keeps"
    # test echo with continue_final_message parameter
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": "tell me a common saying"
        }, {
            "role": "assistant",
            "content": saying
        }],
        extra_body={
            "echo": echo,
            "continue_final_message": True,
            "add_generation_prompt": False
        })
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason in ["stop", "length"]

    message = choice.message
    if echo:
        assert message.content is not None and saying in message.content
    else:
        assert message.content is not None and saying not in message.content
    assert message.role == "assistant"
