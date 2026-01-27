# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio

import openai
import pytest


@pytest.mark.asyncio
async def test_store(client: openai.AsyncOpenAI):
    # By default, store is True.
    response = await client.responses.create(input="Hello!")
    assert response.status == "completed"

    # Retrieve the response.
    response = await client.responses.retrieve(response.id)
    assert response.status == "completed"

    # Test store=False.
    response = await client.responses.create(
        input="Hello!",
        store=False,
    )
    assert response.status == "completed"

    # The response should not be found.
    with pytest.raises(openai.NotFoundError, match="Response with id .* not found."):
        await client.responses.retrieve(response.id)


@pytest.mark.asyncio
async def test_background(client: openai.AsyncOpenAI):
    # NOTE: This query should be easy enough for the model to answer
    # within the 10 seconds.
    response = await client.responses.create(
        input="Hello!",
        background=True,
    )
    assert response.status == "queued"

    max_retries = 10
    for _ in range(max_retries):
        await asyncio.sleep(1)
        response = await client.responses.retrieve(response.id)
        if response.status != "queued":
            break
    print(response)

    assert response.status == "completed"


@pytest.mark.asyncio
async def test_background_error(client: openai.AsyncOpenAI):
    with pytest.raises(
        openai.BadRequestError, match="background can only be used when `store` is true"
    ):
        _ = await client.responses.create(
            input="What is 13 * 24?",
            background=True,
            store=False,
        )


@pytest.mark.asyncio
async def test_background_cancel(client: openai.AsyncOpenAI):
    response = await client.responses.create(
        input="Write a long story about a cat.",
        background=True,
    )
    assert response.status == "queued"

    # Cancel the response before it is completed.
    # Poll until the response is no longer queued (started processing) or timeout
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    max_wait_seconds = 5.0
    poll_interval = 0.1
    while loop.time() - start_time < max_wait_seconds:
        response = await client.responses.retrieve(response.id)
        if response.status != "queued":
            # Started processing or completed - try to cancel
            break
        await asyncio.sleep(poll_interval)

    response = await client.responses.cancel(response.id)
    assert response.status == "cancelled"

    # Make sure the response status remains unchanged after some time.
    max_retries = 10
    for _ in range(max_retries):
        await asyncio.sleep(0.5)
        response = await client.responses.retrieve(response.id)
        # Verify status is still cancelled
        assert response.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_completed(client: openai.AsyncOpenAI):
    response = await client.responses.create(input="Hello")
    assert response.status == "completed"

    with pytest.raises(
        openai.BadRequestError, match="Cannot cancel a synchronous response."
    ):
        await client.responses.cancel(response.id)


@pytest.mark.asyncio
async def test_previous_response_id(client: openai.AsyncOpenAI):
    response1 = await client.responses.create(
        instructions="You are tested on your ability to retrieve the correct "
        "information from the previous response.",
        input="Hello, my name is John.",
    )

    response2 = await client.responses.create(
        input="Actually, my name is not John. My real name is Mark.",
        previous_response_id=response1.id,
    )

    response3 = await client.responses.create(
        input="What is my real name again? Answer in one word.",
        previous_response_id=response2.id,
    )
    print(response3)
    assert "Mark" in response3.output[-1].content[0].text
    assert "John" not in response3.output[-1].content[0].text


@pytest.mark.asyncio
async def test_two_responses_with_same_prev_id(client: openai.AsyncOpenAI):
    response1 = await client.responses.create(
        instructions="You are tested on your ability to retrieve the correct "
        "information from the previous response.",
        input="Hello, my name is John.",
    )

    # Both response 2 and 3 use response 1 as the previous response.
    response2 = client.responses.create(
        input="Actually, my name is not John. My name is Mark.",
        previous_response_id=response1.id,
    )
    response3 = client.responses.create(
        input="What is my name again? Answer in one word.",
        previous_response_id=response1.id,
    )

    _ = await response2
    response3_result = await response3
    print(response3_result)
    assert "John" in response3_result.output[-1].content[0].text
    assert "Mark" not in response3_result.output[-1].content[0].text
