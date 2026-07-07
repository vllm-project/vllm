# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import openai.types.responses as openai_responses_types
import pytest


@pytest.mark.asyncio
async def test_simple_input(client: openai.AsyncOpenAI):
    response = await client.responses.create(input="What is 13 * 24?")
    print(response)

    outputs = response.output
    # Whether the output contains the answer.
    assert outputs[-1].type == "message"
    assert "312" in outputs[-1].content[0].text

    # Whether the output contains the reasoning.
    assert outputs[0].type == "reasoning"
    assert outputs[0].content[0].text != ""


@pytest.mark.asyncio
async def test_instructions(client: openai.AsyncOpenAI):
    response = await client.responses.create(
        instructions="Finish the answer with QED.",
        input="What is 13 * 24?",
    )
    print(response)

    output_text = response.output[-1].content[0].text
    assert "312" in output_text
    assert "QED" in output_text


@pytest.mark.asyncio
async def test_chat(client: openai.AsyncOpenAI):
    response = await client.responses.create(
        input=[
            {"role": "system", "content": "Finish the answer with QED."},
            {"role": "user", "content": "What is 5 * 3?"},
            {"role": "assistant", "content": "15. QED."},
            {"role": "user", "content": "Multiply the result by 2."},
        ],
    )
    print(response)

    output_text = response.output[-1].content[0].text
    assert "30" in output_text
    assert "QED" in output_text


@pytest.mark.asyncio
async def test_chat_with_input_type(client: openai.AsyncOpenAI):
    response = await client.responses.create(
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello!"}],
            },
        ],
    )
    print(response)
    assert response.status == "completed"


@pytest.mark.asyncio
async def test_logprobs(client: openai.AsyncOpenAI):
    response = await client.responses.create(
        include=["message.output_text.logprobs"],
        input="What is 13 * 24?",
        top_logprobs=5,
    )
    print(response)
    outputs = response.output
    assert outputs[-1].content[-1].logprobs
    assert len(outputs[-1].content[-1].logprobs[0].top_logprobs) == 5


@pytest.mark.asyncio
async def test_streaming(client: openai.AsyncOpenAI):
    stream = await client.responses.create(
        input="What is 13 * 24?",
        stream=True,
    )
    events = [event async for event in stream]
    assert isinstance(events[0], openai_responses_types.ResponseCreatedEvent)
    assert any(
        isinstance(event, openai_responses_types.ResponseTextDeltaEvent)
        for event in events
    )
    assert isinstance(events[-1], openai_responses_types.ResponseCompletedEvent)
