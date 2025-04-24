# SPDX-License-Identifier: Apache-2.0

import openai
import pytest

from tests.tool_use.utils import MESSAGES_ASKING_FOR_TOOLS, WEATHER_TOOL


# test: a tool_choice with mistral-tokenizer results in an ID of length 9
@pytest.mark.asyncio
async def test_tool_call_with_tool_choice(client: openai.AsyncOpenAI):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(
        messages=MESSAGES_ASKING_FOR_TOOLS,
        temperature=0,
        max_completion_tokens=100,
        model=model_name,
        tools=[WEATHER_TOOL],
        tool_choice=WEATHER_TOOL,
        logprobs=False)

    choice = chat_completion.choices[0]

    assert choice.finish_reason != "tool_calls"  # "stop" or "length"
    assert choice.message.role == "assistant"
    assert choice.message.tool_calls is None \
           or len(choice.message.tool_calls) == 1
    assert len(choice.message.tool_calls[0].id) == 9  # length of 9 for mistral
