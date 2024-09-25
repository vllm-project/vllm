import json
from typing import Dict, List, Optional

import openai
import pytest

from .utils import (MESSAGES_ASKING_FOR_TOOLS, MESSAGES_WITH_TOOL_RESPONSE,
                    SEARCH_TOOL, WEATHER_TOOL)


# test: request a chat completion that should return tool calls, so we know they
# are parsable
@pytest.mark.asyncio
async def test_tool_call_and_choice(client: openai.AsyncOpenAI):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(
        messages=MESSAGES_ASKING_FOR_TOOLS,
        temperature=0,
        max_tokens=100,
        model=model_name,
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        logprobs=False)

    choice = chat_completion.choices[0]
    stop_reason = chat_completion.choices[0].finish_reason
    tool_calls = chat_completion.choices[0].message.tool_calls

    # make sure a tool call is present
    assert choice.message.role == 'assistant'
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].type == 'function'
    assert tool_calls[0].function is not None
    assert isinstance(tool_calls[0].id, str)
    assert len(tool_calls[0].id) > 16

    # make sure the weather tool was called (classic example) with arguments
    assert tool_calls[0].function.name == WEATHER_TOOL["function"]["name"]
    assert tool_calls[0].function.arguments is not None
    assert isinstance(tool_calls[0].function.arguments, str)

    # make sure the arguments parse properly
    parsed_arguments = json.loads(tool_calls[0].function.arguments)
    assert isinstance(parsed_arguments, Dict)
    assert isinstance(parsed_arguments.get("city"), str)
    assert isinstance(parsed_arguments.get("state"), str)
    assert parsed_arguments.get("city") == "Dallas"
    assert parsed_arguments.get("state") == "TX"

    assert stop_reason == "tool_calls"

    function_name: Optional[str] = None
    function_args_str: str = ''
    tool_call_id: Optional[str] = None
    role_name: Optional[str] = None
    finish_reason_count: int = 0

    # make the same request, streaming
    stream = await client.chat.completions.create(
        model=model_name,
        messages=MESSAGES_ASKING_FOR_TOOLS,
        temperature=0,
        max_tokens=100,
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        logprobs=False,
        stream=True)

    async for chunk in stream:
        assert chunk.choices[0].index == 0

        if chunk.choices[0].finish_reason:
            finish_reason_count += 1
            assert chunk.choices[0].finish_reason == 'tool_calls'

        # if a role is being streamed make sure it wasn't already set to
        # something else
        if chunk.choices[0].delta.role:
            assert not role_name or role_name == 'assistant'
            role_name = 'assistant'

        # if a tool call is streamed make sure there's exactly one
        # (based on the request parameters
        streamed_tool_calls = chunk.choices[0].delta.tool_calls

        if streamed_tool_calls and len(streamed_tool_calls) > 0:
            assert len(streamed_tool_calls) == 1
            tool_call = streamed_tool_calls[0]

            # if a tool call ID is streamed, make sure one hasn't been already
            if tool_call.id:
                assert not tool_call_id
                tool_call_id = tool_call.id

            # if parts of the function start being streamed
            if tool_call.function:
                # if the function name is defined, set it. it should be streamed
                # IN ENTIRETY, exactly one time.
                if tool_call.function.name:
                    assert function_name is None
                    assert isinstance(tool_call.function.name, str)
                    function_name = tool_call.function.name
                if tool_call.function.arguments:
                    assert isinstance(tool_call.function.arguments, str)
                    function_args_str += tool_call.function.arguments

    assert finish_reason_count == 1
    assert role_name == 'assistant'
    assert isinstance(tool_call_id, str) and (len(tool_call_id) > 16)

    # validate the name and arguments
    assert function_name == WEATHER_TOOL["function"]["name"]
    assert function_name == tool_calls[0].function.name
    assert isinstance(function_args_str, str)

    # validate arguments
    streamed_args = json.loads(function_args_str)
    assert isinstance(streamed_args, Dict)
    assert isinstance(streamed_args.get("city"), str)
    assert isinstance(streamed_args.get("state"), str)
    assert streamed_args.get("city") == "Dallas"
    assert streamed_args.get("state") == "TX"

    # make sure everything matches non-streaming except for ID
    assert function_name == tool_calls[0].function.name
    assert choice.message.role == role_name
    assert choice.message.tool_calls[0].function.name == function_name

    # compare streamed with non-streamed args Dict-wise, not string-wise
    # because character-to-character comparison might not work e.g. the tool
    # call parser adding extra spaces or something like that. we care about the
    # dicts matching not byte-wise match
    assert parsed_arguments == streamed_args


# test: providing tools and results back to model to get a non-tool response
# (streaming/not)
@pytest.mark.asyncio
async def test_tool_call_with_results(client: openai.AsyncOpenAI):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(
        messages=MESSAGES_WITH_TOOL_RESPONSE,
        temperature=0,
        max_tokens=100,
        model=model_name,
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        logprobs=False)

    choice = chat_completion.choices[0]

    assert choice.finish_reason != "tool_calls"  # "stop" or "length"
    assert choice.message.role == "assistant"
    assert choice.message.tool_calls is None \
           or len(choice.message.tool_calls) == 0
    assert choice.message.content is not None
    assert "98" in choice.message.content  # the temperature from the response

    stream = await client.chat.completions.create(
        messages=MESSAGES_WITH_TOOL_RESPONSE,
        temperature=0,
        max_tokens=100,
        model=model_name,
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        logprobs=False,
        stream=True)

    chunks: List[str] = []
    finish_reason_count = 0
    role_sent: bool = False

    async for chunk in stream:
        delta = chunk.choices[0].delta

        if delta.role:
            assert not role_sent
            assert delta.role == "assistant"
            role_sent = True

        if delta.content:
            chunks.append(delta.content)

        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
            assert chunk.choices[0].finish_reason == choice.finish_reason

        assert not delta.tool_calls or len(delta.tool_calls) == 0

    assert role_sent
    assert finish_reason_count == 1
    assert len(chunks)
    assert "".join(chunks) == choice.message.content
