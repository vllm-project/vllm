import json
from typing import Dict, List, Optional

import openai
import pytest
from openai.types.chat import (ChatCompletionMessageParam,
                               ChatCompletionToolParam)
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import TypedDict

from ...utils import VLLM_PATH, RemoteOpenAIServer


class ServerConfig(TypedDict):
    model: str
    arguments: List[str]


class TestConfig(TypedDict):
    client: openai.AsyncOpenAI
    model: str


ARGS: List[str] = [
    "--dtype",
    "half",  # TODO change to BF16
    "--kv-cache-dtype",
    "fp8",
    "--enable-auto-tool-choice"
]

CONFIGS: Dict[str, ServerConfig] = {
    "hermes": {
        "model":
            "NousResearch/Hermes-2-Pro-Llama-3-8B",
        "arguments": [
            "--tool-call-parser", "hermes", "--chat-template",
            str(VLLM_PATH / "examples/tool_chat_template_hermes.jinja")
        ]
    },
    "mistral": {
        "model":
            "mistralai/Mistral-7B-Instruct-v0.3",
        "arguments": [
            "--tool-call-parser", "mistral", "--chat-template",
            str(VLLM_PATH / "examples/tool_chat_template_mistral.jinja")
        ]
    }
}

WEATHER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type":
                        "string",
                    "description":
                        "The city to find the weather for, "
                        "e.g. 'San Francisco'"
                },
                "state": {
                    "type":
                        "string",
                    "description":
                        "the two-letter abbreviation for the state "
                        "that the city is in, e.g. 'CA' which would "
                        "mean 'California'"
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"]
                }
            }
        }
    }
}

SEARCH_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name":
            "web_search",
        "description":
            "Search the internet and get a summary of the top "
            "10 webpages. Should only be used if you don't know "
            "the answer to a user query, and the results are likely"
            "to be able to be found with a web search",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type":
                        "string",
                    "description":
                        "The term to use in the search. This should"
                        "ideally be keywords to search for, not a"
                        "natural-language question"
                }
            },
            "required": ["search_term"]
        }
    }
}

MESSAGES_WITHOUT_TOOLS: List[ChatCompletionMessageParam] = [{
    "role":
        "system",
    "content":
        "You are a helpful assistant with access to tools. If a tool"
        " that you have would be helpful to answer a user query, "
        "call the tool. Otherwise, answer the user's query directly "
        "without calling a tool. DO NOT CALL A TOOL THAT IS IRRELEVANT "
        "to the user's question - just respond to it normally."
}, {
    "role":
        "user",
    "content":
        "Hi! How are you?"
}, {
    "role":
        "assistant",
    "content":
        "I'm doing great! How can I assist you?"
}, {
    "role":
        "user",
    "content":
        "Can you tell me a joke please?"
}]

MESSAGES_ASKING_FOR_TOOLS: List[ChatCompletionMessageParam] = [{
    "role":
        "user",
    "content":
        "What is the weather in Dallas, Texas in Fahrenheit?"
}]

MESSAGES_WITH_TOOL_RESPONSE: List[ChatCompletionMessageParam] = [{
        "role": "user",
        "content": "What is the weather in Dallas, Texas in Fahrenheit?"
    },{
        "role": "assistant",
        "tool_calls": [{
            "id": "chatcmpl-tool-03e6481b146e408e9523d9c956696295",
            "type": "function",
            "function": {
                "name": WEATHER_TOOL["function"]["name"],
                "arguments": '{"city": "Dallas", "state": "TX", '
                             '"unit": "fahrenheit"}'
            }
        }]
    },{
        "role": "tool",
        "tool_call_id": "chatcmpl-tool-03e6481b146e408e9523d9c956696295",
        "content": "The weather in Dallas is 98 degrees fahrenheit, with partly"
                   "cloudy skies and a low chance of rain."
    }]


# for each server config, download the model and return the config
@pytest.fixture(scope="module", params=CONFIGS.keys())
def server_config(request):
    config = CONFIGS[request.param]

    print(f'downloading model for {config["model"]}')

    # download model and tokenizer using transformers
    AutoTokenizer.from_pretrained(config["model"])
    AutoModelForCausalLM.from_pretrained(config["model"])
    yield CONFIGS[request.param]


# run this for each server config
@pytest.fixture(scope="module")
def server(request, server_config: ServerConfig):
    model = server_config["model"]
    args_for_model = server_config["arguments"]
    with RemoteOpenAIServer(model, ARGS + args_for_model) as server:
        yield server

@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_async_client()


# test: make sure chat completions without tools provided work even when tools
# are enabled. This makes sure tool call chat templates work, AND that the tool
# parser stream processing doesn't change the output of the model.
@pytest.mark.asyncio
async def test_chat_completion_without_tools(client: openai.AsyncOpenAI):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(
        messages=MESSAGES_WITHOUT_TOOLS,
        temperature=0,
        max_tokens=128,
        model=model_name,
        logprobs=False)
    choice = chat_completion.choices[0]
    stop_reason = chat_completion.choices[0].finish_reason
    output_text = chat_completion.choices[0].message.content

    # check to make sure we got text
    assert output_text is not None
    assert len(output_text) > 0

    # check to make sure no tool calls were returned
    assert (choice.message.tool_calls is None
            or len(choice.message.tool_calls) == 0)

    # make the same request, streaming
    stream = await client.chat.completions.create(
        messages=MESSAGES_WITHOUT_TOOLS,
        temperature=0,
        max_tokens=128,
        model=model_name,
        logprobs=False,
        stream=True,
    )
    chunks: List[str] = []
    finish_reason_count = 0
    role_sent: bool = False

    # assemble streamed chunks
    async for chunk in stream:
        delta = chunk.choices[0].delta

        # make sure the role is assistant
        if delta.role:
            assert not role_sent
            assert delta.role == 'assistant'
            role_sent = True

        if delta.content:
            chunks.append(delta.content)

        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
            assert chunk.choices[0].finish_reason == choice.finish_reason

        # make sure tool call chunks aren't being streamed
        assert not delta.tool_calls or len(delta.tool_calls) == 0

    # make sure the role was sent, only 1 finish reason was sent, that chunks
    # were in fact sent, and that the chunks match non-streaming
    assert role_sent
    assert finish_reason_count == 1
    assert len(chunks)
    assert "".join(chunks) == output_text


# test: conversation with tools enabled and provided that should not invoke
# tools, to make sure we can still get normal chat completion responses
# and that they won't be parsed as tools
@pytest.mark.asyncio
async def test_chat_completion_with_tools(client: openai.AsyncOpenAI):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(
        messages=MESSAGES_WITHOUT_TOOLS,
        temperature=0,
        max_tokens=128,
        model=model_name,
        tools=[WEATHER_TOOL],
        logprobs=False)
    choice = chat_completion.choices[0]
    stop_reason = chat_completion.choices[0].finish_reason
    output_text = chat_completion.choices[0].message.content

    # check to make sure we got text
    assert output_text is not None
    assert stop_reason != 'tool_calls'
    assert len(output_text) > 0

    # check to make sure no tool calls were returned
    assert (choice.message.tool_calls is None
            or len(choice.message.tool_calls) == 0)

    # make the same request, streaming
    stream = await client.chat.completions.create(
        messages=MESSAGES_WITHOUT_TOOLS,
        temperature=0,
        max_tokens=128,
        model=model_name,
        logprobs=False,
        tools=[WEATHER_TOOL],
        stream=True,
    )

    chunks: List[str] = []
    finish_reason_count = 0
    role_sent: bool = False

    # assemble streamed chunks
    async for chunk in stream:
        delta = chunk.choices[0].delta

        # make sure the role is assistant
        if delta.role:
            assert delta.role == 'assistant'
            role_sent = True

        if delta.content:
            chunks.append(delta.content)

        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1

        # make sure tool call chunks aren't being streamed
        assert not delta.tool_calls or len(delta.tool_calls) == 0

    # make sure the role was sent, only 1 finish reason was sent, that chunks
    # were in fact sent, and that the chunks match non-streaming
    assert role_sent
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == stop_reason
    assert chunk.choices[0].finish_reason != 'tool_calls'
    assert len(chunks)
    assert "".join(chunks) == output_text


# test: request a chat completion that should return tool calls, so we know they
# are parsable
@pytest.mark.asyncio
async def test_tool_call_and_choice(client: openai.AsyncOpenAI):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(
        messages=MESSAGES_ASKING_FOR_TOOLS,
        temperature=0,
        max_tokens=500,
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
        max_tokens=500,
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
        max_tokens=500,
        model=model_name,
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        logprobs=False
    )

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
        max_tokens=500,
        model=model_name,
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        logprobs=False,
        stream=True
    )

    chunks: List[str] = []
    finish_reason: Optional[str] == None
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


# test: getting the model to generate parallel tool calls (streaming/not)

# test: providing parallel tool calls back to the model to get a response
# (streaming/not)
