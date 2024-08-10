from typing import Dict, List, TypedDict

import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam

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

MESSAGES_WITHOUT_TOOLS: List[ChatCompletionMessageParam] = [{
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
    "Can you write a simple 'hello world' program in python?"
}]

WEATHER_TOOL = {
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

SEARCH_TOOL = {
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

configKeys = CONFIGS.keys()


@pytest.fixture(scope="module", params=configKeys)
def client_config(request):
    print('param', request.param)
    server_config: ServerConfig = CONFIGS["hermes"]
    model = server_config["model"]
    args_for_model = server_config["arguments"]
    with RemoteOpenAIServer(model, ARGS + args_for_model) as server:
        client = server.get_async_client()
        yield TestConfig(client=client, model=model)


@pytest.mark.asyncio
async def test_get_models(client_config: TestConfig):
    client = client_config["client"]
    model = client_config["model"]
    print('Running test_get_models for ', model)
    assert client is not None
    assert isinstance(client, openai.AsyncOpenAI)

    models = await client.models.list()
    assert len(models.data) == 1


# test: make sure chat completions without tools provided work even when tools
# are enabled. This makes sure tool call chat templates work, AND that the tool
# parser stream processing doesn't change the output of the model.
@pytest.mark.asyncio
async def test_chat_completion_without_tools(client_config: TestConfig):
    chat_completion = await client_config["client"].chat.completions.create(
        messages=MESSAGES_WITHOUT_TOOLS,
        temperature=0,
        max_tokens=16,
        model=client_config["model"],
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
    stream = await client_config["client"].chat.completions.create(
        messages=MESSAGES_WITHOUT_TOOLS,
        temperature=0,
        max_tokens=16,
        model=client_config["model"],
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
    assert len(chunks)
    assert "".join(chunks) == output_text


# test: conversation with tools enabled and provided that should not invoke
# tools, to make sure we can still get normal chat completion responses
# and that they won't be parsed as tools

# test: request a chat completion that should return tool calls, so we know they
# are parsable

# test: providing tools and results back to model to get a non-tool response
# (streaming/not)

# test: getting the model to generate parallel tool calls (streaming/not)

# test: providing parallel tool calls back to the model to get a response
# (streaming/not)
