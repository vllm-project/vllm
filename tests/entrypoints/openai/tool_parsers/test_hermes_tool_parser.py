# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

from ....utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LORA_MODEL = "minpeter/LoRA-Llama-3.2-1B-tool-vllm-ci"

SERVER_ARGS = [
    "--enforce-eager",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
    "--enable-lora",
    "--lora-modules",
    f"{LORA_MODEL}={LORA_MODEL}",
    "--tokenizer",
    f"{LORA_MODEL}",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]

PRODUCT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Get detailed information of a product based on its "
            "product ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "inserted": {
                        "type": "boolean",
                        "description": "inserted.",
                    },
                    "product_id": {
                        "type": "integer",
                        "description": "The product ID of the product.",
                    },
                },
                "required": ["product_id", "inserted"],
            },
        },
    }
]

MESSAGES = [{"role": "user", "content": "What's the weather like in Boston?"}]

PRODUCT_MESSAGES = [
    {
        "role": "user",
        "content": "Hi! Do you have any detailed information about the product id "
        "7355608 and inserted true?",
    }
]


@pytest.mark.asyncio
async def test_non_streaming_tool_call():
    """Test tool call in non-streaming mode."""
    with RemoteOpenAIServer(MODEL_NAME, SERVER_ARGS) as server:
        client = server.get_async_client()

        response = await client.chat.completions.create(
            model=LORA_MODEL,
            messages=MESSAGES,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
        )

        assert response.choices
        choice = response.choices[0]
        message = choice.message

        assert choice.finish_reason == "tool_calls"
        assert message.tool_calls is not None

        tool_call = message.tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_current_weather"

        arguments = json.loads(tool_call.function.arguments)
        assert "location" in arguments
        assert "Boston" in arguments["location"]
        print("\n[Non-Streaming Test Passed]")
        print(f"Tool Call: {tool_call.function.name}")
        print(f"Arguments: {arguments}")


@pytest.mark.asyncio
async def test_streaming_tool_call():
    """Test tool call in streaming mode."""
    with RemoteOpenAIServer(MODEL_NAME, SERVER_ARGS) as server:
        client = server.get_async_client()

        stream = await client.chat.completions.create(
            model=LORA_MODEL,
            messages=MESSAGES,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
            stream=True,
        )

        tool_call_chunks = {}
        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if not delta or not delta.tool_calls:
                continue

            for tool_chunk in delta.tool_calls:
                index = tool_chunk.index
                if index not in tool_call_chunks:
                    tool_call_chunks[index] = {"name": "", "arguments": ""}

                if tool_chunk.function.name:
                    tool_call_chunks[index]["name"] += tool_chunk.function.name
                if tool_chunk.function.arguments:
                    tool_call_chunks[index]["arguments"] += (
                        tool_chunk.function.arguments
                    )

        assert len(tool_call_chunks) == 1
        reconstructed_tool_call = tool_call_chunks[0]

        assert reconstructed_tool_call["name"] == "get_current_weather"

        arguments = json.loads(reconstructed_tool_call["arguments"])
        assert "location" in arguments
        assert "Boston" in arguments["location"]
        print("\n[Streaming Test Passed]")
        print(f"Reconstructed Tool Call: {reconstructed_tool_call['name']}")
        print(f"Reconstructed Arguments: {arguments}")


@pytest.mark.asyncio
async def test_non_streaming_product_tool_call():
    """Test tool call integer and boolean parameters in non-streaming mode."""
    with RemoteOpenAIServer(MODEL_NAME, SERVER_ARGS) as server:
        client = server.get_async_client()

        response = await client.chat.completions.create(
            model=LORA_MODEL,
            messages=PRODUCT_MESSAGES,
            tools=PRODUCT_TOOLS,
            tool_choice="auto",
            temperature=0.66,
        )

        assert response.choices
        choice = response.choices[0]
        message = choice.message

        assert choice.finish_reason == "tool_calls"
        assert message.tool_calls is not None

        tool_call = message.tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_product_info"

        arguments = json.loads(tool_call.function.arguments)
        assert "product_id" in arguments
        assert "inserted" in arguments

        product_id = arguments.get("product_id")
        inserted = arguments.get("inserted")

        assert isinstance(product_id, int)
        assert product_id == 7355608
        assert isinstance(inserted, bool)
        assert inserted is True

        print("\n[Non-Streaming Product Test Passed]")
        print(f"Tool Call: {tool_call.function.name}")
        print(f"Arguments: {arguments}")


@pytest.mark.asyncio
async def test_streaming_product_tool_call():
    """Test tool call integer and boolean parameters in streaming mode."""
    with RemoteOpenAIServer(MODEL_NAME, SERVER_ARGS) as server:
        client = server.get_async_client()

        stream = await client.chat.completions.create(
            model=LORA_MODEL,
            messages=PRODUCT_MESSAGES,
            tools=PRODUCT_TOOLS,
            tool_choice="auto",
            temperature=0.66,
            stream=True,
        )

        tool_call_chunks = {}
        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if not delta or not delta.tool_calls:
                continue

            for tool_chunk in delta.tool_calls:
                index = tool_chunk.index
                if index not in tool_call_chunks:
                    tool_call_chunks[index] = {"name": "", "arguments": ""}

                if tool_chunk.function.name:
                    tool_call_chunks[index]["name"] += tool_chunk.function.name
                if tool_chunk.function.arguments:
                    tool_call_chunks[index]["arguments"] += (
                        tool_chunk.function.arguments
                    )

        assert len(tool_call_chunks) == 1
        reconstructed_tool_call = tool_call_chunks[0]

        assert reconstructed_tool_call["name"] == "get_product_info"

        arguments = json.loads(reconstructed_tool_call["arguments"])
        assert "product_id" in arguments
        assert "inserted" in arguments

        # Handle type coercion for streaming test as well
        product_id = arguments.get("product_id")
        inserted = arguments.get("inserted")

        assert isinstance(product_id, int)
        assert product_id == 7355608
        assert isinstance(inserted, bool)
        assert inserted is True

        print("\n[Streaming Product Test Passed]")
        print(f"Reconstructed Tool Call: {reconstructed_tool_call['name']}")
        print(f"Reconstructed Arguments: {arguments}")


@pytest.fixture
def qwen_tokenizer() -> TokenizerLike:
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


@pytest.fixture
def hermes_parser(qwen_tokenizer: TokenizerLike) -> Hermes2ProToolParser:
    return Hermes2ProToolParser(qwen_tokenizer)


@pytest.fixture
def any_chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        seed=42,
        model="Qwen/Qwen3-32B",
        messages=[],
    )


def test_hermes_parser_streaming_just_forward_text(
    qwen_tokenizer: TokenizerLike,
    hermes_parser: Hermes2ProToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """This is some prior text that has nothing to do with tool calling."""
    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for token in tokens:
        delta_text = qwen_tokenizer.decode([token])
        current_text = previous_text + delta_text
        delta = hermes_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        delta_messages.append(delta)

    for delta in delta_messages:
        assert delta is not None
        assert not delta.tool_calls

    print(delta_messages)
    assert "".join([delta.content for delta in delta_messages]) == text


def test_hermes_parser_streaming_failure_case_bug_19056(
    qwen_tokenizer: TokenizerLike,
    hermes_parser: Hermes2ProToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>"""
    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for token in tokens:
        text = qwen_tokenizer.decode([token])
        current_text = previous_text + text
        delta = hermes_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)

    assert delta_messages[0].tool_calls[0].function.name == "final_answer"
    tool_call_args = "".join(
        delta.tool_calls[0].function.arguments or "" for delta in delta_messages
    )
    assert tool_call_args == '{"trigger": true}'


def test_hermes_parser_streaming(
    qwen_tokenizer: TokenizerLike,
    hermes_parser: Hermes2ProToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = '<tool_call>\
{"name": "get_current_temperature",\
"arguments": {"location":\
"San Francisco, California, United States", "unit": "celsius"}}\
</tool_call>'

    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for token in tokens:
        text = qwen_tokenizer.decode([token])
        current_text = previous_text + text
        delta = hermes_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)
    print(delta_messages)
    assert delta_messages[0].tool_calls[0].function.name == "get_current_temperature"
    tool_call_args = "".join(
        delta.tool_calls[0].function.arguments or "" for delta in delta_messages
    )
    assert tool_call_args == (
        '{"location":"San Francisco, California, United States", "unit": "celsius"}'
    )


def test_hermes_parser_non_streaming_no_tool_call(
    hermes_parser: Hermes2ProToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """This is not a tool call."""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert not tool_call.tools_called


def test_hermes_parser_non_streaming_tool_call_between_tags(
    hermes_parser: Hermes2ProToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>"""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert tool_call.tool_calls[0].function.name == "final_answer"
    assert tool_call.tool_calls[0].function.arguments == '{"trigger": true}'


def test_hermes_parser_non_streaming_tool_call_until_eos(
    hermes_parser: Hermes2ProToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}"""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert tool_call.tool_calls[0].function.name == "final_answer"
    assert tool_call.tool_calls[0].function.arguments == '{"trigger": true}'


def test_hermes_parser_non_streaming_tool_call_invalid_json(
    hermes_parser: Hermes2ProToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    # Missing closing brace to trigger exception
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}"""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert not tool_call.tools_called


def test_hermes_parser_streaming_state_reset(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
) -> None:
    """Test that streaming state is properly reset between requests.

    This test verifies the fix for issue #31871 where state from a previous
    streaming session could interfere with a new session.
    """
    parser = Hermes2ProToolParser(qwen_tokenizer)

    # First streaming session
    text1 = '<tool_call>\n{"name": "func1", "arguments": {"a": 1}}\n</tool_call>'
    tokens1 = qwen_tokenizer.encode(text1)
    previous_text = ""
    for token in tokens1:
        delta_text = qwen_tokenizer.decode([token])
        current_text = previous_text + delta_text
        parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text

    # Verify state after first session
    assert parser.current_tool_id >= 0

    # Second streaming session (simulating new request)
    # When previous_text is empty, state should be reset
    text2 = '<tool_call>\n{"name": "func2", "arguments": {"b": 2}}\n</tool_call>'
    tokens2 = qwen_tokenizer.encode(text2)
    previous_text = ""  # Empty indicates new request
    delta_messages = []
    for token in tokens2:
        delta_text = qwen_tokenizer.decode([token])
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)

    # Verify second session parsed correctly (not affected by first session)
    tool_call_deltas = [d for d in delta_messages if d.tool_calls]
    assert len(tool_call_deltas) > 0
    # The function name should be from the second session
    assert tool_call_deltas[0].tool_calls[0].function.name == "func2"


def test_hermes_parser_streaming_prev_tool_call_arr_populated_early(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
) -> None:
    """Test that prev_tool_call_arr is populated early for finish_reason.

    This test verifies the fix for issue #31871 where finish_reason was
    'stop' instead of 'tool_calls' in streaming mode because prev_tool_call_arr
    was not populated early enough.
    """
    parser = Hermes2ProToolParser(qwen_tokenizer)

    text = (
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "Beijing"}}\n'
        "</tool_call>"
    )
    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    found_tool_name = False

    for token in tokens:
        delta_text = qwen_tokenizer.decode([token])
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[token],
            request=any_chat_request,
        )
        previous_text = current_text

        # Check if we just received the tool name
        if delta is not None and delta.tool_calls and delta.tool_calls[0].function.name:
            found_tool_name = True
            # CRITICAL: prev_tool_call_arr should be populated immediately
            # when tool name is detected, not later
            assert len(parser.prev_tool_call_arr) > 0, (
                "prev_tool_call_arr should be populated when tool name is sent"
            )
            assert parser.prev_tool_call_arr[0].get("name") == "get_weather"
            break

    assert found_tool_name, "Should have found tool name in streaming output"
