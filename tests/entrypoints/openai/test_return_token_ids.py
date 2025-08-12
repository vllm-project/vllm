# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.tokenizer import get_tokenizer

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--enforce-eager",
        # For debugging
        "--gpu-memory-utilization",
        "0.7",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_basic_completion_with_emoji(server):
    """Test basic completion with emoji to verify token_ids field."""
    async with server.get_async_client() as client:
        # Test with return_token_ids enabled
        completion = await client.completions.create(
            model=MODEL_NAME,
            prompt="Complete this sentence with emojis: I love coding 🚀",
            max_tokens=10,
            temperature=0,
            logprobs=1,
            extra_body={"return_token_ids": True},
        )

        # Check the raw response to see the structure
        completion_dict = completion.model_dump()

        # Verify prompt_token_ids field is present in the completion response
        assert "prompt_token_ids" in completion_dict["choices"][0]
        assert isinstance(completion.choices[0].prompt_token_ids, list)

        # Check against the expected prompt token IDs
        tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
        encoded_tokens = tokenizer.encode(
            "Complete this sentence with emojis: I love coding 🚀")
        # Check that encoded_tokens is a subsequence of prompt_token_ids
        assert any(completion.choices[0].prompt_token_ids[i:i +
                                                          len(encoded_tokens)]
                   == encoded_tokens for i in range(
                       len(completion.choices[0].prompt_token_ids) -
                       len(encoded_tokens) + 1))

        # Verify token_ids field is present in the choice
        assert completion.choices[0].token_ids is not None
        assert isinstance(completion.choices[0].token_ids, list)
        assert len(completion.choices[0].token_ids) > 0

        # Verify decoding works correctly
        decoded_text = tokenizer.decode(completion.choices[0].token_ids)
        # The decoded text should contain a <|im_end|> at the end
        assert decoded_text.startswith(completion.choices[0].text)

        # Test without return_token_ids (should be None)
        completion_without = await client.completions.create(
            model=MODEL_NAME,
            prompt="Complete this sentence with emojis: I love coding 🚀",
            max_tokens=10,
            temperature=0,
            logprobs=1,
            extra_body={"return_token_ids": False},
        )

        completion_without_dict = completion_without.model_dump()
        assert completion_without_dict["choices"][0].get("token_ids") is None
        assert completion_without_dict.get("prompt_token_ids") is None


@pytest.mark.asyncio
async def test_chat_completion_with_tool_use(server):
    """Test chat completion with tool use (get_weather function)."""
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type":
                        "string",
                        "description":
                        "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature",
                    },
                },
                "required": ["location"],
            },
        },
    }]

    async with server.get_async_client() as client:
        # Test with return_token_ids enabled
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What's the weather like in Paris?"
                },
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=100,
            temperature=0,
            logprobs=True,
            extra_body={"return_token_ids": True},
        )

        # Verify token_ids field is present in choices
        assert response.choices[0].token_ids is not None
        assert isinstance(response.choices[0].token_ids, list)

        # Verify prompt_token_ids field is present
        assert response.prompt_token_ids is not None
        assert isinstance(response.prompt_token_ids, list)

        # Verify the prompt texts and response texts
        tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
        prompt_text = tokenizer.decode(response.prompt_token_ids)
        assert prompt_text.startswith(
            "<|im_start|>system\nYou are a helpful assistant.")
        assert prompt_text.endswith(
            "What's the weather like in Paris?<|im_end|>\n"
            "<|im_start|>assistant\n")

        response_text = tokenizer.decode(response.choices[0].token_ids)
        assert response_text.startswith(
            "<tool_call>\n{\"name\": \"get_weather\"")
        assert response_text.endswith("</tool_call><|im_end|>")

        # If tool call was made, verify the response structure
        if response.choices[0].message.tool_calls:
            assert len(response.choices[0].message.tool_calls) > 0
            tool_call = response.choices[0].message.tool_calls[0]
            assert tool_call.function.name == "get_weather"

        # Test without return_token_ids
        response_without = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What's the weather like in Paris?"
                },
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=100,
            temperature=0,
            logprobs=True,
            extra_body={"return_token_ids": False},
        )

        assert response_without.choices[0].token_ids is None
        assert response_without.prompt_token_ids is None


@pytest.mark.asyncio
async def test_comparison_with_prompt_logprobs_and_logprobs(server):
    """
    Test that token_ids align with prompt_logprobs and
    logprobs when return_tokens_as_token_ids is enabled.
    """
    async with server.get_async_client() as client:
        # Test with both return_token_ids and return_tokens_as_token_ids enabled
        completion = await client.completions.create(
            model=MODEL_NAME,
            prompt="Hello, world! How are you today?",
            max_tokens=20,
            temperature=0,
            echo=True,
            logprobs=1,
            extra_body={
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
                "prompt_logprobs": 1
            },
        )

        # Verify all fields are present
        assert completion.choices[0].token_ids is not None
        assert completion.choices[0].prompt_token_ids is not None
        assert completion.choices[0].prompt_logprobs is not None
        assert completion.choices[0].logprobs is not None

        # Extract token IDs from logprobs
        # (when return_tokens_as_token_ids is True)
        logprobs_token_ids = []
        for token_str in completion.choices[0].logprobs.tokens:
            # Token format is "token_id:12345" when
            # return_tokens_as_token_ids is True
            if token_str.startswith("token_id:"):
                token_id = int(token_str.removeprefix("token_id:"))
                logprobs_token_ids.append(token_id)

        # When echo=True, the logprobs include both prompt and response tokens
        # The token_ids field should match the response portion
        # The prompt_token_ids should match the prompt portion

        # Verify tokenizer consistency
        tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)

        # Decode prompt tokens
        if completion.choices[0].prompt_token_ids:
            prompt_text = tokenizer.decode(
                completion.choices[0].prompt_token_ids)
            # The decoded prompt should match or close to original prompt
            assert "Hello, world" in prompt_text

        # Decode response tokens
        if completion.choices[0].token_ids:
            response_text = tokenizer.decode(completion.choices[0].token_ids)
            assert response_text == completion.choices[0].text

        # Test streaming mode
        stream = await client.completions.create(
            model=MODEL_NAME,
            prompt="Tell me a short fact about Python:",
            max_tokens=30,
            temperature=0,
            stream=True,
            logprobs=1,
            extra_body={
                "return_token_ids": True,
                "return_tokens_as_token_ids": True
            },
        )

        # Collect streamed tokens
        streamed_token_ids = []
        async for chunk in stream:
            print(chunk)
            if chunk.choices and chunk.choices[0].logprobs:
                for token_str in chunk.choices[0].logprobs.tokens:
                    if token_str.startswith("token_id:"):
                        token_id = int(token_str.removeprefix("token_id:"))
                        streamed_token_ids.append(token_id)

        # Verify we collected some tokens
        assert len(streamed_token_ids) > 0


@pytest.mark.asyncio
async def test_chat_completion_with_emoji_and_token_ids(server):
    """Test chat completion with emojis to verify token_ids handling."""
    async with server.get_async_client() as client:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You like to use emojis in your responses."
                },
                {
                    "role": "user",
                    "content": "Repeat after me: I love cats 🐱"
                },
            ],
            max_tokens=50,
            temperature=0,
            logprobs=True,
            extra_body={"return_token_ids": True},
        )

        # Verify token_ids are present
        response_dict = response.model_dump()
        assert response.choices[0].token_ids is not None
        assert "prompt_token_ids" in response_dict

        # Verify the response contains the expected fields
        assert response.choices[0].message.content is not None

        # Decode token_ids and verify consistency
        tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
        decoded_response = tokenizer.decode(response.choices[0].token_ids)
        # The content should match the response text
        # except the ending <|im_end|>
        assert decoded_response == response.choices[
            0].message.content + "<|im_end|>"

        # Test with streaming
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": "Say hello with an emoji 👋"
            }],
            max_tokens=20,
            temperature=0,
            stream=True,
            logprobs=True,
            extra_body={"return_token_ids": True},
        )

        collected_content = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                collected_content += chunk.choices[0].delta.content

        # Verify we got some response
        assert len(collected_content) > 0
