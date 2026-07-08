# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tokenizers import get_tokenizer

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
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize("return_token_ids", [True, False, None])
async def test_basic_completion_with_emoji(server, return_token_ids: bool | None):
    """Test basic completion with emoji to verify token_ids field."""
    extra_body = None
    if return_token_ids is not None:
        extra_body = {"return_token_ids": return_token_ids}
    async with server.get_async_client() as client:
        # Test with return_token_ids enabled
        completion = await client.completions.create(
            model=MODEL_NAME,
            prompt="Complete this sentence with emojis: I love coding 🚀",
            max_tokens=10,
            temperature=0,
            logprobs=1,
            extra_body=extra_body,
        )

        # Check the raw response to see the structure
        completion_dict = completion.model_dump()

        # Verify prompt_token_ids field is present in the completion response
        assert "prompt_token_ids" in completion_dict["choices"][0]
        if not return_token_ids:
            # If return_token_ids is False, token_ids should not be present
            assert completion_dict["choices"][0].get("token_ids") is None
            assert completion_dict["choices"][0].get("prompt_token_ids") is None
            # Skip further checks
            return
        assert isinstance(completion.choices[0].prompt_token_ids, list)

        # Check against the expected prompt token IDs
        tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
        encoded_tokens = tokenizer.encode(
            "Complete this sentence with emojis: I love coding 🚀"
        )
        # Check that encoded_tokens is a subsequence of prompt_token_ids
        assert any(
            completion.choices[0].prompt_token_ids[i : i + len(encoded_tokens)]
            == encoded_tokens
            for i in range(
                len(completion.choices[0].prompt_token_ids) - len(encoded_tokens) + 1
            )
        )

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
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
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
                            "description": "The unit of temperature",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    async with server.get_async_client() as client:
        # Test with return_token_ids enabled
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather like in Paris?"},
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
            "<|im_start|>system\nYou are a helpful assistant."
        )
        assert prompt_text.endswith(
            "What's the weather like in Paris?<|im_end|>\n<|im_start|>assistant\n"
        )

        response_text = tokenizer.decode(response.choices[0].token_ids)
        assert response_text.startswith('<tool_call>\n{"name": "get_weather"')
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
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather like in Paris?"},
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
                "prompt_logprobs": 1,
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
        # The token_ids field should match the suffix of response portion
        # The prompt_token_ids should match the prompt portion
        assert len(completion.choices[0].token_ids) < len(logprobs_token_ids)
        response_token_ids_length = len(completion.choices[0].token_ids)
        assert (
            logprobs_token_ids[-response_token_ids_length:]
            == completion.choices[0].token_ids
        )

        # Verify tokenizer consistency
        tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)

        # Decode prompt tokens
        if completion.choices[0].prompt_token_ids:
            prompt_text = tokenizer.decode(completion.choices[0].prompt_token_ids)
            # The decoded prompt should match or close to original prompt
            assert "Hello, world" in prompt_text

        # Decode response tokens
        if completion.choices[0].token_ids:
            response_text = tokenizer.decode(completion.choices[0].token_ids)
            assert completion.choices[0].text.endswith(response_text)

        # Test streaming mode
        stream = await client.completions.create(
            model=MODEL_NAME,
            prompt="Tell me a short fact about Python:",
            max_tokens=30,
            temperature=0,
            stream=True,
            echo=False,
            logprobs=1,
            extra_body={"return_token_ids": True, "return_tokens_as_token_ids": True},
        )

        # Collect streamed tokens
        streamed_prompt_token_ids = []
        streamed_token_ids = []
        streamed_logprob_token_ids = []
        first_chunk = True
        async for chunk in stream:
            for token_str in chunk.choices[0].logprobs.tokens:
                # Token format is "token_id:12345" when
                # return_tokens_as_token_ids is True
                if token_str.startswith("token_id:"):
                    token_id = int(token_str.removeprefix("token_id:"))
                    streamed_logprob_token_ids.append(token_id)
            if first_chunk:
                streamed_prompt_token_ids = chunk.choices[0].prompt_token_ids
                first_chunk = False
            streamed_token_ids += chunk.choices[0].token_ids

        # Verify we collected some tokens and first chunk had prompt_token_ids
        assert len(streamed_prompt_token_ids) > 0
        assert streamed_token_ids == streamed_logprob_token_ids


@pytest.mark.asyncio
async def test_chat_completion_with_emoji_and_token_ids(server):
    """Test chat completion with emojis to verify token_ids handling."""
    chat_messages = [
        {"role": "system", "content": "You like to use emojis in your responses."},
        {"role": "user", "content": "Repeat after me: I love cats 🐱"},
    ]
    async with server.get_async_client() as client:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_messages,
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

        decoded_prompt = tokenizer.decode(response.prompt_token_ids)
        assert decoded_prompt.startswith(
            "<|im_start|>system\nYou like to use emojis in your responses."
        )
        assert decoded_prompt.endswith(
            "I love cats 🐱<|im_end|>\n<|im_start|>assistant\n"
        )

        decoded_response = tokenizer.decode(response.choices[0].token_ids)
        # The content should match the response text
        # except the ending <|im_end|>
        assert decoded_response == response.choices[0].message.content + "<|im_end|>"

        # Test with streaming
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_messages,
            max_tokens=50,
            temperature=0,
            stream=True,
            extra_body={"return_token_ids": True},
        )

        collected_content = ""
        collected_token_ids = []
        first_chunk = True

        async for chunk in stream:
            if first_chunk:
                assert chunk.prompt_token_ids is not None
                assert isinstance(chunk.prompt_token_ids, list)
                # Check the prompt_token_ids match the initial prompt
                decoded_prompt_stream = tokenizer.decode(chunk.prompt_token_ids)
                assert decoded_prompt_stream == decoded_prompt
                first_chunk = False
            else:
                chunk_dump = chunk.model_dump()
                assert "prompt_token_ids" not in chunk_dump, (
                    "Subsequent chunks should not have prompt_token_ids"
                )

            if chunk.choices:
                if chunk.choices[0].delta.content:
                    collected_content += chunk.choices[0].delta.content
                # token_ids may not present in all chunks
                choice_dump = chunk.choices[0].model_dump()
                if "token_ids" in choice_dump:
                    collected_token_ids.extend(chunk.choices[0].token_ids)

        # Verify we got response and token_ids
        assert len(collected_content) > 0
        assert len(collected_token_ids) > 0

        # Verify token_ids decode properly
        decoded_response = tokenizer.decode(collected_token_ids)
        assert decoded_response == collected_content + "<|im_end|>"
