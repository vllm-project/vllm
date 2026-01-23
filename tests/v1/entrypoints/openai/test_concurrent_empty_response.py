# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that concurrent requests with large payloads do not return empty responses.

Regression test for a bug where `with_cancellation` returned None when the
disconnect listener completed before the handler under high concurrent load,
causing FastAPI to send HTTP 200 with an empty body.
"""

import asyncio

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def default_server_args():
    return [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def _build_large_messages(num_turns: int = 50, chars_per_msg: int = 2048):
    """Build a large conversation history to create request bodies >100KB."""
    messages = []
    for i in range(num_turns):
        role = "user" if i % 2 == 0 else "assistant"
        # Generate padding text to reach target size
        content = f"Turn {i}: " + ("x" * chars_per_msg)
        messages.append({"role": role, "content": content})
    # Ensure the last message is from the user
    if messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "Please respond."})
    return messages


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_concurrent_large_requests_no_empty_response(
    client: openai.AsyncOpenAI, model_name: str
) -> None:
    """Verify that concurrent non-streaming requests with large payloads
    never return empty responses."""
    messages = _build_large_messages(num_turns=50, chars_per_msg=2048)
    num_concurrent = 50
    num_iterations = 3

    for iteration in range(num_iterations):

        async def make_request(req_id: int):
            completion = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=5,
                temperature=1.0,
            )
            return req_id, completion

        tasks = [asyncio.create_task(make_request(i)) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)

        for req_id, completion in results:
            assert completion.choices is not None, (
                f"Iteration {iteration}, request {req_id}: "
                "response.choices is None (empty response body)"
            )
            assert len(completion.choices) > 0, (
                f"Iteration {iteration}, request {req_id}: response.choices is empty"
            )
            assert completion.choices[0].message.content, (
                f"Iteration {iteration}, request {req_id}: message.content is empty"
            )
            assert completion.usage is not None, (
                f"Iteration {iteration}, request {req_id}: usage is None"
            )
            assert completion.usage.completion_tokens > 0, (
                f"Iteration {iteration}, request {req_id}: completion_tokens is 0"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_concurrent_large_requests_streaming_no_empty_response(
    client: openai.AsyncOpenAI, model_name: str
) -> None:
    """Verify that concurrent streaming requests with large payloads
    never return empty responses."""
    messages = _build_large_messages(num_turns=50, chars_per_msg=2048)
    num_concurrent = 50

    async def make_streaming_request(req_id: int):
        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=5,
            temperature=1.0,
            stream=True,
        )
        collected_text = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                collected_text += chunk.choices[0].delta.content
        return req_id, collected_text

    tasks = [
        asyncio.create_task(make_streaming_request(i)) for i in range(num_concurrent)
    ]
    results = await asyncio.gather(*tasks)

    for req_id, text in results:
        assert text, f"Request {req_id}: collected streaming text is empty"
