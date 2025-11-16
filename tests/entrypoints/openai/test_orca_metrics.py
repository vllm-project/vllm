# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="module", params=[True])
def server(request, monkeypatch_module):
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_chat_completion_with_orca_header(server: RemoteOpenAIServer):
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "what is 1+1?"},
    ]

    client = openai.OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{server.port}/v1",
        default_headers={"endpoint-load-metrics-format": "TEXT"},
    )

    # 1. Use raw client to get response headers.
    raw_client = client.with_raw_response

    # 2. Make the API call using the raw_client
    response_with_raw = raw_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_headers={"endpoint-load-metrics-format": "TEXT"},
    )

    # 3. Access the raw httpx.Response object
    raw_http_response = response_with_raw.http_response

    # 4. Get the headers from the httpx.Response object
    response_headers = raw_http_response.headers

    assert "endpoint-load-metrics" in response_headers


@pytest.mark.asyncio
async def test_completion_with_orca_header(client: openai.AsyncOpenAI):
    # 1. Use raw client to get response headers.
    raw_client = client.with_raw_response

    # 2. Make the API call using the raw_client
    completion = await raw_client.completions.create(
        model=MODEL_NAME,
        prompt="Hello, my name is",
        max_tokens=5,
        extra_headers={"endpoint-load-metrics-format": "JSON"},
    )

    # 3. Access the raw httpx.Response object
    raw_http_response = completion.http_response

    # 4. Get the headers from the httpx.Response object
    response_headers = raw_http_response.headers

    assert "endpoint-load-metrics" in response_headers


@pytest.mark.asyncio
async def test_single_completion(client: openai.AsyncOpenAI):
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt="Hello, my name is",
        max_tokens=5,
        extra_headers={"endpoint-load-metrics-format": "JSON"},
        temperature=0.0,
    )

    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 1

    choice = completion.choices[0]
    assert len(choice.text) >= 5
    assert choice.finish_reason == "length"
    assert completion.usage == openai.types.CompletionUsage(
        completion_tokens=5, prompt_tokens=6, total_tokens=11
    )

    # test using token IDs
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(completion.choices[0].text) >= 1
    assert completion.choices[0].prompt_logprobs is None
