"""
This file tests significant load on the vLLM server.
Inside vLLM, we use a zeromq based messaging protocol
to enable multiprocessing between the API server and 
the AsyncLLMEngine.

This test confirms that even at high load with >20k 
active requests, zmq does not drop any messages.
"""

import asyncio
import json

import aiohttp
import pytest

from ...utils import RemoteOpenAIServer

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
NUM_REQUESTS = 10000
MAX_TOKENS = 50
MESSAGES = [{
    "role": "system",
    "content": "you are a helpful assistant"
}, {
    "role": "user",
    "content": "The meaning of life is"
}]


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len", "4096", "--enable-chunked-prefill",
        "--disable-log-requests", "--enforce-eager"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def server_data(server):
    return {
        "url": f"{server.url_for('v1')}/chat/completions",
        "api_key": server.DUMMY_API_KEY
    }


# Cannot use Async OpenAIClient due to limitations in maximum
# number of concurrent requests that can be sent to the server 
# from the client.
async def async_openai_chat(model_name, url, api_key):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": model_name,
            "messages": MESSAGES,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        async with session.post(url=url, json=payload,
                                headers=headers) as response:
            assert response.status == 200
            # data = json.loads(response.text)
            data = json.loads(await response.text())
            completion_tokens = data["usage"]["completion_tokens"]
            text = data["choices"][0]["message"]

        return (completion_tokens, text)


async def get_request(model_name, url, api_key):
    for _ in range(NUM_REQUESTS):
        yield async_openai_chat(model_name, url, api_key)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_load(server_data, model_name):
    # Make requests to the server.
    tasks = []
    async for request in get_request(model_name, server_data["url"],
                                     server_data["api_key"]):
        tasks.append(asyncio.create_task(request))
    outputs = await asyncio.gather(*tasks)

    # Check that each client generated exactly 50 tokens.
    # If this is true, then we are not seeing any message dropping in zeromq.
    for idx, (completion_tokens, text) in enumerate(outputs):
        assert completion_tokens == MAX_TOKENS, (
            f"Request {idx}: Expected {MAX_TOKENS} completion tokens but "
            f"found only {completion_tokens} were generated. "
            f"zeromq multiprocessing frontend is likely dropping messages. "
            f"Full text:\n\n\n {text}")
