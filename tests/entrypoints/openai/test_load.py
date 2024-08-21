"""
This file tests significant load on the vLLM server.
Inside vLLM, we use a zeromq based messaging protocol
to enable multiprocessing between the API server and 
the AsyncLLMEngine.

This test confirms that even at high load with >20k 
active requests, zmq does not drop any messages.
"""

import asyncio
import openai
import pytest

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
NUM_REQUESTS = 25000
QPS_RATE = 1000.
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
        "--max-model-len",
        "4096",
        "--enable-chunked-prefill",
        "--disable-log-requests"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server):
    return server.get_async_client()


async def get_request(client, model_name):
    for _ in range(NUM_REQUESTS):
        yield client.chat.completions.create(
            model=model_name,
            messages=MESSAGES,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        # Send 200
        await asyncio.sleep(1./QPS_RATE)

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", [MODEL_NAME],
)
async def test_load(client: openai.AsyncOpenAI, model_name: str):
    # Make requests to the server.
    tasks = []
    async for request in get_request(client, model_name):
        tasks.append(asyncio.create_task(request))
    outputs = await asyncio.gather(*tasks)
    
    # Check that each client generated exactly 50 tokens.
    # If this is true, then we are not seeing any message dropping in zeromq.
    for idx, output in enumerate(outputs):
        assert output.usage.completion_tokens == MAX_TOKENS, (
            f"Request {idx}: Expected {MAX_TOKENS} completion tokens but "
            f"found only {output.usage.completion_tokens} were generated. "
            f"zeromq multiprocessing frontend is likely dropping messages. "
        )
    
