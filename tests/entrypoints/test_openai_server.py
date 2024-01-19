import time
import subprocess

import sys
import pytest
import requests
import ray  # using Ray for overall ease of process management, parallel requests, and debugging.
import openai  # use the official client for correctness check

MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 60 seconds
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"  # any model with a chat template should work here

pytestmark = pytest.mark.asyncio


@ray.remote(num_gpus=1)
class ServerRunner:

    def __init__(self, args):
        self.proc = subprocess.Popen(
            ["python3", "-m", "vllm.entrypoints.openai.api_server"] + args,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_server()

    def ready(self):
        return True

    def _wait_for_server(self):
        # run health check
        start = time.time()
        while True:
            try:
                if requests.get(
                        "http://localhost:8000/health").status_code == 200:
                    break
            except Exception as err:
                if self.proc.poll() is not None:
                    raise RuntimeError("Server exited unexpectedly.") from err

                time.sleep(0.5)
                if time.time() - start > MAX_SERVER_START_WAIT_S:
                    raise RuntimeError(
                        "Server failed to start in time.") from err

    def __del__(self):
        if hasattr(self, "proc"):
            self.proc.terminate()


@pytest.fixture(scope="session")
def server():
    ray.init()
    server_runner = ServerRunner.remote([
        "--model",
        MODEL_NAME,
        "--dtype",
        "bfloat16",  # use half precision for speed and memory savings in CI environment
        "--max-model-len",
        "8192"
    ])
    ray.get(server_runner.ready.remote())
    yield server_runner
    ray.shutdown()


@pytest.fixture(scope="session")
def client():
    client = openai.AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    yield client


async def test_single_completion(server, client: openai.AsyncOpenAI):
    completion = await client.completions.create(model=MODEL_NAME,
                                                 prompt="Hello, my name is",
                                                 max_tokens=5,
                                                 temperature=0.0)

    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 1
    assert completion.choices[0].text is not None and len(
        completion.choices[0].text) >= 5
    assert completion.choices[0].finish_reason == "length"
    assert completion.usage == openai.types.CompletionUsage(
        completion_tokens=5, prompt_tokens=6, total_tokens=11)

    # test using token IDs
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    assert completion.choices[0].text is not None and len(
        completion.choices[0].text) >= 5


async def test_single_chat_session(server, client: openai.AsyncOpenAI):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=10,
    )
    assert chat_completion.id is not None
    assert chat_completion.choices is not None and len(
        chat_completion.choices) == 1
    assert chat_completion.choices[0].message is not None
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 10
    assert message.role == "assistant"
    messages.append({"role": "assistant", "content": message.content})

    # test multi-turn dialogue
    messages.append({"role": "user", "content": "express your result in json"})
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=10,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0


async def test_completion_streaming(server, client: openai.AsyncOpenAI):
    prompt = "What is an LLM?"

    single_completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
    )
    single_output = single_completion.choices[0].text
    single_usage = single_completion.usage

    stream = await client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
    )
    chunks = []
    async for chunk in stream:
        chunks.append(chunk.choices[0].text)
    assert chunk.choices[0].finish_reason == "length"
    assert chunk.usage == single_usage
    assert "".join(chunks) == single_output


async def test_chat_streaming(server, client: openai.AsyncOpenAI):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
    )
    output = chat_completion.choices[0].message.content
    stop_reason = chat_completion.choices[0].finish_reason

    # test streaming
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        stream=True,
    )
    chunks = []
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.role:
            assert delta.role == "assistant"
        if delta.content:
            chunks.append(delta.content)
    assert chunk.choices[0].finish_reason == stop_reason
    assert "".join(chunks) == output


if __name__ == "__main__":
    pytest.main([__file__])
