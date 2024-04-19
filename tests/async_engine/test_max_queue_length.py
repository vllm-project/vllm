import asyncio
import os
import subprocess
import time
from vllm.logger import init_logger

import sys
from fastapi.responses import JSONResponse
import pytest
import requests
# using Ray for overall ease of process management, parallel requests,
# and debugging.
import ray
import openai  # use the official client for correctness check
# downloading lora to test lora requests
from huggingface_hub import snapshot_download

logger = init_logger(__name__)

MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 60 seconds
# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
# technically this needs Mistral-7B-v0.1 as base, but we're not testing
# generation quality here
LORA_NAME = "typeof/zephyr-7b-beta-lora"

pytestmark = pytest.mark.asyncio


@ray.remote(num_gpus=1)
class ServerRunner:

    def __init__(self, args):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            ["python3", "-m", "vllm.entrypoints.openai.api_server"] + args,
            env=env,
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
def zephyr_lora_files():
    return snapshot_download(repo_id=LORA_NAME)


@pytest.fixture(scope="session")
def server():
    ray.init()
    server_runner = ServerRunner.remote([
        "--model",
        MODEL_NAME,
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "half",
        "--max-model-len",
        "1024",
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--max-queue-length",
        "3",
        "--max-num-batched-tokens",
        "2048",
        "--gpu-memory-utilization",
        "1"
    ])
    ray.get(server_runner.ready.remote())
    yield server_runner
    ray.shutdown()


@pytest.fixture(scope="session")
def client():
    client = openai.AsyncOpenAI(base_url="http://localhost:8000/v1",
                                api_key="token-abc123",
                                max_retries=0)
    yield client


@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_max_queue_length(server, client: openai.AsyncOpenAI,
                                model_name: str):
    sample_chats = [[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    }],
                    [{
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }, {
                        "role": "user",
                        "content": "Where was the 2020 world series played?"
                    }],
                    [{
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }, {
                        "role": "user",
                        "content": "How long did the 2020 world series last?"
                    }],
                    [{
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }, {
                        "role":
                        "user",
                        "content":
                        "What were some television viewership statistics?"
                    }],
                    [{
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }, {
                        "role": "user",
                        "content": "Why was the 2020 world series so popular?"
                    }]]

    async def make_api_call(sample_chat):
        chat_completion = await client.chat.completions.create(
            messages=sample_chat,
            model=model_name,
            temperature=0.8,
            presence_penalty=0.2,
            max_tokens=400,
        )
        return chat_completion

    async def main():
        coroutines = [
            make_api_call(sample_chat) for sample_chat in sample_chats
        ]

        responses = await asyncio.gather(*coroutines, return_exceptions=True)

        for response in responses:
            logger.info(response)
            if isinstance(response, JSONResponse):
                assert response.status_code == 503

    await main()