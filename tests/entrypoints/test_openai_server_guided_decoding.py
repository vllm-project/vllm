import os
import subprocess
import time

import sys
import pytest
import requests
import ray  # using Ray for overall ease of process management, parallel requests, and debugging.
import openai  # use the official client for correctness check

import json
import jsonschema
import re

MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 60 seconds
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"  # any model with a chat template should work here

TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "age": {
            "type": "integer"
        },
        "skills": {
            "type": "array",
            "items": {
                "type": "string",
                "maxLength": 10
            },
            "minItems": 3
        },
        "work history": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string"
                    },
                    "duration": {
                        "type": "string"
                    },
                    "position": {
                        "type": "string"
                    }
                },
                "required": [
                    "company",
                    "position"
                ]
            }
        }
    },
    "required": [
        "name",
        "age",
        "skills",
        "work history"
    ]
}
# NOTE: outlines' underlying regex library (interegular) doesn't support
# ^ or $ or \b, kinda annoying
TEST_REGEX = r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}" + \
               "(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)"

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


@pytest.fixture(scope="module")
def server():
    ray.init()
    server_runner = ServerRunner.remote([
        "--model",
        MODEL_NAME,
        "--dtype",
        "bfloat16",  # use half precision for speed and memory savings in CI environment
        "--max-model-len",
        "8192",
        "--enforce-eager",
    ])
    ray.get(server_runner.ready.remote())
    yield server_runner
    ray.shutdown()


@pytest.fixture(scope="module")
def client():
    client = openai.AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    yield client


async def test_guided_json_completion(server, client: openai.AsyncOpenAI):
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=f"Give an example JSON for an employee profile that fits this schema: {TEST_SCHEMA}",
        n=3,
        temperature=1.0,
        max_tokens=500,
        extra_body=dict(
            guided_json=TEST_SCHEMA
        )
    )

    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 3
    for i in range(3):
        assert completion.choices[i].text is not None
        output_json = json.loads(completion.choices[i].text)
        jsonschema.validate(instance=output_json, schema=TEST_SCHEMA)


async def test_guided_json_chat(server, client: openai.AsyncOpenAI):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "Give an example JSON for an employee profile that " + \
                    f"fits this schema: {TEST_SCHEMA}"
    }]
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=500,
        extra_body=dict(
            guided_json=TEST_SCHEMA
        )
    )
    message = chat_completion.choices[0].message
    assert message.content is not None
    json1 = json.loads(message.content)
    jsonschema.validate(instance=json1, schema=TEST_SCHEMA)

    messages.append({"role": "assistant", "content": message.content})
    messages.append({
        "role": "user",
        "content": "Give me another one with a different name and age"
    })
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=500,
        extra_body=dict(
            guided_json=TEST_SCHEMA
        )
    )
    message = chat_completion.choices[0].message
    assert message.content is not None
    json2 = json.loads(message.content)
    jsonschema.validate(instance=json2, schema=TEST_SCHEMA)
    assert json1["name"] != json2["name"]
    assert json1["age"] != json2["age"]


async def test_guided_regex_completion(server, client: openai.AsyncOpenAI):
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=f"Give an example IPv4 address with this regex: {TEST_REGEX}",
        n=3,
        temperature=1.0,
        max_tokens=20,
        extra_body=dict(
            guided_regex=TEST_REGEX
        )
    )

    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 3
    for i in range(3):
        assert completion.choices[i].text is not None
        assert re.fullmatch(TEST_REGEX, completion.choices[i].text) is not None


async def test_guided_regex_chat(server, client: openai.AsyncOpenAI):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": f"Give an example IP address with this regex: {TEST_REGEX}"
    }]
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=20,
        extra_body=dict(
            guided_regex=TEST_REGEX
        )
    )
    ip1 = chat_completion.choices[0].message.content
    assert ip1 is not None
    assert re.fullmatch(TEST_REGEX, ip1) is not None

    messages.append({"role": "assistant", "content": ip1})
    messages.append({
        "role": "user",
        "content": "Give me a different one"
    })
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=20,
        extra_body=dict(
            guided_regex=TEST_REGEX
        )
    )
    ip2 = chat_completion.choices[0].message.content
    assert ip2 is not None
    assert re.fullmatch(TEST_REGEX, ip2) is not None
    assert ip1 != ip2


async def test_guided_decoding_type_error(server, client: openai.AsyncOpenAI):
    with pytest.raises(Exception):
        _ = await client.completions.create(
            model=MODEL_NAME,
            prompt="Give an example JSON that fits this schema: 42",
            temperature=0.0,
            extra_body=dict(
                guided_json=42
            )
        )

    with pytest.raises(Exception):
        _ = await client.completions.create(
            model=MODEL_NAME,
            prompt="Give an example string that fits this regex: True",
            temperature=0.0,
            extra_body=dict(
                guided_regex=True
            )
        )


if __name__ == "__main__":
    pytest.main([__file__])
