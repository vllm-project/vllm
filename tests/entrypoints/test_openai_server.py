import os
import subprocess
import time

import sys
import pytest
import requests
import ray  # using Ray for overall ease of process management, parallel requests, and debugging.
import openai  # use the official client for correctness check
from huggingface_hub import snapshot_download  # downloading lora to test lora requests

# imports for guided decoding tests
import json
import jsonschema
import re

from vllm.transformers_utils.tokenizer import get_tokenizer

MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 60 seconds
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"  # any model with a chat template should work here
LORA_NAME = "typeof/zephyr-7b-beta-lora"  # technically this needs Mistral-7B-v0.1 as base, but we're not testing generation quality here

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
                "required": ["company", "position"]
            }
        }
    },
    "required": ["name", "age", "skills", "work history"]
}

TEST_REGEX = r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}" + \
             r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)"

TEST_CHOICE = [
    "Python", "Java", "JavaScript", "C++", "C#", "PHP", "TypeScript", "Ruby",
    "Swift", "Kotlin"
]

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
def server(zephyr_lora_files):
    ray.init()
    server_runner = ServerRunner.remote([
        "--model",
        MODEL_NAME,
        "--dtype",
        "bfloat16",  # use half precision for speed and memory savings in CI environment
        "--max-model-len",
        "8192",
        "--enforce-eager",
        # lora config below
        "--enable-lora",
        "--lora-modules",
        f"zephyr-lora={zephyr_lora_files}",
        f"zephyr-lora2={zephyr_lora_files}",
        "--max-lora-rank",
        "64",
        "--max-cpu-loras",
        "2",
        "--max-num-seqs",
        "128"
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


async def test_check_models(server, client: openai.AsyncOpenAI):
    models = await client.models.list()
    models = models.data
    served_model = models[0]
    lora_models = models[1:]
    assert served_model.id == MODEL_NAME
    assert all(model.root == MODEL_NAME for model in models)
    assert lora_models[0].id == "zephyr-lora"
    assert lora_models[1].id == "zephyr-lora2"


@pytest.mark.parametrize(
    # first test base model, then test loras
    "model_name",
    [MODEL_NAME, "zephyr-lora", "zephyr-lora2"],
)
async def test_single_completion(server, client: openai.AsyncOpenAI,
                                 model_name: str):
    completion = await client.completions.create(model=model_name,
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


@pytest.mark.parametrize(
    # just test 1 lora hereafter
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_single_chat_session(server, client: openai.AsyncOpenAI,
                                   model_name: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(model=model_name,
                                                           messages=messages,
                                                           max_tokens=10,
                                                           logprobs=True,
                                                           top_logprobs=10)
    assert chat_completion.id is not None
    assert chat_completion.choices is not None and len(
        chat_completion.choices) == 1
    assert chat_completion.choices[0].message is not None
    assert chat_completion.choices[0].logprobs is not None
    assert chat_completion.choices[0].logprobs.top_logprobs is not None
    assert len(chat_completion.choices[0].logprobs.top_logprobs[0]) == 10
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


@pytest.mark.parametrize(
    # just test 1 lora hereafter
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_completion_streaming(server, client: openai.AsyncOpenAI,
                                    model_name: str):
    prompt = "What is an LLM?"

    single_completion = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
    )
    single_output = single_completion.choices[0].text
    single_usage = single_completion.usage

    stream = await client.completions.create(model=model_name,
                                             prompt=prompt,
                                             max_tokens=5,
                                             temperature=0.0,
                                             stream=True)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk.choices[0].text)
    assert chunk.choices[0].finish_reason == "length"
    assert chunk.usage == single_usage
    assert "".join(chunks) == single_output


@pytest.mark.parametrize(
    # just test 1 lora hereafter
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_chat_streaming(server, client: openai.AsyncOpenAI,
                              model_name: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
    )
    output = chat_completion.choices[0].message.content
    stop_reason = chat_completion.choices[0].finish_reason

    # test streaming
    stream = await client.chat.completions.create(
        model=model_name,
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


@pytest.mark.parametrize(
    # just test 1 lora hereafter
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_batch_completions(server, client: openai.AsyncOpenAI,
                                 model_name: str):
    # test simple list
    batch = await client.completions.create(
        model=model_name,
        prompt=["Hello, my name is", "Hello, my name is"],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(batch.choices) == 2
    assert batch.choices[0].text == batch.choices[1].text

    # test n = 2
    batch = await client.completions.create(
        model=model_name,
        prompt=["Hello, my name is", "Hello, my name is"],
        n=2,
        max_tokens=5,
        temperature=0.0,
        extra_body=dict(
            # NOTE: this has to be true for n > 1 in vLLM, but not necessary for official client.
            use_beam_search=True),
    )
    assert len(batch.choices) == 4
    assert batch.choices[0].text != batch.choices[
        1].text, "beam search should be different"
    assert batch.choices[0].text == batch.choices[
        2].text, "two copies of the same prompt should be the same"
    assert batch.choices[1].text == batch.choices[
        3].text, "two copies of the same prompt should be the same"

    # test streaming
    batch = await client.completions.create(
        model=model_name,
        prompt=["Hello, my name is", "Hello, my name is"],
        max_tokens=5,
        temperature=0.0,
        stream=True,
    )
    texts = [""] * 2
    async for chunk in batch:
        assert len(chunk.choices) == 1
        choice = chunk.choices[0]
        texts[choice.index] += choice.text
    assert texts[0] == texts[1]


async def test_logits_bias(server, client: openai.AsyncOpenAI):
    prompt = "Hello, my name is"
    max_tokens = 5
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)

    # Test exclusive selection
    token_id = 1000
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        logit_bias={str(token_id): 100},
        seed=42,
    )
    assert completion.choices[0].text is not None and len(
        completion.choices[0].text) >= 5
    response_tokens = tokenizer(completion.choices[0].text,
                                add_special_tokens=False)["input_ids"]
    expected_tokens = tokenizer(tokenizer.decode([token_id] * 5),
                                add_special_tokens=False)["input_ids"]
    assert all([
        response == expected
        for response, expected in zip(response_tokens, expected_tokens)
    ])

    # Test ban
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    response_tokens = tokenizer(completion.choices[0].text,
                                add_special_tokens=False)["input_ids"]
    first_response = completion.choices[0].text
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        logit_bias={str(token): -100
                    for token in response_tokens},
    )
    assert first_response != completion.choices[0].text


async def test_guided_json_completion(server, client: openai.AsyncOpenAI):
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=
        f"Give an example JSON for an employee profile that fits this schema: {TEST_SCHEMA}",
        n=3,
        temperature=1.0,
        max_tokens=500,
        extra_body=dict(guided_json=TEST_SCHEMA))

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
        extra_body=dict(guided_json=TEST_SCHEMA))
    message = chat_completion.choices[0].message
    assert message.content is not None
    json1 = json.loads(message.content)
    jsonschema.validate(instance=json1, schema=TEST_SCHEMA)

    messages.append({"role": "assistant", "content": message.content})
    messages.append({
        "role":
        "user",
        "content":
        "Give me another one with a different name and age"
    })
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=500,
        extra_body=dict(guided_json=TEST_SCHEMA))
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
        extra_body=dict(guided_regex=TEST_REGEX))

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
        "role":
        "user",
        "content":
        f"Give an example IP address with this regex: {TEST_REGEX}"
    }]
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=20,
        extra_body=dict(guided_regex=TEST_REGEX))
    ip1 = chat_completion.choices[0].message.content
    assert ip1 is not None
    assert re.fullmatch(TEST_REGEX, ip1) is not None

    messages.append({"role": "assistant", "content": ip1})
    messages.append({"role": "user", "content": "Give me a different one"})
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=20,
        extra_body=dict(guided_regex=TEST_REGEX))
    ip2 = chat_completion.choices[0].message.content
    assert ip2 is not None
    assert re.fullmatch(TEST_REGEX, ip2) is not None
    assert ip1 != ip2


async def test_guided_choice_completion(server, client: openai.AsyncOpenAI):
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt="The best language for type-safe systems programming is ",
        n=2,
        temperature=1.0,
        max_tokens=10,
        extra_body=dict(guided_choice=TEST_CHOICE))

    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 2
    for i in range(2):
        assert completion.choices[i].text in TEST_CHOICE


async def test_guided_choice_chat(server, client: openai.AsyncOpenAI):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
        "user",
        "content":
        "The best language for type-safe systems programming is "
    }]
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=10,
        extra_body=dict(guided_choice=TEST_CHOICE))
    choice1 = chat_completion.choices[0].message.content
    assert choice1 in TEST_CHOICE

    messages.append({"role": "assistant", "content": choice1})
    messages.append({
        "role": "user",
        "content": "I disagree, pick another one"
    })
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=10,
        extra_body=dict(guided_choice=TEST_CHOICE))
    choice2 = chat_completion.choices[0].message.content
    assert choice2 in TEST_CHOICE
    assert choice1 != choice2


async def test_guided_decoding_type_error(server, client: openai.AsyncOpenAI):
    with pytest.raises(openai.BadRequestError):
        _ = await client.completions.create(
            model=MODEL_NAME,
            prompt="Give an example JSON that fits this schema: 42",
            extra_body=dict(guided_json=42))

    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
        "user",
        "content":
        "The best language for type-safe systems programming is "
    }]
    with pytest.raises(openai.BadRequestError):
        _ = await client.chat.completions.create(model=MODEL_NAME,
                                                 messages=messages,
                                                 extra_body=dict(guided_regex={
                                                     1: "Python",
                                                     2: "C++"
                                                 }))

    with pytest.raises(openai.BadRequestError):
        _ = await client.completions.create(
            model=MODEL_NAME,
            prompt="Give an example string that fits this regex",
            extra_body=dict(guided_regex=TEST_REGEX, guided_json=TEST_SCHEMA))


if __name__ == "__main__":
    pytest.main([__file__])
