# imports for guided decoding tests
import json
import re

import jsonschema
import openai  # use the official client for correctness check
import pytest
# using Ray for overall ease of process management, parallel requests,
# and debugging.
import ray
import torch
# downloading lora to test lora requests
from huggingface_hub import snapshot_download
from openai import BadRequestError

from vllm.transformers_utils.tokenizer import get_tokenizer

from ..utils import ServerRunner

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
EMBEDDING_MODEL_NAME = "intfloat/e5-mistral-7b-instruct"
# technically this needs Mistral-7B-v0.1 as base, but we're not testing
# generation quality here
LORA_NAME = "typeof/zephyr-7b-beta-lora"

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

TEST_REGEX = (r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
              r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)")

TEST_CHOICE = [
    "Python", "Java", "JavaScript", "C++", "C#", "PHP", "TypeScript", "Ruby",
    "Swift", "Kotlin"
]

pytestmark = pytest.mark.openai


@pytest.fixture(scope="session")
def zephyr_lora_files():
    return snapshot_download(repo_id=LORA_NAME)


@pytest.fixture(scope="module")
def server(zephyr_lora_files):
    ray.init()
    server_runner = ServerRunner.remote([
        "--model",
        MODEL_NAME,
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.75",
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
        "128",
    ])
    ray.get(server_runner.ready.remote())
    yield server_runner
    ray.shutdown()


@pytest.fixture(scope="module")
def embedding_server(zephyr_lora_files):
    ray.shutdown()
    ray.init()
    server_runner = ServerRunner.remote([
        "--model",
        EMBEDDING_MODEL_NAME,
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.75",
        "--max-model-len",
        "8192",
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


@pytest.mark.asyncio
async def test_check_models(server, client: openai.AsyncOpenAI):
    models = await client.models.list()
    models = models.data
    served_model = models[0]
    lora_models = models[1:]
    assert served_model.id == MODEL_NAME
    assert all(model.root == MODEL_NAME for model in models)
    assert lora_models[0].id == "zephyr-lora"
    assert lora_models[1].id == "zephyr-lora2"


@pytest.mark.asyncio
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

    choice = completion.choices[0]
    assert len(choice.text) >= 5
    assert choice.finish_reason == "length"
    assert completion.usage == openai.types.CompletionUsage(
        completion_tokens=5, prompt_tokens=6, total_tokens=11)

    # test using token IDs
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(completion.choices[0].text) >= 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # first test base model, then test loras
    "model_name",
    [MODEL_NAME, "zephyr-lora", "zephyr-lora2"],
)
async def test_no_logprobs(server, client: openai.AsyncOpenAI,
                           model_name: str):
    # test using token IDs
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
        logprobs=None,
    )
    choice = completion.choices[0]
    assert choice.logprobs is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # just test 1 lora hereafter
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_zero_logprobs(server, client: openai.AsyncOpenAI,
                             model_name: str):
    # test using token IDs
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
        logprobs=0,
    )
    choice = completion.choices[0]
    assert choice.logprobs is not None
    assert choice.logprobs.token_logprobs is not None
    assert choice.logprobs.top_logprobs is not None
    assert len(choice.logprobs.top_logprobs[0]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_some_logprobs(server, client: openai.AsyncOpenAI,
                             model_name: str):
    # test using token IDs
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
        logprobs=5,
    )
    choice = completion.choices[0]
    assert choice.logprobs is not None
    assert choice.logprobs.token_logprobs is not None
    assert choice.logprobs.top_logprobs is not None
    assert 5 <= len(choice.logprobs.top_logprobs[0]) <= 6


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_too_many_completion_logprobs(server, client: openai.AsyncOpenAI,
                                            model_name: str):

    with pytest.raises(
        (openai.BadRequestError, openai.APIError)):  # test using token IDs
        await client.completions.create(
            model=MODEL_NAME,
            prompt=[0, 0, 0, 0, 0],
            max_tokens=5,
            temperature=0.0,
            # vLLM has higher default max_logprobs (20 instead of 5) to support
            # both Completion API and Chat Completion API
            logprobs=21,
        )
        ...
    with pytest.raises(
        (openai.BadRequestError, openai.APIError)):  # test using token IDs
        stream = await client.completions.create(
            model=MODEL_NAME,
            prompt=[0, 0, 0, 0, 0],
            max_tokens=5,
            temperature=0.0,
            # vLLM has higher default max_logprobs (20 instead of 5) to support
            # both Completion API and Chat Completion API
            logprobs=30,
            stream=True,
        )
        async for chunk in stream:
            ...

    # the server should still work afterwards
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(completion.choices[0].text) >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # first test base model, then test loras
    "model_name",
    [MODEL_NAME, "zephyr-lora", "zephyr-lora2"],
)
async def test_no_logprobs_chat(server, client: openai.AsyncOpenAI,
                                model_name: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    chat_completion = await client.chat.completions.create(model=model_name,
                                                           messages=messages,
                                                           max_tokens=5,
                                                           temperature=0.0,
                                                           logprobs=False)

    choice = chat_completion.choices[0]
    assert choice.logprobs is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # just test 1 lora hereafter
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_zero_logprobs_chat(server, client: openai.AsyncOpenAI,
                                  model_name: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    chat_completion = await client.chat.completions.create(model=model_name,
                                                           messages=messages,
                                                           max_tokens=5,
                                                           temperature=0.0,
                                                           logprobs=True,
                                                           top_logprobs=0)

    choice = chat_completion.choices[0]
    assert choice.logprobs is not None
    assert choice.logprobs.content is not None
    assert len(choice.logprobs.content[0].top_logprobs) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_some_logprobs_chat(server, client: openai.AsyncOpenAI,
                                  model_name: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    chat_completion = await client.chat.completions.create(model=model_name,
                                                           messages=messages,
                                                           max_tokens=5,
                                                           temperature=0.0,
                                                           logprobs=True,
                                                           top_logprobs=5)

    choice = chat_completion.choices[0]
    assert choice.logprobs is not None
    assert choice.logprobs.content is not None
    assert len(choice.logprobs.content[0].top_logprobs) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_too_many_chat_logprobs(server, client: openai.AsyncOpenAI,
                                      model_name: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    # Default max_logprobs is 20, so this should raise an error
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        stream = await client.chat.completions.create(model=model_name,
                                                      messages=messages,
                                                      max_tokens=10,
                                                      logprobs=True,
                                                      top_logprobs=21,
                                                      stream=True)
        async for chunk in stream:
            ...

    with pytest.raises(openai.BadRequestError):
        await client.chat.completions.create(model=model_name,
                                             messages=messages,
                                             max_tokens=10,
                                             logprobs=True,
                                             top_logprobs=30,
                                             stream=False)

    # the server should still work afterwards
    chat_completion = await client.chat.completions.create(model=model_name,
                                                           messages=messages,
                                                           max_tokens=10,
                                                           stream=False)
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
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
                                                           top_logprobs=5)
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    assert chat_completion.usage == openai.types.CompletionUsage(
        completion_tokens=10, prompt_tokens=37, total_tokens=47)

    message = choice.message
    assert message.content is not None and len(message.content) >= 10
    assert message.role == "assistant"
    messages.append({"role": "assistant", "content": message.content})

    # test multi-turn dialogue
    messages.append({"role": "user", "content": "express your result in json"})
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
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
    stream = await client.completions.create(model=model_name,
                                             prompt=prompt,
                                             max_tokens=5,
                                             temperature=0.0,
                                             stream=True)
    chunks = []
    finish_reason_count = 0
    async for chunk in stream:
        chunks.append(chunk.choices[0].text)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    # finish reason should only return in last block
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == "length"
    assert chunk.choices[0].text
    assert "".join(chunks) == single_output


@pytest.mark.asyncio
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
    finish_reason_count = 0
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.role:
            assert delta.role == "assistant"
        if delta.content:
            chunks.append(delta.content)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    # finish reason should only return in last block
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == stop_reason
    assert delta.content
    assert "".join(chunks) == output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    ["HuggingFaceH4/zephyr-7b-beta", "zephyr-lora"],
)
async def test_chat_completion_stream_options(server,
                                              client: openai.AsyncOpenAI,
                                              model_name: str):
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "What is the capital of France?"
    }]

    # Test stream=True, stream_options={"include_usage": False}
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": False})
    async for chunk in stream:
        assert chunk.usage is None

    # Test stream=True, stream_options={"include_usage": True}
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True})

    async for chunk in stream:
        if chunk.choices[0].finish_reason is None:
            assert chunk.usage is None
        else:
            assert chunk.usage is None
            final_chunk = await stream.__anext__()
            assert final_chunk.usage is not None
            assert final_chunk.usage.prompt_tokens > 0
            assert final_chunk.usage.completion_tokens > 0
            assert final_chunk.usage.total_tokens == (
                final_chunk.usage.prompt_tokens +
                final_chunk.usage.completion_tokens)
            assert final_chunk.choices == []

    # Test stream=False, stream_options={"include_usage": None}
    with pytest.raises(BadRequestError):
        await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            stream=False,
            stream_options={"include_usage": None})

    # Test stream=False, stream_options={"include_usage": True}
    with pytest.raises(BadRequestError):
        await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            stream=False,
            stream_options={"include_usage": True})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    ["HuggingFaceH4/zephyr-7b-beta", "zephyr-lora"],
)
async def test_completion_stream_options(server, client: openai.AsyncOpenAI,
                                         model_name: str):
    prompt = "What is the capital of France?"

    # Test stream=True, stream_options={"include_usage": False}
    stream = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": False})
    async for chunk in stream:
        assert chunk.usage is None

    # Test stream=True, stream_options={"include_usage": True}
    stream = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True})
    async for chunk in stream:
        if chunk.choices[0].finish_reason is None:
            assert chunk.usage is None
        else:
            assert chunk.usage is None
            final_chunk = await stream.__anext__()
            assert final_chunk.usage is not None
            assert final_chunk.usage.prompt_tokens > 0
            assert final_chunk.usage.completion_tokens > 0
            assert final_chunk.usage.total_tokens == (
                final_chunk.usage.prompt_tokens +
                final_chunk.usage.completion_tokens)
            assert final_chunk.choices == []

    # Test stream=False, stream_options={"include_usage": None}
    with pytest.raises(BadRequestError):
        await client.completions.create(model=model_name,
                                        prompt=prompt,
                                        max_tokens=5,
                                        temperature=0.0,
                                        stream=False,
                                        stream_options={"include_usage": None})

    # Test stream=False, stream_options={"include_usage": True}
    with pytest.raises(BadRequestError):
        await client.completions.create(model=model_name,
                                        prompt=prompt,
                                        max_tokens=5,
                                        temperature=0.0,
                                        stream=False,
                                        stream_options={"include_usage": True})


@pytest.mark.asyncio
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
            # NOTE: this has to be true for n > 1 in vLLM, but not necessary
            # for official client.
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


@pytest.mark.asyncio
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
    assert len(completion.choices[0].text) >= 5
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


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_json_completion(server, client: openai.AsyncOpenAI,
                                      guided_decoding_backend: str):
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=f"Give an example JSON for an employee profile "
        f"that fits this schema: {TEST_SCHEMA}",
        n=3,
        temperature=1.0,
        max_tokens=500,
        extra_body=dict(guided_json=TEST_SCHEMA,
                        guided_decoding_backend=guided_decoding_backend))

    assert completion.id is not None
    assert len(completion.choices) == 3
    for i in range(3):
        output_json = json.loads(completion.choices[i].text)
        jsonschema.validate(instance=output_json, schema=TEST_SCHEMA)


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_json_chat(server, client: openai.AsyncOpenAI,
                                guided_decoding_backend: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
        "user",
        "content":
        f"Give an example JSON for an employee profile that "
        f"fits this schema: {TEST_SCHEMA}"
    }]
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1000,
        extra_body=dict(guided_json=TEST_SCHEMA,
                        guided_decoding_backend=guided_decoding_backend))
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
        max_tokens=1000,
        extra_body=dict(guided_json=TEST_SCHEMA,
                        guided_decoding_backend=guided_decoding_backend))
    message = chat_completion.choices[0].message
    assert message.content is not None
    json2 = json.loads(message.content)
    jsonschema.validate(instance=json2, schema=TEST_SCHEMA)
    assert json1["name"] != json2["name"]
    assert json1["age"] != json2["age"]


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_regex_completion(server, client: openai.AsyncOpenAI,
                                       guided_decoding_backend: str):
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=f"Give an example IPv4 address with this regex: {TEST_REGEX}",
        n=3,
        temperature=1.0,
        max_tokens=20,
        extra_body=dict(guided_regex=TEST_REGEX,
                        guided_decoding_backend=guided_decoding_backend))

    assert completion.id is not None
    assert len(completion.choices) == 3
    for i in range(3):
        assert re.fullmatch(TEST_REGEX, completion.choices[i].text) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_regex_chat(server, client: openai.AsyncOpenAI,
                                 guided_decoding_backend: str):
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
        extra_body=dict(guided_regex=TEST_REGEX,
                        guided_decoding_backend=guided_decoding_backend))
    ip1 = chat_completion.choices[0].message.content
    assert ip1 is not None
    assert re.fullmatch(TEST_REGEX, ip1) is not None

    messages.append({"role": "assistant", "content": ip1})
    messages.append({"role": "user", "content": "Give me a different one"})
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=20,
        extra_body=dict(guided_regex=TEST_REGEX,
                        guided_decoding_backend=guided_decoding_backend))
    ip2 = chat_completion.choices[0].message.content
    assert ip2 is not None
    assert re.fullmatch(TEST_REGEX, ip2) is not None
    assert ip1 != ip2


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_choice_completion(server, client: openai.AsyncOpenAI,
                                        guided_decoding_backend: str):
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt="The best language for type-safe systems programming is ",
        n=2,
        temperature=1.0,
        max_tokens=10,
        extra_body=dict(guided_choice=TEST_CHOICE,
                        guided_decoding_backend=guided_decoding_backend))

    assert completion.id is not None
    assert len(completion.choices) == 2
    for i in range(2):
        assert completion.choices[i].text in TEST_CHOICE


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_choice_chat(server, client: openai.AsyncOpenAI,
                                  guided_decoding_backend: str):
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
        extra_body=dict(guided_choice=TEST_CHOICE,
                        guided_decoding_backend=guided_decoding_backend))
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
        extra_body=dict(guided_choice=TEST_CHOICE,
                        guided_decoding_backend=guided_decoding_backend))
    choice2 = chat_completion.choices[0].message.content
    assert choice2 in TEST_CHOICE
    assert choice1 != choice2


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_decoding_type_error(server, client: openai.AsyncOpenAI,
                                          guided_decoding_backend: str):
    with pytest.raises(openai.BadRequestError):
        _ = await client.completions.create(
            model=MODEL_NAME,
            prompt="Give an example JSON that fits this schema: 42",
            extra_body=dict(guided_json=42,
                            guided_decoding_backend=guided_decoding_backend))

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


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_choice_chat_logprobs(server, client: openai.AsyncOpenAI,
                                           guided_decoding_backend: str):
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
        logprobs=True,
        top_logprobs=5,
        extra_body=dict(guided_choice=TEST_CHOICE,
                        guided_decoding_backend=guided_decoding_backend))

    assert chat_completion.choices[0].logprobs is not None
    assert chat_completion.choices[0].logprobs.content is not None
    top_logprobs = chat_completion.choices[0].logprobs.content[0].top_logprobs

    # -9999.0 is the minimum logprob returned by OpenAI
    for item in top_logprobs:
        assert item.logprob >= -9999.0, f"Failed (top_logprobs={top_logprobs})"


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_named_tool_use(server, client: openai.AsyncOpenAI,
                              guided_decoding_backend: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
        "user",
        "content":
        f"Give an example JSON for an employee profile that "
        f"fits this schema: {TEST_SCHEMA}"
    }]

    # non-streaming

    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1000,
        tools=[{
            "type": "function",
            "function": {
                "name": "dummy_function_name",
                "description": "This is a dummy function",
                "parameters": TEST_SCHEMA
            }
        }],
        tool_choice={
            "type": "function",
            "function": {
                "name": "dummy_function_name"
            }
        })
    message = chat_completion.choices[0].message
    assert len(message.content) == 0
    json_string = message.tool_calls[0].function.arguments
    json1 = json.loads(json_string)
    jsonschema.validate(instance=json1, schema=TEST_SCHEMA)

    messages.append({"role": "assistant", "content": json_string})
    messages.append({
        "role":
        "user",
        "content":
        "Give me another one with a different name and age"
    })

    # streaming

    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1000,
        tools=[{
            "type": "function",
            "function": {
                "name": "dummy_function_name",
                "description": "This is a dummy function",
                "parameters": TEST_SCHEMA
            }
        }],
        tool_choice={
            "type": "function",
            "function": {
                "name": "dummy_function_name"
            }
        },
        stream=True)

    output = []
    finish_reason_count = 0
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.role:
            assert delta.role == "assistant"
        assert delta.content is None or len(delta.content) == 0
        if delta.tool_calls:
            output.append(delta.tool_calls[0].function.arguments)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    # finish reason should only return in last block
    assert finish_reason_count == 1
    json2 = json.loads("".join(output))
    jsonschema.validate(instance=json2, schema=TEST_SCHEMA)
    assert json1["name"] != json2["name"]
    assert json1["age"] != json2["age"]


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend", ["outlines"])
async def test_required_tool_use_not_yet_supported(
        server, client: openai.AsyncOpenAI, guided_decoding_backend: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
        "user",
        "content":
        f"Give an example JSON for an employee profile that "
        f"fits this schema: {TEST_SCHEMA}"
    }]

    with pytest.raises(openai.BadRequestError):
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1000,
            tools=[{
                "type": "function",
                "function": {
                    "name": "dummy_function_name",
                    "description": "This is a dummy function",
                    "parameters": TEST_SCHEMA
                }
            }],
            tool_choice="required")

    with pytest.raises(openai.BadRequestError):
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1000,
            tools=[{
                "type": "function",
                "function": {
                    "name": "dummy_function_name",
                    "description": "This is a dummy function",
                    "parameters": TEST_SCHEMA
                }
            }],
            tool_choice="auto")


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend", ["outlines"])
async def test_inconsistent_tool_choice_and_tools(
        server, client: openai.AsyncOpenAI, guided_decoding_backend: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
        "user",
        "content":
        f"Give an example JSON for an employee profile that "
        f"fits this schema: {TEST_SCHEMA}"
    }]

    with pytest.raises(openai.BadRequestError):
        await client.chat.completions.create(model=MODEL_NAME,
                                             messages=messages,
                                             max_tokens=1000,
                                             tool_choice={
                                                 "type": "function",
                                                 "function": {
                                                     "name":
                                                     "dummy_function_name"
                                                 }
                                             })

    with pytest.raises(openai.BadRequestError):
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1000,
            tools=[{
                "type": "function",
                "function": {
                    "name": "dummy_function_name",
                    "description": "This is a dummy function",
                    "parameters": TEST_SCHEMA
                }
            }],
            tool_choice={
                "type": "function",
                "function": {
                    "name": "nondefined_function_name"
                }
            })


@pytest.mark.asyncio
async def test_response_format_json_object(server, client: openai.AsyncOpenAI):
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role":
                "user",
                "content": ('what is 1+1? please respond with a JSON object, '
                            'the format is {"result": 2}')
            }],
            response_format={"type": "json_object"})

        content = resp.choices[0].message.content
        assert content is not None

        loaded = json.loads(content)
        assert loaded == {"result": 2}, loaded


@pytest.mark.asyncio
async def test_extra_fields(server, client: openai.AsyncOpenAI):
    with pytest.raises(BadRequestError) as exc_info:
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant.",
                "extra_field": "0",
            }],  # type: ignore
            temperature=0,
            seed=0)

    assert "extra_forbidden" in exc_info.value.message


@pytest.mark.asyncio
async def test_complex_message_content(server, client: openai.AsyncOpenAI):
    resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role":
            "user",
            "content": [{
                "type":
                "text",
                "text":
                "what is 1+1? please provide the result without any other text."
            }]
        }],
        temperature=0,
        seed=0)
    content = resp.choices[0].message.content
    assert content == "2"


@pytest.mark.asyncio
async def test_custom_role(server, client: openai.AsyncOpenAI):
    # Not sure how the model handles custom roles so we just check that
    # both string and complex message content are handled in the same way

    resp1 = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "my-custom-role",
            "content": "what is 1+1?",
        }],  # type: ignore
        temperature=0,
        seed=0)

    resp2 = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "my-custom-role",
            "content": [{
                "type": "text",
                "text": "what is 1+1?"
            }]
        }],  # type: ignore
        temperature=0,
        seed=0)

    content1 = resp1.choices[0].message.content
    content2 = resp2.choices[0].message.content
    assert content1 == content2


@pytest.mark.asyncio
async def test_guided_grammar(server, client: openai.AsyncOpenAI):
    simple_sql_grammar = """
start: select_statement

select_statement: "SELECT" column "from" table "where" condition

column: "col_1" | "col_2"
table: "table_1" | "table_2"
condition: column "=" number

number: "1" | "2"
"""

    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=("Generate a sql state that select col_1 from "
                "table_1 where it is equals to 1"),
        temperature=1.0,
        max_tokens=500,
        extra_body=dict(guided_grammar=simple_sql_grammar))

    content = completion.choices[0].text

    # use Lark to parse the output, and make sure it's a valid parse tree
    from lark import Lark
    parser = Lark(simple_sql_grammar)
    parser.parse(content)

    # remove spaces for comparison b/c we removed them in the grammar
    ground_truth = "SELECT col_1 from table_1 where col_1 = 1".replace(" ", "")

    assert content.strip() == ground_truth


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # first test base model, then test loras
    "model_name",
    [MODEL_NAME, "zephyr-lora", "zephyr-lora2"],
)
@pytest.mark.parametrize("logprobs_arg", [1, 0])
async def test_echo_logprob_completion(server, client: openai.AsyncOpenAI,
                                       model_name: str, logprobs_arg: int):
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
    # test using text and token IDs
    for prompt in ("Hello, my name is", [0, 0, 0, 0, 0]):
        completion = await client.completions.create(model=model_name,
                                                     prompt=prompt,
                                                     max_tokens=5,
                                                     temperature=0.0,
                                                     echo=True,
                                                     logprobs=logprobs_arg)

        prompt_text = tokenizer.decode(prompt) if isinstance(prompt,
                                                             list) else prompt
        assert re.search(r"^" + prompt_text, completion.choices[0].text)
        logprobs = completion.choices[0].logprobs
        assert logprobs is not None
        assert len(logprobs.text_offset) > 5
        assert (len(logprobs.token_logprobs) > 5
                and logprobs.token_logprobs[0] is None)
        assert (len(logprobs.top_logprobs) > 5
                and logprobs.top_logprobs[0] is None)
        for top_logprobs in logprobs.top_logprobs[1:]:
            assert max(logprobs_arg,
                       1) <= len(top_logprobs) <= logprobs_arg + 1
        assert len(logprobs.tokens) > 5


@pytest.mark.asyncio
async def test_long_seed(server, client: openai.AsyncOpenAI):
    for seed in [
            torch.iinfo(torch.long).min - 1,
            torch.iinfo(torch.long).max + 1
    ]:
        with pytest.raises(BadRequestError) as exc_info:
            await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "system",
                    "content": "You are a helpful assistant.",
                }],
                temperature=0,
                seed=seed)

        assert ("greater_than_equal" in exc_info.value.message
                or "less_than_equal" in exc_info.value.message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [EMBEDDING_MODEL_NAME],
)
async def test_single_embedding(embedding_server, client: openai.AsyncOpenAI,
                                model_name: str):
    input_texts = [
        "The chef prepared a delicious meal.",
    ]

    # test single embedding
    embeddings = await client.embeddings.create(
        model=model_name,
        input=input_texts,
        encoding_format="float",
    )
    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) == 4096
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 9
    assert embeddings.usage.total_tokens == 9

    # test using token IDs
    input_tokens = [1, 1, 1, 1, 1]
    embeddings = await client.embeddings.create(
        model=model_name,
        input=input_tokens,
        encoding_format="float",
    )
    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) == 4096
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 5
    assert embeddings.usage.total_tokens == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [EMBEDDING_MODEL_NAME],
)
async def test_batch_embedding(embedding_server, client: openai.AsyncOpenAI,
                               model_name: str):
    # test List[str]
    input_texts = [
        "The cat sat on the mat.", "A feline was resting on a rug.",
        "Stars twinkle brightly in the night sky."
    ]
    embeddings = await client.embeddings.create(
        model=model_name,
        input=input_texts,
        encoding_format="float",
    )
    assert embeddings.id is not None
    assert len(embeddings.data) == 3
    assert len(embeddings.data[0].embedding) == 4096

    # test List[List[int]]
    input_tokens = [[4, 5, 7, 9, 20], [15, 29, 499], [24, 24, 24, 24, 24],
                    [25, 32, 64, 77]]
    embeddings = await client.embeddings.create(
        model=model_name,
        input=input_tokens,
        encoding_format="float",
    )
    assert embeddings.id is not None
    assert len(embeddings.data) == 4
    assert len(embeddings.data[0].embedding) == 4096
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 17
    assert embeddings.usage.total_tokens == 17


if __name__ == "__main__":
    pytest.main([__file__])
