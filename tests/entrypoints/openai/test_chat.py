# imports for guided decoding tests
import json
import re
from typing import List

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

from ...utils import VLLM_PATH, RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
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


@pytest.fixture(scope="module")
def zephyr_lora_files():
    return snapshot_download(repo_id=LORA_NAME)


@pytest.fixture(scope="module")
def ray_ctx():
    ray.init(runtime_env={"working_dir": VLLM_PATH})
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def server(zephyr_lora_files, ray_ctx):
    return RemoteOpenAIServer([
        "--model",
        MODEL_NAME,
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
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
        "128",
    ])


@pytest.fixture(scope="module")
def client(server):
    return server.get_async_client()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # first test base model, then test loras
    "model_name",
    [MODEL_NAME, "zephyr-lora", "zephyr-lora2"],
)
async def test_no_logprobs_chat(client: openai.AsyncOpenAI, model_name: str):
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
async def test_zero_logprobs_chat(client: openai.AsyncOpenAI, model_name: str):
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
async def test_some_logprobs_chat(client: openai.AsyncOpenAI, model_name: str):
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
async def test_too_many_chat_logprobs(client: openai.AsyncOpenAI,
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
async def test_single_chat_session(client: openai.AsyncOpenAI,
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
    # just test 1 lora hereafter
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_chat_streaming(client: openai.AsyncOpenAI, model_name: str):
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
    chunks: List[str] = []
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
async def test_chat_completion_stream_options(client: openai.AsyncOpenAI,
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


# NOTE: Not sure why, but when I place this after `test_guided_regex_chat`
# (i.e. using the same ordering as in the Completions API tests), the test
# will fail on the second `guided_decoding_backend` even when I swap their order
# (ref: https://github.com/vllm-project/vllm/pull/5526#issuecomment-2173772256)
@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_choice_chat(client: openai.AsyncOpenAI,
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
async def test_guided_json_chat(client: openai.AsyncOpenAI,
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
async def test_guided_regex_chat(client: openai.AsyncOpenAI,
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
async def test_guided_decoding_type_error(client: openai.AsyncOpenAI):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("guided_decoding_backend",
                         ["outlines", "lm-format-enforcer"])
async def test_guided_choice_chat_logprobs(client: openai.AsyncOpenAI,
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
async def test_named_tool_use(client: openai.AsyncOpenAI,
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
        client: openai.AsyncOpenAI, guided_decoding_backend: str):
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
        client: openai.AsyncOpenAI, guided_decoding_backend: str):
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
async def test_response_format_json_object(client: openai.AsyncOpenAI):
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
async def test_extra_fields(client: openai.AsyncOpenAI):
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
async def test_complex_message_content(client: openai.AsyncOpenAI):
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
async def test_custom_role(client: openai.AsyncOpenAI):
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
async def test_long_seed(client: openai.AsyncOpenAI):
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
