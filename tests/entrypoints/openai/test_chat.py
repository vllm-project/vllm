# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# imports for structured outputs tests
import json
from collections import defaultdict

import jsonschema
import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import regex as re
import requests
import torch
from openai import BadRequestError

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.sampling_params import SamplingParams

from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.fixture(scope="module")
def zephyr_lora_files():
    """Download zephyr LoRA files once per test session."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id="typeof/zephyr-7b-beta-lora")


@pytest.fixture(scope="module")
def server(zephyr_lora_files):
    args = [
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
        "--max-lora-rank",
        "64",
        "--max-cpu-loras",
        "2",
        "--max-num-seqs",
        "128",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # first test base model, then test loras
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_no_logprobs_chat(client: openai.AsyncOpenAI, model_name: str):
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "what is 1+1?"},
    ]

    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=5,
        temperature=0.0,
        logprobs=False,
    )

    choice = chat_completion.choices[0]
    assert choice.logprobs is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # just test 1 lora hereafter
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_zero_logprobs_chat(client: openai.AsyncOpenAI, model_name: str):
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "what is 1+1?"},
    ]

    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=5,
        temperature=0.0,
        logprobs=True,
        top_logprobs=0,
    )

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
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "what is 1+1?"},
    ]

    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=5,
        temperature=0.0,
        logprobs=True,
        top_logprobs=5,
    )

    choice = chat_completion.choices[0]
    assert choice.logprobs is not None
    assert choice.logprobs.content is not None
    assert len(choice.logprobs.content[0].top_logprobs) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_too_many_chat_logprobs(client: openai.AsyncOpenAI, model_name: str):
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "what is 1+1?"},
    ]

    # Default max_logprobs is 20, so this should raise an error
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=10,
            logprobs=True,
            top_logprobs=21,
            stream=True,
        )
        async for chunk in stream:
            ...

    with pytest.raises(openai.BadRequestError):
        await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=10,
            logprobs=True,
            top_logprobs=30,
            stream=False,
        )

    # the server should still work afterwards
    chat_completion = await client.chat.completions.create(
        model=model_name, messages=messages, max_completion_tokens=10, stream=False
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name, prompt_logprobs",
    [(MODEL_NAME, 1), (MODEL_NAME, 0), (MODEL_NAME, -1), (MODEL_NAME, None)],
)
async def test_prompt_logprobs_chat(
    client: openai.AsyncOpenAI, model_name: str, prompt_logprobs: int | None
):
    params: dict = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020.",
            },
            {"role": "user", "content": "Where was it played?"},
        ],
        "model": model_name,
    }

    if prompt_logprobs is not None:
        params["extra_body"] = {"prompt_logprobs": prompt_logprobs}

    if prompt_logprobs is not None and prompt_logprobs < 0:
        with pytest.raises(BadRequestError):
            await client.chat.completions.create(**params)
    else:
        completion = await client.chat.completions.create(**params)
        if prompt_logprobs is not None:
            assert completion.prompt_logprobs is not None
            assert len(completion.prompt_logprobs) > 0
        else:
            assert completion.prompt_logprobs is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_more_than_one_prompt_logprobs_chat(
    client: openai.AsyncOpenAI, model_name: str
):
    params: dict = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020.",
            },
            {"role": "user", "content": "Where was it played?"},
        ],
        "model": model_name,
        "extra_body": {"prompt_logprobs": 1},
    }

    completion_1 = await client.chat.completions.create(**params)

    params["extra_body"] = {"prompt_logprobs": 2}
    completion_2 = await client.chat.completions.create(**params)

    assert len(completion_1.prompt_logprobs[3]) == 1
    assert len(completion_2.prompt_logprobs[3]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME, "zephyr-lora"],
)
async def test_single_chat_session(client: openai.AsyncOpenAI, model_name: str):
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "what is 1+1?"},
    ]
    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=5,
        logprobs=True,
        top_logprobs=5,
    )
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]

    assert choice.finish_reason == "length"
    assert chat_completion.usage == openai.types.CompletionUsage(
        completion_tokens=5, prompt_tokens=37, total_tokens=42
    )

    message = choice.message
    assert message.content is not None and len(message.content) >= 5
    assert message.role == "assistant"
    messages.append({"role": "assistant", "content": message.content})

    # test multi-turn dialogue
    messages.append({"role": "user", "content": "express your result in json"})
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=5,
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
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "what is 1+1?"},
    ]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
    )
    output = chat_completion.choices[0].message.content
    stop_reason = chat_completion.choices[0].finish_reason

    # test streaming
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        stream=True,
    )
    chunks: list[str] = []
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
async def test_chat_completion_stream_options(
    client: openai.AsyncOpenAI, model_name: str
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    # Test stream=True, stream_options={"include_usage": False}
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": False},
    )
    async for chunk in stream:
        assert chunk.usage is None

    # Test stream=True, stream_options={"include_usage": True,
    #                                   "continuous_usage_stats": False}}
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True, "continuous_usage_stats": False},
    )

    async for chunk in stream:
        if chunk.choices[0].finish_reason is None:
            assert chunk.usage is None
        else:
            assert chunk.usage is None
            final_chunk = await anext(stream)
            assert final_chunk.usage is not None
            assert final_chunk.usage.prompt_tokens > 0
            assert final_chunk.usage.completion_tokens > 0
            assert final_chunk.usage.total_tokens == (
                final_chunk.usage.prompt_tokens + final_chunk.usage.completion_tokens
            )
            assert final_chunk.choices == []

    # Test stream=False, stream_options={"include_usage": None}
    with pytest.raises(BadRequestError):
        await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=10,
            temperature=0.0,
            stream=False,
            stream_options={"include_usage": None},
        )

    # Test stream=False, stream_options={"include_usage": True}
    with pytest.raises(BadRequestError):
        await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=10,
            temperature=0.0,
            stream=False,
            stream_options={"include_usage": True},
        )

    # Test stream=True, stream_options={"include_usage": True,
    #                           "continuous_usage_stats": True}
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        extra_body=dict(min_tokens=10),
        temperature=0.0,
        stream=True,
        stream_options={
            "include_usage": True,
            "continuous_usage_stats": True,
        },
    )
    last_completion_tokens = 0
    async for chunk in stream:
        assert chunk.usage.prompt_tokens >= 0
        assert (
            last_completion_tokens == 0
            or chunk.usage.completion_tokens > last_completion_tokens
            or (
                not chunk.choices
                and chunk.usage.completion_tokens == last_completion_tokens
            )
        )
        assert chunk.usage.total_tokens == (
            chunk.usage.prompt_tokens + chunk.usage.completion_tokens
        )
        last_completion_tokens = chunk.usage.completion_tokens

    assert last_completion_tokens == 10


@pytest.mark.asyncio
async def test_structured_outputs_choice_chat(
    client: openai.AsyncOpenAI,
    sample_structured_outputs_choices,
):
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {
            "role": "user",
            "content": "The best language for type-safe systems programming is ",
        },
    ]
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.7,
        extra_body=dict(
            structured_outputs={"choice": sample_structured_outputs_choices}
        ),
    )
    choice1 = chat_completion.choices[0].message.content
    assert choice1 in sample_structured_outputs_choices

    messages.append({"role": "assistant", "content": choice1})
    messages.append({"role": "user", "content": "I disagree, pick another one"})
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.7,
        extra_body=dict(
            structured_outputs={"choice": sample_structured_outputs_choices}
        ),
    )
    choice2 = chat_completion.choices[0].message.content
    assert choice2 in sample_structured_outputs_choices
    assert choice1 != choice2


@pytest.mark.asyncio
async def test_structured_outputs_json_chat(
    client: openai.AsyncOpenAI,
    sample_json_schema,
):
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {
            "role": "user",
            "content": f"Give an example JSON for an employee profile that "
            f"fits this schema: {sample_json_schema}",
        },
    ]
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=1000,
        extra_body=dict(structured_outputs={"json": sample_json_schema}),
    )
    message = chat_completion.choices[0].message
    assert message.content is not None
    json1 = json.loads(message.content)
    jsonschema.validate(instance=json1, schema=sample_json_schema)

    messages.append({"role": "assistant", "content": message.content})
    messages.append(
        {"role": "user", "content": "Give me another one with a different name and age"}
    )
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=1000,
        extra_body=dict(structured_outputs={"json": sample_json_schema}),
    )
    message = chat_completion.choices[0].message
    assert message.content is not None
    json2 = json.loads(message.content)
    jsonschema.validate(instance=json2, schema=sample_json_schema)
    assert json1["name"] != json2["name"]
    assert json1["age"] != json2["age"]


@pytest.mark.asyncio
async def test_structured_outputs_regex_chat(
    client: openai.AsyncOpenAI,
    sample_regex,
):
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {
            "role": "user",
            "content": f"Give an example IP address with this regex: {sample_regex}",
        },
    ]
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=20,
        extra_body=dict(structured_outputs={"regex": sample_regex}),
    )
    ip1 = chat_completion.choices[0].message.content
    assert ip1 is not None
    assert re.fullmatch(sample_regex, ip1) is not None

    messages.append({"role": "assistant", "content": ip1})
    messages.append({"role": "user", "content": "Give me a different one"})
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=20,
        extra_body=dict(structured_outputs={"regex": sample_regex}),
    )
    ip2 = chat_completion.choices[0].message.content
    assert ip2 is not None
    assert re.fullmatch(sample_regex, ip2) is not None
    assert ip1 != ip2


@pytest.mark.asyncio
async def test_structured_outputs_type_error(client: openai.AsyncOpenAI):
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {
            "role": "user",
            "content": "The best language for type-safe systems programming is ",
        },
    ]

    with pytest.raises(openai.BadRequestError):
        _ = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_body=dict(structured_outputs={"regex": {1: "Python", 2: "C++"}}),
        )


@pytest.mark.asyncio
async def test_structured_outputs_choice_chat_logprobs(
    client: openai.AsyncOpenAI, sample_structured_outputs_choices
):
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {
            "role": "user",
            "content": "The best language for type-safe systems programming is ",
        },
    ]
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=10,
        logprobs=True,
        top_logprobs=5,
        extra_body=dict(
            structured_outputs={"choice": sample_structured_outputs_choices}
        ),
    )

    assert chat_completion.choices[0].logprobs is not None
    assert chat_completion.choices[0].logprobs.content is not None
    top_logprobs = chat_completion.choices[0].logprobs.content[0].top_logprobs

    # -9999.0 is the minimum logprob returned by OpenAI
    for item in top_logprobs:
        assert item.logprob >= -9999.0, f"Failed (top_logprobs={top_logprobs})"


@pytest.mark.asyncio
async def test_response_format_json_object(client: openai.AsyncOpenAI):
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "what is 1+1? please respond with a JSON object, "
                        'the format is {"result": 2}'
                    ),
                }
            ],
            response_format={"type": "json_object"},
        )

        content = resp.choices[0].message.content
        assert content is not None

        loaded = json.loads(content)
        assert loaded == {"result": 2}, loaded


@pytest.mark.asyncio
async def test_response_format_json_schema(client: openai.AsyncOpenAI):
    prompt = 'what is 1+1? The format is "result": 2'
    # Check that this prompt cannot lead to a valid JSON without json_schema
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content
        assert content is not None
        with pytest.raises((json.JSONDecodeError, AssertionError)):
            loaded = json.loads(content)
            assert loaded == {"result": 2}, loaded

    for _ in range(2):
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "foo_test",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "result": {"type": "integer"},
                        },
                    },
                },
            },
        )

        content = resp.choices[0].message.content
        assert content is not None

        loaded = json.loads(content)
        assert loaded == {"result": 2}, loaded


@pytest.mark.asyncio
async def test_response_format_text(client: openai.AsyncOpenAI):
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "what is 1+1?",
                }
            ],
            max_completion_tokens=10,
            response_format={"type": "text"},
        )

        content = resp.choices[0].message.content
        assert content is not None


@pytest.mark.asyncio
async def test_extra_fields_allowed(client: openai.AsyncOpenAI):
    resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "what is 1+1?",
                "extra_field": "0",
            }
        ],  # type: ignore
        temperature=0,
        seed=0,
    )

    content = resp.choices[0].message.content
    assert content is not None


@pytest.mark.asyncio
async def test_complex_message_content(client: openai.AsyncOpenAI):
    content = [
        {
            "type": "text",
            "text": "what is 1+1? please provide the result without any other text.",
        }
    ]
    resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        temperature=0,
        seed=0,
    )
    content = resp.choices[0].message.content
    assert content == "2"


@pytest.mark.asyncio
async def test_custom_role(client: openai.AsyncOpenAI):
    # Not sure how the model handles custom roles so we just check that
    # both string and complex message content are handled in the same way

    resp1 = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "my-custom-role",
                "content": "what is 1+1?",
            }
        ],  # type: ignore
        temperature=0,
        seed=0,
    )

    resp2 = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "my-custom-role",
                "content": [{"type": "text", "text": "what is 1+1?"}],
            }
        ],  # type: ignore
        temperature=0,
        seed=0,
    )

    content1 = resp1.choices[0].message.content
    content2 = resp2.choices[0].message.content
    assert content1 == content2


@pytest.mark.asyncio
async def test_long_seed(client: openai.AsyncOpenAI):
    for seed in [torch.iinfo(torch.long).min - 1, torch.iinfo(torch.long).max + 1]:
        with pytest.raises(BadRequestError) as exc_info:
            await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    }
                ],
                temperature=0,
                seed=seed,
            )

        assert (
            "greater_than_equal" in exc_info.value.message
            or "less_than_equal" in exc_info.value.message
        )


@pytest.mark.asyncio
async def test_invocations(server: RemoteOpenAIServer, client: openai.AsyncOpenAI):
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "what is 1+1?"},
    ]

    request_args = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_completion_tokens": 5,
        "temperature": 0.0,
        "logprobs": False,
    }

    chat_completion = await client.chat.completions.create(**request_args)

    invocation_response = requests.post(
        server.url_for("invocations"), json=request_args
    )
    invocation_response.raise_for_status()

    chat_output = chat_completion.model_dump()
    invocation_output = invocation_response.json()

    assert chat_output.keys() == invocation_output.keys()
    assert chat_output["choices"] == invocation_output["choices"]


# Test n parameter for chat completions
# Tests that the n parameter works correctly for regular sampling
# (non-beam search) in chat completions, addressing issue #34305.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_chat_completion_n_parameter_non_streaming(
    client: openai.AsyncOpenAI, model_name: str
):
    """Test that n parameter returns multiple choices for non-streaming requests."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the opposite of big?"},
    ]

    # Test with n=3
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=20,
        temperature=0.7,
        n=3,
        stream=False,
    )

    assert len(chat_completion.choices) == 3

    # Verify each choice has content and correct index
    for i, choice in enumerate(chat_completion.choices):
        assert choice.index == i
        assert choice.message.content is not None
        assert len(choice.message.content) > 0

    # Verify all responses are different (highly likely with temperature > 0)
    contents = [choice.message.content for choice in chat_completion.choices]
    assert len(set(contents)) > 1, "Expected different responses with n=3"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_chat_completion_n_parameter_streaming(
    client: openai.AsyncOpenAI, model_name: str
):
    """Test that n parameter returns multiple choices for streaming requests."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=15,
        temperature=0.7,
        n=2,
        stream=True,
    )

    # Collect all chunks using defaultdict for dynamic handling
    chunks_by_index = defaultdict(list)
    async for chunk in stream:
        for choice in chunk.choices:
            if choice.delta.content:
                chunks_by_index[choice.index].append(choice.delta.content)

    # Verify both choices received content
    assert len(chunks_by_index[0]) > 0, "Choice 0 received no content chunks"
    assert len(chunks_by_index[1]) > 0, "Choice 1 received no content chunks"

    # Reconstruct full responses
    response_0 = "".join(chunks_by_index[0])
    response_1 = "".join(chunks_by_index[1])

    assert len(response_0) > 0, "Choice 0 has empty response"
    assert len(response_1) > 0, "Choice 1 has empty response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_chat_completion_n_with_seed(client: openai.AsyncOpenAI, model_name: str):
    """Test that n parameter works correctly with seed parameter."""
    messages = [
        {"role": "user", "content": "Say hello."},
    ]

    # Test that seed parameter is accepted and works with n > 1
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.8,
        n=2,
        seed=42,
        stream=False,
    )

    # Verify we get n=2 choices
    assert len(chat_completion.choices) == 2

    # Verify both choices have valid content
    for i, choice in enumerate(chat_completion.choices):
        assert choice.index == i
        assert choice.message.content is not None
        assert len(choice.message.content) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_chat_completion_n_equals_1(client: openai.AsyncOpenAI, model_name: str):
    """Test that n=1 (default) still works correctly."""
    messages = [
        {"role": "user", "content": "Hello!"},
    ]

    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.7,
        n=1,
        stream=False,
    )

    assert len(chat_completion.choices) == 1
    assert chat_completion.choices[0].index == 0
    assert chat_completion.choices[0].message.content is not None


# Unit tests for n parameter in ChatCompletionRequest.to_sampling_params()
def test_chat_completion_request_n_parameter_to_sampling_params():
    """Test that n parameter is correctly passed to SamplingParams."""
    # Test with n=3
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        n=3,
        max_tokens=10,
    )

    sampling_params = request.to_sampling_params(
        max_tokens=10,
        default_sampling_params={},
    )

    assert isinstance(sampling_params, SamplingParams)
    assert sampling_params.n == 3, f"Expected n=3, got n={sampling_params.n}"


def test_chat_completion_request_n_parameter_default():
    """Test that n parameter defaults to 1."""
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        # n not specified, should default to 1
        max_tokens=10,
    )

    assert request.n == 1, "n should default to 1"
    sampling_params = request.to_sampling_params(
        max_tokens=10,
        default_sampling_params={},
    )

    # SamplingParams.from_optional converts None to 1
    assert sampling_params.n == 1, f"Expected n=1 (default), got n={sampling_params.n}"


def test_chat_completion_request_n_parameter_various_values():
    """Test n parameter with various values."""
    for n_value in [1, 2, 5, 10]:
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Test"}],
            n=n_value,
            max_tokens=10,
        )

        sampling_params = request.to_sampling_params(
            max_tokens=10,
            default_sampling_params={},
        )

        assert sampling_params.n == n_value, (
            f"Expected n={n_value}, got n={sampling_params.n}"
        )
