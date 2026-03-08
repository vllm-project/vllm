# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import io
import json

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import torch

# downloading lora to test lora requests
from openai import BadRequestError
from transformers import AutoConfig

from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "facebook/opt-125m"
LORA_SERVING_MODEL_NAME = "opt125m-lora"

CONFIG = AutoConfig.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module", params=["use-lora"])
def default_server_args(
    request: pytest.FixtureRequest, opt125_lora_files: str
) -> list[str]:
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        # Prompt Embeds server args
        "--enable-prompt-embeds",
    ]

    if request.param == "use-lora":
        lora_module_1 = {
            "name": LORA_SERVING_MODEL_NAME,
            "path": opt125_lora_files,
            "base_model_name": MODEL_NAME,
        }

        args.extend(
            [
                "--enable-lora",
                "--lora-module",
                json.dumps(lora_module_1),
                "--max-lora-rank",
                "64",
                "--max-cpu-loras",
                "2",
            ]
        )

    return args


EXAMPLE_PROMPTS = [
    "Hello, my name is",
    "What is an LLM?",
]


def _encode_embeds(embeds: torch.Tensor):
    buffer = io.BytesIO()
    torch.save(embeds, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture(scope="module")
def example_prompt_embeds(hf_runner):
    """Create example embeddings and return them as base64 encoded string."""
    with hf_runner(MODEL_NAME) as hf_model:
        example_embeddings = hf_model.get_prompt_embeddings(EXAMPLE_PROMPTS)

    return [_encode_embeds(item) for item in example_embeddings]


@pytest.fixture(scope="module", params=["", "--disable-frontend-multiprocessing"])
def server_with_prompt_embeds(default_server_args, request):
    if request.param:
        default_server_args.append(request.param)

    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client_with_prompt_embeds(server_with_prompt_embeds):
    async with server_with_prompt_embeds.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME, LORA_SERVING_MODEL_NAME])
async def test_completions_with_prompt_embeds(
    example_prompt_embeds,
    client_with_prompt_embeds: openai.AsyncOpenAI,
    model_name: str,
):
    encoded_embeds, encoded_embeds2 = example_prompt_embeds

    # Test case: Single prompt embeds input
    completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt=None,
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": encoded_embeds},
    )
    assert len(completion.choices[0].text) >= 1
    assert completion.choices[0].prompt_logprobs is None

    # Test case: batch completion with prompt_embeds
    completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt=None,
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": [encoded_embeds, encoded_embeds2]},
    )
    assert len(completion.choices) == 2
    assert len(completion.choices[0].text) >= 1
    assert len(completion.choices[1].text) >= 1

    # Test case: streaming with prompt_embeds
    single_completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt=None,
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": encoded_embeds},
    )
    single_output = single_completion.choices[0].text

    stream = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt=None,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        extra_body={"prompt_embeds": encoded_embeds},
    )
    chunks = []
    finish_reason_count = 0
    async for chunk in stream:
        chunks.append(chunk.choices[0].text)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == "length"
    assert chunk.choices[0].text
    assert "".join(chunks) == single_output

    # Test case: batch streaming with prompt_embeds
    stream = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt=None,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        extra_body={"prompt_embeds": [encoded_embeds, encoded_embeds2]},
    )
    chunks_stream_embeds: list[list[str]] = [[], []]
    finish_reason_count = 0
    async for chunk in stream:
        chunks_stream_embeds[chunk.choices[0].index].append(chunk.choices[0].text)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    assert finish_reason_count == 2
    assert chunk.choices[0].finish_reason == "length"
    assert chunk.choices[0].text
    assert len(chunks_stream_embeds[0]) > 0
    assert len(chunks_stream_embeds[1]) > 0

    # Test case: mixed text and prompt_embeds
    completion_mixed = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="This is a prompt",
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": encoded_embeds},
    )
    assert len(completion.choices) == 2
    completion_text_only = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="This is a prompt",
        max_tokens=5,
        temperature=0.0,
    )
    completion_embeds_only = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt=None,
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": encoded_embeds},
    )
    # Embeddings responses should be handled first
    assert completion_mixed.choices[0].text == completion_embeds_only.choices[0].text
    assert completion_mixed.choices[1].text == completion_text_only.choices[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME, LORA_SERVING_MODEL_NAME])
async def test_completions_errors_with_prompt_embeds(
    client_with_prompt_embeds: openai.AsyncOpenAI, model_name: str
):
    # Test error case: invalid prompt_embeds
    with pytest.raises(BadRequestError):
        await client_with_prompt_embeds.completions.create(
            prompt=None,
            model=model_name,
            max_tokens=5,
            temperature=0.0,
            extra_body={"prompt_embeds": "invalid_base64"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("logprobs_arg", [1, 0])
@pytest.mark.parametrize("model_name", [MODEL_NAME, LORA_SERVING_MODEL_NAME])
async def test_completions_with_logprobs_and_prompt_embeds(
    example_prompt_embeds,
    client_with_prompt_embeds: openai.AsyncOpenAI,
    logprobs_arg: int,
    model_name: str,
):
    encoded_embeds, encoded_embeds2 = example_prompt_embeds

    # Test case: Logprobs using prompt_embeds
    completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt=None,
        max_tokens=5,
        temperature=0.0,
        echo=False,
        logprobs=logprobs_arg,
        extra_body={"prompt_embeds": encoded_embeds},
    )

    logprobs = completion.choices[0].logprobs
    assert logprobs is not None
    assert len(logprobs.text_offset) == 5
    assert len(logprobs.token_logprobs) == 5
    assert len(logprobs.top_logprobs) == 5
    for top_logprobs in logprobs.top_logprobs[1:]:
        assert max(logprobs_arg, 1) <= len(top_logprobs) <= logprobs_arg + 1
    assert len(logprobs.tokens) == 5

    # Test case: Log probs with batch completion and prompt_embeds
    completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt=None,
        max_tokens=5,
        temperature=0.0,
        echo=False,
        logprobs=logprobs_arg,
        extra_body={"prompt_embeds": [encoded_embeds, encoded_embeds2]},
    )

    assert len(completion.choices) == 2
    for choice in completion.choices:
        logprobs = choice.logprobs
        assert logprobs is not None
        assert len(logprobs.text_offset) == 5
        assert len(logprobs.token_logprobs) == 5
        assert len(logprobs.top_logprobs) == 5
        for top_logprobs in logprobs.top_logprobs[1:]:
            assert max(logprobs_arg, 1) <= len(top_logprobs) <= logprobs_arg + 1
        assert len(logprobs.tokens) == 5


@pytest.mark.asyncio
async def test_prompt_logprobs_raises_error(
    example_prompt_embeds,
    client_with_prompt_embeds: openai.AsyncOpenAI,
):
    encoded_embeds, _ = example_prompt_embeds

    with pytest.raises(BadRequestError, match="not compatible"):
        await client_with_prompt_embeds.completions.create(
            model=MODEL_NAME,
            prompt=None,
            max_tokens=5,
            temperature=0.0,
            extra_body={"prompt_embeds": encoded_embeds, "prompt_logprobs": True},
        )


@pytest.mark.asyncio
async def test_empty_prompt_embeds(
    client_with_prompt_embeds: openai.AsyncOpenAI,
) -> None:
    await client_with_prompt_embeds.completions.create(
        model=MODEL_NAME,
        prompt="Hello",
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": []},
    )
