# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import io
import shutil
from tempfile import TemporaryDirectory

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import torch
# downloading lora to test lora requests
from huggingface_hub import snapshot_download
from openai import BadRequestError
from transformers import AutoConfig, AutoTokenizer

from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
LORA_NAME = "typeof/zephyr-7b-beta-lora"

CONFIG = AutoConfig.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def zephyr_lora_files():
    return snapshot_download(repo_id=LORA_NAME)


@pytest.fixture(scope="module")
def zephyr_lora_added_tokens_files(zephyr_lora_files):
    tmp_dir = TemporaryDirectory()
    tmp_model_dir = f"{tmp_dir.name}/zephyr"
    shutil.copytree(zephyr_lora_files, tmp_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Copy tokenizer to adapter and add some unique tokens
    # 32000, 32001, 32002
    added = tokenizer.add_tokens(["vllm1", "vllm2", "vllm3"],
                                 special_tokens=True)
    assert added == 3
    tokenizer.save_pretrained(tmp_model_dir)
    yield tmp_model_dir
    tmp_dir.cleanup()


@pytest.fixture(scope="module")
def default_server_args(
    zephyr_lora_files,
    zephyr_lora_added_tokens_files,
) -> list[str]:
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        # Prompt Embeds server args
        "--enable-prompt-embeds",
        "--no-enable-chunked-prefill",
    ]


@pytest.fixture(scope="module",
                params=["", "--disable-frontend-multiprocessing"])
def server_with_prompt_embeds(default_server_args, request):
    if request.param:
        default_server_args.append(request.param)

    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client_with_prompt_embeds(server_with_prompt_embeds):
    async with server_with_prompt_embeds.get_async_client() as async_client:
        yield async_client


def create_dummy_embeds(num_tokens: int = 5) -> str:
    """Create dummy embeddings and return them as base64 encoded string."""
    dummy_embeds = torch.randn(num_tokens, CONFIG.hidden_size)
    buffer = io.BytesIO()
    torch.save(dummy_embeds, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_completions_with_prompt_embeds(
        client_with_prompt_embeds: openai.AsyncOpenAI, model_name: str):
    # Test case: Single prompt embeds input
    encoded_embeds = create_dummy_embeds()
    completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": encoded_embeds})
    assert len(completion.choices[0].text) >= 1
    assert completion.choices[0].prompt_logprobs is None

    # Test case: batch completion with prompt_embeds
    encoded_embeds2 = create_dummy_embeds()
    completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": [encoded_embeds, encoded_embeds2]})
    assert len(completion.choices) == 2
    assert len(completion.choices[0].text) >= 1
    assert len(completion.choices[1].text) >= 1

    # Test case: streaming with prompt_embeds
    encoded_embeds = create_dummy_embeds()
    single_completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": encoded_embeds})
    single_output = single_completion.choices[0].text

    stream = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        stream=True,
        extra_body={"prompt_embeds": encoded_embeds})
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
    encoded_embeds2 = create_dummy_embeds()
    stream = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        stream=True,
        extra_body={"prompt_embeds": [encoded_embeds, encoded_embeds2]})
    chunks_stream_embeds: list[list[str]] = [[], []]
    finish_reason_count = 0
    async for chunk in stream:
        chunks_stream_embeds[chunk.choices[0].index].append(
            chunk.choices[0].text)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    assert finish_reason_count == 2
    assert chunk.choices[0].finish_reason == "length"
    assert chunk.choices[0].text
    assert len(chunks_stream_embeds[0]) > 0
    assert len(chunks_stream_embeds[1]) > 0

    # Test case: mixed text and prompt_embeds
    encoded_embeds = create_dummy_embeds()
    completion_mixed = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="This is a prompt",
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": encoded_embeds})
    assert len(completion.choices) == 2
    completion_text_only = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="This is a prompt",
        max_tokens=5,
        temperature=0.0,
    )
    completion_embeds_only = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_embeds": encoded_embeds})
    # Embeddings responses should be handled first
    assert completion_mixed.choices[0].text == completion_embeds_only.choices[
        0].text
    assert completion_mixed.choices[1].text == completion_text_only.choices[
        0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_completions_errors_with_prompt_embeds(
        client_with_prompt_embeds: openai.AsyncOpenAI, model_name: str):
    # Test error case: invalid prompt_embeds
    with pytest.raises(BadRequestError):
        await client_with_prompt_embeds.completions.create(
            prompt="",
            model=model_name,
            max_tokens=5,
            temperature=0.0,
            extra_body={"prompt_embeds": "invalid_base64"})


@pytest.mark.asyncio
@pytest.mark.parametrize("logprobs_arg", [1, 0])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_completions_with_logprobs_and_prompt_embeds(
        client_with_prompt_embeds: openai.AsyncOpenAI, logprobs_arg: int,
        model_name: str):
    # Test case: Logprobs using prompt_embeds
    encoded_embeds = create_dummy_embeds()
    completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        echo=False,
        logprobs=logprobs_arg,
        extra_body={"prompt_embeds": encoded_embeds})

    logprobs = completion.choices[0].logprobs
    assert logprobs is not None
    assert len(logprobs.text_offset) == 5
    assert len(logprobs.token_logprobs) == 5
    assert len(logprobs.top_logprobs) == 5
    for top_logprobs in logprobs.top_logprobs[1:]:
        assert max(logprobs_arg, 1) <= len(top_logprobs) <= logprobs_arg + 1
    assert len(logprobs.tokens) == 5

    # Test case: Log probs with batch completion and prompt_embeds
    encoded_embeds2 = create_dummy_embeds()
    completion = await client_with_prompt_embeds.completions.create(
        model=model_name,
        prompt="",  # Add empty prompt as required parameter
        max_tokens=5,
        temperature=0.0,
        echo=False,
        logprobs=logprobs_arg,
        extra_body={"prompt_embeds": [encoded_embeds, encoded_embeds2]})

    assert len(completion.choices) == 2
    for choice in completion.choices:
        logprobs = choice.logprobs
        assert logprobs is not None
        assert len(logprobs.text_offset) == 5
        assert len(logprobs.token_logprobs) == 5
        assert len(logprobs.top_logprobs) == 5
        for top_logprobs in logprobs.top_logprobs[1:]:
            assert max(logprobs_arg,
                       1) <= len(top_logprobs) <= logprobs_arg + 1
        assert len(logprobs.tokens) == 5
