# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile

import openai
import pytest

from tests.utils import RemoteOpenAIServer
from vllm import envs
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "qwen3_06b")


@pytest.fixture(scope="module")
def server():
    global MODEL_PATH
    MODEL_PATH = download_weights_from_hf(
        MODEL_NAME,
        allow_patterns=["*"],
        cache_dir=MODEL_PATH,
        ignore_patterns=["tokenizer*", "vocab*", "*.safetensors"],
    )
    args = [
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        "--skip-tokenizer-init",
        "--load-format",
        "dummy",
    ]
    with RemoteOpenAIServer(MODEL_PATH, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_token_in_token_out_and_logprobs(server):
    """Test token-in-token-out and token_ids align with prompt_logprobs
    & logprobs when return_tokens_as_token_ids is enabled.
    """
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
    text = "Hello, world! How are you today?"
    token_ids = tokenizer.encode(text)
    async with server.get_async_client() as client:
        # Test with both return_token_ids and return_tokens_as_token_ids enabled
        completion = await client.completions.create(
            model=MODEL_PATH,
            prompt=token_ids,
            max_tokens=20,
            temperature=0,
            echo=True,
            extra_body={
                "return_token_ids": True,
            },
        )

        # Verify all fields are present
        assert (
            completion.choices[0].token_ids is not None
            and 0 < len(completion.choices[0].token_ids) <= 20
        )
        assert completion.choices[0].prompt_token_ids is not None

        # Decode prompt tokens
        if completion.choices[0].prompt_token_ids:
            prompt_text = tokenizer.decode(completion.choices[0].prompt_token_ids)
            # The decoded prompt should match or close to original prompt
            assert prompt_text == text


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["END", ["END", "STOP"]])
@pytest.mark.parametrize("stream", [False, True])
async def test_rejects_stop_strings_without_tokenizer(server, stop, stream):
    async with server.get_async_client() as client:
        with pytest.raises(openai.BadRequestError, match="stop strings require") as exc:
            await client.completions.create(
                model=MODEL_PATH,
                prompt=[1, 2, 3],
                max_tokens=1,
                temperature=0,
                stop=stop,
                stream=stream,
            )
    if not envs.VLLM_USE_RUST_FRONTEND:
        assert exc.value.param == "stop"


@pytest.mark.asyncio
async def test_stop_token_ids_work_without_tokenizer(server):
    token_ids = [4]
    options = {"return_token_ids": True, "allowed_token_ids": token_ids}
    async with server.get_async_client() as client:
        completion = await client.completions.create(
            model=MODEL_PATH,
            prompt=[1, 2, 3],
            max_tokens=2,
            temperature=0,
            extra_body=options,
        )
        stopped = await client.completions.create(
            model=MODEL_PATH,
            prompt=[1, 2, 3],
            max_tokens=2,
            temperature=0,
            extra_body={**options, "stop_token_ids": token_ids},
        )
    assert completion.choices[0].token_ids == [4, 4]
    assert completion.choices[0].finish_reason == "length"
    assert stopped.choices[0].token_ids == token_ids
    assert stopped.choices[0].finish_reason == "stop"
